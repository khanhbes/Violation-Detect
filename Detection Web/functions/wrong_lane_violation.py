"""
wrong_lane.py (FULL) - Minimize false positives in real traffic
===============================================================
Mục tiêu: hạn chế phạt sai nhất có thể trong thực tế

A) 5 giây đầu: HIỂN THỊ HẾT MASK (giống detect_sidewalk_violation.py)
B) Sau 5 giây: FREEZE
   - Stopline vẽ cố định khớp mask (giống redlight_violation.py): fit line bằng RANSAC từ mask stopline
   - Vạch lane cố định: dash_white_line + solid_white_line (giữ như bản bạn đã ổn)
   - Lane = khoảng giữa các vạch biên (dash/solid)
   - Luật lane: lấy từ arrow masks (1..5), 4/5 ưu tiên "Straight_Or_Left"/"Straight_Or_Right"
C) Khi xe đang đi trong lane: KHÔNG phạt WrongDir
D) Chỉ khi xe CẮT stopline mới xét WrongDir so với luật của lane BEFORE crossing
E) ChamVach:
   - solid_white_line & solid_yellow_line: 2 bánh (2 góc đáy bbox) chạm vạch => "Cham Vach"
   - dash_*: chạm không lỗi
F) Hạn chế phạt sai:
   - Không dùng slope. Dùng VECTOR/ANGLE.
   - Reference hướng đi thẳng dùng theo "trục lane" (lane centerline from fixed boundaries), fallback stopline normal.
   - "Straight" cho phép lệch một chút: STRAIGHT_ANGLE_DEG.
   - Vùng mơ hồ giữa STRAIGHT_ANGLE và TURN_ANGLE => ưu tiên STRAIGHT (giảm false)
   - Thêm safeguard: nếu luật có STRAIGHT và góc < SAFE_STRAIGHT_MAX thì ép STRAIGHT (tránh phạt sai lane ngoài)
   - Debug overlay + console để truy lỗi.

Phím:
- q : quit
- d : toggle debug

Uses shared Config class from config/config.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import cv2
import numpy as np
from ultralytics import YOLO

# Add parent directory to path so we can import config module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared config and draw utilities
from config.config import config
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, draw_calibration_hud, save_violation_snapshot


# =========================
# LANE-SPECIFIC CONFIG
# =========================
IOU_THRESHOLD = config.IOU_THRESHOLD
IMG_SIZE = config.IMG_SIZE
TRACKER = config.TRACKER

CALIBRATION_DURATION_SEC = 5.0
CONF_CALIB = 0.25
CONF_TRACK = 0.45

WINDOW_NAME = "Wrong Lane Detection"

# Seg classes from config
DASH_WHITE = config.DASHED_WHITE_LINE[0] if config.DASHED_WHITE_LINE else 7
DASH_YELLOW = config.DASHED_YELLOW_LINE[0] if config.DASHED_YELLOW_LINE else 8
SOLID_WHITE = config.SOLID_WHITE_LINE[0] if config.SOLID_WHITE_LINE else 37
SOLID_YELLOW = config.SOLID_YELLOW_LINE[0] if config.SOLID_YELLOW_LINE else 38
STOPLINE_ID = config.STOPLINE_CLASS[0] if config.STOPLINE_CLASS else 39

# Arrows (dataset)
ARROW_CLASSES = {1, 2, 3, 4, 5}

# Vehicles from config
VEHICLE_CLASSES = set(config.VEHICLE_CLASSES)
PRIORITY_VEHICLES = {0, 9, 26}  # ambulance, fire_truck, police_car

# Colors (BGR) - Lane-specific colors hardcoded
COLOR_DASH_WHITE = (255, 255, 200)     # Light cyan
COLOR_DASH_YELLOW = (0, 200, 255)      # Yellow
COLOR_SOLID_WHITE = (255, 255, 255)    # White
COLOR_SOLID_YELLOW = (0, 140, 255)     # Orange-yellow
COLOR_STOPLINE = config.COLOR_STOPLINE # From config
COLOR_ARROW = (255, 0, 255)            # Magenta
COLOR_SAFE = config.COLOR_SAFE         # From config
COLOR_VIOLATION = config.COLOR_VIOLATION # From config

# Cham vach thresholds (px)
LINE_TOUCH_DIST_PX = 6.0
LINE_TOUCH_Y_MARGIN = 15

# Motion window (used only when cross stopline)
ACTION_WINDOW = 30
MIN_ACTION_FRAMES = 12
STEP_DMIN = 2.0
MIN_MOVE_MAG_SUM = 80.0

# Direction thresholds (degrees)
# "vuông góc hoặc sai lệch 1 chút thì vẫn tính thẳng"
STRAIGHT_ANGLE_DEG = 26.0
TURN_ANGLE_DEG = 34.0
SAFE_STRAIGHT_MAX = 40.0   # safeguard: if lane rule allows STRAIGHT and angle < this => treat STRAIGHT

DEBUG_DEFAULT = True


# =========================
# UTILITIES
# =========================
def mask_data_to_binary(mask_2d: np.ndarray, w: int, h: int, thr: float = 0.5) -> np.ndarray:
    m = mask_2d.astype(np.float32)
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return (m > thr).astype(np.uint8)


def smooth_binary(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return (m > 0).astype(np.uint8)


def polygon_centroid(poly_xy: np.ndarray) -> Optional[Tuple[float, float]]:
    if poly_xy is None or len(poly_xy) < 3:
        return None
    pts = poly_xy.astype(np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def draw_polygon_overlay(frame: np.ndarray, poly: np.ndarray,
                         fill_color: Tuple[int, int, int],
                         alpha: float = 0.30,
                         border_color: Tuple[int, int, int] = (255, 255, 255),
                         border_thickness: int = 2) -> np.ndarray:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly.astype(np.int32)], fill_color)
    out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(out, [poly.astype(np.int32)], True, border_color, border_thickness)
    return out


# =========================
# LINE FIT (RANSAC)
# =========================
def fit_line_ransac(points_xy: np.ndarray,
                    max_iters: int = 600,
                    inlier_thresh: float = 3.5,
                    min_inliers: int = 220) -> Optional[Tuple[float, float, float]]:
    if points_xy is None or len(points_xy) < 2:
        return None

    pts = points_xy.astype(np.float32)
    n = len(pts)
    if n > 25000:
        idx = np.random.choice(n, size=25000, replace=False)
        pts = pts[idx]
        n = len(pts)

    rng = np.random.default_rng(42)
    best_inliers = 0
    best_abc = None

    for _ in range(max_iters):
        i1, i2 = rng.integers(0, n, size=2)
        if i1 == i2:
            continue
        x1, y1 = pts[i1]
        x2, y2 = pts[i2]
        if abs(x2 - x1) + abs(y2 - y1) < 1e-6:
            continue

        a = (y1 - y2)
        b = (x2 - x1)
        c = (x1 * y2 - x2 * y1)

        norm = float(np.sqrt(a * a + b * b) + 1e-12)
        a, b, c = a / norm, b / norm, c / norm

        dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
        inliers = int(np.sum(dist <= inlier_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_abc = (float(a), float(b), float(c))

    if best_abc is None or best_inliers < min_inliers:
        return None
    return best_abc


def signed_distance(px: float, py: float, a: float, b: float, c: float) -> float:
    return (a * px + b * py + c) / (np.sqrt(a * a + b * b) + 1e-12)


def line_x_at_y(a: float, b: float, c: float, y: float) -> Optional[float]:
    if abs(a) < 1e-9:
        return None
    return float(-(b * y + c) / a)


# =========================
# STOPLINE CALIBRATOR (copy style from redlight)
# =========================
class StoplineCalibrator:
    def __init__(self, duration: float = 5.0):
        self.duration = duration
        self.start_time: Optional[float] = None
        self.points: List[np.ndarray] = []
        self.line_abc: Optional[Tuple[float, float, float]] = None
        self.sign_flip: float = 1.0
        self.min_x: int = 0
        self.max_x: int = 0

    def start(self):
        self.start_time = time.time()

    def is_calibrated(self) -> bool:
        return self.line_abc is not None

    def add_mask_points(self, mask01: np.ndarray):
        ys, xs = np.where(mask01 > 0)
        if len(xs) > 0:
            pts = np.stack([xs, ys], axis=1)
            self.points.append(pts)

    def maybe_finish(self, frame_h: int, frame_w: int, min_points: int = 800):
        if self.start_time is None or self.is_calibrated():
            return
        elapsed = time.time() - self.start_time
        if elapsed < self.duration:
            return
        if not self.points:
            return

        all_pts = np.concatenate(self.points, axis=0)
        if len(all_pts) < min_points:
            return

        self.min_x = int(np.min(all_pts[:, 0]))
        self.max_x = int(np.max(all_pts[:, 0]))

        abc = fit_line_ransac(all_pts, max_iters=800, inlier_thresh=3.0, min_inliers=max(260, min_points // 2))
        if abc is None:
            return

        self.line_abc = abc
        a, b, c = abc

        # sign flip so that "before stopline" is negative at bottom center
        ref_x, ref_y = frame_w / 2.0, frame_h - 10.0
        s = a * ref_x + b * ref_y + c
        self.sign_flip = -1.0 if s > 0 else 1.0

        print(f"[✓] Stopline calibrated: {a:.4f}x + {b:.4f}y + {c:.2f} = 0 | x-range=({self.min_x},{self.max_x})")


# =========================
# FROZEN LINES
# =========================
@dataclass
class FrozenLine:
    cls_id: int
    abc: Tuple[float, float, float]
    y_min: int
    y_max: int
    color: Tuple[int, int, int]

    def x_at(self, y: float) -> Optional[float]:
        a, b, c = self.abc
        return line_x_at_y(a, b, c, y)


def road_roi_bounds(w: int, h: int) -> Tuple[int, int]:
    return int(0.18 * w), int(0.94 * w)


def class_color(cls_id: int) -> Tuple[int, int, int]:
    if cls_id == DASH_WHITE:
        return COLOR_DASH_WHITE
    if cls_id == DASH_YELLOW:
        return COLOR_DASH_YELLOW
    if cls_id == SOLID_WHITE:
        return COLOR_SOLID_WHITE
    if cls_id == SOLID_YELLOW:
        return COLOR_SOLID_YELLOW
    return (255, 255, 255)


def extract_component_lines(union01: np.ndarray,
                            cls_id: int,
                            frame_w: int,
                            frame_h: int,
                            min_area: int = 550,
                            min_points: int = 170) -> List[FrozenLine]:
    lines: List[FrozenLine] = []
    if union01 is None:
        return lines

    roi_l, roi_r = road_roi_bounds(frame_w, frame_h)
    bin01 = (union01 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)

    for k in range(1, num):
        x, y, w, h, area = stats[k]
        if area < min_area:
            continue

        x_mid = x + w * 0.5
        if not (roi_l <= x_mid <= roi_r):
            continue

        # bottom-half preference
        if (y + h) < 0.55 * frame_h:
            continue

        comp = (labels == k)
        ys, xs = np.where(comp)
        if len(xs) < min_points:
            continue

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        abc = fit_line_ransac(
            pts,
            max_iters=600,
            inlier_thresh=3.6,
            min_inliers=min(320, max(150, len(pts) // 6))
        )
        if abc is None:
            continue

        y_min = int(max(0, y))
        y_max = int(min(frame_h - 1, y + h - 1))
        lines.append(FrozenLine(cls_id=cls_id, abc=abc, y_min=y_min, y_max=y_max, color=class_color(cls_id)))

    return lines


def draw_lines(frame: np.ndarray, lines: List[FrozenLine], thickness: int = 3) -> None:
    h, w = frame.shape[:2]
    for ln in lines:
        a, b, c = ln.abc
        y1, y2 = ln.y_min, ln.y_max
        x1 = line_x_at_y(a, b, c, y1)
        x2 = line_x_at_y(a, b, c, y2)
        if x1 is None or x2 is None:
            continue
        p1 = (int(np.clip(round(x1), 0, w - 1)), int(np.clip(y1, 0, h - 1)))
        p2 = (int(np.clip(round(x2), 0, w - 1)), int(np.clip(y2, 0, h - 1)))
        cv2.line(frame, p1, p2, ln.color, thickness)


# =========================
# LANE RULES FROM ARROWS
# =========================
def allowed_to_label(allowed: Set[str]) -> str:
    if allowed == {"LEFT"}:
        return "Only_Left"
    if allowed == {"RIGHT"}:
        return "Only_Right"
    if allowed == {"STRAIGHT"}:
        return "Only_Straight"
    if allowed == {"STRAIGHT", "LEFT"}:
        return "Straight_Or_Left"
    if allowed == {"STRAIGHT", "RIGHT"}:
        return "Straight_Or_Right"
    if not allowed:
        return "UNKNOWN"
    return "OR".join(sorted(allowed))


def arrow_allowed_raw(cls_id: int) -> Set[str]:
    if cls_id == 1:
        return {"LEFT"}
    if cls_id == 2:
        return {"RIGHT"}
    if cls_id == 3:
        return {"STRAIGHT"}
    if cls_id == 4:
        return {"STRAIGHT", "LEFT"}
    if cls_id == 5:
        return {"STRAIGHT", "RIGHT"}
    return set()


def lane_of_point_by_y(px: float, py: float, boundaries: List[FrozenLine]) -> Optional[int]:
    xs = []
    for ln in boundaries:
        x = ln.x_at(py)
        if x is not None and np.isfinite(x):
            xs.append(float(x))
    if len(xs) < 2:
        return None
    xs.sort()
    for i in range(len(xs) - 1):
        if xs[i] <= px < xs[i + 1]:
            return i
    return None


def lane_of_polygon(poly: np.ndarray, boundaries: List[FrozenLine]) -> Optional[int]:
    if poly is None or len(poly) < 3:
        return None
    pts = poly.astype(np.float32)
    step = max(1, len(pts) // 25)
    votes: Dict[int, int] = {}
    for p in pts[::step]:
        li = lane_of_point_by_y(float(p[0]), float(p[1]), boundaries)
        if li is None:
            continue
        votes[li] = votes.get(li, 0) + 1
    if not votes:
        cen = polygon_centroid(poly)
        if cen is None:
            return None
        return lane_of_point_by_y(cen[0], cen[1], boundaries)
    return max(votes.items(), key=lambda kv: kv[1])[0]


def finalize_lane_rule(present_arrow_classes: Set[int], present_dirs: Set[str]) -> Set[str]:
    # prioritize combined arrows 4/5
    if 5 in present_arrow_classes:
        return {"STRAIGHT", "RIGHT"}
    if 4 in present_arrow_classes:
        return {"STRAIGHT", "LEFT"}

    # fallback combine if both present
    if "STRAIGHT" in present_dirs and "RIGHT" in present_dirs:
        return {"STRAIGHT", "RIGHT"}
    if "STRAIGHT" in present_dirs and "LEFT" in present_dirs:
        return {"STRAIGHT", "LEFT"}

    if "STRAIGHT" in present_dirs:
        return {"STRAIGHT"}
    if "RIGHT" in present_dirs:
        return {"RIGHT"}
    if "LEFT" in present_dirs:
        return {"LEFT"}
    return set()


def build_lane_rules_from_arrows(poly_acc: Dict[int, List[np.ndarray]],
                                 lane_boundaries: List[FrozenLine]) -> Dict[int, Set[str]]:
    lane_dirs: Dict[int, Set[str]] = {}
    lane_arrow_classes: Dict[int, Set[int]] = {}

    for cid in ARROW_CLASSES:
        raw_dirs = arrow_allowed_raw(cid)
        for poly in poly_acc.get(cid, []):
            li = lane_of_polygon(poly, lane_boundaries)
            if li is None:
                continue
            lane_dirs.setdefault(li, set()).update(raw_dirs)
            lane_arrow_classes.setdefault(li, set()).add(cid)

    lane_rules: Dict[int, Set[str]] = {}
    for li in lane_dirs.keys():
        lane_rules[li] = finalize_lane_rule(lane_arrow_classes.get(li, set()), lane_dirs.get(li, set()))
    return lane_rules


# =========================
# BUILD PER-LANE REF VECTORS (to reduce false positives)
# =========================
def build_lane_ref_vectors(boundaries: List[FrozenLine], fh: int) -> Dict[int, Tuple[float, float]]:
    """
    Compute per-lane reference direction (unit vector) using lane centerline.
    This is crucial for lanes near the right edge where "straight" drifts to center in the image.
    """
    lane_ref: Dict[int, Tuple[float, float]] = {}
    if len(boundaries) < 2:
        return lane_ref

    # choose two y-levels: near stopline and near camera
    y_high = int(0.55 * fh)  # nearer stopline
    y_low = int(0.86 * fh)   # nearer camera
    y_high = max(0, min(fh - 1, y_high))
    y_low = max(0, min(fh - 1, y_low))
    if y_high >= y_low:
        y_high = int(0.52 * fh)
        y_low = int(0.88 * fh)

    for i in range(len(boundaries) - 1):
        xL1 = boundaries[i].x_at(y_low)
        xR1 = boundaries[i + 1].x_at(y_low)
        xL2 = boundaries[i].x_at(y_high)
        xR2 = boundaries[i + 1].x_at(y_high)
        if any(v is None or not np.isfinite(v) for v in [xL1, xR1, xL2, xR2]):
            continue

        cx1 = 0.5 * (float(xL1) + float(xR1))
        cx2 = 0.5 * (float(xL2) + float(xR2))

        dx = cx2 - cx1
        dy = float(y_high - y_low)  # negative => upward

        n = float(np.hypot(dx, dy) + 1e-9)
        lane_ref[i] = (dx / n, dy / n)

    return lane_ref


# =========================
# CHAM VACH
# =========================
def point_near_line(px: float, py: float, ln: FrozenLine) -> bool:
    if py < ln.y_min - LINE_TOUCH_Y_MARGIN or py > ln.y_max + LINE_TOUCH_Y_MARGIN:
        return False
    a, b, c = ln.abc
    d = abs(a * px + b * py + c)
    return d <= LINE_TOUCH_DIST_PX


def cham_vach_for_bbox(x1: float, y1: float, x2: float, y2: float, solid_lines: List[FrozenLine]) -> bool:
    """
    solid_white_line / solid_yellow_line:
      both bottom wheels touch => Cham Vach
    """
    bl = (float(x1), float(y2))
    br = (float(x2), float(y2))
    for ln in solid_lines:
        if point_near_line(bl[0], bl[1], ln) and point_near_line(br[0], br[1], ln):
            return True
    return False


# =========================
# DIRECTION BY REFERENCE VECTOR (ANGLE)
# =========================
def robust_direction_by_ref_vector(
    positions: List[Tuple[float, float]],
    vref: Tuple[float, float]
) -> Tuple[str, Dict[str, float]]:
    """
    Determine action by comparing motion vector v with reference vector vref (unit-ish).
    - Align vref direction with motion (flip if dot < 0)
    - Use median dx/dy over forward steps to stabilize
    - Compute dot/cross/angle
    Decision:
      angle <= STRAIGHT_ANGLE_DEG => STRAIGHT
      angle >= TURN_ANGLE_DEG     => LEFT/RIGHT by sign(cross)
      else                         => STRAIGHT (bias to reduce false positives)
    """
    if len(positions) < MIN_ACTION_FRAMES:
        return "UNKNOWN", {"magSum": 0.0, "dxMed": 0.0, "dyMed": 0.0, "dot": 0.0, "cross": 0.0, "angle": 0.0, "used": 0.0}

    v0x, v0y = float(vref[0]), float(vref[1])
    v0n = np.hypot(v0x, v0y) + 1e-9
    v0x /= v0n
    v0y /= v0n

    steps = []
    for (x0, y0), (x1, y1) in zip(positions[:-1], positions[1:]):
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        if np.hypot(dx, dy) < STEP_DMIN:
            continue
        steps.append((dx, dy))

    if len(steps) < max(6, MIN_ACTION_FRAMES // 2):
        return "UNKNOWN", {"magSum": 0.0, "dxMed": 0.0, "dyMed": 0.0, "dot": 0.0, "cross": 0.0, "angle": 0.0, "used": float(len(steps))}

    # rough motion
    dx0 = float(np.median([s[0] for s in steps]))
    dy0 = float(np.median([s[1] for s in steps]))
    if (dx0 * dx0 + dy0 * dy0) < 1e-6:
        return "UNKNOWN", {"magSum": 0.0, "dxMed": dx0, "dyMed": dy0, "dot": 0.0, "cross": 0.0, "angle": 0.0, "used": float(len(steps))}

    # flip vref to face forward along motion
    if (dx0 * v0x + dy0 * v0y) < 0:
        v0x, v0y = -v0x, -v0y

    # keep forward steps
    fsteps = []
    mag_sum = 0.0
    for dx, dy in steps:
        if (dx * v0x + dy * v0y) <= 0.0:
            continue
        fsteps.append((dx, dy))
        mag_sum += float(np.hypot(dx, dy))

    if len(fsteps) < max(6, MIN_ACTION_FRAMES // 2) or mag_sum < MIN_MOVE_MAG_SUM:
        return "UNKNOWN", {"magSum": mag_sum, "dxMed": dx0, "dyMed": dy0, "dot": 0.0, "cross": 0.0, "angle": 0.0, "used": float(len(fsteps))}

    dx_med = float(np.median([s[0] for s in fsteps]))
    dy_med = float(np.median([s[1] for s in fsteps]))
    vn = np.hypot(dx_med, dy_med) + 1e-9
    vx = dx_med / vn
    vy = dy_med / vn

    dot = float(vx * v0x + vy * v0y)
    dot = max(-1.0, min(1.0, dot))
    cross = float(v0x * vy - v0y * vx)  # >0 => RIGHT , <0 => LEFT
    angle = float(np.degrees(np.arctan2(abs(cross), dot)))

    if angle <= STRAIGHT_ANGLE_DEG:
        act = "STRAIGHT"
    elif angle >= TURN_ANGLE_DEG:
        act = "RIGHT" if cross > 0 else "LEFT"
    else:
        act = "STRAIGHT"  # bias to STRAIGHT

    return act, {
        "magSum": mag_sum,
        "dxMed": dx_med,
        "dyMed": dy_med,
        "dot": dot,
        "cross": cross,
        "angle": angle,
        "used": float(len(fsteps)),
        "refX": float(v0x),
        "refY": float(v0y),
    }


# =========================
# TRACK STATE
# =========================
@dataclass
class TrackState:
    positions: List[Tuple[float, float]] = field(default_factory=list)
    last_lane: Optional[int] = None
    wrong_dir_violation: bool = False
    touch_violation: bool = False
    dbg_text: str = ""
    dbg_time: float = 0.0


# =========================
# HUD
# =========================
def draw_info_panel(frame, status: str, fps: float, wrong: int, cham: int,
                    calibrating: bool, remain_s: float = 0.0, debug_on: bool = False):
    """HUD sử dụng draw_info_hud thống nhất"""
    if calibrating:
        # Sử dụng draw_calibration_hud cho giai đoạn calibration
        progress = (CALIBRATION_DURATION_SEC - remain_s) / CALIBRATION_DURATION_SEC * 100
        draw_calibration_hud(frame, progress, CALIBRATION_DURATION_SEC)
    else:
        # Sử dụng draw_info_hud thống nhất
        hud_lines = [
            (f"FPS: {fps:.1f}", config.HUD_TEXT_COLOR),
            (f"WrongDir: {wrong} | ChamVach: {cham}", config.COLOR_VIOLATION),
            (f"DEBUG: {'ON' if debug_on else 'OFF'} (press 'd')", config.HUD_TEXT_COLOR),
        ]
        draw_info_hud(frame, hud_lines, title=f"STATUS: {status}", title_color=config.COLOR_SAFE, width=450)


# =========================
# REUSABLE DETECTOR CLASS (for Web)
# =========================
class WrongLaneDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time: Optional[float] = None
        self.frame_w: Optional[int] = None
        self.frame_h: Optional[int] = None

        self.poly_acc: Dict[int, List[np.ndarray]] = {
            cid: [] for cid in (ARROW_CLASSES | {DASH_WHITE, DASH_YELLOW, SOLID_WHITE, SOLID_YELLOW, STOPLINE_ID})
        }
        self.union_line: Dict[int, Optional[np.ndarray]] = {
            DASH_WHITE: None,
            DASH_YELLOW: None,
            SOLID_WHITE: None,
            SOLID_YELLOW: None
        }
        self.union_stop: Optional[np.ndarray] = None

        self.stop_cal = StoplineCalibrator(duration=CALIBRATION_DURATION_SEC)
        self.all_lines: List[FrozenLine] = []
        self.lane_boundaries: List[FrozenLine] = []
        self.lane_rules: Dict[int, Set[str]] = {}
        self.lane_ref_vecs: Dict[int, Tuple[float, float]] = {}

        self.stopline_segment: Optional[
            Tuple[Tuple[int, int], Tuple[int, int], Tuple[float, float, float], int, int]
        ] = None
        self.stop_vref: Tuple[float, float] = (0.0, -1.0)

        self.tracks: Dict[int, TrackState] = {}
        self.wrong_count = 0
        self.cham_count = 0
        self.is_frozen = False

        self.fps_t0 = time.time()
        self.fps_cnt = 0
        self.fps_now = 0.0

    def _ensure_start(self):
        if self.start_time is None:
            self.start_time = time.time()
            self.stop_cal.start()
            self.fps_t0 = self.start_time
            self.fps_cnt = 0
            self.fps_now = 0.0

    def _update_fps(self, now: float):
        self.fps_cnt += 1
        if now - self.fps_t0 >= 1.0:
            self.fps_now = self.fps_cnt / (now - self.fps_t0)
            self.fps_cnt = 0
            self.fps_t0 = now

    def _parse_results(self, r0):
        masks_xy = None
        masks_data = None
        if r0 is not None and r0.masks is not None:
            if hasattr(r0.masks, "xy") and r0.masks.xy is not None:
                masks_xy = r0.masks.xy
            if hasattr(r0.masks, "data") and r0.masks.data is not None:
                masks_data = r0.masks.data.cpu().numpy()

        if r0 is None or r0.boxes is None or len(r0.boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls_ids = np.zeros((0,), dtype=np.int32)
            tids = np.zeros((0,), dtype=np.int32)
        else:
            boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_ids = r0.boxes.cls.cpu().numpy().astype(np.int32)
            tids = (
                r0.boxes.id.cpu().numpy().astype(np.int32)
                if (hasattr(r0.boxes, "id") and r0.boxes.id is not None)
                else np.array([-1] * len(cls_ids), dtype=np.int32)
            )

        return boxes, cls_ids, tids, masks_xy, masks_data

    def process_frame(
        self,
        frame: np.ndarray,
        r0=None,
        model=None,
        conf_track: float = 0.25,
        conf_calib: float = 0.25,
        debug: bool = False
    ) -> Tuple[np.ndarray, List[dict]]:
        if frame is None:
            return frame, []

        fh, fw = frame.shape[:2]
        if self.frame_w is None or self.frame_h is None:
            self.frame_w, self.frame_h = fw, fh
        elif self.frame_w != fw or self.frame_h != fh:
            self.reset()
            self.frame_w, self.frame_h = fw, fh

        self._ensure_start()
        now = time.time()
        self._update_fps(now)

        elapsed = now - self.start_time
        calibrating = (elapsed < CALIBRATION_DURATION_SEC) and (not self.is_frozen)
        remain = max(0.0, CALIBRATION_DURATION_SEC - elapsed)

        # enforce stable calibration confidence
        conf_calib = min(conf_calib, 0.25)

        # Ensure we have usable results
        if calibrating:
            if r0 is None or r0.masks is None:
                if model is None:
                    return frame, []
                results = model.predict(frame, imgsz=IMG_SIZE, conf=conf_calib, verbose=False)
                r0 = results[0]
        else:
            if r0 is None or r0.boxes is None or r0.boxes.id is None:
                if model is None:
                    return frame, []
                results = model.track(
                    frame, imgsz=IMG_SIZE, conf=conf_track, iou=IOU_THRESHOLD,
                    persist=True, tracker=TRACKER, verbose=False
                )
                r0 = results[0]

        boxes, cls_ids, tids, masks_xy, masks_data = self._parse_results(r0)
        vis = frame.copy()
        violations: List[dict] = []

        # ================= CALIBRATION (5s) =================
        if calibrating:
            if masks_xy is not None:
                for i, cid in enumerate(cls_ids):
                    cid = int(cid)
                    if cid not in self.poly_acc:
                        continue
                    if i >= len(masks_xy):
                        continue
                    poly = np.array(masks_xy[i], dtype=np.int32)
                    if poly is None or len(poly) < 3:
                        continue

                    self.poly_acc[cid].append(poly)

                    if cid in ARROW_CLASSES:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(255, 120, 255), alpha=0.25,
                                                   border_color=COLOR_ARROW, border_thickness=1)
                    elif cid == DASH_WHITE:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(180, 255, 255), alpha=0.25,
                                                   border_color=COLOR_DASH_WHITE, border_thickness=2)
                    elif cid == DASH_YELLOW:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(160, 220, 255), alpha=0.25,
                                                   border_color=COLOR_DASH_YELLOW, border_thickness=2)
                    elif cid == SOLID_WHITE:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(120, 120, 255), alpha=0.25,
                                                   border_color=COLOR_SOLID_WHITE, border_thickness=2)
                    elif cid == SOLID_YELLOW:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(120, 180, 255), alpha=0.25,
                                                   border_color=COLOR_SOLID_YELLOW, border_thickness=2)
                    elif cid == STOPLINE_ID:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(255, 180, 180), alpha=0.30,
                                                   border_color=COLOR_STOPLINE, border_thickness=2)

            if masks_data is not None and masks_data.shape[0] == len(cls_ids):
                for cls_line in [DASH_WHITE, DASH_YELLOW, SOLID_WHITE, SOLID_YELLOW]:
                    idxs = np.where(cls_ids == cls_line)[0]
                    if len(idxs) > 0:
                        m = masks_data[idxs].max(axis=0)
                        m01 = smooth_binary(mask_data_to_binary(m, fw, fh, thr=0.5))
                        self.union_line[cls_line] = m01 if self.union_line[cls_line] is None else (self.union_line[cls_line] | m01)

                idx39 = np.where(cls_ids == STOPLINE_ID)[0]
                if len(idx39) > 0:
                    m = masks_data[idx39].max(axis=0)
                    m01 = smooth_binary(mask_data_to_binary(m, fw, fh, thr=0.5))
                    self.union_stop = m01 if self.union_stop is None else (self.union_stop | m01)
                    self.stop_cal.add_mask_points(m01)

            self.stop_cal.maybe_finish(fh, fw)

        # ================= FREEZE AFTER 5s =================
        if (not calibrating) and (not self.is_frozen):
            if not self.stop_cal.is_calibrated() and self.union_stop is not None:
                self.stop_cal.add_mask_points(self.union_stop)
                self.stop_cal.maybe_finish(fh, fw, min_points=600)

            if self.stop_cal.is_calibrated():
                a, b, c = self.stop_cal.line_abc
                if abs(b) > 1e-9:
                    y1 = int(-(a * self.stop_cal.min_x + c) / b)
                    y2 = int(-(a * self.stop_cal.max_x + c) / b)
                    y1 = max(0, min(fh - 1, y1))
                    y2 = max(0, min(fh - 1, y2))
                    self.stopline_segment = (
                        (self.stop_cal.min_x, y1),
                        (self.stop_cal.max_x, y2),
                        (a, b, c),
                        self.stop_cal.min_x,
                        self.stop_cal.max_x
                    )
                    self.stop_vref = (float(a), float(b))

            self.all_lines = []
            for cls_line in [DASH_WHITE, DASH_YELLOW, SOLID_WHITE, SOLID_YELLOW]:
                self.all_lines.extend(extract_component_lines(self.union_line.get(cls_line), cls_line, fw, fh))

            self.lane_boundaries = [ln for ln in self.all_lines if ln.cls_id in (DASH_WHITE, SOLID_WHITE)]

            y_ref = 0.75 * fh
            tmp = []
            for ln in self.lane_boundaries:
                x = ln.x_at(y_ref)
                if x is None or not np.isfinite(x):
                    continue
                tmp.append((float(x), ln))
            tmp.sort(key=lambda t: t[0])
            self.lane_boundaries = [ln for _, ln in tmp]

            self.lane_rules = build_lane_rules_from_arrows(self.poly_acc, self.lane_boundaries)
            self.lane_ref_vecs = build_lane_ref_vectors(self.lane_boundaries, fh)

            self.is_frozen = True

        # ================= MONITORING =================
        if self.is_frozen:
            if self.stopline_segment is not None:
                p1, p2, _, _, _ = self.stopline_segment
                cv2.line(vis, p1, p2, COLOR_STOPLINE, 3)

            draw_lines(vis, self.lane_boundaries, thickness=3)

            y_label = int(0.80 * fh)
            xs_lab = []
            for ln in self.lane_boundaries:
                x = ln.x_at(y_label)
                if x is not None and np.isfinite(x):
                    xs_lab.append(float(x))
            xs_lab.sort()
            for i in range(len(xs_lab) - 1):
                cx = int((xs_lab[i] + xs_lab[i + 1]) / 2.0)
                allowed = self.lane_rules.get(i, set())
                cv2.putText(vis, f"Lane {i}: {allowed_to_label(allowed)}",
                            (max(10, cx - 160), y_label),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            solid_lines_for_touch = [ln for ln in self.all_lines if ln.cls_id in (SOLID_WHITE, SOLID_YELLOW)]

            if self.stopline_segment is not None:
                a_s, b_s, c_s = self.stopline_segment[2]
                flip = self.stop_cal.sign_flip
            else:
                a_s = b_s = c_s = 0.0
                flip = 1.0

            for xyxy, cid, tid in zip(boxes, cls_ids, tids):
                cid = int(cid)
                tid = int(tid)
                if tid < 0 or cid not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = [float(v) for v in xyxy]
                px = (x1 + x2) / 2.0
                py = y2

                st = self.tracks.get(tid)
                if st is None:
                    st = TrackState()
                    self.tracks[tid] = st

                st.positions.append((px, py))
                if len(st.positions) > 180:
                    st.positions.pop(0)

                lane_now = lane_of_point_by_y(px, py, self.lane_boundaries)
                if lane_now is not None:
                    st.last_lane = lane_now

                # ---- Cham Vach ----
                if (not st.touch_violation) and (cid not in PRIORITY_VEHICLES):
                    if cham_vach_for_bbox(x1, y1, x2, y2, solid_lines_for_touch):
                        st.touch_violation = True
                        self.cham_count += 1
                        vclass = config.CLASS_NAMES.get(cid, "vehicle").lower()
                        save_violation_snapshot(frame, "cham_vach", tid, (x1, y1, x2, y2), vehicle_class=vclass)
                        violations.append({
                            'type': 'wrong_lane',
                            'id': tid,
                            'label': 'Cham Vach'
                        })

                # ---- WrongDir only at CROSS stopline ----
                if (not st.wrong_dir_violation) and (self.stopline_segment is not None) and (cid not in PRIORITY_VEHICLES):
                    if len(st.positions) >= 2:
                        px_prev, py_prev = st.positions[-2]
                        s_prev = signed_distance(px_prev, py_prev, a_s, b_s, c_s) * flip
                        s_curr = signed_distance(px, py, a_s, b_s, c_s) * flip

                        r_prev = -1 if s_prev < 0 else +1
                        r_curr = -1 if s_curr < 0 else +1

                        crossed = (r_prev == -1 and r_curr == +1)
                        if crossed:
                            lane_before = lane_of_point_by_y(px_prev, py_prev, self.lane_boundaries)
                            if lane_before is None:
                                lane_before = st.last_lane

                            allowed = self.lane_rules.get(lane_before, set()) if lane_before is not None else set()

                            hist = st.positions[:-1]
                            win = hist[-ACTION_WINDOW:] if len(hist) >= ACTION_WINDOW else hist

                            ref = self.lane_ref_vecs.get(lane_before, self.stop_vref)
                            act, stats = robust_direction_by_ref_vector(win, ref)

                            if allowed and "STRAIGHT" in allowed and act in {"LEFT", "RIGHT"} and stats.get("angle", 999.0) < SAFE_STRAIGHT_MAX:
                                act = "STRAIGHT"

                            if allowed and act != "UNKNOWN":
                                if act not in allowed:
                                    st.wrong_dir_violation = True
                                    self.wrong_count += 1
                                    st.dbg_text = ""
                                    st.dbg_time = time.time()
                                    vclass = config.CLASS_NAMES.get(cid, "vehicle").lower()
                                    save_violation_snapshot(frame, "wrong_lane", tid, (x1, y1, x2, y2), vehicle_class=vclass)
                                    violations.append({
                                        'type': 'wrong_lane',
                                        'id': tid,
                                        'label': 'Wrong Lane'
                                    })

                # ---- Draw bbox ----
                vehicle_name = config.CLASS_NAMES.get(cid, "Vehicle")
                col = COLOR_SAFE
                label = f"{vehicle_name}:{tid}"

                if st.wrong_dir_violation or st.touch_violation:
                    col = COLOR_VIOLATION
                    label = "Violation"

                draw_bbox_with_label(vis, (x1, y1, x2, y2), label, col)
                cv2.circle(vis, (int(px), int(py)), 4, col, -1)

                if debug:
                    allow_now = allowed_to_label(self.lane_rules.get(lane_now, set()))
                    cv2.putText(vis, f"laneNow={lane_now} allowNow={allow_now}",
                                (int(x1), int(y2) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        draw_info_panel(vis, "MONITORING" if self.is_frozen else "CALIBRATING",
                        self.fps_now, self.wrong_count, self.cham_count,
                        calibrating, remain, debug)

        return vis, violations

    def get_stats(self) -> Dict[str, Any]:
        return {
            'wrong': self.wrong_count,
            'cham': self.cham_count,
            'total': self.wrong_count + self.cham_count,
            'is_frozen': self.is_frozen
        }


# =========================
# MAIN
# =========================
def run(video_path: str = None, model_path: str = None):
    # Use config paths if not specified
    if video_path is None:
        video_path = config.DEFAULT_VIDEO
    if model_path is None:
        model_path = config.MODEL_PATH
    
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # accumulate polygons in calibration phase
    poly_acc: Dict[int, List[np.ndarray]] = {
        cid: [] for cid in (ARROW_CLASSES | {DASH_WHITE, DASH_YELLOW, SOLID_WHITE, SOLID_YELLOW, STOPLINE_ID})
    }

    union_line: Dict[int, Optional[np.ndarray]] = {DASH_WHITE: None, DASH_YELLOW: None, SOLID_WHITE: None, SOLID_YELLOW: None}
    union_stop: Optional[np.ndarray] = None

    stop_cal = StoplineCalibrator(duration=CALIBRATION_DURATION_SEC)
    stop_cal.start()

    all_lines: List[FrozenLine] = []
    lane_boundaries: List[FrozenLine] = []
    lane_rules: Dict[int, Set[str]] = {}
    lane_ref_vecs: Dict[int, Tuple[float, float]] = {}

    stopline_segment: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[float, float, float], int, int]] = None
    stop_vref: Tuple[float, float] = (0.0, -1.0)

    tracks: Dict[int, TrackState] = {}
    wrong_count = 0
    cham_count = 0

    t0 = time.time()
    fps_t0 = time.time()
    fps_cnt = 0
    fps_now = 0.0
    is_frozen = False

    debug_on = False  # Mặc định tắt, bấm 'd' để bật

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        elapsed = now - t0
        calibrating = (elapsed < CALIBRATION_DURATION_SEC) and (not is_frozen)
        remain = max(0.0, CALIBRATION_DURATION_SEC - elapsed)

        fps_cnt += 1
        if now - fps_t0 >= 1.0:
            fps_now = fps_cnt / (now - fps_t0)
            fps_cnt = 0
            fps_t0 = now

        # predict vs track
        if calibrating:
            results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_CALIB, verbose=False)
        else:
            results = model.track(
                frame, imgsz=IMG_SIZE, conf=CONF_TRACK, iou=IOU_THRESHOLD,
                persist=True, tracker=TRACKER, verbose=False
            )

        r0 = results[0]
        vis = frame.copy()

        masks_xy = None
        masks_data = None
        if r0.masks is not None:
            if hasattr(r0.masks, "xy") and r0.masks.xy is not None:
                masks_xy = r0.masks.xy
            if hasattr(r0.masks, "data") and r0.masks.data is not None:
                masks_data = r0.masks.data.cpu().numpy()

        if r0.boxes is None or len(r0.boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls_ids = np.zeros((0,), dtype=np.int32)
            tids = np.zeros((0,), dtype=np.int32)
        else:
            boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_ids = r0.boxes.cls.cpu().numpy().astype(np.int32)
            tids = (
                r0.boxes.id.cpu().numpy().astype(np.int32)
                if (hasattr(r0.boxes, "id") and r0.boxes.id is not None)
                else np.array([-1] * len(cls_ids), dtype=np.int32)
            )

        # ================= CALIBRATION (5s) =================
        if calibrating:
            # show all masks overlays and accumulate polygons
            if masks_xy is not None:
                for i, cid in enumerate(cls_ids):
                    cid = int(cid)
                    if cid not in poly_acc:
                        continue
                    if i >= len(masks_xy):
                        continue
                    poly = np.array(masks_xy[i], dtype=np.int32)
                    if poly is None or len(poly) < 3:
                        continue

                    poly_acc[cid].append(poly)

                    if cid in ARROW_CLASSES:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(255, 120, 255), alpha=0.25,
                                                   border_color=COLOR_ARROW, border_thickness=1)
                    elif cid == DASH_WHITE:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(180, 255, 255), alpha=0.25,
                                                   border_color=COLOR_DASH_WHITE, border_thickness=2)
                    elif cid == DASH_YELLOW:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(160, 220, 255), alpha=0.25,
                                                   border_color=COLOR_DASH_YELLOW, border_thickness=2)
                    elif cid == SOLID_WHITE:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(120, 120, 255), alpha=0.25,
                                                   border_color=COLOR_SOLID_WHITE, border_thickness=2)
                    elif cid == SOLID_YELLOW:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(120, 180, 255), alpha=0.25,
                                                   border_color=COLOR_SOLID_YELLOW, border_thickness=2)
                    elif cid == STOPLINE_ID:
                        vis = draw_polygon_overlay(vis, poly, fill_color=(255, 180, 180), alpha=0.30,
                                                   border_color=COLOR_STOPLINE, border_thickness=2)

            # build union masks for lines + stopline
            if masks_data is not None and masks_data.shape[0] == len(cls_ids):
                for cls_line in [DASH_WHITE, DASH_YELLOW, SOLID_WHITE, SOLID_YELLOW]:
                    idxs = np.where(cls_ids == cls_line)[0]
                    if len(idxs) > 0:
                        m = masks_data[idxs].max(axis=0)
                        m01 = smooth_binary(mask_data_to_binary(m, fw, fh, thr=0.5))
                        union_line[cls_line] = m01 if union_line[cls_line] is None else (union_line[cls_line] | m01)

                idx39 = np.where(cls_ids == STOPLINE_ID)[0]
                if len(idx39) > 0:
                    m = masks_data[idx39].max(axis=0)
                    m01 = smooth_binary(mask_data_to_binary(m, fw, fh, thr=0.5))
                    union_stop = m01 if union_stop is None else (union_stop | m01)
                    stop_cal.add_mask_points(m01)

            stop_cal.maybe_finish(fh, fw)
            # Calibration ngầm - không hiển thị panel

        # ================= FREEZE AFTER 5s =================
        if (not calibrating) and (not is_frozen):
            # finalize stopline if needed
            if not stop_cal.is_calibrated() and union_stop is not None:
                stop_cal.add_mask_points(union_stop)
                stop_cal.maybe_finish(fh, fw, min_points=600)

            if stop_cal.is_calibrated():
                a, b, c = stop_cal.line_abc
                if abs(b) > 1e-9:
                    y1 = int(-(a * stop_cal.min_x + c) / b)
                    y2 = int(-(a * stop_cal.max_x + c) / b)
                    y1 = max(0, min(fh - 1, y1))
                    y2 = max(0, min(fh - 1, y2))
                    stopline_segment = ((stop_cal.min_x, y1), (stop_cal.max_x, y2), (a, b, c),
                                        stop_cal.min_x, stop_cal.max_x)
                    stop_vref = (float(a), float(b))  # normal

            all_lines = []
            for cls_line in [DASH_WHITE, DASH_YELLOW, SOLID_WHITE, SOLID_YELLOW]:
                all_lines.extend(extract_component_lines(union_line.get(cls_line), cls_line, fw, fh))

            # lane boundaries: ONLY dash_white + solid_white
            lane_boundaries = [ln for ln in all_lines if ln.cls_id in (DASH_WHITE, SOLID_WHITE)]

            # sort boundaries by x at a reference y
            y_ref = 0.75 * fh
            tmp = []
            for ln in lane_boundaries:
                x = ln.x_at(y_ref)
                if x is None or not np.isfinite(x):
                    continue
                tmp.append((float(x), ln))
            tmp.sort(key=lambda t: t[0])
            lane_boundaries = [ln for _, ln in tmp]

            lane_rules = build_lane_rules_from_arrows(poly_acc, lane_boundaries)
            lane_ref_vecs = build_lane_ref_vectors(lane_boundaries, fh)

            is_frozen = True
            print("[✓] FROZEN READY")
            print(f"  Stopline: {'OK' if stopline_segment is not None else 'MISSING'}")
            print(f"  Lane boundaries: {len(lane_boundaries)} | lanes: {max(0, len(lane_boundaries)-1)}")
            for li in range(max(0, len(lane_boundaries) - 1)):
                allowed = lane_rules.get(li, set())
                print(f"  Lane {li}: {allowed_to_label(allowed)}")
                if li in lane_ref_vecs:
                    v = lane_ref_vecs[li]
                    print(f"    refVec=({v[0]:.3f},{v[1]:.3f})")
            if stopline_segment is not None:
                a_s, b_s, _ = stopline_segment[2]
                print(f"  Stopline normal v0=(a,b)=({a_s:.4f},{b_s:.4f})")
                print(f"  STRAIGHT_ANGLE_DEG={STRAIGHT_ANGLE_DEG:.1f} | TURN_ANGLE_DEG={TURN_ANGLE_DEG:.1f} | SAFE_STRAIGHT_MAX={SAFE_STRAIGHT_MAX:.1f}")

        # ================= MONITORING =================
        if is_frozen:
            # draw stopline
            if stopline_segment is not None:
                p1, p2, _, _, _ = stopline_segment
                cv2.line(vis, p1, p2, COLOR_STOPLINE, 3)

            # draw lane boundaries
            draw_lines(vis, lane_boundaries, thickness=3)

            # lane labels
            y_label = int(0.80 * fh)
            xs_lab = []
            for ln in lane_boundaries:
                x = ln.x_at(y_label)
                if x is not None and np.isfinite(x):
                    xs_lab.append(float(x))
            xs_lab.sort()
            for i in range(len(xs_lab) - 1):
                cx = int((xs_lab[i] + xs_lab[i + 1]) / 2.0)
                allowed = lane_rules.get(i, set())
                cv2.putText(vis, f"Lane {i}: {allowed_to_label(allowed)}",
                            (max(10, cx - 160), y_label),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # cham vach only SOLID lines
            solid_lines_for_touch = [ln for ln in all_lines if ln.cls_id in (SOLID_WHITE, SOLID_YELLOW)]

            # stopline signed distance
            if stopline_segment is not None:
                a_s, b_s, c_s = stopline_segment[2]
                flip = stop_cal.sign_flip
            else:
                a_s = b_s = c_s = 0.0
                flip = 1.0

            # process tracked vehicles
            for xyxy, cid, tid in zip(boxes, cls_ids, tids):
                cid = int(cid)
                tid = int(tid)
                if tid < 0 or cid not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = [float(v) for v in xyxy]
                px = (x1 + x2) / 2.0
                py = y2  # bottom center

                st = tracks.get(tid)
                if st is None:
                    st = TrackState()
                    tracks[tid] = st

                st.positions.append((px, py))
                if len(st.positions) > 180:
                    st.positions.pop(0)

                lane_now = lane_of_point_by_y(px, py, lane_boundaries)
                if lane_now is not None:
                    st.last_lane = lane_now

                # ---- Cham Vach ----
                if (not st.touch_violation) and (cid not in PRIORITY_VEHICLES):
                    if cham_vach_for_bbox(x1, y1, x2, y2, solid_lines_for_touch):
                        st.touch_violation = True
                        cham_count += 1
                        # Chụp screenshot khi phát hiện cham vach (dùng frame gốc)
                        vclass = config.CLASS_NAMES.get(cid, "vehicle").lower()
                        save_violation_snapshot(frame, "cham_vach", tid, (x1, y1, x2, y2), vehicle_class=vclass)
                        if debug_on:
                            print(f"[CHAMVACH] ID={tid}")

                # ---- WrongDir only at CROSS stopline ----
                if (not st.wrong_dir_violation) and (stopline_segment is not None) and (cid not in PRIORITY_VEHICLES):
                    if len(st.positions) >= 2:
                        px_prev, py_prev = st.positions[-2]
                        s_prev = signed_distance(px_prev, py_prev, a_s, b_s, c_s) * flip
                        s_curr = signed_distance(px, py, a_s, b_s, c_s) * flip

                        r_prev = -1 if s_prev < 0 else +1
                        r_curr = -1 if s_curr < 0 else +1

                        crossed = (r_prev == -1 and r_curr == +1)
                        if crossed:
                            lane_before = lane_of_point_by_y(px_prev, py_prev, lane_boundaries)
                            if lane_before is None:
                                lane_before = st.last_lane

                            allowed = lane_rules.get(lane_before, set()) if lane_before is not None else set()

                            hist = st.positions[:-1]
                            win = hist[-ACTION_WINDOW:] if len(hist) >= ACTION_WINDOW else hist

                            # choose reference: per-lane ref vector if available, else stopline normal
                            ref = lane_ref_vecs.get(lane_before, stop_vref)
                            act, stats = robust_direction_by_ref_vector(win, ref)
                            ref_used = "LANE" if lane_before in lane_ref_vecs else "STOP"

                            # SAFEGUARD: if STRAIGHT is allowed and the angle is not too large => force STRAIGHT
                            if allowed and "STRAIGHT" in allowed and act in {"LEFT", "RIGHT"} and stats.get("angle", 999.0) < SAFE_STRAIGHT_MAX:
                                act = "STRAIGHT"

                            dbg = (
                                f"ID={tid} laneB={lane_before} allow={allowed_to_label(allowed)} "
                                f"act={act} angle={stats.get('angle',0):.1f} "
                                f"ref={ref_used} vRef=({stats.get('refX',0):.2f},{stats.get('refY',0):.2f}) "
                                f"dot={stats.get('dot',0):.3f} cross={stats.get('cross',0):.3f} "
                                f"dxMed={stats.get('dxMed',0):.2f} dyMed={stats.get('dyMed',0):.2f} "
                                f"magSum={stats.get('magSum',0):.1f} used={stats.get('used',0):.0f} "
                                f"sPrev={s_prev:.2f} sCur={s_curr:.2f}"
                            )

                            if debug_on:
                                print(f"[CROSS] {dbg}")

                            if allowed and act != "UNKNOWN":
                                if act not in allowed:
                                    st.wrong_dir_violation = True
                                    wrong_count += 1
                                    st.dbg_text = dbg
                                    st.dbg_time = time.time()
                                    # Chụp screenshot khi phát hiện wrong_dir (dùng frame gốc)
                                    vclass = config.CLASS_NAMES.get(cid, "vehicle").lower()
                                    save_violation_snapshot(frame, "wrong_lane", tid, (x1, y1, x2, y2), vehicle_class=vclass)
                                    if debug_on:
                                        print(f"[WRONGDIR] {dbg}")

                # ---- Draw bbox - sử dụng draw_utils ----
                vehicle_name = config.CLASS_NAMES.get(cid, "Vehicle")
                col = COLOR_SAFE
                label = f"{vehicle_name}:{tid}"  # Không có Safe

                if st.wrong_dir_violation:
                    col = COLOR_VIOLATION
                    label = "Violation"  # Ngắn gọn
                elif st.touch_violation:
                    col = COLOR_VIOLATION
                    label = "Violation"  # Ngắn gọn

                draw_bbox_with_label(vis, (x1, y1, x2, y2), label, col)
                cv2.circle(vis, (int(px), int(py)), 4, col, -1)

                # Debug overlay
                if debug_on:
                    allow_now = allowed_to_label(lane_rules.get(lane_now, set()))
                    cv2.putText(vis, f"laneNow={lane_now} allowNow={allow_now}",
                                (int(x1), int(y2) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # show last CROSS debug 3s
                if debug_on and st.dbg_text and (time.time() - st.dbg_time) < 3.0:
                    s = st.dbg_text
                    cv2.putText(vis, s[:120], (int(x1), int(y2) + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(vis, s[120:240], (int(x1), int(y2) + 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            draw_info_panel(vis, "MONITORING", fps_now, wrong_count, cham_count, False, 0.0, debug_on)

        cv2.imshow(WINDOW_NAME, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug_on = not debug_on

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. WrongDir={wrong_count} | ChamVach={cham_count}")


if __name__ == "__main__":
    run()
