"""
Red Light Violation Detection System
=====================================
Phát hiện vi phạm vượt đèn đỏ sử dụng YOLOv8 Segmentation + Object Tracking

Features:
1. Calibration stopline bằng RANSAC (5 giây đầu)
2. Hệ tọa độ tương đối: Âm = đang tới, Dương = đã qua
3. Phát hiện vi phạm khi xe chuyển từ Âm sang Dương lúc đèn đỏ

Author: Traffic Violation Detection System
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Import config and draw utilities
from config import config
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, draw_calibration_hud, save_violation_snapshot


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def get_bottom_center(xyxy: np.ndarray) -> Tuple[float, float]:
    """Lấy điểm đáy-giữa của bounding box"""
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float(y2)


def get_signed_distance(px: float, py: float, a: float, b: float, c: float) -> float:
    """
    Tính khoảng cách có dấu từ điểm (px, py) đến đường thẳng ax + by + c = 0
    
    Returns:
        Giá trị có dấu: âm = một phía, dương = phía kia
    """
    return (a * px + b * py + c) / (np.sqrt(a * a + b * b) + 1e-12)


def get_distance_to_line(px: float, py: float, a: float, b: float, c: float) -> float:
    """Tính khoảng cách tuyệt đối từ điểm đến đường thẳng"""
    return abs(get_signed_distance(px, py, a, b, c))


# =============================================================================
# RANSAC STOPLINE FITTING
# =============================================================================

def fit_stopline_ransac(
    points_xy: np.ndarray,
    max_iters: int = 400,
    inlier_thresh: float = 3.0,
    min_inliers: int = 100
) -> Optional[Tuple[float, float, float]]:
    """
    Fit đường thẳng ax + by + c = 0 sử dụng RANSAC
    
    Args:
        points_xy: Mảng điểm (N, 2)
        max_iters: Số vòng lặp tối đa
        inlier_thresh: Ngưỡng khoảng cách để tính inlier
        min_inliers: Số inlier tối thiểu để chấp nhận
        
    Returns:
        (a, b, c) đã normalize (sqrt(a²+b²)=1) hoặc None nếu thất bại
    """
    if points_xy is None or len(points_xy) < 2:
        return None

    pts = points_xy.astype(np.float32)
    n = len(pts)

    # Subsample nếu quá nhiều điểm
    if n > 15000:
        idx = np.random.choice(n, size=15000, replace=False)
        pts = pts[idx]
        n = len(pts)

    best_inliers = 0
    best_abc = None
    rng = np.random.default_rng(42)

    for _ in range(max_iters):
        # Chọn 2 điểm ngẫu nhiên
        i1, i2 = rng.integers(0, n, size=2)
        if i1 == i2:
            continue
            
        x1, y1 = pts[i1]
        x2, y2 = pts[i2]
        dx, dy = x2 - x1, y2 - y1
        
        if abs(dx) + abs(dy) < 1e-6:
            continue

        # Đường thẳng qua 2 điểm: (y1-y2)x + (x2-x1)y + (x1*y2 - x2*y1) = 0
        a = (y1 - y2)
        b = (x2 - x1)
        c = (x1 * y2 - x2 * y1)

        # Normalize
        norm = np.sqrt(a * a + b * b) + 1e-12
        a, b, c = a / norm, b / norm, c / norm

        # Đếm inliers
        distances = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
        inlier_count = int(np.sum(distances <= inlier_thresh))

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_abc = (float(a), float(b), float(c))

    if best_abc is None or best_inliers < min_inliers:
        return None

    return best_abc


def line_to_segment(a: float, b: float, c: float, w: int, h: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Chuyển đường thẳng vô hạn thành đoạn thẳng trong frame"""
    pts = []

    def add_if_valid(x, y):
        if 0 <= x < w and 0 <= y < h:
            pts.append((int(round(x)), int(round(y))))

    # Giao với các biên
    if abs(b) > 1e-9:
        add_if_valid(0, -c / b)
        add_if_valid(w - 1, -(a * (w - 1) + c) / b)
    if abs(a) > 1e-9:
        add_if_valid(-c / a, 0)
        add_if_valid(-(b * (h - 1) + c) / a, h - 1)

    # Loại trùng
    pts = list(set(pts))
    if len(pts) < 2:
        return None

    # Chọn 2 điểm xa nhất
    max_dist = -1
    p1, p2 = pts[0], pts[1]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d > max_dist:
                max_dist = d
                p1, p2 = pts[i], pts[j]
    return p1, p2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrackState:
    """Trạng thái tracking của mỗi xe"""
    last_region: Optional[int] = None  # -1: Âm (đang tới), +1: Dương (đã qua)
    first_seen: bool = False           # Đã thấy lần đầu chưa
    last_event_frame: int = -1000      # Frame cuối xảy ra event
    label: str = "Safe"                # Safe / WARNING / VIOLATION
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 255, 0))
    positions: List[Tuple[float, float]] = field(default_factory=list)  # Lịch sử vị trí để phát hiện hướng đi
    direction: str = "UNKNOWN"         # STRAIGHT, LEFT, RIGHT, UNKNOWN


@dataclass
class TrafficLightState:
    """Trạng thái chi tiết của tất cả đèn giao thông"""
    # Đèn tròn (circle)
    circle_red: bool = False           # light_straight_circle_red (class 18)
    circle_yellow: bool = False        # light_straight_circle_yellow (class 19)
    circle_green: bool = False         # light_straight_circle_green (class 17)
    
    # Đèn mũi tên rẽ trái
    left_red: bool = False             # light_left_red (class 11)
    left_yellow: bool = False          # light_left_yellow (class 12)
    left_green: bool = False           # light_left_green (class 10)
    
    # Đèn mũi tên đi thẳng
    straight_red: bool = False         # light_straight_arrow_red (class 15)
    straight_yellow: bool = False      # light_straight_arrow_yellow (class 16)
    straight_green: bool = False       # light_straight_arrow_green (class 14)
    
    # Đèn mũi tên rẽ phải
    right_green: bool = False          # light_right_green (class 13)
    
    def get_allowed_directions(self) -> Dict[str, bool]:
        """
        Trả về các hướng được phép đi dựa trên tổ hợp đèn
        Returns: {'STRAIGHT': bool, 'LEFT': bool, 'RIGHT': bool}
        """
        allowed = {'STRAIGHT': True, 'LEFT': True, 'RIGHT': True}
        
        # Rule 1: circle_red + right_green -> tất cả được rẽ phải
        if self.circle_red and self.right_green:
            allowed = {'STRAIGHT': False, 'LEFT': False, 'RIGHT': True}
            return allowed
        
        # Rule 2: circle_green + left_red -> đi thẳng OK, rẽ trái vi phạm
        if self.circle_green and self.left_red:
            allowed = {'STRAIGHT': True, 'LEFT': False, 'RIGHT': True}
            return allowed
        
        # Rule 3: circle_red + left_green -> rẽ trái OK, đi thẳng vi phạm
        if self.circle_red and self.left_green:
            allowed = {'STRAIGHT': False, 'LEFT': True, 'RIGHT': True}
            return allowed
        
        # Rule 5: straight_red + left_green -> rẽ trái/phải OK, đi thẳng vi phạm
        if self.straight_red and self.left_green:
            allowed = {'STRAIGHT': False, 'LEFT': True, 'RIGHT': True}
            return allowed
        
        # Rule 6: chỉ có straight_red -> rẽ trái/phải OK, đi thẳng vi phạm
        if self.straight_red and not self.left_green and not self.circle_green:
            allowed = {'STRAIGHT': False, 'LEFT': True, 'RIGHT': True}
            return allowed
        
        # Đèn đỏ tròn thông thường (không có đèn xanh nào khác)
        if self.circle_red and not self.right_green and not self.left_green:
            allowed = {'STRAIGHT': False, 'LEFT': False, 'RIGHT': False}
            return allowed
        
        return allowed
    
    def has_any_yellow(self) -> bool:
        """Kiểm tra có bất kỳ đèn vàng nào sáng không"""
        return self.circle_yellow or self.left_yellow or self.straight_yellow
    
    def has_any_red(self) -> bool:
        """Kiểm tra có bất kỳ đèn đỏ nào sáng không"""
        return self.circle_red or self.left_red or self.straight_red
    
    def get_simple_state(self) -> str:
        """Trả về trạng thái đơn giản cho HUD"""
        if self.has_any_red():
            return "RED"
        if self.has_any_yellow():
            return "YELLOW"
        return "GREEN"


@dataclass  
class LightMemory:
    """Bộ nhớ trạng thái đèn giao thông"""
    last_state: TrafficLightState = field(default_factory=TrafficLightState)
    last_update_time: float = 0.0
    
    def update(self, state: TrafficLightState, current_time: float):
        if state.has_any_red() or state.has_any_yellow() or state.circle_green or state.left_green or state.straight_green or state.right_green:
            self.last_state = state
            self.last_update_time = current_time
    
    def get(self, current_time: float) -> TrafficLightState:
        # Nhớ trạng thái trong 2 giây
        if (current_time - self.last_update_time) <= config.LIGHT_MEMORY_DURATION:
            return self.last_state
        return TrafficLightState()  # Mặc định tất cả False


# =============================================================================
# STOPLINE CALIBRATOR
# =============================================================================

class StoplineCalibrator:
    """Bộ calibration stopline sử dụng RANSAC"""
    
    def __init__(self, duration: float = 5.0):
        self.duration = duration
        self.start_time = None
        self.points: List[np.ndarray] = []
        self.line_abc: Optional[Tuple[float, float, float]] = None
        self.sign_flip: float = 1.0  # Để auto-orient dấu
        self.min_x: int = 0  # Giới hạn trái của stopline
        self.max_x: int = 0  # Giới hạn phải của stopline
        
    def is_calibrated(self) -> bool:
        return self.line_abc is not None
    
    def start(self):
        self.start_time = time.time()
        
    def add_mask_points(self, mask: np.ndarray):
        """Thêm các điểm từ mask stopline"""
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            pts = np.stack([xs, ys], axis=1)
            self.points.append(pts)
    
    def maybe_finish(self, frame_height: int, frame_width: int):
        """Kiểm tra và hoàn thành calibration"""
        if self.start_time is None or self.is_calibrated():
            return
            
        elapsed = time.time() - self.start_time
        if elapsed < self.duration:
            return
        
        # Ghép tất cả điểm
        if not self.points:
            print("[ERROR] No stopline points collected!")
            return
            
        all_pts = np.concatenate(self.points, axis=0)
        print(f"[CALIBRATION] Collected {len(all_pts)} points")
        
        if len(all_pts) < config.STOPLINE_MIN_POINTS:
            print(f"[ERROR] Not enough points: {len(all_pts)} < {config.STOPLINE_MIN_POINTS}")
            return
        
        # Lưu giới hạn x của mask
        self.min_x = int(np.min(all_pts[:, 0]))
        self.max_x = int(np.max(all_pts[:, 0]))
        
        # Fit đường thẳng bằng RANSAC
        abc = fit_stopline_ransac(all_pts)
        if abc is None:
            print("[ERROR] RANSAC failed to fit stopline!")
            return
        
        self.line_abc = abc
        
        # Auto-orient: điểm dưới frame phải là ÂM (xe đang tới)
        a, b, c = abc
        ref_x, ref_y = frame_width / 2.0, frame_height - 10
        signed = a * ref_x + b * ref_y + c
        self.sign_flip = -1.0 if signed > 0 else 1.0
        
        print(f"[✓] Stopline calibrated: {a:.4f}x + {b:.4f}y + {c:.2f} = 0")


# =============================================================================
# TRAFFIC LIGHT DETECTION
# =============================================================================

def get_all_light_classes():
    """Trả về tất cả class ID của đèn giao thông"""
    return (config.RED_LIGHTS + config.YELLOW_LIGHTS + config.GREEN_LIGHTS +
            config.ARROW_LEFT_RED + config.ARROW_LEFT_YELLOW + config.ARROW_LEFT_GREEN +
            config.ARROW_STRAIGHT_RED + config.ARROW_STRAIGHT_YELLOW + config.ARROW_STRAIGHT_GREEN +
            config.ARROW_RIGHT_GREEN)


def get_light_color(cls_id: int) -> Tuple[Tuple[int, int, int], str]:
    """Trả về màu và label cho đèn giao thông"""
    all_red = config.RED_LIGHTS + config.ARROW_LEFT_RED + config.ARROW_STRAIGHT_RED
    all_yellow = config.YELLOW_LIGHTS + config.ARROW_LEFT_YELLOW + config.ARROW_STRAIGHT_YELLOW
    all_green = config.GREEN_LIGHTS + config.ARROW_LEFT_GREEN + config.ARROW_STRAIGHT_GREEN + config.ARROW_RIGHT_GREEN
    
    if cls_id in all_red:
        return (0, 0, 255), "RED"
    elif cls_id in all_yellow:
        return (0, 255, 255), "YELLOW"
    elif cls_id in all_green:
        return (0, 255, 0), "GREEN"
    return (128, 128, 128), "UNKNOWN"


def draw_traffic_lights(frame, boxes, cls_ids, confs):
    """Vẽ bounding box cho đèn giao thông tốt nhất (confidence cao nhất cho mỗi loại, thêm NMS)"""
    all_light_classes = get_all_light_classes()
    all_red = config.RED_LIGHTS + config.ARROW_LEFT_RED + config.ARROW_STRAIGHT_RED
    all_yellow = config.YELLOW_LIGHTS + config.ARROW_LEFT_YELLOW + config.ARROW_STRAIGHT_YELLOW
    all_green = config.GREEN_LIGHTS + config.ARROW_LEFT_GREEN + config.ARROW_STRAIGHT_GREEN + config.ARROW_RIGHT_GREEN
    
    # Tìm đèn tốt nhất cho mỗi loại (RED, YELLOW, GREEN)
    best_lights = {'RED': None, 'YELLOW': None, 'GREEN': None}
    best_confs = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
    
    for xyxy, cls_id, conf in zip(boxes, cls_ids, confs):
        if cls_id not in all_light_classes:
            continue
        
        # Xác định loại đèn
        if cls_id in all_red:
            light_type = 'RED'
        elif cls_id in all_yellow:
            light_type = 'YELLOW'
        elif cls_id in all_green:
            light_type = 'GREEN'
        else:
            continue
        
        # Lưu đèn có confidence cao nhất
        if conf > best_confs[light_type]:
            best_confs[light_type] = conf
            best_lights[light_type] = (xyxy, cls_id, conf)
    
    # Helper function tính IoU
    def calc_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    # Lọc đèn bị overlap - chỉ giữ đèn có confidence cao nhất
    lights_to_draw = []
    for light_type, light_data in best_lights.items():
        if light_data is not None:
            lights_to_draw.append((light_type, light_data))
    
    # Sắp xếp theo confidence giảm dần
    lights_to_draw.sort(key=lambda x: x[1][2], reverse=True)
    
    # Lọc overlap
    final_lights = []
    for light_type, light_data in lights_to_draw:
        xyxy, cls_id, conf = light_data
        is_overlap = False
        for _, existing_data in final_lights:
            if calc_iou(xyxy, existing_data[0]) > 0.3:  # Threshold IoU = 0.3
                is_overlap = True
                break
        if not is_overlap:
            final_lights.append((light_type, light_data))
    
    # Vẽ chỉ các đèn không bị overlap
    for light_type, light_data in final_lights:
        xyxy, cls_id, conf = light_data
        color, label = get_light_color(cls_id)
        draw_bbox_with_label(frame, tuple(xyxy), f"{label} {conf:.2f}", color)


def detect_traffic_lights(cls_ids: np.ndarray) -> TrafficLightState:
    """Phát hiện chi tiết trạng thái các đèn giao thông"""
    state = TrafficLightState()
    
    if cls_ids is None or len(cls_ids) == 0:
        return state
    
    # Đèn tròn (Circle lights)
    state.circle_red = np.isin(cls_ids, config.RED_LIGHTS).any()        # class 18
    state.circle_yellow = np.isin(cls_ids, config.YELLOW_LIGHTS).any()  # class 19
    state.circle_green = np.isin(cls_ids, config.GREEN_LIGHTS).any()    # class 17
    
    # Đèn mũi tên rẽ trái
    state.left_red = np.isin(cls_ids, config.ARROW_LEFT_RED).any()      # class 11
    state.left_yellow = np.isin(cls_ids, config.ARROW_LEFT_YELLOW).any() # class 12
    state.left_green = np.isin(cls_ids, config.ARROW_LEFT_GREEN).any()  # class 10
    
    # Đèn mũi tên đi thẳng
    state.straight_red = np.isin(cls_ids, config.ARROW_STRAIGHT_RED).any()      # class 15
    state.straight_yellow = np.isin(cls_ids, config.ARROW_STRAIGHT_YELLOW).any() # class 16
    state.straight_green = np.isin(cls_ids, config.ARROW_STRAIGHT_GREEN).any()  # class 14
    
    # Đèn mũi tên rẽ phải
    state.right_green = np.isin(cls_ids, config.ARROW_RIGHT_GREEN).any()  # class 13
    
    return state


def detect_vehicle_direction(positions: List[Tuple[float, float]], min_frames: int = 10) -> str:
    """
    Phát hiện hướng đi của xe dựa trên trajectory
    
    Returns:
        'STRAIGHT': Đi thẳng (dx nhỏ, dy lớn)
        'LEFT': Rẽ trái (dx < 0 đáng kể)
        'RIGHT': Rẽ phải (dx > 0 đáng kể)
        'UNKNOWN': Chưa đủ dữ liệu
    """
    if len(positions) < min_frames:
        return "UNKNOWN"
    
    # Lấy vị trí đầu và cuối
    start_x, start_y = positions[0]
    end_x, end_y = positions[-1]
    
    dx = end_x - start_x
    dy = end_y - start_y
    
    # Tính tỷ lệ dx/dy để xác định hướng
    # Thường xe đi từ dưới lên (dy < 0)
    if abs(dy) < 20:  # Di chuyển dọc quá ít
        return "UNKNOWN"
    
    ratio = dx / (abs(dy) + 1e-6)
    
    # Ngưỡng để xác định rẽ (có thể cần điều chỉnh)
    if ratio < -0.3:
        return "LEFT"
    elif ratio > 0.3:
        return "RIGHT"
    else:
        return "STRAIGHT"


# =============================================================================


def check_violation(
    track_state: TrackState,
    signed_distance: float,
    light_state: TrafficLightState,
    frame_idx: int,
    px: float,
    py: float,
    debounce_frames: int = 8
) -> Tuple[str, Tuple[int, int, int]]:
    """
    Kiểm tra vi phạm dựa trên chuyển vùng và trạng thái đèn chi tiết
    
    Args:
        track_state: Trạng thái tracking của xe
        signed_distance: Khoảng cách có dấu đến stopline
        light_state: Trạng thái đèn chi tiết (TrafficLightState)
        frame_idx: Frame hiện tại
        px, py: Vị trí hiện tại của xe
        debounce_frames: Số frame tối thiểu giữa các event
        
    Returns:
        (label, color) để hiển thị
    """
    current_region = -1 if signed_distance < 0 else +1
    
    # Cập nhật vị trí để theo dõi hướng đi
    track_state.positions.append((px, py))
    if len(track_state.positions) > 30:  # Giữ tối đa 30 vị trí
        track_state.positions.pop(0)
    
    # Lần đầu thấy xe
    if not track_state.first_seen:
        track_state.first_seen = True
        track_state.last_region = current_region
        return track_state.label, track_state.color
    
    prev_region = track_state.last_region
    
    # Chỉ trigger khi: Âm → Dương (xe vượt vạch)
    if prev_region == -1 and current_region == +1:
        # Debounce để tránh trigger liên tục
        if (frame_idx - track_state.last_event_frame) >= debounce_frames:
            
            # Rule 4: Đèn vàng bất kỳ -> WARNING
            if light_state.has_any_yellow():
                track_state.label = "WARNING"
                track_state.color = config.COLOR_WARNING
                track_state.last_event_frame = frame_idx
                track_state.last_region = current_region
                return track_state.label, track_state.color
            
            # Phát hiện hướng đi của xe
            direction = detect_vehicle_direction(track_state.positions)
            track_state.direction = direction
            
            # Lấy các hướng được phép
            allowed = light_state.get_allowed_directions()
            
            # Kiểm tra vi phạm dựa trên hướng đi
            is_violation = False
            
            if direction == "STRAIGHT" and not allowed['STRAIGHT']:
                is_violation = True
            elif direction == "LEFT" and not allowed['LEFT']:
                is_violation = True
            elif direction == "RIGHT" and not allowed['RIGHT']:
                is_violation = True
            elif direction == "UNKNOWN":
                # Nếu không xác định được hướng, kiểm tra đèn đỏ chung
                if light_state.circle_red and not light_state.right_green and not light_state.left_green:
                    is_violation = True
            
            if is_violation:
                track_state.label = "VIOLATION"
                track_state.color = config.COLOR_VIOLATION
            else:
                track_state.label = "Safe"
                track_state.color = config.COLOR_SAFE
            
            track_state.last_event_frame = frame_idx
    
    track_state.last_region = current_region
    return track_state.label, track_state.color


def dedup_boxes(xyxy: np.ndarray, cls_ids: np.ndarray, confs: np.ndarray, track_ids: np.ndarray):
    """
    Loại bỏ box trùng lặp theo track_id (giữ box có conf cao nhất)
    """
    if track_ids is None or len(track_ids) == 0:
        return xyxy, cls_ids, confs, track_ids
    
    best = {}
    for i, tid in enumerate(track_ids):
        if tid is None or tid < 0:
            continue
        tid = int(tid)
        if tid not in best or confs[i] > confs[best[tid]]:
            best[tid] = i
    
    if not best:
        return xyxy, cls_ids, confs, track_ids
        
    keep_idx = np.array(sorted(best.values()), dtype=int)
    return xyxy[keep_idx], cls_ids[keep_idx], confs[keep_idx], track_ids[keep_idx]


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def run(
    video_path: str = None,
    output_path: str = None,
    model_path: str = None,
    show_preview: bool = True
):
    """
    Chạy detection vi phạm vượt đèn đỏ
    
    Args:
        video_path: Đường dẫn video input
        output_path: Đường dẫn video output  
        model_path: Đường dẫn model YOLO
        show_preview: Có hiển thị preview không
    """
    import os
    from datetime import datetime
    from pathlib import Path
    
    video_path = video_path or config.DEFAULT_VIDEO
    model_path = model_path or config.MODEL_PATH
    
    # Tự động tạo output folder nếu chưa có
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo tên file với timestamp để không bị ghi đè
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"redlight_violation_{timestamp}.mp4"
        output_path = str(output_dir / output_filename)
    else:
        # Đảm bảo output_path nằm trong output folder
        output_path = str(output_dir / Path(output_path).name)
    
    print(f"[INFO] Output will be saved to: {output_path}")
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer - sử dụng kích thước gốc của video
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)  # Giữ nguyên độ phân giải gốc
    )
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")
    
    # Load model
    print(f"[INIT] Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Initialize components
    calibrator = StoplineCalibrator(duration=config.STOPLINE_CALIBRATION_DURATION)
    calibrator.start()
    
    light_memory = LightMemory()
    tracks: Dict[int, TrackState] = {}
    
    # Counters
    violations = 0
    warnings = 0
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0
    
    frame_idx = 0
    
    # Debug mode - mặc định tắt, bấm 'd' để bật
    debug_on = False
    
    print("\n[START] Processing video...")
    print("Press 'q' to quit, 'd' to toggle debug\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        fps_frame_count += 1
        current_time = time.time()
        
        # Cập nhật FPS mỗi giây
        if current_time - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time
        
        # YOLO inference với tracking
        results = model.track(
            frame,
            persist=True,
            conf=0.25,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMG_SIZE,
            tracker=config.TRACKER,
            verbose=False
        )
        r0 = results[0]
        
        # Parse detections
        if r0.boxes is None or len(r0.boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls_ids = np.zeros((0,), dtype=np.int32)
            confs = np.zeros((0,), dtype=np.float32)
            track_ids = np.zeros((0,), dtype=np.int32)
        else:
            boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_ids = r0.boxes.cls.cpu().numpy().astype(np.int32)
            confs = r0.boxes.conf.cpu().numpy().astype(np.float32)
            track_ids = r0.boxes.id.cpu().numpy().astype(np.int32) if r0.boxes.id is not None else np.array([-1] * len(cls_ids))
        
        # Update light state - dùng detect_traffic_lights để lấy chi tiết
        light_state = detect_traffic_lights(cls_ids)
        light_memory.update(light_state, current_time)
        light_state = light_memory.get(current_time)
        
        # Parse stopline mask
        stopline_mask = np.zeros((h, w), dtype=np.uint8)
        if r0.masks is not None and r0.masks.data is not None:
            masks_data = r0.masks.data.cpu().numpy()
            orig_cls = r0.boxes.cls.cpu().numpy().astype(np.int32) if r0.boxes is not None else np.array([])
            stopline_idxs = np.where(np.isin(orig_cls, config.STOPLINE_CLASS))[0]
            
            if len(stopline_idxs) > 0 and stopline_idxs.max() < len(masks_data):
                m = masks_data[stopline_idxs].max(axis=0)
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m.astype(np.float32), (w, h))
                stopline_mask = (m > 0.5).astype(np.uint8)
        
        # Frame để vẽ
        frame_vis = frame.copy()
        
        # Vẽ traffic light boxes
        draw_traffic_lights(frame_vis, boxes, cls_ids, confs)
        
        # Calibration phase - Vẽ mask stopline trong 5s đầu
        if not calibrator.is_calibrated():
            calibrator.add_mask_points(stopline_mask)
            calibrator.maybe_finish(h, w)
            
            # Vẽ mask stopline với overlay xanh lá
            if np.any(stopline_mask > 0):
                overlay = frame_vis.copy()
                overlay[stopline_mask > 0] = [0, 255, 0]  # Green mask
                frame_vis = cv2.addWeighted(overlay, 0.4, frame_vis, 0.6, 0)
                # Vẽ viền mask
                contours, _ = cv2.findContours(stopline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_vis, contours, -1, (0, 255, 0), 2)
            
            # Hiển thị progress
            elapsed = time.time() - calibrator.start_time
            progress = min(elapsed / calibrator.duration * 100, 100)
            remaining = max(0, calibrator.duration - elapsed)
            # Calibration ngầm - không hiển thị text
        
        # Vẽ stopline cố định sau khi calibrate (giới hạn trong phạm vi mask)
        if calibrator.is_calibrated():
            a, b, c = calibrator.line_abc
            # Tính y tại min_x và max_x
            if abs(b) > 1e-9:
                y1 = int(-(a * calibrator.min_x + c) / b)
                y2 = int(-(a * calibrator.max_x + c) / b)
                # Giới hạn y trong frame
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))
                cv2.line(frame_vis, (calibrator.min_x, y1), (calibrator.max_x, y2), config.COLOR_STOPLINE, 3)
        
        # Dedup boxes
        boxes, cls_ids, confs, track_ids = dedup_boxes(boxes, cls_ids, confs, track_ids)
        
        # Process vehicles
        for xyxy, cls_id, conf, tid in zip(boxes, cls_ids, confs, track_ids):
            # Chỉ xử lý xe
            if cls_id not in config.VEHICLE_CLASSES:
                continue
            if conf < config.CONF_THRESHOLD_VEHICLE:
                continue
            if tid < 0:
                continue
            
            tid = int(tid)
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            px, py = get_bottom_center(xyxy)
            
            # Init track state
            if tid not in tracks:
                tracks[tid] = TrackState()
            
            label = tracks[tid].label
            color = tracks[tid].color
            
            # Check violation nếu đã calibrate
            if calibrator.is_calibrated():
                a, b, c = calibrator.line_abc
                signed = get_signed_distance(px, py, a, b, c) * calibrator.sign_flip
                
                label, color = check_violation(
                    tracks[tid], signed, light_state, frame_idx, px, py
                )
                
                # Cập nhật counter
                if label == "VIOLATION" and tracks[tid].last_event_frame == frame_idx:
                    violations += 1
                    direction = tracks[tid].direction
                    # Chụp screenshot ngay khi phát hiện violation (dùng frame gốc)
                    save_violation_snapshot(frame, "redlight", tid, (x1, y1, x2, y2))
                    print(f"[VIOLATION] Vehicle ID {tid} | Direction: {direction}")
                elif label == "WARNING" and tracks[tid].last_event_frame == frame_idx:
                    warnings += 1
            
            # Vẽ box và label - ngắn gọn
            vehicle_name = config.CLASS_NAMES.get(cls_id, "Vehicle")
            # Chỉ hiện Warning hoặc Violation ngắn gọn
            if label == "VIOLATION":
                display_label = "Violation"
            elif label == "WARNING":
                display_label = "Warning"
            else:
                display_label = f"{vehicle_name}:{tid}"
            draw_bbox_with_label(frame_vis, (x1, y1, x2, y2), display_label, color)
            cv2.circle(frame_vis, (int(px), int(py)), 4, color, -1)
            
            # Debug overlay - hiển thị thông tin chi tiết
            if debug_on:
                track = tracks[tid]
                # Hiển thị signed distance và state
                if calibrator.is_calibrated():
                    a, b, c = calibrator.line_abc
                    signed = get_signed_distance(px, py, a, b, c) * calibrator.sign_flip
                    debug_text = f"d={signed:.0f} dir={track.direction}"
                    cv2.putText(frame_vis, debug_text, (int(x1), int(y2) + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # HUD - sử dụng draw_utils
        simple_light = light_state.get_simple_state()
        light_color = config.COLOR_VIOLATION if simple_light == "RED" else config.COLOR_WARNING if simple_light == "YELLOW" else config.COLOR_SAFE
        
        hud_lines = [
            (f"Light: {simple_light}", light_color),
            (f"Violations: {violations}", config.COLOR_VIOLATION),
            (f"Warnings: {warnings}", config.COLOR_WARNING),
            (f"FPS: {current_fps:.1f}", config.HUD_TEXT_COLOR),
        ]
        draw_info_hud(frame_vis, hud_lines, title="RED LIGHT DETECTION", title_color=config.COLOR_VIOLATION)
        
        # Output - giữ nguyên độ phân giải gốc (không resize)
        writer.write(frame_vis)
        
        # Preview
        if show_preview:
            cv2.imshow("Red Light Violation Detection", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                debug_on = not debug_on
                print(f"[DEBUG] Debug mode: {'ON' if debug_on else 'OFF'}")
        
        # Progress log
        if frame_idx % 100 == 0:
            print(f"[PROGRESS] Frame {frame_idx} | Violations: {violations} | Warnings: {warnings}")
    
    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "=" * 50)
    print("DETECTION SUMMARY")
    print("=" * 50)
    print(f"Total Frames: {frame_idx}")
    print(f"Red Light Violations: {violations}")
    print(f"Yellow Warnings: {warnings}")
    print(f"\nOutput saved: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    run()