"""
Helmet Violation Detection (Motorcycle - No Helmet)
Uses shared Config class from config/config.py

Features:
- Motorcycle detection with rider association
- Helmet detection (latch SAFE when helmet detected once)
- Violation snapshot saving
"""

import sys
import cv2
import time
import math
import os
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path so we can import config module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared config and draw utilities
from config.config import config
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, save_violation_snapshot

# ================= HELMET-SPECIFIC CONFIG =================
# These class IDs are specific to helmet detection and not in shared config
CLS_MOTORCYCLE = 21
CLS_PERSON = 23  
CLS_PERSON_NO_HELMET = 24
CLS_PERSON_WITH_HELMET = 25

# Confidence thresholds
CONF_MOTORCYCLE = 0.35
CONF_RIDER_NO_HELMET = 0.30
CONF_RIDER_WITH_HELMET = 0.30
CONF_RIDER_FALLBACK = 0.35

# Association parameters
MOTO_BOX_EXPAND_RATIO = 0.20
MIN_IOU_RIDER_MOTO = 0.02
MAX_CENTER_DIST_RATIO = 1.30
CENTER_INSIDE_BONUS = 0.35
VERTICAL_GATE_TOP_RATIO = 0.85

# Event cooldown
EVENT_COOLDOWN_SEC = 2.0

# Snapshot directory
SNAPSHOT_DIR = str(config.OUTPUT_DIR / "violations")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Colors (BGR)
C_BG = (20, 20, 20)
C_BORDER = (230, 230, 230)
C_GREEN = config.COLOR_SAFE
C_RED = config.COLOR_VIOLATION
C_ORANGE = config.COLOR_STOPLINE
C_HELMET = C_GREEN
C_NO_HELMET = C_RED
C_PERSON = config.COLOR_WARNING

# Debug mode - mặc định tắt, bấm 'd' để bật
DRAW_DEBUG_ASSOC = False


# ================ UTILS ================
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def bbox_area(x1, y1, x2, y2) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = bbox_area(ix1, iy1, ix2, iy2)
    if inter <= 0:
        return 0.0
    ua = bbox_area(ax1, ay1, ax2, ay2) + bbox_area(bx1, by1, bx2, by2) - inter
    return inter / ua if ua > 0 else 0.0

def bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def bbox_diag(b: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return math.hypot(x2 - x1, y2 - y1)

def expand_bbox(b: Tuple[float, float, float, float], ratio: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    ex, ey = bw * ratio, bh * ratio
    nx1 = clamp(x1 - ex, 0, w - 1)
    ny1 = clamp(y1 - ey, 0, h - 1)
    nx2 = clamp(x2 + ex, 0, w - 1)
    ny2 = clamp(y2 + ey, 0, h - 1)
    return (nx1, ny1, nx2, ny2)

def point_in_bbox(px: float, py: float, b: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = b
    return (x1 <= px <= x2) and (y1 <= py <= y2)

# draw_box_tag và put_info_box đã được thay thế bởi draw_utils.draw_bbox_with_label và draw_info_hud


# ================ STATE ================
@dataclass
class TrackState:
    safe_latched: bool = False
    last_snapshot_time: float = 0.0
    last_rider_cls: Optional[int] = None
    last_rider_bbox: Optional[Tuple[float, float, float, float]] = None


# ================ ASSOCIATION ================
def associate_rider_to_motorcycle(
    moto_bbox: Tuple[float, float, float, float],
    riders: List[Tuple[Tuple[float, float, float, float], int, float]],
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[Tuple[float, float, float, float], int, float]]:
    mexp = expand_bbox(moto_bbox, MOTO_BOX_EXPAND_RATIO, frame_w, frame_h)
    mx, my = bbox_center(mexp)
    mdiag = max(1.0, bbox_diag(mexp))

    mx1, my1, mx2, my2 = mexp
    m_h = max(1.0, my2 - my1)
    y_gate = my1 + VERTICAL_GATE_TOP_RATIO * m_h

    best = None
    best_score = -1e9

    for rb, rcls, rconf in riders:
        rx, ry = bbox_center(rb)
        inside = point_in_bbox(rx, ry, mexp)
        iou = bbox_iou(mexp, rb)

        if (not inside) and (iou < MIN_IOU_RIDER_MOTO):
            continue
        if ry > y_gate:
            continue

        dist = math.hypot(rx - mx, ry - my)
        if dist > MAX_CENTER_DIST_RATIO * mdiag:
            continue

        dist_score = 1.0 - (dist / (MAX_CENTER_DIST_RATIO * mdiag))
        score = (2.0 * rconf) + (1.2 * iou) + (1.0 * dist_score) + (CENTER_INSIDE_BONUS if inside else 0.0)

        if rcls in (CLS_PERSON_NO_HELMET, CLS_PERSON_WITH_HELMET):
            score += 0.20

        if score > best_score:
            best_score = score
            best = (rb, rcls, rconf)

    return best


# ================ MAIN ================
def main():
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(
            "Missing ultralytics. Install with:\n"
            "  pip install ultralytics opencv-python\n\n"
            f"Import error: {e}"
        )

    print(f"[INFO] Loading model: {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)

    print(f"[INFO] Opening video: {config.DEFAULT_VIDEO}")
    cap = cv2.VideoCapture(config.DEFAULT_VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {config.DEFAULT_VIDEO}")

    prev_time = time.time()
    fps_smooth = 0.0

    moto_states: Dict[int, TrackState] = defaultdict(TrackState)

    total_violations = 0
    total_safe = 0
    
    # Debug mode - mặc định tắt, bấm 'd' để bật
    debug_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        results = model.track(
            source=frame,
            imgsz=config.IMG_SIZE,
            conf=0.25,
            iou=config.IOU_THRESHOLD,
            persist=True,
            verbose=False,
            tracker=config.TRACKER,
        )
        r0 = results[0]

        motos: List[Tuple[Tuple[float, float, float, float], int, float]] = []
        riders: List[Tuple[Tuple[float, float, float, float], int, float]] = []

        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
            boxes_cls = r0.boxes.cls.cpu().numpy().astype(int)
            boxes_conf = r0.boxes.conf.cpu().numpy()
            boxes_id = None
            if hasattr(r0.boxes, "id") and r0.boxes.id is not None:
                boxes_id = r0.boxes.id.cpu().numpy().astype(int)

            for i in range(len(boxes_xyxy)):
                bbox = tuple(boxes_xyxy[i].tolist())
                cls_id = int(boxes_cls[i])
                conf = float(boxes_conf[i])

                if cls_id == CLS_MOTORCYCLE and conf >= CONF_MOTORCYCLE:
                    track_id = int(boxes_id[i]) if boxes_id is not None else i
                    motos.append((bbox, track_id, conf))
                elif cls_id == CLS_PERSON_NO_HELMET and conf >= CONF_RIDER_NO_HELMET:
                    riders.append((bbox, cls_id, conf))
                elif cls_id == CLS_PERSON_WITH_HELMET and conf >= CONF_RIDER_WITH_HELMET:
                    riders.append((bbox, cls_id, conf))
                elif cls_id == CLS_PERSON and conf >= CONF_RIDER_FALLBACK:
                    riders.append((bbox, cls_id, conf))

        # Draw rider boxes - sử dụng draw_utils
        for rb, rcls, rconf in riders:
            if rcls == CLS_PERSON_WITH_HELMET:
                draw_bbox_with_label(frame, rb, f"Helmet {rconf:.2f}", C_HELMET)
            elif rcls == CLS_PERSON_NO_HELMET:
                draw_bbox_with_label(frame, rb, f"NoHelmet {rconf:.2f}", C_NO_HELMET)
            else:
                draw_bbox_with_label(frame, rb, f"Person {rconf:.2f}", C_PERSON)

        now = time.time()
        vehicles_count = len(motos)
        live_violations = 0
        live_safe = 0

        for moto_bbox, moto_id, _ in motos:
            state = moto_states[moto_id]

            best_rider = associate_rider_to_motorcycle(moto_bbox, riders, w, h)

            rider_cls = None
            if best_rider is not None:
                rb, rider_cls, rc = best_rider
                state.last_rider_bbox = rb
                state.last_rider_cls = rider_cls

                # Helmet latch: once -> forever
                if rider_cls == CLS_PERSON_WITH_HELMET and not state.safe_latched:
                    state.safe_latched = True
                    total_safe += 1

                # Immediate snapshot on no-helmet (ONLY if not safe-latched)
                if rider_cls == CLS_PERSON_NO_HELMET and not state.safe_latched:
                    if (now - state.last_snapshot_time) >= EVENT_COOLDOWN_SEC:
                        save_violation_snapshot(frame, "no_helmet", moto_id, moto_bbox, vehicle_class="motorcycle")
                        state.last_snapshot_time = now
                        total_violations += 1

            # Display rule - sử dụng draw_utils
            if state.safe_latched:
                live_safe += 1
                draw_bbox_with_label(frame, moto_bbox, f"Motorcycle:{moto_id} Helmet", C_GREEN)
            else:
                if rider_cls == CLS_PERSON_NO_HELMET:
                    live_violations += 1
                    draw_bbox_with_label(frame, moto_bbox, f"Motorcycle:{moto_id} NO HELMET", C_RED)
                else:
                    draw_bbox_with_label(frame, moto_bbox, f"Motorcycle:{moto_id}", C_ORANGE)

            if debug_on and state.last_rider_bbox is not None:
                mx, my = map(int, bbox_center(moto_bbox))
                rxc, ryc = map(int, bbox_center(state.last_rider_bbox))
                cv2.line(frame, (mx, my), (rxc, ryc), (255, 255, 255), 2)

        # FPS
        cur_time = time.time()
        dt = max(1e-6, cur_time - prev_time)
        inst_fps = 1.0 / dt
        prev_time = cur_time
        fps_smooth = (0.85 * fps_smooth) + (0.15 * inst_fps) if fps_smooth > 0 else inst_fps

        # HUD - sử dụng draw_utils
        hud_lines = [
            (f"FPS: {fps_smooth:.1f}", config.HUD_TEXT_COLOR),
            (f"Vehicles: {vehicles_count}", config.HUD_TEXT_COLOR),
            (f"Violations: {live_violations} (total {total_violations})", C_RED),
            (f"Safe: {live_safe} (total {total_safe})", C_GREEN),
        ]
        draw_info_hud(frame, hud_lines, title="HELMET DETECTION", title_color=config.COLOR_WARNING)

        cv2.imshow("Helmet Violation - Motorcycle", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord('d'):
            debug_on = not debug_on

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()