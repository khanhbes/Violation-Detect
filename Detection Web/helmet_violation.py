"""
UPDATE per request:
- Default motorcycle label: "Motorcycle ID <id> ..."
- If helmet detected ONCE for that track => latch SAFE:
    - Motorcycle box turns GREEN
    - Label becomes: "Helmet ID <id>"
- Rider boxes still shown:
    - Helmet (25) green, NoHelmet (24) red

Violation snapshot:
- If NoHelmet detected and NOT helmet-latched yet => save screenshot immediately (cooldown).
"""

import cv2
import time
import math
import os
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ================= USER CONFIG =================
MODEL_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"
VIDEO_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/video/test_2.mp4"
IMG_SIZE = 1280

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ================= CLASSES =================
CLS_MOTORCYCLE = 21
CLS_PERSON = 23
CLS_PERSON_NO_HELMET = 24
CLS_PERSON_WITH_HELMET = 25

# ================= THRESHOLDS =================
CONF_MOTORCYCLE = 0.35
CONF_RIDER_NO_HELMET = 0.30
CONF_RIDER_WITH_HELMET = 0.30
CONF_RIDER_FALLBACK = 0.35

# ================= ASSOCIATION =================
MOTO_BOX_EXPAND_RATIO = 0.20
MIN_IOU_RIDER_MOTO = 0.02
MAX_CENTER_DIST_RATIO = 1.30
CENTER_INSIDE_BONUS = 0.35
VERTICAL_GATE_TOP_RATIO = 0.85

# ================= EVENT COOLDOWN =================
EVENT_COOLDOWN_SEC = 2.0

# ================= UI / COLORS (BGR) =================
C_BG = (20, 20, 20)
C_BORDER = (230, 230, 230)

C_GREEN = (60, 220, 60)
C_RED = (50, 50, 255)
C_ORANGE = (255, 180, 80)

C_HELMET = C_GREEN
C_NO_HELMET = C_RED
C_PERSON = (0, 220, 255)

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

def draw_box_tag(frame, bbox, text, color, thickness=3):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.65
    th = 2
    (tw, th_text), _ = cv2.getTextSize(text, font, fs, th)
    tag_h = th_text + 10
    tag_w = tw + 14

    tx1, ty1 = x1, max(0, y1 - tag_h - 3)
    tx2, ty2 = x1 + tag_w, y1
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(frame, text, (tx1 + 7, ty2 - 6), font, fs, (10, 10, 10), th, cv2.LINE_AA)

def put_info_box(frame, lines: List[str], x=15, y=15):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    pad = 10
    line_h = 22
    widths = [cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t in lines]
    box_w = max(widths) + pad * 2
    box_h = line_h * len(lines) + pad
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), C_BG, -1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), C_BORDER, 2)
    ty = y + pad + 16
    for t in lines:
        cv2.putText(frame, t, (x + pad, ty), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)
        ty += line_h


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
    riders: List[Tuple[Tuple[float, float, float, float], int, float]],  # (bbox, cls, conf)
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


def save_violation_snapshot(frame_bgr, moto_bbox, moto_id: int):
    snap = frame_bgr.copy()
    draw_box_tag(snap, moto_bbox, f"No Helmet | ID {moto_id}", C_RED, thickness=4)

    text = "NO HELMET"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 2.0
    th = 5
    (tw, thh), _ = cv2.getTextSize(text, font, fs, th)
    x = max(10, (snap.shape[1] - tw) // 2)
    y = max(thh + 20, 80)
    cv2.putText(snap, text, (x, y), font, fs, C_RED, th, cv2.LINE_AA)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(SNAPSHOT_DIR, f"no_helmet_id{moto_id}_{ts}.jpg")
    cv2.imwrite(out_path, snap)
    return out_path


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

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

    prev_time = time.time()
    fps_smooth = 0.0

    moto_states: Dict[int, TrackState] = defaultdict(TrackState)

    total_violations = 0
    total_safe = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        results = model.track(
            source=frame,
            imgsz=IMG_SIZE,
            conf=0.25,
            iou=0.5,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
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

        # Draw rider boxes
        for rb, rcls, rconf in riders:
            if rcls == CLS_PERSON_WITH_HELMET:
                draw_box_tag(frame, rb, f"Helmet {rconf:.2f}", C_HELMET, thickness=2)
            elif rcls == CLS_PERSON_NO_HELMET:
                draw_box_tag(frame, rb, f"NoHelmet {rconf:.2f}", C_NO_HELMET, thickness=2)
            else:
                draw_box_tag(frame, rb, f"Person {rconf:.2f}", C_PERSON, thickness=1)

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
                        out_path = save_violation_snapshot(frame, moto_bbox, moto_id)
                        state.last_snapshot_time = now
                        total_violations += 1
                        print(f"[SNAPSHOT] {out_path}")

            # ======== YOUR DISPLAY RULE ========
            # Default: "Motorcycle ID ..."
            # If helmet detected once: GREEN + "Helmet ID ..."
            if state.safe_latched:
                live_safe += 1
                draw_box_tag(frame, moto_bbox, f"Helmet ID {moto_id}", C_GREEN, thickness=3)
            else:
                if rider_cls == CLS_PERSON_NO_HELMET:
                    live_violations += 1
                    draw_box_tag(frame, moto_bbox, f"No Helmet ID {moto_id}", C_RED, thickness=3)
                else:
                    draw_box_tag(frame, moto_bbox, f"Motorcycle ID {moto_id} ...", C_ORANGE, thickness=3)

            if DRAW_DEBUG_ASSOC and state.last_rider_bbox is not None:
                mx, my = map(int, bbox_center(moto_bbox))
                rxc, ryc = map(int, bbox_center(state.last_rider_bbox))
                cv2.line(frame, (mx, my), (rxc, ryc), (255, 255, 255), 2)

        # FPS
        cur_time = time.time()
        dt = max(1e-6, cur_time - prev_time)
        inst_fps = 1.0 / dt
        prev_time = cur_time
        fps_smooth = (0.85 * fps_smooth) + (0.15 * inst_fps) if fps_smooth > 0 else inst_fps

        info_lines = [
            f"FPS: {fps_smooth:.1f}",
            f"Vehicles Count: {vehicles_count}",
            f"Violations: {live_violations} (total {total_violations})",
            f"Safe: {live_safe} (total {total_safe})",
        ]
        put_info_box(frame, info_lines, x=15, y=15)

        cv2.imshow("Helmet Violation - Motorcycle", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
