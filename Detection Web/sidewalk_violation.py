"""
Multi-Class Traffic Violation Detection - Sidewalk Violation Detection
Model: YOLOv12s-seg
Structure: Calibration (10s) → Lock → Detection → Export Video
Uses shared Config class from config/config.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os

# Import shared config
from config.config import config

# --- CALIBRATION CONFIG ---
CONF_THRESHOLD_CALIBRATION = 0.35
CONF_THRESHOLD_TRACKING = 0.45
CALIBRATION_DURATION = 10.0

# --- OUTPUT ---
OUTPUT_VIDEO_PATH = str(config.OUTPUT_DIR / "output_sidewalk_violation.avi")

# --- GLOBAL STATE ---
accum_sidewalk = None
accum_median = None
calibration_frame_count = 0
is_calibrated = False
calibration_start_time = None

forbidden_zones = []
zone_labels = []
cached_overlay = None

vehicle_status = defaultdict(lambda: "Safe")
violated_ids = set()
violation_details = {}

frame_count = 0
fps_history = []

# Video Writer
out = None


# --- PROCESSING FUNCTIONS ---
def process_mask_to_polygons_from_mask(binary_mask, zone_type, min_area=10000):
    if binary_mask is None or np.sum(binary_mask) == 0:
        print(f"⚠️ No {zone_type} mask after voting!")
        return []

    kernel_small = np.ones((5, 5), np.uint8)
    kernel_med = np.ones((9, 9), np.uint8)

    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_med, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    if zone_type == "Sidewalk":
        cleaned = cv2.erode(cleaned, kernel_small, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        hull = cv2.convexHull(contour)

        epsilon = 0.012 * cv2.arcLength(hull, True)
        approx_polygon = cv2.approxPolyDP(hull, epsilon, True)

        rect = cv2.minAreaRect(hull)
        w, h = rect[1]
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 999
        if aspect > 12:
            continue

        polygons.append(approx_polygon)
        print(f"   {zone_type} polygon: {len(approx_polygon)} points, area: {area:.0f}")

    return polygons


def check_violation_with_points(box, threshold_points=2, offset_ratio=0.3):
    if not is_calibrated:
        return False, None

    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1

    y_check = int(y2 - height * offset_ratio)

    check_points = [
        (x1, y_check),
        (int((x1 + x2) / 2), y_check),
        (x2, y_check)
    ]

    in_zone_count = 0
    zone_type_found = None

    for point in check_points:
        for i, polygon in enumerate(forbidden_zones):
            dist = cv2.pointPolygonTest(polygon, point, False)
            if dist > 0:
                in_zone_count += 1
                zone_type_found = zone_labels[i]
                break

    if in_zone_count >= threshold_points:
        return True, zone_type_found

    return False, None


def draw_forbidden_zones(frame):
    global cached_overlay

    if not is_calibrated or len(forbidden_zones) == 0:
        return frame

    if cached_overlay is None:
        cached_overlay = np.zeros_like(frame, dtype=np.uint8)

        for polygon, label in zip(forbidden_zones, zone_labels):
            if label == "SIDEWALK":
                color = (255, 200, 100)
            else:
                color = (100, 255, 200)

            cv2.fillPoly(cached_overlay, [polygon], color)

    frame = cv2.addWeighted(cached_overlay, 0.3, frame, 0.7, 0, frame)

    for polygon, label in zip(forbidden_zones, zone_labels):
        border_color = (255, 150, 0) if label == "SIDEWALK" else (0, 200, 150)
        cv2.polylines(frame, [polygon], True, border_color, 2)

    return frame


def draw_info_box(frame, fps, vehicle_count, calibration_progress=None):
    cv2.rectangle(frame, (10, 10), (380, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (380, 180), (255, 255, 255), 2)

    if not is_calibrated:
        cv2.putText(frame, "STATUS: CALIBRATING", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        progress_text = f"Progress: {int(calibration_progress * 100)}%"
        cv2.putText(frame, progress_text, (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        remaining = max(0, CALIBRATION_DURATION * (1 - calibration_progress))
        cv2.putText(frame, f"Time: {remaining:.1f}s", (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        bar_x = 25
        bar_y = 130
        bar_w = 330
        bar_h = 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        fill_w = int(bar_w * calibration_progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)

    else:
        cv2.putText(frame, "STATUS: MONITORING", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_SAFE, 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Violations: {len(violated_ids)}", (25, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_VIOLATION, 2)


def main():
    global is_calibrated, calibration_start_time, frame_count
    global forbidden_zones, zone_labels, cached_overlay
    global accum_sidewalk, accum_median, calibration_frame_count
    global out
    global OUTPUT_VIDEO_PATH

    print("=" * 60)
    print("🚀 SIDEWALK VIOLATION DETECTION SYSTEM (with Video Export)")
    print("=" * 60)
    print(f"📦 Loading model: {config.MODEL_PATH}")

    model = YOLO(config.MODEL_PATH)
    print("✅ Model loaded!")

    print(f"📹 Opening video: {config.DEFAULT_VIDEO}")
    cap = cv2.VideoCapture(config.DEFAULT_VIDEO)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {config.DEFAULT_VIDEO}")
        return

    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Video: {frame_width}x{frame_height} @ {video_fps} FPS")
    print(f"⏱️  Calibration: {CALIBRATION_DURATION}s")
    print(f"🎯 Target classes: Sidewalk({config.SIDEWALK_CLASS}), Median({config.MEDIAN_CLASS})")
    print(f"💾 Output video: {OUTPUT_VIDEO_PATH}")
    print("=" * 60 + "\n")

    # Initialize VideoWriter
    if OUTPUT_VIDEO_PATH:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (frame_width, frame_height))
        if not out.isOpened():
            print("⚠️ XVID failed → trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = OUTPUT_VIDEO_PATH.replace('.avi', '.mp4')
            out = cv2.VideoWriter(out_path, fourcc, video_fps, (frame_width, frame_height))
        if not out.isOpened():
            print("❌ Cannot create VideoWriter!")
            out = None
        else:
            print(f"📼 Recording output: {OUTPUT_VIDEO_PATH}")

    accum_sidewalk = np.zeros((frame_height, frame_width), dtype=np.float32)
    accum_median = np.zeros((frame_height, frame_width), dtype=np.float32)
    calibration_frame_count = 0

    calibration_start_time = time.time()

    # Get class IDs from config
    sidewalk_class = config.SIDEWALK_CLASS[0] if config.SIDEWALK_CLASS else 27
    median_class = config.MEDIAN_CLASS[0] if config.MEDIAN_CLASS else 20

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_start = time.time()
        frame_count += 1
        current_time = time.time()

        if not is_calibrated:
            elapsed = current_time - calibration_start_time
            calibration_progress = min(elapsed / CALIBRATION_DURATION, 1.0)

            results = model.predict(
                frame,
                imgsz=config.IMG_SIZE,
                conf=CONF_THRESHOLD_CALIBRATION,
                verbose=False
            )
            result = results[0]

            if result.masks is not None:
                for i, cls_id in enumerate(result.boxes.cls):
                    cls_id = int(cls_id)
                    mask_xy = result.masks.xy[i]
                    if len(mask_xy) < 5:
                        continue

                    temp_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    pts = np.array(mask_xy, dtype=np.int32)
                    cv2.fillPoly(temp_mask, [pts], 255)

                    if cls_id == sidewalk_class:
                        accum_sidewalk += temp_mask.astype(np.float32) / 255.0
                    elif cls_id == median_class:
                        accum_median += temp_mask.astype(np.float32) / 255.0

            if result.masks is not None and len(result.masks) > 0:
                calibration_frame_count += 1

            if elapsed >= CALIBRATION_DURATION:
                print("\n🔒 LOCKING FORBIDDEN ZONES...")

                voting_threshold_sidewalk = 0.38
                voting_threshold_median = 0.30

                if np.any(accum_sidewalk > 0):
                    sidewalk_final = (accum_sidewalk >= voting_threshold_sidewalk * calibration_frame_count).astype(np.uint8) * 255
                    sidewalk_polygons = process_mask_to_polygons_from_mask(sidewalk_final, "Sidewalk", min_area=10000)
                    for poly in sidewalk_polygons:
                        forbidden_zones.append(poly)
                        zone_labels.append("SIDEWALK")

                if np.any(accum_median > 0):
                    median_final = (accum_median >= voting_threshold_median * calibration_frame_count).astype(np.uint8) * 255
                    median_polygons = process_mask_to_polygons_from_mask(median_final, "Median", min_area=8000)
                    for poly in median_polygons:
                        forbidden_zones.append(poly)
                        zone_labels.append("MEDIAN")

                is_calibrated = True
                print(f"✅ LOCKED {len(forbidden_zones)} forbidden zones!")
                print(f"   - Sidewalk zones: {zone_labels.count('SIDEWALK')}")
                print(f"   - Median zones: {zone_labels.count('MEDIAN')}")
                print(f"   Frames with detection: {calibration_frame_count}")
                print("🟢 Starting violation detection...\n")

            draw_info_box(frame, 0, 0, calibration_progress)

        else:
            results = model.track(
                frame,
                imgsz=config.IMG_SIZE,
                conf=CONF_THRESHOLD_TRACKING,
                iou=config.IOU_THRESHOLD,
                persist=True,
                verbose=False,
                tracker=config.TRACKER
            )
            result = results[0]

            vehicle_count = 0

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                    if cls_id not in config.VEHICLE_CLASSES:
                        continue

                    vehicle_count += 1

                    is_violation, zone_type = check_violation_with_points(box, threshold_points=2, offset_ratio=0.3)

                    if is_violation:
                        box_color = config.COLOR_VIOLATION
                        label = f"VIOLATION #{track_id} ({zone_type})"

                        if track_id not in violated_ids:
                            violated_ids.add(track_id)
                            violation_details[track_id] = {
                                "zone": zone_type,
                                "frame": frame_count
                            }
                            print(f"🚨 NEW VIOLATION: Vehicle #{track_id} on {zone_type} (frame {frame_count})")

                        vehicle_status[track_id] = "Violation"
                    else:
                        box_color = config.COLOR_SAFE
                        label = f"Safe #{track_id}"

                        if vehicle_status[track_id] != "Violation":
                            vehicle_status[track_id] = "Safe"

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (label_w, label_h), _ = cv2.getTextSize(label, font, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 8),
                                  (x1 + label_w + 6, y1), box_color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                font, 0.6, (255, 255, 255), 2)

                    # Debug points (yellow)
                    height = y2 - y1
                    y_check = int(y2 - height * 0.3)
                    cv2.circle(frame, (x1, y_check), 6, (0, 255, 255), -1)
                    cv2.circle(frame, (int((x1 + x2)/2), y_check), 7, (0, 255, 255), -1)
                    cv2.circle(frame, (x2, y_check), 6, (0, 255, 255), -1)

                    cv2.circle(frame, ((x1 + x2)//2, y2), 5, box_color, -1)

            frame = draw_forbidden_zones(frame)

            frame_time = time.time() - frame_start
            fps_history.append(frame_time)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = 1.0 / (sum(fps_history) / len(fps_history)) if fps_history else 0

            draw_info_box(frame, avg_fps, vehicle_count)

        # Write frame to video
        if out is not None:
            out.write(frame)

        cv2.imshow("Sidewalk Violation Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
        print(f"\n✅ Video output saved: {os.path.abspath(OUTPUT_VIDEO_PATH)}")

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Violations: {len(violated_ids)}")

    if len(violated_ids) > 0:
        print(f"\nViolation Details:")
        for track_id in sorted(violated_ids):
            detail = violation_details.get(track_id, {})
            zone = detail.get("zone", "UNKNOWN")
            frame_num = detail.get("frame", "?")
            print(f"   Vehicle #{track_id}: {zone} violation at frame {frame_num}")

    print("=" * 60)


if __name__ == "__main__":
    main()