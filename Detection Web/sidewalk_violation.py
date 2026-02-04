"""
Multi-Class Traffic Violation Detection - Sidewalk Violation Detection
Model: YOLOv12s-seg
Cấu trúc: Calibration (5s) → Lock → Detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# --- CẤU HÌNH ---
MODEL_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"
VIDEO_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/video/test_2.mp4"

IMG_SIZE = 1280
CONF_THRESHOLD_CALIBRATION = 0.30  # Conf thấp để thu thập nhiều mask
CONF_THRESHOLD_TRACKING = 0.45     # Conf cao để tracking chính xác
IOU_THRESHOLD = 0.5

# --- MAPPING CLASS ---
VEHICLE_CLASSES = [0, 6, 9, 21, 26]  # ambulance, car, fire_truck, motorcycle, police_car
SIDEWALK_CLASS = 27
MEDIAN_CLASS = 20

# --- BIẾN TOÀN CỤC ---
# Calibration
sidewalk_mask_accumulated = []
median_mask_accumulated = []
is_calibrated = False
calibration_start_time = None
CALIBRATION_DURATION = 10.0  # 10 giây

# Forbidden zones (sau khi lock)
forbidden_zones = []  # List các polygon
zone_labels = []      # Label cho từng zone (SIDEWALK/MEDIAN)
cached_overlay = None  # Cache overlay để vẽ nhanh

# Tracking violations
vehicle_status = defaultdict(lambda: "Safe")
violated_ids = set()
violation_details = {}

# Stats
frame_count = 0
fps_history = []


# --- HÀM XỬ LÝ ---
def process_mask_to_polygons(accumulated_mask, zone_type, min_area=1000):
    """
    Xử lý mask thành polygons (tương tự RANSAC filtering)
    """
    if accumulated_mask is None or len(accumulated_mask) == 0:
        print(f"⚠️ No {zone_type} mask accumulated!")
        return []
    
    # Tạo mask binary từ các điểm
    h, w = 720, 1280  # Kích thước frame
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for points in accumulated_mask:
        if len(points) > 0:
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    
    # Morphological Operations - CỰC KỲ NHẸ để giữ nguyên các vùng riêng biệt
    kernel_medium = np.ones((10, 10), np.uint8)  # Giảm thêm từ 15x15
    kernel_small = np.ones((5, 5), np.uint8)     # Giảm thêm từ 10x10
    
    # Chỉ làm mịn nhẹ, không merge
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)  # Lấp lỗ nhỏ
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small)              # Loại nhiễu
    
    # Tìm contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print(f"⚠️ No {zone_type} contour found!")
        return []
    
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Xấp xỉ polygon
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        
        polygons.append(approx_polygon)
        print(f"   {zone_type} polygon: {len(approx_polygon)} points, area: {area:.0f}")
    
    return polygons


def is_point_in_forbidden_zone(point):
    """
    Kiểm tra điểm có trong vùng cấm không
    """
    if not is_calibrated:
        return False, None
    
    for i, polygon in enumerate(forbidden_zones):
        dist = cv2.pointPolygonTest(polygon, point, False)
        if dist > 0:
            return True, zone_labels[i]
    
    return False, None


def draw_forbidden_zones(frame):
    """
    Vẽ vùng cấm đã lock (với cache)
    """
    global cached_overlay
    
    if not is_calibrated or len(forbidden_zones) == 0:
        return frame
    
    # Tạo cached overlay 1 lần duy nhất
    if cached_overlay is None:
        cached_overlay = np.zeros_like(frame, dtype=np.uint8)
        
        for polygon, label in zip(forbidden_zones, zone_labels):
            if label == "SIDEWALK":
                color = (255, 200, 100)  # Xanh dương nhạt
            else:  # MEDIAN
                color = (100, 255, 200)  # Xanh lá nhạt
            
            cv2.fillPoly(cached_overlay, [polygon], color)
    
    # Blend nhanh
    frame = cv2.addWeighted(cached_overlay, 0.3, frame, 0.7, 0)
    
    # Vẽ viền
    for polygon, label in zip(forbidden_zones, zone_labels):
        border_color = (255, 150, 0) if label == "SIDEWALK" else (0, 200, 150)
        cv2.polylines(frame, [polygon], True, border_color, 2)
    
    return frame


def draw_info_box(frame, fps, vehicle_count, calibration_progress=None):
    """
    Vẽ bảng thông tin
    """
    # Background
    cv2.rectangle(frame, (10, 10), (380, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (380, 180), (255, 255, 255), 2)
    
    if not is_calibrated:
        # Đang calibration
        cv2.putText(frame, "STATUS: CALIBRATING", (25, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        progress_text = f"Progress: {int(calibration_progress * 100)}%"
        cv2.putText(frame, progress_text, (25, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        remaining = max(0, CALIBRATION_DURATION * (1 - calibration_progress))
        cv2.putText(frame, f"Time: {remaining:.1f}s", (25, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress bar
        bar_x = 25
        bar_y = 130
        bar_w = 330
        bar_h = 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        fill_w = int(bar_w * calibration_progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
        
    else:
        # Đang monitoring
        cv2.putText(frame, "STATUS: MONITORING", (25, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (25, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (25, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Violations: {len(violated_ids)}", (25, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main():
    global is_calibrated, calibration_start_time, frame_count
    global forbidden_zones, zone_labels, cached_overlay
    
    print("="*60)
    print("🚀 SIDEWALK VIOLATION DETECTION SYSTEM")
    print("="*60)
    print(f"📦 Loading model: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded!")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return
    
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {frame_width}x{frame_height} @ {video_fps} FPS")
    print(f"⏱️  Calibration: {CALIBRATION_DURATION}s")
    print(f"🎯 Target classes: Sidewalk(27), Median(20)")
    print("="*60 + "\n")
    
    calibration_start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_start = time.time()
        frame_count += 1
        current_time = time.time()
        
        # ================================================
        # PHASE 1: CALIBRATION (5 giây đầu)
        # ================================================
        if not is_calibrated:
            elapsed = current_time - calibration_start_time
            calibration_progress = min(elapsed / CALIBRATION_DURATION, 1.0)
            
            # Inference để thu thập mask
            results = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD_CALIBRATION,
                verbose=False
            )
            result = results[0]
            
            # Thu thập mask sidewalk và median
            if result.masks is not None:
                for i, cls_id in enumerate(result.boxes.cls):
                    cls_id = int(cls_id)
                    
                    if cls_id == SIDEWALK_CLASS:
                        mask_points = result.masks.xy[i]
                        sidewalk_mask_accumulated.append(mask_points)
                        
                        # Vẽ mask đang thu thập (màu xanh lá nhạt)
                        if len(mask_points) > 0:
                            mask_polygon = np.array(mask_points, dtype=np.int32)
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [mask_polygon], (100, 255, 100))
                            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                            cv2.polylines(frame, [mask_polygon], True, (0, 255, 0), 1)
                    
                    elif cls_id == MEDIAN_CLASS:
                        mask_points = result.masks.xy[i]
                        median_mask_accumulated.append(mask_points)
                        
                        # Vẽ mask đang thu thập (màu xanh dương nhạt)
                        if len(mask_points) > 0:
                            mask_polygon = np.array(mask_points, dtype=np.int32)
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [mask_polygon], (255, 200, 100))
                            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                            cv2.polylines(frame, [mask_polygon], True, (255, 150, 0), 1)
            
            # Sau đủ thời gian → LOCK
            if elapsed >= CALIBRATION_DURATION:
                print("\n🔒 LOCKING FORBIDDEN ZONES...")
                
                # Xử lý Sidewalk
                if len(sidewalk_mask_accumulated) > 0:
                    sidewalk_polygons = process_mask_to_polygons(
                        sidewalk_mask_accumulated, "Sidewalk", min_area=1000
                    )
                    for poly in sidewalk_polygons:
                        forbidden_zones.append(poly)
                        zone_labels.append("SIDEWALK")
                
                # Xử lý Median
                if len(median_mask_accumulated) > 0:
                    median_polygons = process_mask_to_polygons(
                        median_mask_accumulated, "Median", min_area=1000
                    )
                    for poly in median_polygons:
                        forbidden_zones.append(poly)
                        zone_labels.append("MEDIAN")
                
                is_calibrated = True
                print(f"✅ LOCKED {len(forbidden_zones)} forbidden zones!")
                print(f"   - Sidewalk zones: {zone_labels.count('SIDEWALK')}")
                print(f"   - Median zones: {zone_labels.count('MEDIAN')}")
                print("🟢 Starting violation detection...\n")
            
            # Vẽ UI calibration
            draw_info_box(frame, 0, 0, calibration_progress)
        
        # ================================================
        # PHASE 2: TRACKING & DETECTION
        # ================================================
        else:
            # Tracking vehicles
            results = model.track(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD_TRACKING,
                iou=IOU_THRESHOLD,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
            )
            result = results[0]
            
            vehicle_count = 0
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                
                for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    
                    vehicle_count += 1
                    
                    # Lấy điểm chạm (bottom-center)
                    x1, y1, x2, y2 = map(int, box)
                    bottom_center = (int((x1 + x2) / 2), y2)
                    
                    # Kiểm tra vi phạm
                    is_violation, zone_type = is_point_in_forbidden_zone(bottom_center)
                    
                    if is_violation:
                        # Vi phạm
                        box_color = (0, 0, 255)
                        label = f"VIOLATION #{track_id} ({zone_type})"
                        
                        # Chỉ đếm 1 lần
                        if track_id not in violated_ids:
                            violated_ids.add(track_id)
                            violation_details[track_id] = {
                                "zone": zone_type,
                                "frame": frame_count
                            }
                            print(f"🚨 NEW VIOLATION: Vehicle #{track_id} on {zone_type} (frame {frame_count})")
                        
                        vehicle_status[track_id] = "Violation"
                    else:
                        # An toàn
                        box_color = (0, 255, 0)
                        label = f"Safe #{track_id}"
                        
                        if vehicle_status[track_id] != "Violation":
                            vehicle_status[track_id] = "Safe"
                    
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Vẽ label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (label_w, label_h), _ = cv2.getTextSize(label, font, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 8), 
                                 (x1 + label_w + 6, y1), box_color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 4), 
                               font, 0.6, (255, 255, 255), 2)
                    
                    # Vẽ điểm chạm
                    cv2.circle(frame, bottom_center, 6, box_color, -1)
                    cv2.circle(frame, bottom_center, 8, (255, 255, 255), 1)
            
            # Vẽ vùng cấm đã lock
            frame = draw_forbidden_zones(frame)
            
            # Tính FPS
            frame_time = time.time() - frame_start
            fps_history.append(frame_time)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = 1.0 / (sum(fps_history) / len(fps_history))
            
            # Vẽ UI
            draw_info_box(frame, avg_fps, vehicle_count)
        
        # Hiển thị
        cv2.imshow("Sidewalk Violation Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Violations: {len(violated_ids)}")
    
    if len(violated_ids) > 0:
        print(f"\nViolation Details:")
        for track_id in sorted(violated_ids):
            detail = violation_details.get(track_id, {})
            zone = detail.get("zone", "UNKNOWN")
            frame_num = detail.get("frame", "?")
            print(f"   Vehicle #{track_id}: {zone} violation at frame {frame_num}")
    
    print("="*60)


if __name__ == "__main__":
    main()