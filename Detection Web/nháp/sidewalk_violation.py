"""
Multi-Class Traffic Violation Detection - Sidewalk Violation Detection
Model: YOLOv12s-seg
Structure: Calibration (10s) → Lock → Detection
Uses shared Config class from config/config.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# Import shared config
from config.config import config

# --- CALIBRATION CONFIG ---
CONF_THRESHOLD_CALIBRATION = 0.30  # Low conf for mask collection
CONF_THRESHOLD_TRACKING = 0.45     # High conf for accurate tracking
CALIBRATION_DURATION = 10.0  # 10 seconds

# --- GLOBAL STATE ---
sidewalk_mask_accumulated = []
median_mask_accumulated = []
is_calibrated = False
calibration_start_time = None

# Forbidden zones (after lock)
forbidden_zones = []  # List of polygons
zone_labels = []      # Label for each zone (SIDEWALK/MEDIAN)
cached_overlay = None  # Cache overlay for fast drawing

# Tracking violations
vehicle_status = defaultdict(lambda: "Safe")
violated_ids = set()
violation_details = {}

# Stats
frame_count = 0
fps_history = []


# --- PROCESSING FUNCTIONS ---
def process_mask_to_polygons(accumulated_mask, zone_type, min_area=1000):
    """Process mask into polygons (similar to RANSAC filtering)"""
    if accumulated_mask is None or len(accumulated_mask) == 0:
        print(f"⚠️ No {zone_type} mask accumulated!")
        return []
    
    # Create binary mask from points
    h, w = 720, 1280  # Frame dimensions
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for points in accumulated_mask:
        if len(points) > 0:
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    
    # Morphological Operations - very light to keep separate zones
    kernel_medium = np.ones((10, 10), np.uint8)
    kernel_small = np.ones((5, 5), np.uint8)
    
    # Light smoothing, no merge
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print(f"⚠️ No {zone_type} contour found!")
        return []
    
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Approximate polygon
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        
        polygons.append(approx_polygon)
        print(f"   {zone_type} polygon: {len(approx_polygon)} points, area: {area:.0f}")
    
    return polygons


def is_point_in_forbidden_zone(point):
    """Check if point is in forbidden zone"""
    if not is_calibrated:
        return False, None
    
    for i, polygon in enumerate(forbidden_zones):
        dist = cv2.pointPolygonTest(polygon, point, False)
        if dist > 0:
            return True, zone_labels[i]
    
    return False, None


def draw_forbidden_zones(frame):
    """Draw locked forbidden zones (with cache)"""
    global cached_overlay
    
    if not is_calibrated or len(forbidden_zones) == 0:
        return frame
    
    # Create cached overlay once
    if cached_overlay is None:
        cached_overlay = np.zeros_like(frame, dtype=np.uint8)
        
        for polygon, label in zip(forbidden_zones, zone_labels):
            if label == "SIDEWALK":
                color = (255, 200, 100)  # Light blue
            else:  # MEDIAN
                color = (100, 255, 200)  # Light green
            
            cv2.fillPoly(cached_overlay, [polygon], color)
    
    # Fast blend
    frame = cv2.addWeighted(cached_overlay, 0.3, frame, 0.7, 0)
    
    # Draw borders
    for polygon, label in zip(forbidden_zones, zone_labels):
        border_color = (255, 150, 0) if label == "SIDEWALK" else (0, 200, 150)
        cv2.polylines(frame, [polygon], True, border_color, 2)
    
    return frame


def draw_info_box(frame, fps, vehicle_count, calibration_progress=None):
    """Draw info panel"""
    # Background
    cv2.rectangle(frame, (10, 10), (380, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (380, 180), (255, 255, 255), 2)
    
    if not is_calibrated:
        # Calibrating
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
        # Monitoring
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
    
    print("="*60)
    print("🚀 SIDEWALK VIOLATION DETECTION SYSTEM")
    print("="*60)
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
    print("="*60 + "\n")
    
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
        
        # ================================================
        # PHASE 1: CALIBRATION (first 10 seconds)
        # ================================================
        if not is_calibrated:
            elapsed = current_time - calibration_start_time
            calibration_progress = min(elapsed / CALIBRATION_DURATION, 1.0)
            
            # Inference to collect mask
            results = model.predict(
                frame,
                imgsz=config.IMG_SIZE,
                conf=CONF_THRESHOLD_CALIBRATION,
                verbose=False
            )
            result = results[0]
            
            # Collect sidewalk and median masks
            if result.masks is not None:
                for i, cls_id in enumerate(result.boxes.cls):
                    cls_id = int(cls_id)
                    
                    if cls_id == sidewalk_class:
                        mask_points = result.masks.xy[i]
                        sidewalk_mask_accumulated.append(mask_points)
                        
                        # Draw mask being collected (light green)
                        if len(mask_points) > 0:
                            mask_polygon = np.array(mask_points, dtype=np.int32)
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [mask_polygon], (100, 255, 100))
                            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                            cv2.polylines(frame, [mask_polygon], True, (0, 255, 0), 1)
                    
                    elif cls_id == median_class:
                        mask_points = result.masks.xy[i]
                        median_mask_accumulated.append(mask_points)
                        
                        # Draw mask being collected (light blue)
                        if len(mask_points) > 0:
                            mask_polygon = np.array(mask_points, dtype=np.int32)
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [mask_polygon], (255, 200, 100))
                            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                            cv2.polylines(frame, [mask_polygon], True, (255, 150, 0), 1)
            
            # After time → LOCK
            if elapsed >= CALIBRATION_DURATION:
                print("\n🔒 LOCKING FORBIDDEN ZONES...")
                
                # Process Sidewalk
                if len(sidewalk_mask_accumulated) > 0:
                    sidewalk_polygons = process_mask_to_polygons(
                        sidewalk_mask_accumulated, "Sidewalk", min_area=1000
                    )
                    for poly in sidewalk_polygons:
                        forbidden_zones.append(poly)
                        zone_labels.append("SIDEWALK")
                
                # Process Median
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
            
            # Draw UI calibration
            draw_info_box(frame, 0, 0, calibration_progress)
        
        # ================================================
        # PHASE 2: TRACKING & DETECTION
        # ================================================
        else:
            # Tracking vehicles
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
                    
                    # Get contact point (bottom-center)
                    x1, y1, x2, y2 = map(int, box)
                    bottom_center = (int((x1 + x2) / 2), y2)
                    
                    # Check violation
                    is_violation, zone_type = is_point_in_forbidden_zone(bottom_center)
                    
                    if is_violation:
                        # Violation
                        box_color = config.COLOR_VIOLATION
                        label = f"VIOLATION #{track_id} ({zone_type})"
                        
                        # Count only once
                        if track_id not in violated_ids:
                            violated_ids.add(track_id)
                            violation_details[track_id] = {
                                "zone": zone_type,
                                "frame": frame_count
                            }
                            print(f"🚨 NEW VIOLATION: Vehicle #{track_id} on {zone_type} (frame {frame_count})")
                        
                        vehicle_status[track_id] = "Violation"
                    else:
                        # Safe
                        box_color = config.COLOR_SAFE
                        label = f"Safe #{track_id}"
                        
                        if vehicle_status[track_id] != "Violation":
                            vehicle_status[track_id] = "Safe"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Draw label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (label_w, label_h), _ = cv2.getTextSize(label, font, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 8), 
                                 (x1 + label_w + 6, y1), box_color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 4), 
                               font, 0.6, (255, 255, 255), 2)
                    
                    # Draw contact point
                    cv2.circle(frame, bottom_center, 6, box_color, -1)
                    cv2.circle(frame, bottom_center, 8, (255, 255, 255), 1)
            
            # Draw locked forbidden zones
            frame = draw_forbidden_zones(frame)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps_history.append(frame_time)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = 1.0 / (sum(fps_history) / len(fps_history))
            
            # Draw UI
            draw_info_box(frame, avg_fps, vehicle_count)
        
        # Display
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