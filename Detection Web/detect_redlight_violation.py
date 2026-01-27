# --- CẤU HÌNH ---
#MODEL_PATH = 'Detection Web/assets/best_yolo12s_seg.pt'  # Đường dẫn model của bạn
#VIDEO_SOURCE = 'Detection Web/assets/test_2.mp4' # Thay bằng đường dẫn video hoặc 0 cho webcam
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# --- CẤU HÌNH ---
MODEL_PATH = 'Detection Web/assets/best_yolo12s_seg.pt'  # Đường dẫn model của bạn
VIDEO_SOURCE = 'Detection Web/assets/test_2.mp4' # Thay bằng đường dẫn video hoặc 0 cho webcam
IMG_SIZE = 1280 

# Ngưỡng Confidence
CONF_THRESHOLD_LIGHT = 0.25 
CONF_THRESHOLD_VEHICLE = 0.50 
IOU_THRESHOLD = 0.45

# --- MAPPING CLASS ID ---
VEHICLE_CLASSES = [0, 6, 9, 21, 26] 
RED_LIGHTS = [18]    # light_straight_circle_red
YELLOW_LIGHTS = [19] # light_straight_circle_yellow
GREEN_LIGHTS = [17]  # light_straight_circle_green
STOPLINE_CLASS = 39 

# --- BIẾN TOÀN CỤC ---
stopline_mask_accumulated = [] 
best_stopline_line = None 
line_coeffs = None
is_calibrated = False
calibration_start_time = None

vehicle_last_position = defaultdict(lambda: None) 
vehicle_status = defaultdict(lambda: "Safe") 
violation_count = 0
warning_count = 0

last_traffic_state = "GREEN"
last_light_seen_time = 0
LIGHT_MEMORY_DURATION = 2.0 

# --- HÀM HỖ TRỢ ---
def ransac_filter_stopline(points, threshold=15, iterations=100):
    """
    Sử dụng RANSAC để loại bỏ outliers (phần cong vỉa hè) và giữ lại stopline thẳng.
    
    Args:
        points: numpy array shape (N, 2) với [x, y]
        threshold: Khoảng cách tối đa (pixels) để coi là inlier
        iterations: Số lần lặp RANSAC
    
    Returns:
        cleaned_points: Các điểm inliers (không có phần cong)
    """
    if len(points) < 2:
        return points
    
    best_inliers = []
    best_line = None
    
    for _ in range(iterations):
        # Chọn ngẫu nhiên 2 điểm
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        
        # Tính phương trình đường thẳng: ax + by + c = 0
        # Từ 2 điểm (x1,y1) và (x2,y2)
        x1, y1 = p1
        x2, y2 = p2
        
        # Vector chỉ phương
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        
        # Phương trình: dy*(x - x1) - dx*(y - y1) = 0
        # => dy*x - dx*y + (dx*y1 - dy*x1) = 0
        a = dy
        b = -dx
        c = dx * y1 - dy * x1
        
        # Chuẩn hóa
        norm = np.sqrt(a**2 + b**2)
        if norm < 1e-6:
            continue
        a /= norm
        b /= norm
        c /= norm
        
        # Tính khoảng cách từ tất cả điểm đến đường thẳng
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        
        # Đếm inliers
        inliers = points[distances < threshold]
        
        # Cập nhật best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (a, b, c)
    
    if len(best_inliers) > 0:
        return best_inliers
    else:
        return points

def get_line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    return a, b, c

def calculate_relative_position(point, line_coeffs):
    # < 0: Trục Âm (Dưới vạch/Đang tới)
    # > 0: Trục Dương (Trên vạch/Đã qua)
    x_veh, y_veh = point
    a, b, c = line_coeffs
    if b == 0: return 0
    y_line_at_x = (-c - a * x_veh) / b
    return y_line_at_x - y_veh

def draw_info_box(img, text, pos, bg_color=(0,0,0), txt_color=(255,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (t_w, t_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - t_h - 5), (x + t_w + 5, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, txt_color, thickness)

def main():
    global best_stopline_line, line_coeffs, is_calibrated, calibration_start_time, violation_count, warning_count, last_traffic_state, last_light_seen_time

    print("Loading YOLOv12 with RANSAC Stopline Detection...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    calibration_start_time = time.time()
    last_light_seen_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        current_time = time.time()
        
        # Inference
        results = model.track(frame, persist=True, conf=0.20, iou=IOU_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        result = results[0]

        # ------------------------------------------------
        # 1. LOGIC ĐÈN GIAO THÔNG (PRIORITY ONLY)
        # ------------------------------------------------
        detected_colors = set()
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in RED_LIGHTS: detected_colors.add("RED")
                elif cls_id in YELLOW_LIGHTS: detected_colors.add("YELLOW")
                elif cls_id in GREEN_LIGHTS: detected_colors.add("GREEN")
                
                if cls_id in RED_LIGHTS + YELLOW_LIGHTS + GREEN_LIGHTS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    c_color = (0,0,255) if cls_id in RED_LIGHTS else (0,255,255) if cls_id in YELLOW_LIGHTS else (0,255,0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), c_color, 2)

        instant_state = None
        if "RED" in detected_colors: instant_state = "RED"
        elif "YELLOW" in detected_colors: instant_state = "YELLOW"
        elif "GREEN" in detected_colors: instant_state = "GREEN"
        
        final_traffic_state = last_traffic_state
        if instant_state:
            final_traffic_state = instant_state
            last_traffic_state = instant_state
            last_light_seen_time = current_time
        else:
            if (current_time - last_light_seen_time) > LIGHT_MEMORY_DURATION:
                final_traffic_state = "GREEN"

        # ------------------------------------------------
        # 2. CALIBRATION STOPLINE (RANSAC OUTLIER REMOVAL)
        # ------------------------------------------------
        if not is_calibrated:
            if result.masks is not None:
                for i, cls_id in enumerate(result.boxes.cls):
                    if int(cls_id) == STOPLINE_CLASS:
                        stopline_mask_accumulated.extend(result.masks.xy[i])
            
            # Sau 5 giây thu thập điểm
            if (current_time - calibration_start_time) > 5:
                if len(stopline_mask_accumulated) > 100:
                    points_np = np.array(stopline_mask_accumulated, dtype=np.float32)
                    
                    # ===== BƯỚC 1: RANSAC LỌC NHIỄU (Loại bỏ phần cong) =====
                    print(f"RANSAC: Filtering {len(points_np)} points...")
                    cleaned_points = ransac_filter_stopline(points_np, threshold=8, iterations=100)
                    print(f"RANSAC: Kept {len(cleaned_points)} inliers (removed {len(points_np) - len(cleaned_points)} outliers)")
                    
                    # ===== BƯỚC 2: TÌM VIỀN DƯỚI TỪ ĐIỂM ĐÃ LỌC =====
                    cleaned_points_int = cleaned_points.astype(np.int32)
                    bottom_edge_points = []
                    unique_xs = np.unique(cleaned_points_int[:, 0])
                    
                    for ux in unique_xs:
                        ys_for_ux = cleaned_points_int[cleaned_points_int[:, 0] == ux, 1]
                        max_y = np.max(ys_for_ux)  # Lấy điểm THẤP NHẤT (viền dưới)
                        bottom_edge_points.append([ux, max_y])
                    
                    bottom_points_np = np.array(bottom_edge_points, dtype=np.int32)

                    # ===== BƯỚC 3: VẼ ĐƯỜNG THẲNG TỪ 2 ĐIỂM ĐẦU MÚT =====
                    if len(bottom_points_np) > 2:
                        sorted_indices = np.argsort(bottom_points_np[:, 0])
                        sorted_bottom_points = bottom_points_np[sorted_indices]
                        
                        leftmost_point = sorted_bottom_points[0]
                        rightmost_point = sorted_bottom_points[-1]
                        
                        a, b, c = get_line_equation(leftmost_point, rightmost_point)
                        
                        if b != 0:
                            lefty = int(-c / b)
                            righty = int((-c - a * frame_width) / b)
                            best_stopline_line = ((0, lefty), (frame_width, righty))
                        else:
                            best_stopline_line = (tuple(leftmost_point), tuple(rightmost_point))

                        line_coeffs = (a, b, c)
                        is_calibrated = True
                        print("✓ Stopline (RANSAC Cleaned) Calibrated!")
                else:
                    calibration_start_time = current_time 

        # Vẽ mask stopline trong quá trình calibration
        if not is_calibrated:
            if result.masks is not None:
                for i, cls_id in enumerate(result.boxes.cls):
                    if int(cls_id) == STOPLINE_CLASS:
                        mask_points = result.masks.xy[i]
                        if len(mask_points) > 0:
                            mask_polygon = np.array(mask_points, dtype=np.int32)
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [mask_polygon], (0, 255, 0))
                            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                            cv2.polylines(frame, [mask_polygon], True, (0, 255, 0), 2)
        
        if is_calibrated:
            # Vẽ đường stopline cố định (Màu cam)
            cv2.line(frame, best_stopline_line[0], best_stopline_line[1], (0, 165, 255), 3)

        # ------------------------------------------------
        # 3. XỬ LÝ XE & VI PHẠM (HỆ TỌA ĐỘ)
        # ------------------------------------------------
        count_vehicles_frame = 0
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, track_id, cls_id, conf in zip(boxes, ids, clss, confs):
                track_id = int(track_id)
                cls_id = int(cls_id)
                
                if cls_id in VEHICLE_CLASSES and conf >= CONF_THRESHOLD_VEHICLE:
                    count_vehicles_frame += 1
                    x1, y1, x2, y2 = map(int, box)
                    bottom_center = (int((x1 + x2) / 2), y2)
                    
                    if is_calibrated and line_coeffs:
                        pos_val = calculate_relative_position(bottom_center, line_coeffs)
                        current_pos_state = "NEGATIVE" if pos_val < 0 else "POSITIVE"
                        last_pos_state = vehicle_last_position[track_id]
                        
                        if last_pos_state == "NEGATIVE" and current_pos_state == "POSITIVE":
                            if final_traffic_state == "RED":
                                vehicle_status[track_id] = "Violation"
                                violation_count += 1
                            elif final_traffic_state == "YELLOW":
                                vehicle_status[track_id] = "Warning"
                                warning_count += 1
                            else:
                                vehicle_status[track_id] = "Safe"
                        
                        if last_pos_state is None and current_pos_state == "POSITIVE":
                            vehicle_status[track_id] = "Safe"
                            
                        vehicle_last_position[track_id] = current_pos_state
                    
                    # VẼ UI
                    status = vehicle_status[track_id]
                    if status == "Violation":
                        box_color = (0, 0, 255)
                        label = f"ID:{track_id} VIOLATION"
                    elif status == "Warning":
                        box_color = (0, 255, 255)
                        label = f"ID:{track_id} WARNING"
                    else:
                        box_color = (0, 255, 0)
                        label = f"ID:{track_id}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    draw_info_box(frame, label, (x1, y1), box_color, (0,0,0))
                    cv2.circle(frame, bottom_center, 4, (255, 0, 255), -1)

        # ------------------------------------------------
        # 4. THÔNG TIN
        # ------------------------------------------------
        fps = 1 / (time.time() - current_time) if (time.time() - current_time) > 0 else 0
        
        cv2.rectangle(frame, (10, 10), (320, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 180), (255, 255, 255), 2)
        
        l_color = (0, 255, 0)
        if final_traffic_state == "RED": l_color = (0, 0, 255)
        elif final_traffic_state == "YELLOW": l_color = (0, 255, 255)
        
        cv2.putText(frame, f"FPS: {int(fps)}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Signal: {final_traffic_state}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, l_color, 2)
        cv2.putText(frame, f"Vehicles: {count_vehicles_frame}", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Violations: {violation_count}", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Warnings: {warning_count}", (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("RANSAC Stopline Detection", cv2.resize(frame, (1280, 720)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()