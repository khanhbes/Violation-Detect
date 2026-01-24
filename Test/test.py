import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# --- CẤU HÌNH ---
MODEL_PATH = 'Detection Web/assets/best_yolo12s_seg.pt'  # Đường dẫn model của bạn
VIDEO_SOURCE = 'Detection Web/assets/test_2.mp4' # Thay bằng đường dẫn video hoặc 0 cho webcam

# GIẢM NGƯỠNG TỰ TIN XUỐNG ĐỂ BẮT VẬT NHỎ
CONF_THRESHOLD = 0.25 
IOU_THRESHOLD = 0.45
IMG_SIZE = 1280  # <-- Tăng lên để nhìn rõ đèn và vạch (Default là 640)

# Mapping Class ID (Theo dataset của bạn)
# ID 39 là stop_line
# ID 17, 18, 19 là đèn tròn
# ID 1, 2, 3, 4, 5 là các loại mũi tên
CLASS_NAMES = [
    'ambulance', 'arrow_left', 'arrow_right', 'arrow_straight', 'arrow_straight_and_left', 
    'arrow_straight_and_right', 'car', 'dashed_white_line', 'dashed_yellow_line', 'fire_truck', 
    'light_left_green', 'light_left_red', 'light_left_yellow', 'light_right_green', 
    'light_straight_arrow_green', 'light_straight_arrow_red', 'light_straight_arrow_yellow', 
    'light_straight_circle_green', 'light_straight_circle_red', 'light_straight_circle_yellow', 
    'median', 'motorcycle', 'pedestrian_crossing', 'person', 'person_no_helmet', 
    'person_with_helmet', 'police_car', 'sidewalk', 'sign_no_car', 'sign_no_entry', 
    'sign_no_left_and_return', 'sign_no_left_turn', 'sign_no_parking', 'sign_no_return', 
    'sign_no_right_and_return', 'sign_no_right_turn', 'sign_no_stopping', 'solid_white_line', 
    'solid_yellow_line', 'stop_line'
]

# Định nghĩa nhóm màu sắc để vẽ cho dễ nhìn
def get_color(cls_id):
    # Xe cộ: Xanh lá
    if cls_id in [0, 6, 9, 21, 26]: return (0, 255, 0)
    # Đèn Đỏ: Đỏ
    if cls_id in [11, 15, 18]: return (0, 0, 255)
    # Đèn Vàng: Vàng
    if cls_id in [12, 16, 19]: return (0, 255, 255)
    # Đèn Xanh: Xanh Cyan
    if cls_id in [10, 13, 14, 17]: return (255, 255, 0)
    # Stopline: Tím
    if cls_id == 39: return (255, 0, 255)
    # Mũi tên: Cam
    if cls_id in [1, 2, 3, 4, 5]: return (0, 165, 255)
    # Còn lại: Xám
    return (200, 200, 200)

def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Kiểm tra xem video có mở được không
    if not cap.isOpened():
        print("Error: Không thể mở video.")
        return

    # Lấy thông số video để tính toán resize nếu cần
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Resolution: {width}x{height}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # --- INFERENCE ---
        # Quan trọng: imgsz=IMG_SIZE giúp detect vật nhỏ tốt hơn
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        result = results[0]

        # --- VẼ MASK (SEGMENTATION) ---
        # Vẽ mask trước để nó nằm dưới bounding box
        if result.masks is not None:
            # Code này sẽ vẽ đè màu lên vùng segmentation
            masks = result.masks.xy
            for i, mask in enumerate(masks):
                cls_id = int(result.boxes.cls[i])
                # Chỉ vẽ mask cho Stopline (39) hoặc Mũi tên (1-5) để kiểm tra
                if cls_id == 39 or cls_id in [1,2,3,4,5]:
                    cnt = mask.astype(np.int32)
                    cv2.drawContours(frame, [cnt], -1, get_color(cls_id), 2) # Vẽ viền mask

        # --- VẼ BOUNDING BOX ---
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Bỏ qua logic phức tạp, vẽ TOÀN BỘ để debug
                color = get_color(cls_id)
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ nhãn nền đen chữ trắng
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Show kết quả
        # Resize lại cửa sổ hiển thị nếu video quá to (4K)
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("DEBUG MODE - Check All Classes", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()