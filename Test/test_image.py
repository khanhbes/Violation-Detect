import cv2
from ultralytics import YOLO

# ================= CẤU HÌNH TRỰC TIẾP TẠI ĐÂY =================
MODEL_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"       # Đường dẫn file model của bạn (.pt)
IMAGE_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/image/anh1.png"      # Đường dẫn ảnh cần test
OUTPUT_PATH = "result.jpg"   # Tên file ảnh kết quả
CONF_THRESHOLD = 0.5         # Độ tin cậy (0.0 - 1.0)
IMG_SIZE = 1280              # Kích thước ảnh đầu vào (nên để 1280 hoặc 640)
# ===============================================================

def detect_image():
    # 1. Tải model
    print(f"🔄 Đang tải model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi không tìm thấy model: {e}")
        return

    # 2. Dự đoán (Inference)
    print(f"🖼️ Đang xử lý ảnh: {IMAGE_PATH}...")
    results = model.predict(
        source=IMAGE_PATH,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        retina_masks=True,  # <--- QUAN TRỌNG: Giúp mask mịn, không bị răng cưa
    )

    # 3. Vẽ và lưu kết quả
    for result in results:
        # Vẽ box và mask lên ảnh
        annotated_frame = result.plot()
        
        # Lưu ảnh
        cv2.imwrite(OUTPUT_PATH, annotated_frame)
        print(f"✅ Đã lưu kết quả tại: {OUTPUT_PATH}")
        
        # (Tùy chọn) Hiển thị lên màn hình
        cv2.imshow("Result", annotated_frame)
        cv2.waitKey(0) # Nhấn phím bất kỳ để tắt cửa sổ
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_image()