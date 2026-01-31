import cv2
from ultralytics import YOLO

# ================= CẤU HÌNH TẠI CHỖ =================
MODEL_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"       # Đường dẫn file model của bạn (.pt)
IMAGE_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/image/anh1.png"      # Đường dẫn ảnh cần test
OUTPUT_PATH = "result_smooth.jpg"
CONF_THRESHOLD = 0.5
IMG_SIZE = 1280              # (Quan trọng) Tăng độ phân giải đầu vào
# ====================================================

def detect_image_smooth():
    print(f"🔄 Đang tải model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return

    print(f"🖼️ Đang xử lý ảnh: {IMAGE_PATH}...")
    
    # CHẠY NHẬN DIỆN
    results = model.predict(
        source=IMAGE_PATH,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,      # 1. Ảnh đầu vào nét
        retina_masks=True,   # 2. (QUAN TRỌNG NHẤT) Mask độ phân giải gốc, không bị răng cưa
    )

    # VẼ VÀ LƯU
    for result in results:
        # 3. Vẽ kết quả với nét vẽ được khử răng cưa (mặc định của hàm plot)
        # line_width=None: Tự động điều chỉnh độ dày nét cho đẹp
        annotated_frame = result.plot(line_width=None, font_size=None)
        
        cv2.imwrite(OUTPUT_PATH, annotated_frame)
        print(f"✅ Đã lưu ảnh siêu nét tại: {OUTPUT_PATH}")
        
        # Hiển thị (Thu nhỏ lại để vừa màn hình nếu ảnh quá to)
        h, w = annotated_frame.shape[:2]
        if h > 800:
            scale = 800 / h
            annotated_frame = cv2.resize(annotated_frame, (int(w*scale), 800))
            
        cv2.imshow("Smooth Result", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_image_smooth()