import cv2
from ultralytics import YOLO

# ================= CẤU HÌNH TẠI CHỖ =================
MODEL_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"         # Đường dẫn file model (.pt)
VIDEO_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/video/test_2.mp4"  # Đường dẫn video cần test
OUTPUT_PATH = "output_smooth.mp4"
CONF_THRESHOLD = 0.5
IMG_SIZE = 1280              # Tăng lên 1280 để nét hơn (nhưng sẽ chạy chậm hơn 640)
# ====================================================

def detect_video_smooth():
    print(f"🔄 Đang tải model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Không mở được video.")
        return

    # Lấy thông số video gốc
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Khởi tạo VideoWriter
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("🎥 Đang chạy... (Nhấn 'q' để dừng)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # CHẠY INFERENCE
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,     # Giữ độ phân giải cao
            retina_masks=True,  # <--- KHÓA CHỐNG RĂNG CƯA
            stream=True,        # Tiết kiệm RAM
            verbose=False
        )

        for result in results:
            # Vẽ kết quả lên frame
            annotated_frame = result.plot()
            out.write(annotated_frame)
            
            # Hiển thị (Resize nhỏ để xem trước cho mượt)
            view_frame = cv2.resize(annotated_frame, (1024, int(1024*h/w)))
            cv2.imshow("Anti-aliasing Detection", view_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Xong! Video đã lưu tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    detect_video_smooth()