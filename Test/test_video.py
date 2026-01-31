import cv2
from ultralytics import YOLO

# ================= CẤU HÌNH TRỰC TIẾP TẠI ĐÂY =================
MODEL_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"         # Đường dẫn file model (.pt)
VIDEO_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/video/test_2.mp4"  # Đường dẫn video cần test
OUTPUT_PATH = "output.mp4"     # Tên video kết quả xuất ra
CONF_THRESHOLD = 0.5           # Độ tin cậy
IMG_SIZE = 1280                # Kích thước xử lý
# ===============================================================

def detect_video():
    # 1. Tải model
    print(f"🔄 Đang tải model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return

    # 2. Mở video đầu vào
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {VIDEO_PATH}")
        return

    # Lấy thông số video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Khởi tạo bộ ghi video (VideoWriter)
    # Định dạng mp4v cho file .mp4
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("🎥 Bắt đầu xử lý video... Nhấn 'q' để dừng sớm.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Hết video

        # 4. Chạy nhận diện trên từng khung hình
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            retina_masks=True,  # <--- QUAN TRỌNG: Chống răng cưa
            verbose=False,      # Tắt log spam trên terminal
            stream=True         # Giúp tiết kiệm bộ nhớ khi chạy video dài
        )

        # 5. Vẽ kết quả
        for result in results:
            annotated_frame = result.plot()
            
            # Ghi khung hình đã vẽ vào file output
            out.write(annotated_frame)

            # Hiển thị trực tiếp (Optional)
            cv2.imshow("YOLO Detection", annotated_frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Dọn dẹp
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Hoàn tất! Video đã lưu tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    detect_video()