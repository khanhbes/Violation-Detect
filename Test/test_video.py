import cv2
from ultralytics import YOLO
import torch.nn as nn
import ultralytics.utils.loss

# 1. FIX LỖI LOSS
try:
    class DummyLoss(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
    for attr in ['BCEDiceLoss', 'MultiChannelDiceLoss']:
        if not hasattr(ultralytics.utils.loss, attr):
            setattr(ultralytics.utils.loss, attr, DummyLoss)
except: pass

# 2. CẤU HÌNH
VIDEO_PATH = "Detection Web/assets/test_2.mp4"
MODEL_PATH = "Detection Web/assets/best_yolo12s_seg.pt"  # Đổi thành đường dẫn video của bạn

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Chạy dự đoán (stream=True giúp mượt hơn)
        results = model(frame, stream=True)

        for result in results:
            annotated_frame = result.plot()
            cv2.imshow("YOLOv12 Video Test", annotated_frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()