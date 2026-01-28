import cv2
import torch.nn as nn
from ultralytics import YOLO

# --- IMPORT CÁC MODULE CẦN VÁ ---
import ultralytics.nn.modules.head
import ultralytics.nn.modules.block
import ultralytics.utils.loss

# ====================================================
# KHU VỰC VÁ LỖI (MONKEY PATCHING AREA)
# ====================================================
print(">>> Đang thực hiện vá lỗi thư viện để tương thích model custom...")

try:
    # 1. VÁ LỖI BLOCK: Proto26 -> Proto
    # Proto là module tạo Prototype Masks trong bài toán Segmentation
    if not hasattr(ultralytics.nn.modules.block, 'Proto26'):
        if hasattr(ultralytics.nn.modules.block, 'Proto'):
            setattr(ultralytics.nn.modules.block, 'Proto26', ultralytics.nn.modules.block.Proto)
            print("   + Đã map: Proto26 -> Proto")

    # 2. VÁ LỖI HEAD: Segment26 -> Segment
    if not hasattr(ultralytics.nn.modules.head, 'Segment26'):
        if hasattr(ultralytics.nn.modules.head, 'Segment'):
            setattr(ultralytics.nn.modules.head, 'Segment26', ultralytics.nn.modules.head.Segment)
            print("   + Đã map: Segment26 -> Segment")
    
    # 3. VÁ LỖI HEAD: Detect26 -> Detect (Dự phòng nếu gặp lỗi Detect26)
    if not hasattr(ultralytics.nn.modules.head, 'Detect26'):
        if hasattr(ultralytics.nn.modules.head, 'Detect'):
            setattr(ultralytics.nn.modules.head, 'Detect26', ultralytics.nn.modules.head.Detect)

    # 4. VÁ LỖI BLOCK: C2f26 -> C2f (Dự phòng)
    if not hasattr(ultralytics.nn.modules.block, 'C2f26'):
         if hasattr(ultralytics.nn.modules.block, 'C2f'):
            setattr(ultralytics.nn.modules.block, 'C2f26', ultralytics.nn.modules.block.C2f)

    # 5. VÁ LỖI LOSS FUNCTION
    class DummyLoss(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, *args, **kwargs): return 0
        
    missing_losses = ['BCEDiceLoss', 'MultiChannelDiceLoss', 'FocalLoss', 'VarifocalLoss']
    for attr in missing_losses:
        if not hasattr(ultralytics.utils.loss, attr):
            setattr(ultralytics.utils.loss, attr, DummyLoss)
    print("   + Đã vá xong các hàm Loss.")

except Exception as e:
    print(f"!!! CẢNH BÁO: Lỗi trong quá trình vá thư viện: {e}")

print(">>> Hoàn tất quá trình vá lỗi. Đang load model...\n")
# ====================================================

# 2. CẤU HÌNH
MODEL_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"
IMAGE_PATH = "C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/image/Screenshot 2026-01-26 150959.png"

def main():
    try:
        # Load model
        model = YOLO(MODEL_PATH)
        
        # Dự đoán
        print(f"Processing image: {IMAGE_PATH}...")
        results = model(IMAGE_PATH)

        for result in results:
            im_array = result.plot() 
            cv2.imwrite("result_image.jpg", im_array)
            print("Thành công! Đã lưu ảnh tại: result_image.jpg")
            
            cv2.imshow("Result", im_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except AttributeError as e:
        print(f"\nCRITICAL ERROR: Vẫn còn thiếu module: {e}")
        print("Model của bạn dùng tên module quá lạ. Hãy copy tên module trong lỗi trên và báo lại cho tôi.")
    except Exception as e:
        print(f"Lỗi khác: {e}")

if __name__ == "__main__":
    main()