import sys
import platform
import os

def print_header(title):
    print("\n" + "="*40)
    print(f"  {title}")
    print("="*40)

def check_library(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        lib = __import__(import_name)
        version = getattr(lib, '__version__', 'Unknown')
        print(f"✅  {name:<15} : {version}")
        return True, lib
    except ImportError:
        print(f"❌  {name:<15} : CHƯA CÀI ĐẶT")
        return False, None

print_header("HỆ THỐNG & PYTHON")
print(f"• OS             : {platform.system()} {platform.release()}")
print(f"• Python Version : {sys.version.split()[0]}")
print(f"• Python Path    : {sys.executable}")
# Kiểm tra xem có đang chạy trong môi trường ảo không
is_venv = (sys.prefix != sys.base_prefix)
print(f"• Virtual Env    : {'✅ Đang bật (.venv)' if is_venv else '⚠️  Chưa bật / System Python'}")

print_header("THƯ VIỆN ĐÃ CÀI")
has_numpy, _ = check_library("Numpy", "numpy")
has_cv2, _ = check_library("OpenCV", "cv2")
has_ultra, _ = check_library("Ultralytics", "ultralytics")
has_torch, torch = check_library("PyTorch", "torch")

print_header("KIỂM TRA GPU (CUDA)")

if has_torch:
    import torch
    
    # 1. Kiểm tra CUDA có khả dụng không
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"✅  CUDA Available : TRUE (Ngon lành!)")
        
        # 2. Số lượng GPU
        device_count = torch.cuda.device_count()
        print(f"• Số lượng GPU     : {device_count}")
        
        # 3. Thông tin chi tiết GPU 0
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"• Tên GPU          : {gpu_name}")
        
        # 4. Kiểm tra phiên bản CUDA mà PyTorch đang dùng
        print(f"• CUDA Version     : {torch.version.cuda}")
        
        # 5. Kiểm tra VRAM (Bộ nhớ card hình)
        # Lưu ý: Hàm