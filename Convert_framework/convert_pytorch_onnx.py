from ultralytics import YOLO

# Load model đã train
model = YOLO('Detection Web/assets/best_yolo12s_seg.pt')

# Export sang ONNX
model.export(format='onnx', imgsz=640)
print("✅ Model exported to ONNX format")
