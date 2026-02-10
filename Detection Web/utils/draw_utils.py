"""
Unified Drawing Utilities for Traffic Violation Detection System
Provides consistent bbox and label styling across all detection scripts.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# Import shared config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


def draw_bbox_with_label(
    frame: np.ndarray,
    box: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
    thickness: Optional[int] = None
) -> None:
    """
    Vẽ bounding box với label gọn đẹp, rõ ràng.
    
    Style:
    - BBox: viền màu với thickness từ config hoặc tùy chỉnh
    - Label: nền filled màu bbox, chữ đen (dễ đọc)
    - Font nhỏ gọn, padding vừa phải
    
    Args:
        frame: Frame để vẽ
        box: Tuple (x1, y1, x2, y2)
        label: Text hiển thị
        color: Màu BGR của bbox và label background
        thickness: Override thickness nếu cần (None = dùng config)
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Bbox thickness
    t = thickness if thickness is not None else config.BBOX_THICKNESS
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, t)
    
    # Label style - gọn đẹp như traffic light
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5          # Nhỏ gọn
    text_thickness = 1        # Mỏng, rõ nét
    
    # Get label dimensions
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    
    # Label box position (above bbox)
    label_y1 = max(0, y1 - label_h - 6)
    label_y2 = y1
    label_x1 = x1
    label_x2 = x1 + label_w + 4
    
    # Draw label background (filled with bbox color)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    
    # Draw label text (black for better readability)
    text_x = label_x1 + 2
    text_y = label_y2 - 4
    cv2.putText(
        frame, label, (text_x, text_y),
        font, font_scale, (0, 0, 0),  # Black text
        text_thickness, cv2.LINE_AA
    )


def draw_info_hud(
    frame: np.ndarray,
    lines: List[Tuple[str, Tuple[int, int, int]]],
    position: Tuple[int, int] = (10, 10),
    width: int = 380,
    title: Optional[str] = None,
    title_color: Optional[Tuple[int, int, int]] = None
) -> None:
    """
    Vẽ HUD (Head-Up Display) thông tin đồng bộ.
    """
    # HUD disabled by request
    return


def draw_calibration_hud(
    frame: np.ndarray,
    progress: float,
    duration: float,
    position: Tuple[int, int] = (10, 10),
    width: int = 380
) -> None:
    """
    Vẽ HUD cho giai đoạn calibration.
    """
    # HUD disabled by request
    return


def save_violation_snapshot(
    original_frame: np.ndarray,
    violation_type: str,
    vehicle_id: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    label: str = None,
    color: Tuple[int, int, int] = None,
    vehicle_class: str = "vehicle"
) -> str:
    """
    Lưu screenshot khi phát hiện violation.
    Chỉ vẽ bbox của phương tiện vi phạm, ẩn các bbox khác.
    Sử dụng cùng style bbox như khi chạy video.
    
    Tên file: {ViolationType}_{VehicleClass}_ID{id}_{timestamp}.jpg
    VD: Helmet_motorcycle_ID12_20260211_143025.jpg
    
    Args:
        original_frame: Frame gốc (chưa vẽ bbox)
        violation_type: Loại lỗi (vd: "no_helmet", "redlight", "sidewalk", "wrong_way", "wrong_lane", "sign")
        vehicle_id: ID của xe vi phạm
        bbox: Bounding box của xe vi phạm
        label: Label hiển thị trên bbox (mặc định: "VIOLATION #ID")
        color: Màu bbox (mặc định: COLOR_VIOLATION)
        vehicle_class: Loại phương tiện (vd: "motorcycle", "car", "bus", "truck")
    
    Returns:
        Đường dẫn file đã lưu
    """
    import os
    from datetime import datetime
    
    # Sử dụng SNAPSHOT_DIR từ config
    snapshot_base = config.SNAPSHOT_DIR
    violations_dir = snapshot_base / violation_type
    os.makedirs(violations_dir, exist_ok=True)
    
    # Tạo tên file theo format: ViolationType_VehicleClass_IDxx_timestamp.jpg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    # Capitalize violation type, clean vehicle class
    vtype_name = violation_type.replace("_", " ").title().replace(" ", "")
    vclass_name = vehicle_class.replace(" ", "_") if vehicle_class else "vehicle"
    filename = f"{vtype_name}_{vclass_name}_ID{vehicle_id}_{timestamp}.jpg"
    filepath = violations_dir / filename
    
    # Copy frame để không ảnh hưởng frame gốc
    snapshot = original_frame.copy()
    
    # Vẽ CHỈ bbox của xe vi phạm - dùng cùng style như video đang chạy
    if bbox is not None:
        if color is None:
            color = config.COLOR_VIOLATION
        if label is None:
            label = f"VIOLATION #{vehicle_id}"
        draw_bbox_with_label(snapshot, bbox, label, color)
    
    # Lưu full frame với chỉ bbox violation
    cv2.imwrite(str(filepath), snapshot)
    
    print(f"📸 [SNAPSHOT] {vtype_name}_{vclass_name}_ID{vehicle_id} -> {filepath}")
    
    return str(filepath)
