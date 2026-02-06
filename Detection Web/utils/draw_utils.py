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
    
    Args:
        frame: Frame để vẽ
        lines: List các tuple (text, color) cho từng dòng
        position: Vị trí góc trên trái của HUD
        width: Chiều rộng HUD
        title: Tiêu đề HUD (optional)
        title_color: Màu tiêu đề (default: COLOR_SAFE)
    """
    x, y = position
    line_height = 30
    padding = 15
    
    # Calculate height
    num_lines = len(lines) + (1 if title else 0)
    height = num_lines * line_height + padding * 2
    
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + height), config.HUD_BG_COLOR, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), config.HUD_BORDER_COLOR, 2)
    
    font = config.FONT
    font_scale = config.HUD_FONT_SCALE
    text_thickness = config.HUD_TEXT_THICKNESS
    
    current_y = y + padding + 20
    
    # Draw title if provided
    if title:
        t_color = title_color if title_color else config.COLOR_SAFE
        cv2.putText(frame, title, (x + padding, current_y), font, font_scale, t_color, text_thickness, cv2.LINE_AA)
        current_y += line_height
    
    # Draw each line
    for text, color in lines:
        cv2.putText(frame, text, (x + padding, current_y), font, font_scale, color, text_thickness, cv2.LINE_AA)
        current_y += line_height


def draw_calibration_hud(
    frame: np.ndarray,
    progress: float,
    duration: float,
    position: Tuple[int, int] = (10, 10),
    width: int = 380
) -> None:
    """
    Vẽ HUD cho giai đoạn calibration.
    
    Args:
        frame: Frame để vẽ
        progress: Tiến độ từ 0.0 đến 1.0
        duration: Tổng thời gian calibration (seconds)
        position: Vị trí HUD
        width: Chiều rộng HUD
    """
    x, y = position
    
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + 170), config.HUD_BG_COLOR, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 170), config.HUD_BORDER_COLOR, 2)
    
    font = config.FONT
    fs = config.HUD_FONT_SCALE
    th = config.HUD_TEXT_THICKNESS
    padding = 15
    
    # Title
    cv2.putText(frame, "STATUS: CALIBRATING", (x + padding, y + 30), 
                font, fs, config.COLOR_WARNING, th, cv2.LINE_AA)
    
    # Progress percentage
    progress_text = f"Progress: {int(progress * 100)}%"
    cv2.putText(frame, progress_text, (x + padding, y + 65), 
                font, fs, config.HUD_TEXT_COLOR, th, cv2.LINE_AA)
    
    # Remaining time
    remaining = max(0, duration * (1 - progress))
    cv2.putText(frame, f"Time: {remaining:.1f}s", (x + padding, y + 100), 
                font, fs, config.HUD_TEXT_COLOR, th, cv2.LINE_AA)
    
    # Progress bar
    bar_x = x + padding
    bar_y = y + 120
    bar_w = width - padding * 2
    bar_h = 20
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), config.HUD_BORDER_COLOR, 2)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), config.COLOR_SAFE, -1)
