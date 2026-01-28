"""
Traffic Violation Detection System - Configuration File
Cấu hình tập trung cho toàn bộ hệ thống phát hiện vi phạm giao thông
"""

import os
from pathlib import Path


class Config:
    """Cấu hình chính của hệ thống"""
    
    # ========================================
    # PATHS - Đường dẫn
    # ========================================
    BASE_DIR = Path(__file__).parent.parent
    ASSETS_DIR = BASE_DIR / 'assets'
    MODEL_DIR = ASSETS_DIR / 'model'
    VIDEO_DIR = ASSETS_DIR / 'video'
    OUTPUT_DIR = BASE_DIR / 'output'
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    MODEL_PATH = str(MODEL_DIR / 'best_yolo12s_seg.pt')
    DEFAULT_VIDEO = str(VIDEO_DIR / 'test_2.mp4')
    OUTPUT_VIDEO = str(OUTPUT_DIR / 'output_violations.mp4')
    
    # ========================================
    # MODEL PARAMETERS - Tham số YOLO
    # ========================================
    IMG_SIZE = 1280
    IOU_THRESHOLD = 0.45
    TRACKER = "bytetrack.yaml"
    
    # Confidence thresholds
    CONF_THRESHOLD_LIGHT = 0.25
    CONF_THRESHOLD_VEHICLE = 0.50
    CONF_THRESHOLD_LANE = 0.30
    
    # ========================================
    # CALIBRATION - Hiệu chỉnh
    # ========================================
    STOPLINE_CALIBRATION_DURATION = 5.0    # seconds
    STOPLINE_MIN_POINTS = 100
    LIGHT_MEMORY_DURATION = 2.0            # seconds
    
    # ========================================
    # TRACKING - Theo dõi xe
    # ========================================
    MAX_TRACK_HISTORY = 30                 # Số vị trí lưu trong trajectory
    MIN_TRACK_LENGTH = 15                  # Số frame tối thiểu để phân tích hướng
    
    # ========================================
    # CLASS IDS - Mã class từ YOLO model
    # ========================================
    
    # Vehicles
    VEHICLE_CLASSES = [0, 6, 9, 21, 26]    # ambulance, car, fire_truck, motorcycle, police_car
    
    # Traffic Lights - Circular
    RED_LIGHTS = [18]
    YELLOW_LIGHTS = [19]
    GREEN_LIGHTS = [17]
    
    # Traffic Lights - Arrow
    ARROW_LEFT_RED = [11]
    ARROW_LEFT_YELLOW = [12]
    ARROW_LEFT_GREEN = [10]
    ARROW_STRAIGHT_RED = [15]
    ARROW_STRAIGHT_YELLOW = [16]
    ARROW_STRAIGHT_GREEN = [14]
    ARROW_RIGHT_GREEN = [13]
    
    # Lane Arrows (mũi tên trên mặt đường)
    ARROW_LEFT = [1]                       # CHỈ được rẽ trái
    ARROW_RIGHT = [2]                      # CHỈ được rẽ phải
    ARROW_STRAIGHT = [3]                   # CHỈ được đi thẳng
    ARROW_STRAIGHT_LEFT = [4]              # Được đi thẳng HOẶC rẽ trái
    ARROW_STRAIGHT_RIGHT = [5]             # Được đi thẳng HOẶC rẽ phải
    
    # Lane Lines (vạch kẻ đường)
    SOLID_WHITE_LINE = [37]                # Vạch liền trắng - 1 chiều, không được đè
    DASHED_WHITE_LINE = [7]                # Vạch đứt trắng - 1 chiều, được chuyển làn
    SOLID_YELLOW_LINE = [38]               # Vạch liền vàng - 2 chiều, không được sang
    DASHED_YELLOW_LINE = [8]               # Vạch đứt vàng - 2 chiều, được vượt
    
    # Infrastructure
    STOPLINE_CLASS = [39]
    SIDEWALK_CLASS = [27]
    MEDIAN_CLASS = [20]
    
    # ========================================
    # DISPLAY - Hiển thị
    # ========================================
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    
    # Colors (BGR)
    COLOR_SAFE = (0, 255, 0)               # Green
    COLOR_WARNING = (0, 255, 255)          # Yellow
    COLOR_VIOLATION = (0, 0, 255)          # Red
    COLOR_WRONG_LANE = (255, 0, 255)       # Magenta
    COLOR_STOPLINE = (0, 165, 255)         # Orange
    
    # ========================================
    # CLASS NAMES - Tên đầy đủ
    # ========================================
    CLASS_NAMES = {
        0: 'ambulance',
        1: 'arrow_left',
        2: 'arrow_right',
        3: 'arrow_straight',
        4: 'arrow_straight_and_left',
        5: 'arrow_straight_and_right',
        6: 'car',
        7: 'dashed_white_line',
        8: 'dashed_yellow_line',
        9: 'fire_truck',
        10: 'light_left_green',
        11: 'light_left_red',
        12: 'light_left_yellow',
        13: 'light_right_green',
        14: 'light_straight_arrow_green',
        15: 'light_straight_arrow_red',
        16: 'light_straight_arrow_yellow',
        17: 'light_straight_circle_green',
        18: 'light_straight_circle_red',
        19: 'light_straight_circle_yellow',
        20: 'median',
        21: 'motorcycle',
        22: 'no_left_turn',
        23: 'no_right_turn',
        24: 'no_straight',
        25: 'pedestrian',
        26: 'police_car',
        27: 'sidewalk',
        28: 'solid_white_line',
        29: 'solid_yellow_line',
        30: 'stopline',
        31: 'straight_and_left_turn_only',
        32: 'straight_and_right_turn_only',
        33: 'straight_only',
        34: 'turn_left_only',
        35: 'turn_right_only',
        36: 'u_turn',
        37: 'solid_white_line',
        38: 'solid_yellow_line',
        39: 'stopline'
    }

# Instance mặc định
config = Config()
