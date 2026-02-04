"""
Lane Violation Detection System
================================
Phát hiện vi phạm đi sai làn đường sử dụng YOLOv12-seg Segmentation + Object Tracking

2-Phase Processing Pipeline:
- Phase 1 (0-5s): Calibration - Tích lũy mask và tạo Static Map
- Phase 2 (>5s): Monitoring - Phát hiện vi phạm realtime

Author: Traffic Violation Detection System
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from ultralytics import YOLO

# Import config
from config import config


# =============================================================================
# CLASS MAPPING (Theo yêu cầu)
# =============================================================================

# 1. Nhóm Mũi tên chỉ dẫn (LANE RULES)
ARROW_LEFT = 1                  # Bắt buộc rẽ trái
ARROW_RIGHT = 2                 # Bắt buộc rẽ phải
ARROW_STRAIGHT = 3              # Bắt buộc đi thẳng
ARROW_STRAIGHT_LEFT = 4         # Được đi thẳng hoặc rẽ trái
ARROW_STRAIGHT_RIGHT = 5        # Được đi thẳng hoặc rẽ phải

ARROW_CLASSES = [ARROW_LEFT, ARROW_RIGHT, ARROW_STRAIGHT, ARROW_STRAIGHT_LEFT, ARROW_STRAIGHT_RIGHT]

# 2. Nhóm Vạch kẻ đường (LANE MARKINGS)
DASHED_WHITE_LINE = 7           # Vạch đứt trắng - Được đè
DASHED_YELLOW_LINE = 8          # Vạch đứt vàng - Được đè
SOLID_WHITE_LINE = 37           # Vạch liền trắng - Cấm đè
SOLID_YELLOW_LINE = 38          # Vạch liền vàng - Cấm đè
STOP_LINE = 39                  # Vạch dừng - Trigger Line

LINE_CLASSES = [DASHED_WHITE_LINE, DASHED_YELLOW_LINE, SOLID_WHITE_LINE, SOLID_YELLOW_LINE]
ALL_LINE_CLASSES = LINE_CLASSES + [STOP_LINE]

# 3. Nhóm Phương tiện (VEHICLES)
AMBULANCE = 0
CAR = 6
FIRE_TRUCK = 9
MOTORCYCLE = 21
POLICE_CAR = 26

VEHICLE_CLASSES = [CAR, MOTORCYCLE, AMBULANCE, FIRE_TRUCK, POLICE_CAR]
PRIORITY_VEHICLES = {AMBULANCE, FIRE_TRUCK, POLICE_CAR}  # Xe ưu tiên - bỏ qua lỗi

# Màu sắc hiển thị (BGR)
COLOR_SOLID_LINE = (0, 0, 255)      # Red - Vạch liền
COLOR_DASHED_LINE = (0, 255, 255)   # Yellow - Vạch đứt
COLOR_STOP_LINE = (255, 0, 0)       # Blue - Stopline
COLOR_SAFE = (0, 255, 0)            # Green
COLOR_VIOLATION = (0, 0, 255)       # Red

# Tên luật hiển thị
RULE_NAMES = {
    ARROW_LEFT: "MUST LEFT",
    ARROW_RIGHT: "MUST RIGHT",
    ARROW_STRAIGHT: "ONLY STRAIGHT",
    ARROW_STRAIGHT_LEFT: "STR/LEFT",
    ARROW_STRAIGHT_RIGHT: "STR/RIGHT"
}

# Hướng di chuyển
ACTION_LEFT = "ACTION_LEFT"
ACTION_RIGHT = "ACTION_RIGHT"
ACTION_STRAIGHT = "ACTION_STRAIGHT"
ACTION_UNKNOWN = "ACTION_UNKNOWN"


# =============================================================================
# GEOMETRY HELPERS - Các hàm xử lý hình học
# =============================================================================

def fit_line_ransac(
    points: np.ndarray,
    min_samples: int = 20,
    residual_threshold: float = 10.0,
    min_points: int = 50
) -> Optional[Dict]:
    """
    Fit đường thẳng y = ax + b sử dụng RANSAC từ đám mây điểm.
    
    Args:
        points: Mảng điểm (N, 2) với [x, y]
        min_samples: Số mẫu tối thiểu cho RANSAC
        residual_threshold: Ngưỡng sai số để là inlier
        min_points: Số điểm tối thiểu cần có
        
    Returns:
        Dict với {'slope', 'intercept', 'x_min', 'x_max', 'y_min', 'y_max'}
        hoặc None nếu thất bại
    """
    if points is None or len(points) < min_points:
        return None
    
    points = np.array(points)
    X = points[:, 0].reshape(-1, 1)  # x làm biến độc lập
    y = points[:, 1]                  # y làm biến phụ thuộc
    
    try:
        ransac = RANSACRegressor(
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            random_state=42
        )
        ransac.fit(X, y)
        
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "x_min": int(np.min(X)),
            "x_max": int(np.max(X)),
            "y_min": int(np.min(y)),
            "y_max": int(np.max(y))
        }
    except Exception as e:
        print(f"[RANSAC ERROR] {e}")
        return None


def get_y_from_x(line_params: Dict, x: float) -> float:
    """Tính y từ x dựa trên phương trình y = ax + b"""
    if line_params is None:
        return 0
    return line_params["slope"] * x + line_params["intercept"]


def get_x_from_y(line_params: Dict, y: float) -> int:
    """Tính x từ y dựa trên phương trình y = ax + b => x = (y - b) / a"""
    if line_params is None:
        return 0
    if abs(line_params["slope"]) < 1e-6:  # Đường gần như nằm ngang
        return int(line_params["x_min"])
    return int((y - line_params["intercept"]) / line_params["slope"])


def is_point_between_lines(
    point: Tuple[float, float],
    left_line: Dict,
    right_line: Dict,
    tolerance: int = 15
) -> bool:
    """
    Kiểm tra một điểm có nằm giữa 2 đường thẳng dọc không.
    
    Args:
        point: Tọa độ (x, y) của điểm cần kiểm tra
        left_line: Tham số đường bên trái
        right_line: Tham số đường bên phải
        tolerance: Độ dung sai (pixels)
        
    Returns:
        True nếu điểm nằm trong khoảng giữa 2 đường
    """
    if left_line is None or right_line is None:
        return False
        
    px, py = point
    
    # Tính x của 2 đường thẳng tại cùng độ cao py
    x_left = get_x_from_y(left_line, py)
    x_right = get_x_from_y(right_line, py)
    
    # Đảm bảo left < right
    if x_left > x_right:
        x_left, x_right = x_right, x_left
    
    return (x_left - tolerance) <= px <= (x_right + tolerance)


def get_centroid(mask_points: np.ndarray) -> Optional[Tuple[int, int]]:
    """Tính tâm (centroid) của đám mây điểm"""
    if mask_points is None or len(mask_points) == 0:
        return None
    points = np.array(mask_points)
    cx = int(np.mean(points[:, 0]))
    cy = int(np.mean(points[:, 1]))
    return (cx, cy)


def calculate_movement_direction(trajectory: deque, min_length: int = 10) -> str:
    """
    Xác định hướng di chuyển của xe dựa trên quỹ đạo.
    
    Args:
        trajectory: Hàng đợi các vị trí [(x, y), ...]
        min_length: Số điểm tối thiểu để tính toán
        
    Returns:
        ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT, hoặc ACTION_UNKNOWN
    """
    if len(trajectory) < min_length:
        return ACTION_UNKNOWN
    
    # Lấy điểm đầu và cuối
    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    
    dx = end_x - start_x
    dy = end_y - start_y  # dy âm = đi lên (vào giao lộ)
    
    # Tính góc so với trục thẳng đứng
    # atan2(dx, -dy) cho góc so với hướng đi lên
    if abs(dy) < 20:  # Di chuyển dọc quá ít
        return ACTION_UNKNOWN
    
    angle = math.degrees(math.atan2(dx, -dy))
    
    # Phân loại dựa trên góc
    ANGLE_THRESHOLD = 20  # degrees
    
    if angle < -ANGLE_THRESHOLD:
        return ACTION_LEFT
    elif angle > ANGLE_THRESHOLD:
        return ACTION_RIGHT
    else:
        return ACTION_STRAIGHT


def check_violation(action: str, lane_rule: Optional[int]) -> bool:
    """
    Kiểm tra xem hành động có vi phạm luật làn không.
    
    Args:
        action: Hướng di chuyển thực tế (ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT)
        lane_rule: Class ID của mũi tên quy định làn
        
    Returns:
        True nếu vi phạm, False nếu hợp lệ
    """
    if lane_rule is None or action == ACTION_UNKNOWN:
        return False
    
    # ARROW_STRAIGHT (3): Chỉ được đi thẳng
    if lane_rule == ARROW_STRAIGHT:
        return action != ACTION_STRAIGHT
    
    # ARROW_LEFT (1): Chỉ được rẽ trái
    if lane_rule == ARROW_LEFT:
        return action != ACTION_LEFT
    
    # ARROW_RIGHT (2): Chỉ được rẽ phải
    if lane_rule == ARROW_RIGHT:
        return action != ACTION_RIGHT
    
    # ARROW_STRAIGHT_LEFT (4): Được đi thẳng hoặc rẽ trái
    if lane_rule == ARROW_STRAIGHT_LEFT:
        return action == ACTION_RIGHT
    
    # ARROW_STRAIGHT_RIGHT (5): Được đi thẳng hoặc rẽ phải
    if lane_rule == ARROW_STRAIGHT_RIGHT:
        return action == ACTION_LEFT
    
    return False


def check_stopline_crossing(
    prev_point: Tuple[float, float],
    curr_point: Tuple[float, float],
    stopline: Dict
) -> bool:
    """
    Kiểm tra xe có vừa cắt qua stopline theo hướng đi lên không.
    
    Args:
        prev_point: Vị trí trước đó (x, y)
        curr_point: Vị trí hiện tại (x, y)
        stopline: Tham số đường stopline
        
    Returns:
        True nếu xe vừa cắt qua stopline từ dưới lên
    """
    if stopline is None:
        return False
    
    prev_x, prev_y = prev_point
    curr_x, curr_y = curr_point
    
    # Tính y của stopline tại vị trí x tương ứng
    stop_y_prev = get_y_from_x(stopline, prev_x)
    stop_y_curr = get_y_from_x(stopline, curr_x)
    
    # Kiểm tra cắt: Trước đó ở dưới (y lớn), hiện tại ở trên (y nhỏ hơn)
    # (Xe đi từ dưới lên vào giao lộ)
    if prev_y > stop_y_prev and curr_y <= stop_y_curr:
        return True
    
    return False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VehicleTrack:
    """Thông tin tracking của mỗi xe"""
    trajectory: deque = field(default_factory=lambda: deque(maxlen=50))
    has_crossed_stopline: bool = False
    is_violation: bool = False
    action: str = ACTION_UNKNOWN
    lane_id: int = -1
    violation_frame: int = -1


@dataclass
class Lane:
    """Đại diện một làn đường"""
    id: int
    left_line: Dict
    right_line: Dict
    rule: Optional[int] = None  # Class ID của mũi tên
    
    def get_rule_name(self) -> str:
        return RULE_NAMES.get(self.rule, "NO RULE")


# =============================================================================
# LANE CALIBRATOR - Phase 1: Tích lũy và Freeze
# =============================================================================

class LaneCalibrator:
    """Bộ calibration làn đường trong 5 giây đầu"""
    
    def __init__(self, duration: float = 5.0):
        self.duration = duration
        self.start_time = None
        self.is_frozen = False
        
        # Buffers để tích lũy mask points
        self.line_points: Dict[int, List] = defaultdict(list)
        self.arrow_centroids: Dict[int, List] = defaultdict(list)
        
        # Kết quả sau khi freeze
        self.fitted_lines: List[Dict] = []  # Các đường kẻ đã fit
        self.stopline: Optional[Dict] = None
        self.lanes: List[Lane] = []
        
    def start(self):
        """Bắt đầu calibration"""
        self.start_time = time.time()
        print("[CALIBRATION] Starting lane calibration...")
        
    def is_complete(self) -> bool:
        """Kiểm tra đã hoàn thành calibration chưa"""
        return self.is_frozen
    
    def get_progress(self) -> float:
        """Trả về tiến độ calibration (0.0 - 1.0)"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(elapsed / self.duration, 1.0)
    
    def accumulate_mask(self, cls_id: int, mask_polygon: np.ndarray):
        """
        Tích lũy điểm từ mask vào buffer.
        
        Args:
            cls_id: Class ID của đối tượng
            mask_polygon: Polygon của mask (contour points)
        """
        if mask_polygon is None or len(mask_polygon) == 0:
            return
        
        # Nếu là vạch kẻ đường
        if cls_id in ALL_LINE_CLASSES:
            self.line_points[cls_id].extend(mask_polygon.tolist())
        
        # Nếu là mũi tên
        elif cls_id in ARROW_CLASSES:
            centroid = get_centroid(mask_polygon)
            if centroid:
                self.arrow_centroids[cls_id].append(centroid)
    
    def accumulate_bbox_centroid(self, cls_id: int, bbox: np.ndarray):
        """
        Tích lũy tâm từ bounding box (backup cho mũi tên).
        
        Args:
            cls_id: Class ID
            bbox: Bounding box [x1, y1, x2, y2]
        """
        if cls_id in ARROW_CLASSES:
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            self.arrow_centroids[cls_id].append((cx, cy))
    
    def maybe_freeze(self) -> bool:
        """Kiểm tra và thực hiện freeze nếu đến thời điểm"""
        if self.is_frozen:
            return True
        
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        if elapsed < self.duration:
            return False
        
        # Đến lúc freeze
        self._freeze_static_map()
        self.is_frozen = True
        return True
    
    def _freeze_static_map(self):
        """Xử lý hình học và tạo Static Map"""
        print("\n" + "=" * 60)
        print("[FREEZING] Creating Static Map...")
        print("=" * 60)
        
        # A. Fit Lines bằng RANSAC
        temp_lines = []
        
        for cls_id, points in self.line_points.items():
            if len(points) < 100:
                continue
            
            points_array = np.array(points)
            line_params = fit_line_ransac(points_array)
            
            if line_params is None:
                continue
            
            if cls_id == STOP_LINE:
                # Stopline lưu riêng
                self.stopline = line_params
                print(f"  [+] STOP_LINE fitted: y = {line_params['slope']:.4f}x + {line_params['intercept']:.2f}")
            else:
                # Đường dọc
                # Tính x trung bình để sắp xếp từ trái qua phải
                mean_y = (line_params["y_min"] + line_params["y_max"]) / 2
                mean_x = get_x_from_y(line_params, mean_y)
                
                temp_lines.append({
                    "params": line_params,
                    "cls_id": cls_id,
                    "mean_x": mean_x
                })
                
                line_type = "SOLID" if cls_id in [SOLID_WHITE_LINE, SOLID_YELLOW_LINE] else "DASHED"
                print(f"  [+] {line_type}_LINE (cls={cls_id}) fitted at mean_x={mean_x}")
        
        # B. Sắp xếp đường dọc từ trái qua phải
        temp_lines.sort(key=lambda x: x["mean_x"])
        self.fitted_lines = temp_lines
        
        print(f"\n  Total vertical lines: {len(temp_lines)}")
        
        # C. Tạo Lanes từ các đường kẻ liền kề
        for i in range(len(temp_lines) - 1):
            left_l = temp_lines[i]
            right_l = temp_lines[i + 1]
            
            lane = Lane(
                id=i + 1,
                left_line=left_l["params"],
                right_line=right_l["params"],
                rule=None
            )
            
            # D. Gán luật cho làn dựa trên mũi tên
            best_rule = None
            max_count = 0
            
            for arrow_cls, centroids in self.arrow_centroids.items():
                count = 0
                for pt in centroids:
                    if is_point_between_lines(pt, left_l["params"], right_l["params"]):
                        count += 1
                
                if count > max_count:
                    max_count = count
                    best_rule = arrow_cls
            
            lane.rule = best_rule
            self.lanes.append(lane)
            
            rule_name = lane.get_rule_name()
            print(f"  [LANE {lane.id}] Rule: {rule_name} (matches: {max_count})")
        
        print(f"\n  Total lanes created: {len(self.lanes)}")
        print("=" * 60)
        print("[FREEZING COMPLETE] Switching to monitoring mode...\n")


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def get_line_color(cls_id: int) -> Tuple[int, int, int]:
    """Trả về màu vẽ cho từng loại vạch kẻ"""
    if cls_id in [SOLID_WHITE_LINE, SOLID_YELLOW_LINE]:
        return COLOR_SOLID_LINE  # Red
    elif cls_id in [DASHED_WHITE_LINE, DASHED_YELLOW_LINE]:
        return COLOR_DASHED_LINE  # Yellow
    elif cls_id == STOP_LINE:
        return COLOR_STOP_LINE  # Blue
    return (200, 200, 200)  # Gray default


def draw_fitted_lines(
    frame: np.ndarray,
    fitted_lines: List[Dict],
    stopline: Optional[Dict]
):
    """
    Vẽ các đường kẻ đã fit lên frame.
    
    Args:
        frame: Frame để vẽ
        fitted_lines: Danh sách đường dọc đã fit
        stopline: Thông tin stopline
    """
    # Vẽ các đường dọc
    for line_info in fitted_lines:
        params = line_info["params"]
        cls_id = line_info["cls_id"]
        color = get_line_color(cls_id)
        
        # Tính 2 điểm đầu cuối để vẽ
        y1 = params["y_min"]
        y2 = params["y_max"]
        x1 = get_x_from_y(params, y1)
        x2 = get_x_from_y(params, y2)
        
        # Vẽ đường với anti-aliasing
        cv2.line(frame, (x1, int(y1)), (x2, int(y2)), color, 3, cv2.LINE_AA)
    
    # Vẽ stopline
    if stopline:
        color = COLOR_STOP_LINE
        y1 = stopline["y_min"]
        y2 = stopline["y_max"]
        x1 = get_x_from_y(stopline, y1)
        x2 = get_x_from_y(stopline, y2)
        
        cv2.line(frame, (x1, int(y1)), (x2, int(y2)), color, 4, cv2.LINE_AA)


def draw_lane_rules(frame: np.ndarray, lanes: List[Lane], y_position_ratio: float = 0.7):
    """
    Vẽ text luật lên từng làn đường.
    
    Args:
        frame: Frame để vẽ
        lanes: Danh sách làn đường
        y_position_ratio: Vị trí y tính theo tỷ lệ chiều cao frame
    """
    h, w = frame.shape[:2]
    y_pos = int(h * y_position_ratio)
    
    for lane in lanes:
        if lane.rule is None:
            continue
        
        # Tính vị trí giữa làn
        x_left = get_x_from_y(lane.left_line, y_pos)
        x_right = get_x_from_y(lane.right_line, y_pos)
        x_center = (x_left + x_right) // 2
        
        text = lane.get_rule_name()
        
        # Vẽ text với background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background
        cv2.rectangle(
            frame,
            (x_center - tw // 2 - 5, y_pos - th - 5),
            (x_center + tw // 2 + 5, y_pos + 5),
            (0, 0, 0),
            -1
        )
        
        # Text
        cv2.putText(
            frame,
            text,
            (x_center - tw // 2, y_pos),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )


def draw_hud(
    frame: np.ndarray,
    is_calibrated: bool,
    calibration_progress: float,
    violation_count: int,
    fps: float
):
    """Vẽ HUD thông tin lên góc frame"""
    h, w = frame.shape[:2]
    
    # Background box
    cv2.rectangle(frame, (10, 10), (320, 150), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (320, 150), (255, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if not is_calibrated:
        # Đang calibration
        progress_pct = calibration_progress * 100
        remaining = max(0, 5.0 - calibration_progress * 5.0)
        
        cv2.putText(frame, "CALIBRATING...", (20, 40), font, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Progress: {progress_pct:.0f}%", (20, 70), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Remaining: {remaining:.1f}s", (20, 100), font, 0.6, (255, 255, 255), 1)
        
        # Progress bar
        bar_width = int(280 * calibration_progress)
        cv2.rectangle(frame, (20, 115), (300, 135), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 115), (20 + bar_width, 135), (0, 255, 0), -1)
    else:
        # Đang monitoring
        cv2.putText(frame, "MONITORING", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Violations: {violation_count}", (20, 75), font, 0.7, COLOR_VIOLATION, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110), font, 0.6, (255, 255, 255), 1)


# =============================================================================
# MAIN LANE VIOLATION SYSTEM
# =============================================================================

class LaneViolationSystem:
    """Hệ thống phát hiện vi phạm làn đường"""
    
    def __init__(
        self,
        model_path: str = None,
        video_path: str = None,
        output_path: str = None,
        calibration_duration: float = 5.0,
        show_preview: bool = True
    ):
        self.model_path = model_path or str(config.MODEL_PATH)
        self.video_path = video_path or str(config.DEFAULT_VIDEO)
        self.output_path = output_path
        self.calibration_duration = calibration_duration
        self.show_preview = show_preview
        
        # Load model
        print(f"[INIT] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Calibrator
        self.calibrator = LaneCalibrator(duration=calibration_duration)
        
        # Tracking
        self.tracks: Dict[int, VehicleTrack] = {}
        self.violation_count = 0
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
    
    def get_bottom_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Lấy điểm giữa cạnh đáy của bounding box"""
        x1, y1, x2, y2 = bbox
        return float((x1 + x2) / 2), float(y2)
    
    def find_lane_for_point(self, point: Tuple[float, float]) -> Optional[Lane]:
        """Tìm làn đường chứa điểm"""
        for lane in self.calibrator.lanes:
            if is_point_between_lines(point, lane.left_line, lane.right_line):
                return lane
        return None
    
    def process_frame_calibration(self, frame: np.ndarray, results):
        """Xử lý frame trong giai đoạn calibration"""
        if results[0].masks is None or results[0].boxes is None:
            return
        
        for box, mask in zip(results[0].boxes, results[0].masks):
            cls_id = int(box.cls[0])
            
            # Lấy polygon của mask
            if mask.xy is not None and len(mask.xy) > 0:
                polygon = mask.xy[0]
                if len(polygon) > 0:
                    self.calibrator.accumulate_mask(cls_id, polygon)
            
            # Backup: dùng bbox cho mũi tên
            if cls_id in ARROW_CLASSES:
                bbox = box.xyxy[0].cpu().numpy()
                self.calibrator.accumulate_bbox_centroid(cls_id, bbox)
        
        # Kiểm tra freeze
        self.calibrator.maybe_freeze()
    
    def process_frame_monitoring(self, frame: np.ndarray, results):
        """Xử lý frame trong giai đoạn monitoring"""
        if results[0].boxes is None:
            return
        
        boxes = results[0].boxes
        track_ids = boxes.id
        
        if track_ids is None:
            return
        
        for box, tid in zip(boxes, track_ids.cpu().numpy()):
            tid = int(tid)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Chỉ xử lý phương tiện
            if cls_id not in VEHICLE_CLASSES:
                continue
            
            if conf < config.CONF_THRESHOLD_VEHICLE:
                continue
            
            # Lấy bbox và điểm đáy giữa
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in bbox]
            bottom_center = self.get_bottom_center(bbox)
            
            # Khởi tạo track nếu chưa có
            if tid not in self.tracks:
                self.tracks[tid] = VehicleTrack()
            
            track = self.tracks[tid]
            
            # Cập nhật trajectory
            track.trajectory.append(bottom_center)
            
            # Kiểm tra vi phạm nếu chưa cross stopline
            if not track.has_crossed_stopline and len(track.trajectory) >= 2:
                prev_point = track.trajectory[-2]
                curr_point = track.trajectory[-1]
                
                # Kiểm tra cắt stopline
                if check_stopline_crossing(prev_point, curr_point, self.calibrator.stopline):
                    track.has_crossed_stopline = True
                    
                    # Phân tích hành vi
                    track.action = calculate_movement_direction(track.trajectory)
                    
                    # Tìm làn đường hiện tại
                    lane = self.find_lane_for_point(curr_point)
                    
                    if lane:
                        track.lane_id = lane.id
                        
                        # Bỏ qua xe ưu tiên
                        if cls_id not in PRIORITY_VEHICLES:
                            # Kiểm tra vi phạm
                            if check_violation(track.action, lane.rule):
                                track.is_violation = True
                                track.violation_frame = self.fps_frame_count
                                self.violation_count += 1
                                
                                rule_name = lane.get_rule_name()
                                print(f"[VIOLATION] Vehicle ID={tid} | Action={track.action} | Lane Rule={rule_name}")
            
            # Vẽ bounding box
            if track.is_violation:
                # Xe vi phạm - Màu đỏ + Text cảnh báo
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VIOLATION, 3)
                
                # Text "WRONG LANE" với background
                text = "WRONG LANE"
                font = cv2.FONT_HERSHEY_BOLD
                (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), COLOR_VIOLATION, -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 5), font, 0.8, (255, 255, 255), 2)
            else:
                # Xe bình thường - Màu xanh
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_SAFE, 2)
            
            # Vẽ điểm đáy giữa
            px, py = int(bottom_center[0]), int(bottom_center[1])
            color = COLOR_VIOLATION if track.is_violation else COLOR_SAFE
            cv2.circle(frame, (px, py), 5, color, -1)
    
    def run(self):
        """Chạy hệ thống phát hiện vi phạm"""
        # Mở video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[VIDEO] {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Video writer
        if self.output_path is None:
            from datetime import datetime
            from pathlib import Path
            output_dir = Path(config.OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = str(output_dir / f"lane_violation_{timestamp}.mp4")
        
        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        )
        
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {self.output_path}")
        
        print(f"[OUTPUT] {self.output_path}")
        
        # Bắt đầu calibration
        self.calibrator.start()
        
        frame_idx = 0
        print("\n[START] Processing video...")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                self.fps_frame_count += 1
                
                # Cập nhật FPS
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
                    self.fps_frame_count = 0
                    self.fps_start_time = current_time
                
                # YOLO inference với tracking
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=0.3,
                    iou=config.IOU_THRESHOLD,
                    imgsz=config.IMG_SIZE,
                    tracker=config.TRACKER,
                    retina_masks=True,  # Giảm răng cưa
                    verbose=False
                )
                
                # Xử lý theo giai đoạn
                if not self.calibrator.is_complete():
                    self.process_frame_calibration(frame, results)
                else:
                    self.process_frame_monitoring(frame, results)
                    
                    # Vẽ static map
                    draw_fitted_lines(frame, self.calibrator.fitted_lines, self.calibrator.stopline)
                    draw_lane_rules(frame, self.calibrator.lanes)
                
                # Vẽ HUD
                draw_hud(
                    frame,
                    self.calibrator.is_complete(),
                    self.calibrator.get_progress(),
                    self.violation_count,
                    self.current_fps
                )
                
                # Resize và output
                frame_out = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
                writer.write(frame_out)
                
                # Preview
                if self.show_preview:
                    cv2.imshow("Lane Violation Detection", frame_out)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress log
                if frame_idx % 100 == 0:
                    print(f"[PROGRESS] Frame {frame_idx}/{total_frames} | Violations: {self.violation_count}")
        
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
        
        # Summary
        print("\n" + "=" * 60)
        print("LANE VIOLATION DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total Frames Processed: {frame_idx}")
        print(f"Lanes Detected: {len(self.calibrator.lanes)}")
        print(f"Total Violations: {self.violation_count}")
        print(f"\nOutput saved: {self.output_path}")
        print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

def run(
    video_path: str = None,
    output_path: str = None,
    model_path: str = None,
    show_preview: bool = True
):
    """
    Chạy Lane Violation Detection.
    
    Args:
        video_path: Đường dẫn video input
        output_path: Đường dẫn video output
        model_path: Đường dẫn model YOLO
        show_preview: Có hiển thị preview không
    """
    system = LaneViolationSystem(
        model_path=model_path,
        video_path=video_path,
        output_path=output_path,
        show_preview=show_preview
    )
    system.run()


if __name__ == "__main__":
    run()
