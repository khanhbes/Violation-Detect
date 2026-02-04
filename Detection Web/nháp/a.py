# ============================================================================
# LANE VIOLATION DETECTION SYSTEM
# Phát hiện vi phạm làn đường sử dụng YOLOv12-seg
# ============================================================================
#
# YÊU CẦU CÀI ĐẶT (Dependencies):
# --------------------------------
# pip install ultralytics>=8.0.0
# pip install numpy>=1.24.0
# pip install opencv-python>=4.8.0
# pip install scikit-learn>=1.3.0
# pip install scipy>=1.11.0
#
# CLASS MAPPING:
# --------------
# LANE RULES (Mũi tên chỉ dẫn):
#   1: arrow_left              -> Bắt buộc rẽ trái
#   2: arrow_right             -> Bắt buộc rẽ phải
#   3: arrow_straight          -> Bắt buộc đi thẳng
#   4: arrow_straight_and_left
#   5: arrow_straight_and_right
#
# LANE MARKINGS (Vạch kẻ đường):
#   7:  dashed_white_line  -> Vạch đứt trắng (Được đè) - MÀU VÀNG
#   8:  dashed_yellow_line -> Vạch đứt vàng (Được đè) - MÀU VÀNG
#   37: solid_white_line   -> Vạch liền trắng (Cấm đè) - MÀU ĐỎ
#   38: solid_yellow_line  -> Vạch liền vàng (Cấm đè) - MÀU ĐỎ
#   39: stop_line          -> Vạch dừng (Trigger Line) - MÀU XANH DƯƠNG
#
# VEHICLES (Phương tiện):
#   0:  ambulance      -> Xe cấp cứu (ưu tiên)
#   6:  car            -> Ô tô con
#   9:  fire_truck     -> Xe cứu hỏa (ưu tiên)
#   21: motorcycle     -> Xe máy
#   26: police_car     -> Xe cảnh sát (ưu tiên)
#
# CÁCH SỬ DỤNG:
# -------------
# Video: python lane_violation_detection.py -i video.mp4
# Ảnh:   python lane_violation_detection.py -i image.jpg --image
#
# ============================================================================

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import time
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor
from scipy import stats
import sys


class VehicleClass(Enum):
    """ID các loại phương tiện"""
    AMBULANCE = 0
    CAR = 6
    FIRE_TRUCK = 9
    MOTORCYCLE = 21
    POLICE_CAR = 26


class LaneRule(Enum):
    """Các loại luật làn đường"""
    LEFT = "arrow_left"
    RIGHT = "arrow_right"
    STRAIGHT = "arrow_straight"
    STRAIGHT_AND_LEFT = "arrow_straight_and_left"
    STRAIGHT_AND_RIGHT = "arrow_straight_and_right"


class LineType(Enum):
    """Loại vạch kẻ đường"""
    DASHED_WHITE = 7
    DASHED_YELLOW = 8
    SOLID_WHITE = 37
    SOLID_YELLOW = 38
    STOP_LINE = 39


class ActionType(Enum):
    """Hành động của phương tiện"""
    LEFT = "ACTION_LEFT"
    RIGHT = "ACTION_RIGHT"
    STRAIGHT = "ACTION_STRAIGHT"
    UNKNOWN = "ACTION_UNKNOWN"


@dataclass
class DetectedObject:
    """Đối tượng được phát hiện"""
    class_id: int
    class_name: str
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: Tuple[float, float]
    confidence: float
    track_id: Optional[int] = None
    bottom_center: Tuple[float, float] = None

    def __post_init__(self):
        """Tính toán điểm bottom-center"""
        x1, y1, x2, y2 = self.bbox
        self.bottom_center = ((x1 + x2) / 2, y2)


@dataclass
class FittedLine:
    """Đường thẳng sau khi fit"""
    class_id: int
    line_type: LineType
    slope: float
    intercept: float
    y_min: float
    y_max: float
    points_count: int
    color: Tuple[int, int, int]

    def get_x_at_y(self, y: float) -> Optional[float]:
        """Lấy tọa độ x tại y"""
        if self.slope is None or np.isinf(self.slope):
            return None
        return (y - self.intercept) / self.slope

    def get_y_at_x(self, x: float) -> Optional[float]:
        """Lấy tọa độ y tại x"""
        return self.slope * x + self.intercept


@dataclass
class Lane:
    """Làn đường"""
    lane_id: int
    rule: Optional[LaneRule] = None
    polygon: np.ndarray = None
    left_boundary: Optional[FittedLine] = None
    right_boundary: Optional[FittedLine] = None

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Kiểm tra điểm có nằm trong làn không"""
        if self.polygon is None:
            return False
        x, y = point
        result = cv2.pointPolygonTest(self.polygon, (x, y), False)
        return result >= 0


@dataclass
class VehicleTracking:
    """Theo dõi phương tiện"""
    track_id: int
    class_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    last_action: ActionType = ActionType.UNKNOWN
    crossed_stopline: bool = False
    violation_recorded: bool = False

    def add_position(self, pos: Tuple[float, float]):
        """Thêm vị trí mới"""
        self.positions.append(pos)
        # Giữ lại 10 vị trí gần nhất
        if len(self.positions) > 10:
            self.positions.pop(0)


class GeometryProcessor:
    """Xử lý hình học - Tách riêng để code dễ đọc"""

    @staticmethod
    def fit_line_ransac(points: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit đường thẳng sử dụng RANSAC
        Returns: slope, intercept, inlier_count
        """
        if len(points) < 2:
            return None, None, 0

        # Reshape cho RANSAC
        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        try:
            ransac = RANSACRegressor(
                residual_threshold=5.0,
                max_trials=100,
                min_samples=min(5, len(points) // 2)
            )
            ransac.fit(x, y)

            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            inlier_mask = ransac.inlier_mask_
            inlier_count = np.sum(inlier_mask)

            # Lọc outliers bằng phương pháp đơn giản hơn nếu RANSAC không đủ tốt
            if inlier_count < len(points) * 0.3:
                return GeometryProcessor.fit_line_pca(points)

            return slope, intercept, inlier_count

        except Exception:
            return GeometryProcessor.fit_line_pca(points)

    @staticmethod
    def fit_line_pca(points: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit đường thẳng sử dụng PCA (phương pháp thay thế)
        Returns: slope, intercept, inlier_count
        """
        if len(points) < 2:
            return None, None, 0

        # Tính mean và covariance
        mean = np.mean(points, axis=0)
        centered = points - mean

        # PCA
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Eigenvector với eigenvalue lớn nhất là hướng chính
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Tính slope và intercept
        if np.abs(principal_axis[1]) > 1e-6:
            slope = principal_axis[0] / principal_axis[1]
        else:
            slope = np.inf if principal_axis[0] > 0 else -np.inf

        intercept = mean[1] - slope * mean[0]

        # Đếm inliers (điểm gần đường)
        distances = np.abs(points[:, 1] - slope * points[:, 0] - intercept)
        threshold = 10.0
        inlier_count = np.sum(distances < threshold)

        return slope, intercept, inlier_count

    @staticmethod
    def check_point_in_lane(
        point: Tuple[float, float],
        left_line: FittedLine,
        right_line: FittedLine
    ) -> bool:
        """
        Kiểm tra điểm có nằm giữa 2 đường biên không
        """
        px, py = point

        # Lấy x của 2 đường tại y của điểm
        x_left = left_line.get_x_at_y(py) if left_line else None
        x_right = right_line.get_x_at_y(py) if right_line else None

        if x_left is None or x_right is None:
            return False

        # Kiểm tra điểm nằm giữa
        return min(x_left, x_right) <= px <= max(x_left, x_right)

    @staticmethod
    def calculate_vehicle_action(
        positions: List[Tuple[float, float]]
    ) -> ActionType:
        """
        Tính toán hành động của xe dựa trên trajectory
        """
        if len(positions) < 2:
            return ActionType.UNKNOWN

        # Lấy điểm đầu và cuối
        start_pos = positions[0]
        end_pos = positions[-1]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        # Tính góc di chuyển
        angle = np.arctan2(dy, dx) * 180 / np.pi

        # Phân loại hành động dựa trên góc
        if abs(angle) < 30:
            return ActionType.STRAIGHT
        elif angle > 30 and angle < 150:
            return ActionType.LEFT if dx < 0 else ActionType.RIGHT
        elif angle < -30 and angle > -150:
            return ActionType.RIGHT if dx > 0 else ActionType.LEFT
        else:
            return ActionType.UNKNOWN

    @staticmethod
    def draw_smooth_line(
        image: np.ndarray,
        line: FittedLine,
        thickness: int = 3
    ) -> np.ndarray:
        """
        Vẽ đường thẳng mượt mà
        """
        if line.slope is None or np.isinf(line.slope):
            return image

        # Tính điểm đầu và cuối
        y1 = max(0, int(line.y_min))
        y2 = min(image.shape[0], int(line.y_max))

        if y2 <= y1:
            return image

        x1 = line.get_x_at_y(y1)
        x2 = line.get_x_at_y(y2)

        if x1 is None or x2 is None:
            return image

        # Vẽ đường thẳng
        pt1 = (int(x1), y1)
        pt2 = (int(x2), y2)

        cv2.line(image, pt1, pt2, line.color, thickness)

        return image


class LaneViolationDetector:
    """Hệ thống phát hiện vi phạm làn đường chính"""

    # Mapping màu sắc cho các loại vạch
    LINE_COLORS = {
        LineType.SOLID_WHITE: (0, 0, 255),      # Đỏ
        LineType.SOLID_YELLOW: (0, 0, 255),     # Đỏ
        LineType.DASHED_WHITE: (0, 255, 255),   # Vàng
        LineType.DASHED_YELLOW: (0, 255, 255),  # Vàng
        LineType.STOP_LINE: (255, 0, 0),        # Xanh dương
    }

    # Mapping arrow class ID sang LaneRule
    ARROW_TO_RULE = {
        1: LaneRule.LEFT,
        2: LaneRule.RIGHT,
        3: LaneRule.STRAIGHT,
        4: LaneRule.STRAIGHT_AND_LEFT,
        5: LaneRule.STRAIGHT_AND_RIGHT,
    }

    def __init__(self, model_path: str = r"C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"):
        """
        Khởi tạo detector

        Args:
            model_path: Đường dẫn model YOLO
        """
        print("=" * 60)
        print("KHỞI TẠO HỆ THỐNG PHÁT HIỆN VI PHẠM LÀN ĐƯỜNG")
        print("=" * 60)

        # Load model
        print(f"[1/4] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        # Thời gian freeze (giây)
        self.FREEZE_TIME = 5.0

        # Trạng thái hệ thống
        self.phase = "INIT"  # INIT, ACCUMULATION, MONITORING
        self.accumulated_masks = defaultdict(list)  # {class_id: List[masks]}
        self.fitted_lines: Dict[int, FittedLine] = {}  # {class_id: FittedLine}
        self.lanes: Dict[int, Lane] = {}  # {lane_id: Lane}
        self.stop_line: Optional[FittedLine] = None

        # Tracking
        self.tracked_vehicles: Dict[int, VehicleTracking] = {}

        # Thời gian
        self.start_time = time.time()
        self.freeze_time = None

        # Stats
        self.frame_count = 0

        print("[2/4] Initializing geometry processor...")
        self.geo = GeometryProcessor()

        print("[3/4] Setting up lane detection classes...")
        self.arrow_classes = {1, 2, 3, 4, 5}
        self.line_classes = {7, 8, 37, 38, 39}
        self.vehicle_classes = {0, 6, 9, 21, 26}
        self.priority_vehicles = {0, 9, 26}  # ambulance, fire_truck, police_car

        print("[4/4] Ready!")
        print("-" * 60)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Xử lý một frame video

        Args:
            frame: Ảnh đầu vào (BGR format)

        Returns:
            Frame đã xử lý với annotation
        """
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        # Chuyển đổi BGR sang RGB cho YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inference với tracking
        results = self.model.track(
            frame_rgb,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # Xử lý theo phase
        if elapsed_time < self.FREEZE_TIME:
            self._accumulation_phase(results)
        else:
            self._monitoring_phase(results, elapsed_time, frame)

        # Vẽ visualization
        annotated_frame = self._draw_visualization(frame, elapsed_time)

        return annotated_frame

    def _accumulation_phase(self, results):
        """
        Giai đoạn 1: Tích lũy mask (0-5s)
        """
        if self.phase != "ACCUMULATION":
            print(f"[ACCUMULATION] Phase started at {time.time() - self.start_time:.1f}s")
            self.phase = "ACCUMULATION"

        for result in results:
            if result.masks is None:
                continue

            for mask, det in zip(result.masks.data, result.boxes):
                class_id = int(det.cls)

                # Chỉ tích lũy vạch kẻ và mũi tên
                if class_id in self.line_classes or class_id in self.arrow_classes:
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    self.accumulated_masks[class_id].append(mask_np)

    def _freeze_and_process(self):
        """
        Xử lý và freeze bản đồ tại giây thứ 5
        """
        print(f"\n[FREEZE] Processing accumulated data at {self.FREEZE_TIME}s...")
        print(f"[FREEZE] Total masks accumulated: {sum(len(m) for m in self.accumulated_masks.values())}")

        # Fit lines cho từng class vạch kẻ
        for class_id in self.line_classes:
            if class_id not in self.accumulated_masks:
                continue

            all_points = self._masks_to_points(self.accumulated_masks[class_id])

            if len(all_points) < 10:
                print(f"  [SKIP] Class {class_id}: Not enough points ({len(all_points)})")
                continue

            slope, intercept, inlier_count = self.geo.fit_line_ransac(all_points)

            if slope is not None:
                line_type = LineType(class_id)
                fitted_line = FittedLine(
                    class_id=class_id,
                    line_type=line_type,
                    slope=slope,
                    intercept=intercept,
                    y_min=np.min(all_points[:, 1]),
                    y_max=np.max(all_points[:, 1]),
                    points_count=inlier_count,
                    color=self.LINE_COLORS.get(line_type, (128, 128, 128))
                )
                self.fitted_lines[class_id] = fitted_line
                print(f"  [FIT] Class {class_id} ({line_type.name}): {inlier_count} inliers")

                # Lưu stop line riêng
                if class_id == 39:
                    self.stop_line = fitted_line

        # Xử lý lanes và gán luật từ mũi tên
        self._process_lanes_with_rules()

        self.phase = "MONITORING"
        self.freeze_time = time.time()
        print(f"[FREEZE] Map frozen. Detected {len(self.fitted_lines)} lines and {len(self.lanes)} lanes")

    def _masks_to_points(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Chuyển đổi list of masks thành array of points
        """
        all_points = []

        for mask in masks:
            # Tìm contours từ mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                for point in contour:
                    all_points.append([point[0][0], point[0][1]])

        if not all_points:
            return np.array([])

        return np.array(all_points)

    def _process_lanes_with_rules(self):
        """
        Xử lý làn đường và gán luật từ mũi tên
        """
        # Lấy các đường kẻ dọc (solid/dashed lines)
        vertical_lines = {
            class_id: line for class_id, line in self.fitted_lines.items()
            if class_id in [7, 8, 37, 38]
        }

        # Xử lý từng mũi tên
        for class_id in self.arrow_classes:
            if class_id not in self.accumulated_masks:
                continue

            # Tính centroid của tất cả mask mũi tên
            centroids = self._calculate_mask_centroids(self.accumulated_masks[class_id])

            for centroid in centroids:
                # Tìm làn đường chứa mũi tên này
                lane = self._find_lane_for_point(centroid, vertical_lines)

                if lane is not None:
                    lane.rule = self.ARROW_TO_RULE.get(class_id)
                    print(f"  [RULE] Lane {lane.lane_id} assigned: {lane.rule.value}")

    def _calculate_mask_centroids(self, masks: List[np.ndarray]) -> List[Tuple[float, float]]:
        """
        Tính centroid của các mask
        """
        centroids = []

        for mask in masks:
            # Tìm contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((float(cx), float(cy)))

        return centroids

    def _find_lane_for_point(
        self,
        point: Tuple[float, float],
        lines: Dict[int, FittedLine]
    ) -> Optional[Lane]:
        """
        Tìm làn đường chứa một điểm
        """
        # Sắp xếp các đường theo tọa độ x tại y của điểm
        py = point[1]
        line_x = []

        for class_id, line in lines.items():
            x = line.get_x_at_y(py)
            if x is not None:
                line_x.append((x, class_id))

        if len(line_x) < 2:
            return None

        line_x.sort(key=lambda x: x[0])

        # Tìm khoảng giữa 2 đường liên tiếp
        for i in range(len(line_x) - 1):
            x1 = line_x[i][0]
            x2 = line_x[i + 1][0]

            if x1 <= point[0] <= x2:
                # Tạo hoặc lấy lane
                lane_id = i + 1

                if lane_id not in self.lanes:
                    self.lanes[lane_id] = Lane(
                        lane_id=lane_id,
                        left_boundary=self.fitted_lines.get(line_x[i][1]),
                        right_boundary=self.fitted_lines.get(line_x[i + 1][1])
                    )

                return self.lanes[lane_id]

        return None

    def _monitoring_phase(
        self,
        results,
        elapsed_time: float,
        frame: np.ndarray
    ):
        """
        Giai đoạn 2: Giám sát vi phạm (sau 5s)
        """
        if self.phase == "ACCUMULATION":
            self._freeze_and_process()

        for result in results:
            if result.boxes is None:
                continue

            # Xử lý từng detection
            for box, track_id, cls in zip(
                result.boxes.data,
                result.boxes.id if result.boxes.id is not None else [None] * len(result.boxes),
                result.boxes.cls
            ):
                class_id = int(cls)

                # Chỉ xử lý phương tiện
                if class_id not in self.vehicle_classes:
                    continue

                if track_id is None:
                    continue

                track_id = int(track_id)
                x1, y1, x2, y2, conf = box.cpu().numpy()

                # Tính bottom-center
                bottom_center = ((x1 + x2) / 2, y2)

                # Update hoặc tạo vehicle tracking
                if track_id not in self.tracked_vehicles:
                    self.tracked_vehicles[track_id] = VehicleTracking(
                        track_id=track_id,
                        class_id=class_id
                    )

                vehicle = self.tracked_vehicles[track_id]
                vehicle.add_position(bottom_center)

                # Kiểm tra đã cross stopline chưa
                if self.stop_line is not None and not vehicle.crossed_stopline:
                    if self._crossed_stopline(bottom_center, vehicle.positions):
                        vehicle.crossed_stopline = True

                        # Tính action và kiểm tra vi phạm
                        action = self.geo.calculate_vehicle_action(vehicle.positions)
                        vehicle.last_action = action

                        # Tìm lane hiện tại
                        current_lane = self._find_current_lane(bottom_center)

                        if current_lane and current_lane.rule:
                            self._check_violation(
                                vehicle, action, current_lane, class_id, frame
                            )

    def _crossed_stopline(
        self,
        current_pos: Tuple[float, float],
        positions: List[Tuple[float, float]]
    ) -> bool:
        """
        Kiểm tra xe có cắt stopline không
        """
        if len(positions) < 2 or self.stop_line is None:
            return False

        # Kiểm tra hướng đi lên (y giảm)
        if positions[-1][1] >= positions[0][1]:
            return False

        # Kiểm tra điểm hiện tại đã qua stopline
        x, y = current_pos
        x_stopline = self.stop_line.get_x_at_y(y)

        if x_stopline is None:
            return False

        # Đi qua stopline khi y < y_stopline và x gần stopline
        return y > self.stop_line.y_min and abs(x - x_stopline) < 50

    def _find_current_lane(self, position: Tuple[float, float]) -> Optional[Lane]:
        """
        Tìm làn đường hiện tại của xe
        """
        for lane in self.lanes.values():
            if lane.contains_point(position):
                return lane
        return None

    def _check_violation(
        self,
        vehicle: VehicleTracking,
        action: ActionType,
        lane: Lane,
        vehicle_class_id: int,
        frame: np.ndarray
    ):
        """
        Kiểm tra và ghi nhận vi phạm
        """
        # Bỏ qua xe ưu tiên
        if vehicle_class_id in self.priority_vehicles:
            return

        # Map action sang rule tương ứng
        action_to_rule = {
            ActionType.LEFT: LaneRule.LEFT,
            ActionType.RIGHT: LaneRule.RIGHT,
            ActionType.STRAIGHT: LaneRule.STRAIGHT,
        }

        expected_action = action_to_rule.get(action)

        # Kiểm tra vi phạm
        violation = False
        rule = lane.rule

        if rule is None:
            return

        # Logic so sánh
        if rule == LaneRule.STRAIGHT:
            if action != ActionType.STRAIGHT:
                violation = True
        elif rule == LaneRule.LEFT:
            if action != ActionType.LEFT:
                violation = True
        elif rule == LaneRule.RIGHT:
            if action != ActionType.RIGHT:
                violation = True
        elif rule == LaneRule.STRAIGHT_AND_LEFT:
            if action not in [ActionType.STRAIGHT, ActionType.LEFT]:
                violation = True
        elif rule == LaneRule.STRAIGHT_AND_RIGHT:
            if action not in [ActionType.STRAIGHT, ActionType.RIGHT]:
                violation = True

        if violation and not vehicle.violation_recorded:
            vehicle.violation_recorded = True

            # Log vi phạm
            self._log_violation(vehicle, action, lane)

            # Vẽ annotation vi phạm lên frame
            self._draw_violation(frame, vehicle, action, lane)

    def _log_violation(
        self,
        vehicle: VehicleTracking,
        action: ActionType,
        lane: Lane
    ):
        """Log thông tin vi phạm"""
        print("\n" + "=" * 60)
        print("🚨 PHÁT HIỆN VI PHẠM LÀN ĐƯỜNG")
        print("=" * 60)
        print(f"  Track ID: {vehicle.track_id}")
        print(f"  Lane Rule: {lane.rule.value if lane.rule else 'None'}")
        print(f"  Vehicle Action: {action.value}")
        print(f"  Status: VI PHẠM ❌")
        print("=" * 60 + "\n")

    def _draw_violation(
        self,
        frame: np.ndarray,
        vehicle: VehicleTracking,
        action: ActionType,
        lane: Lane
    ):
        """Vẽ annotation vi phạm lên frame"""
        # Tìm bbox của xe từ positions
        if len(vehicle.positions) < 2:
            return

        positions = vehicle.positions
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        x1 = max(0, int(min(x_coords) - 20))
        y1 = max(0, int(min(y_coords) - 40))
        x2 = min(frame.shape[1], int(max(x_coords) + 20))
        y2 = min(frame.shape[0], int(max(y_coords) + 5))

        # Vẽ bounding box đỏ
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Vẽ text "WRONG LANE"
        label = "WRONG LANE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Background cho text
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            (0, 0, 255),
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    def _draw_visualization(
        self,
        frame: np.ndarray,
        elapsed_time: float
    ) -> np.ndarray:
        """
        Vẽ tất cả visualization lên frame
        """
        # Vẽ các đường kẻ đã fit
        for class_id, line in self.fitted_lines.items():
            frame = self.geo.draw_smooth_line(frame, line, thickness=3)

        # Vẽ luật làn đường
        for lane_id, lane in self.lanes.items():
            if lane.rule:
                # Tìm điểm giữa làn để vẽ text
                if lane.left_boundary and lane.right_boundary:
                    mid_y = (lane.left_boundary.y_min + lane.left_boundary.y_max) / 2
                    x_left = lane.left_boundary.get_x_at_y(mid_y)
                    x_right = lane.right_boundary.get_x_at_y(mid_y)

                    if x_left and x_right:
                        mid_x = (x_left + x_right) / 2

                        # Format rule text
                        rule_text = lane.rule.value.replace("arrow_", "MUST_").upper()

                        # Vẽ text
                        cv2.putText(
                            frame,
                            rule_text,
                            (int(mid_x - 50), int(mid_y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2
                        )

        # Vẽ thông tin phase
        phase_text = f"Phase: {self.phase}"
        if self.phase == "ACCUMULATION":
            phase_text += f" ({self.FREEZE_TIME - elapsed_time:.1f}s)"

        cv2.putText(
            frame,
            phase_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Vẽ frame count
        cv2.putText(
            frame,
            f"Frame: {self.frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (128, 128, 128),
            1
        )

        return frame


def process_video(
    video_path: str,
    model_path: str = r"C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt",
    output_path: Optional[str] = None,
    save_video: bool = True
):
    """
    Xử lý video và phát hiện vi phạm

    Args:
        video_path: Đường dẫn video đầu vào
        model_path: Đường dẫn model YOLO
        output_path: Đường dẫn lưu video kết quả (optional)
        save_video: Có lưu video không
    """
    print("\n" + "=" * 60)
    print("BẮT ĐẦU XỬ LÝ VIDEO")
    print("=" * 60)

    # Khởi tạo detector
    detector = LaneViolationDetector(model_path)

    # Mở video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ LỖI: Không thể mở video {video_path}")
        return

    # Lấy thông tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video: {width}x{height} @ {fps:.1f}fps")
    print(f"📊 Total frames: {total_frames}")
    print("-" * 60)

    # Setup video writer
    writer = None
    if save_video:
        output_path = output_path or video_path.replace(".mp4", "_result.mp4")
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

    # Process frames
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # Process frame
        result_frame = detector.process_frame(frame)

        # Show preview
        cv2.imshow("Lane Violation Detection - Press 'q' to quit", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️ Người dùng đã dừng xử lý!")
            break

        # Write output
        if writer:
            writer.write(result_frame)

        # Show progress
        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_idx / elapsed
            progress = frame_idx / total_frames * 100

            print(f"  Processing: {frame_idx}/{total_frames} frames ({progress:.1f}%) "
                  f"- {fps_processing:.1f} fps")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("HOÀN THÀNH XỬ LÝ VIDEO")
    print("=" * 60)
    print(f"  Total frames: {frame_idx}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Processing speed: {frame_idx/total_time:.1f} fps")

    if save_video:
        print(f"  Output saved: {output_path}")

    print("=" * 60 + "\n")


def process_image(
    image_path: str,
    model_path: str = r"C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt"
) -> np.ndarray:
    """
    Xử lý một ảnh và phát hiện vi phạm

    Args:
        image_path: Đường dẫn ảnh đầu vào
        model_path: Đường dẫn model YOLO

    Returns:
        Ảnh đã xử lý
    """
    print(f"\n📷 Processing image: {image_path}")

    # Khởi tạo detector
    detector = LaneViolationDetector(model_path)

    # Đọc ảnh
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"❌ LỖI: Không thể đọc ảnh {image_path}")
        return None

    # Process frame (sẽ ở trong accumulation phase)
    result_frame = detector.process_frame(frame)

    return result_frame





# ============================================================================
# VÍ DỤ SỬ DỤNG (EXAMPLES)
# ============================================================================

def demo_geometry_processor():
    """
    Demo các hàm xử lý hình học
    """
    print("\n" + "=" * 60)
    print("DEMO: Geometry Processor Functions")
    print("=" * 60)

    geo = GeometryProcessor()

    # Test 1: Fit line với RANSAC
    print("\n[1] Testing RANSAC Line Fitting...")
    # Tạo điểm ngẫu nhiên với noise
    np.random.seed(42)
    x = np.linspace(0, 100, 100)
    y = 2 * x + 50 + np.random.normal(0, 5, 100)  # y = 2x + 50 + noise

    points = np.column_stack([x, y])
    slope, intercept, inliers = geo.fit_line_ransac(points)

    print(f"  Original: y = 2x + 50")
    print(f"  Fitted:   y = {slope:.2f}x + {intercept:.2f}")
    print(f"  Inliers:  {inliers}/100")

    # Test 2: Check point in lane
    print("\n[2] Testing Point in Lane Check...")
    # Tạo 2 đường biên giả lập
    left_line = FittedLine(
        class_id=37,
        line_type=LineType.SOLID_WHITE,
        slope=0.1,
        intercept=100,
        y_min=0,
        y_max=500,
        points_count=100,
        color=(0, 0, 255)
    )

    right_line = FittedLine(
        class_id=38,
        line_type=LineType.SOLID_YELLOW,
        slope=0.1,
        intercept=150,
        y_min=0,
        y_max=500,
        points_count=100,
        color=(0, 255, 255)
    )

    # Test điểm nằm trong lane
    test_point = (130, 300)  # Giữa 100 và 150 tại y=300
    result = geo.check_point_in_lane(test_point, left_line, right_line)
    print(f"  Point {test_point} in lane: {result}")

    # Test điểm nằm ngoài lane
    test_point_outside = (80, 300)
    result_outside = geo.check_point_in_lane(test_point_outside, left_line, right_line)
    print(f"  Point {test_point_outside} in lane: {result_outside}")

    # Test 3: Calculate vehicle action
    print("\n[3] Testing Vehicle Action Detection...")
    # Tạo trajectory đi thẳng
    straight_positions = [(100, 400), (105, 380), (110, 360), (115, 340)]
    action = geo.calculate_vehicle_action(straight_positions)
    print(f"  Straight trajectory: {action.value}")

    # Tạo trajectory rẽ trái
    left_positions = [(100, 400), (95, 380), (90, 360), (85, 340)]
    action_left = geo.calculate_vehicle_action(left_positions)
    print(f"  Left turn trajectory: {action_left.value}")

    # Tạo trajectory rẽ phải
    right_positions = [(100, 400), (105, 380), (110, 360), (115, 340)]
    action_right = geo.calculate_vehicle_action(right_positions)
    print(f"  Right turn trajectory: {action_right.value}")


def demo_usage_in_python():
    """
    Ví dụ sử dụng trong code Python
    """
    print("\n" + "=" * 60)
    print("DEMO: Sử dụng trong Python Code")
    print("=" * 60)

    print("""
# Cách 1: Sử dụng hàm tiện ích process_video
from lane_violation_detection import process_video
process_video('input_video.mp4', 'yolo12s-seg.pt', 'output_video.mp4')

# Cách 2: Sử dụng Detector tùy chỉnh
from lane_violation_detection import LaneViolationDetector
import cv2

detector = LaneViolationDetector('yolo12s-seg.pt')

cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = detector.process_frame(frame)
    cv2.imshow('Result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Cách 3: Xử lý ảnh
from lane_violation_detection import process_image
result = process_image('image.jpg', 'yolo12s-seg.pt')
cv2.imwrite('result.jpg', result)
    """)


def demo_troubleshooting():
    """
    Hướng dẫn xử lý lỗi thường gặp
    """
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING - Xử lý lỗi thường gặp")
    print("=" * 60)

    print("""
❓ Lỗi: 'yolo12s-seg.pt not found'
   → Tải model từ: https://github.com/ultralytics/ultralytics
   → Hoặc sử dụng: yolo12s-seg.pt (tự động tải)

❓ Lỗi: 'No module named ultralytics'
   → Chạy: pip install ultralytics

❓ Lỗi: 'No module named cv2'
   → Chạy: pip install opencv-python

❓ Video không được xử lý đúng?
   → Kiểm tra class_index.txt
   → Đảm bảo class IDs trong video khớp với config
   → Tăng thời gian FREEZE_TIME nếu cần
    """)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    # Nếu không có tham số, chạy demo
    if len(sys.argv) == 1:
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "LANE VIOLATION DETECTION" + " " * 16 + "║")
        print("║" + " " * 20 + "VÍ DỤ SỬ DỤNG" + " " * 25 + "║")
        print("╚" + "═" * 58 + "╝")

        # Chạy các demo
        demo_geometry_processor()
        demo_usage_in_python()
        demo_troubleshooting()

        print("\n" + "=" * 60)
        print("Để chạy hệ thống thực tế:")
        print("  Video: python a.py -i video.mp4")
        print("  Ảnh:  python a.py -i image.jpg --image")
        print("=" * 60 + "\n")
    else:
        parser = argparse.ArgumentParser(
            description="Lane Violation Detection System"
        )

        parser.add_argument(
            "--input", "-i",
            type=str,
            required=True,
            help="Đường dẫn video hoặc ảnh đầu vào"
        )

        parser.add_argument(
            "--model", "-m",
            type=str,
            default=r"C:/Users/khanh/OneDrive/Desktop/Violation Detect/Detection Web/assets/model/best_yolo12s_seg.pt",
            help="Đường dẫn model YOLO (default: best_yolo12s_seg.pt)"
        )

        parser.add_argument(
            "--output", "-o",
            type=str,
            default=None,
            help="Đường dẫn lưu kết quả (optional)"
        )

        parser.add_argument(
            "--image",
            action="store_true",
            help="Chế độ xử lý ảnh (mặc định: video)"
        )

        args = parser.parse_args()

        # Xử lý
        if args.image:
            result = process_image(args.input, args.model)
            if result is not None:
                output_path = args.output or args.input.replace(".", "_result.")
                cv2.imwrite(output_path, result)
                print(f"\n✅ Kết quả lưu tại: {output_path}")
        else:
            process_video(
                args.input,
                args.model,
                args.output,
                save_video=True
            )


