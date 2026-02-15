"""
PRECISION WRONG-WAY DETECTION SYSTEM (MERGED)
=============================================
Combines best features from:
1. Kalman Filter & Spatial Zones (from wrong_way_violation.py)
2. Lane Segmentation & Sidewalk Logic (from detect_wrong_way_with_segmentation1.py)

Uses shared Config class from config/config.py
"""

from __future__ import annotations
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter1d
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path so we can import config module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared config and draw utilities
from config.config import config
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, draw_calibration_hud, save_violation_snapshot


# =====================================================================================
# ENUMS
# =====================================================================================

class FlowDirection(Enum):
    UPWARD = "upward"
    DOWNWARD = "downward"
    LEFTWARD = "leftward"
    RIGHTWARD = "rightward"
    UNDEFINED = "undefined"


# =====================================================================================
# LANE SEGMENTATION MANAGER
# =====================================================================================

class LaneSegmentationManager:
    """
    Enhanced lane manager with sidewalk detection and line type differentiation.
    """
    def __init__(self):
        # Separate line types for better logic
        self.solid_yellow_lines = []   # Class 38 - opposite direction, no crossing
        self.dashed_yellow_lines = []  # Class 8 - opposite direction, crossing allowed
        self.solid_white_lines = []    # Class 37 - same direction, no crossing
        self.dashed_white_lines = []   # Class 7 - same direction, crossing allowed
        
        # Infrastructure
        self.sidewalk_masks = []       # Class 27 - vehicles here = violation
        
    def update_from_results(self, results):
        """Extract segmentation masks from YOLO results with line type separation."""
        self.solid_yellow_lines = []
        self.dashed_yellow_lines = []
        self.solid_white_lines = []
        self.dashed_white_lines = []
        self.sidewalk_masks = []
        
        if results.masks is None:
            return
            
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes
        
        if boxes.cls is None:
            return
            
        cls_ids = boxes.cls.int().cpu().tolist()
        
        for i, cls_id in enumerate(cls_ids):
            if cls_id in config.SOLID_YELLOW_LINE:
                self.solid_yellow_lines.append(masks[i])
            elif cls_id in config.DASHED_YELLOW_LINE:
                self.dashed_yellow_lines.append(masks[i])
            elif cls_id in config.SOLID_WHITE_LINE:
                self.solid_white_lines.append(masks[i])
            elif cls_id in config.DASHED_WHITE_LINE:
                self.dashed_white_lines.append(masks[i])
            elif cls_id in config.SIDEWALK_CLASS:
                self.sidewalk_masks.append(masks[i])
    
    def is_on_sidewalk(self, point, frame_shape: Tuple[int, int] = None):
        """
        Check if a point (vehicle position) is on the sidewalk.
        Returns True if on sidewalk, False otherwise.
        Coordinates are scaled from frame space to mask space.
        """
        if not self.sidewalk_masks:
            return False

        px, py = float(point[0]), float(point[1])
        
        for sidewalk_mask in self.sidewalk_masks:
            mh, mw = sidewalk_mask.shape

            # Scale frame coords -> mask coords
            if frame_shape is not None:
                fh, fw = frame_shape
                mx = int(px * mw / fw)
                my = int(py * mh / fh)
            else:
                mx, my = int(px), int(py)

            if 0 <= mx < mw and 0 <= my < mh:
                if sidewalk_mask[my, mx] > 0.5:
                    return True
        
        return False
    
    def get_lane_direction_at_point(self, point, frame_shape: Tuple[int, int] = None):
        """
        Determine the expected direction at a given point based on YELLOW lane markings ONLY.
        Returns: 'left_side', 'right_side', or 'unknown'
        
        Yellow lines indicate opposite-direction traffic.
        White lines indicate same-direction lanes, so we don't use them for wrong-way detection.
        Influence zone scales with frame width instead of hardcoded pixel value.
        """
        px, py = float(point[0]), float(point[1])
        
        # Calculate influence zone based on frame width
        if frame_shape is not None:
            fh, fw = frame_shape
            influence_zone = fw * 0.15
        else:
            influence_zone = 300  # fallback
        
        # Check both solid and dashed yellow lines
        all_yellow_lines = self.solid_yellow_lines + self.dashed_yellow_lines
        
        for yellow_mask in all_yellow_lines:
            mh, mw = yellow_mask.shape

            # Scale frame coords -> mask coords for pixel lookup
            if frame_shape is not None:
                fh, fw = frame_shape
                scale_x = mw / fw
                scale_y = mh / fh
            else:
                scale_x, scale_y = 1.0, 1.0

            mx = int(px * scale_x)
            my = int(py * scale_y)
            if mx < 0 or mx >= mw or my < 0 or my >= mh:
                continue
                
            # Get yellow line pixels
            yellow_pixels = np.column_stack(np.where(yellow_mask > 0.5))
            if len(yellow_pixels) == 0:
                continue
            
            # Find the median x-coordinate of the yellow line (in mask space)
            yellow_x_coords_mask = yellow_pixels[:, 1]
            median_yellow_x_mask = np.median(yellow_x_coords_mask)

            # Convert median back to frame space for comparison
            median_yellow_x_frame = median_yellow_x_mask / scale_x
            
            # Check if vehicle is within influence zone (in frame space)
            if abs(px - median_yellow_x_frame) < influence_zone:
                if px < median_yellow_x_frame:
                    return 'left_side'  # Wrong side for right-hand traffic
                else:
                    return 'right_side'  # Correct side
        
        return 'unknown'
    
    def draw_debug(self, frame):
        """Draw lane line and sidewalk overlays with color coding."""
        overlay = frame.copy()
        fh, fw = frame.shape[:2]
        
        # Draw solid yellow lines (bright red)
        for mask in self.solid_yellow_lines:
            mask_resized = cv2.resize(mask, (fw, fh))
            colored = np.zeros_like(frame)
            colored[:, :, 2] = (mask_resized * 255).astype(np.uint8)  # Red
            overlay = cv2.addWeighted(overlay, 1, colored, 0.4, 0)
        
        # Draw dashed yellow lines (orange)
        for mask in self.dashed_yellow_lines:
            mask_resized = cv2.resize(mask, (fw, fh))
            colored = np.zeros_like(frame)
            colored[:, :, 2] = (mask_resized * 255).astype(np.uint8)  # Red
            colored[:, :, 1] = (mask_resized * 165).astype(np.uint8)  # Orange
            overlay = cv2.addWeighted(overlay, 1, colored, 0.3, 0)
        
        # Draw sidewalks (purple/magenta - violation zone)
        for mask in self.sidewalk_masks:
            mask_resized = cv2.resize(mask, (fw, fh))
            colored = np.zeros_like(frame)
            colored[:, :, 0] = (mask_resized * 255).astype(np.uint8)  # Blue
            colored[:, :, 2] = (mask_resized * 255).astype(np.uint8)  # Red (magenta)
            overlay = cv2.addWeighted(overlay, 1, colored, 0.4, 0)
        
        return overlay


# =====================================================================================
# KALMAN FILTER
# =====================================================================================

class KalmanTracker:
    def __init__(self, initial_position: np.ndarray):
        self.dt = 1.0
        self.state = np.array([
            initial_position[0], initial_position[1], 0.0, 0.0
        ])
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * config.KALMAN_PROCESS_NOISE
        self.R = np.eye(2) * config.KALMAN_MEASUREMENT_NOISE
        self.P = np.eye(4) * 1.0
        
    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]
        
    def update(self, measurement: np.ndarray) -> np.ndarray:
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2]
        
    def get_velocity(self) -> np.ndarray:
        return self.state[2:4]


# =====================================================================================
# SPATIAL ZONE
# =====================================================================================

class SpatialZone:
    def __init__(self, zone_id: Tuple[int, int], bounds: Tuple[int, int, int, int]):
        self.zone_id = zone_id
        self.bounds = bounds
        self.velocity_samples: List[np.ndarray] = []
        self.angle_samples: List[float] = []
        self.dominant_flow: Optional[FlowDirection] = None
        self.dominant_vector: Optional[np.ndarray] = None
        self.dominant_angle: Optional[float] = None
        self.mean_speed: float = 0.0
        self.confidence: float = 0.0
        self.is_reliable: bool = False
        self.sample_count: int = 0
        
    def add_sample(self, velocity: np.ndarray, bbox_aspect: Optional[float] = None):
        self.velocity_samples.append(velocity)
        angle = np.arctan2(velocity[1], velocity[0])
        self.angle_samples.append(angle)
        self.sample_count += 1
        
    def analyze_flow(self) -> bool:
        if len(self.velocity_samples) < 10:
            return False
            
        velocities = np.array(self.velocity_samples)
        angles = np.array(self.angle_samples)
        
        self.dominant_vector = self._estimate_consensus_flow(velocities, angles)
        if self.dominant_vector is None:
            return False
            
        self.dominant_angle = np.arctan2(self.dominant_vector[1], self.dominant_vector[0])
        self.dominant_flow = self._classify_flow_direction(self.dominant_angle)
        
        speeds = np.linalg.norm(velocities, axis=1)
        self.mean_speed = np.median(speeds)
        
        self.confidence = self._calculate_flow_confidence(angles)
        self.is_reliable = (
            self.sample_count >= 15 and
            self.confidence > 0.65 and
            self.dominant_flow != FlowDirection.UNDEFINED
        )
        return self.is_reliable
        
    def _estimate_consensus_flow(self, velocities: np.ndarray, angles: np.ndarray) -> Optional[np.ndarray]:
        if len(velocities) < 5:
            return None
        angle_bins = np.linspace(-np.pi, np.pi, 9)
        angle_indices = np.digitize(angles, angle_bins)
        unique, counts = np.unique(angle_indices, return_counts=True)
        dominant_bin = unique[np.argmax(counts)]
        adjacent_bins = [dominant_bin - 1, dominant_bin, dominant_bin + 1]
        mask = np.isin(angle_indices, adjacent_bins)
        if np.sum(mask) < 3:
            return None
        consensus_velocities = velocities[mask]
        mean_velocity = np.mean(consensus_velocities, axis=0)
        norm = np.linalg.norm(mean_velocity)
        if norm > 1e-6:
            return mean_velocity / norm
        return None
        
    def _classify_flow_direction(self, angle: float) -> FlowDirection:
        """Classify flow direction in IMAGE coordinate system (Y-axis points DOWN)."""
        angle_deg = np.degrees(angle)
        angle_deg = (angle_deg + 360) % 360
        # In image coords: angle 45-135 means dy>0 (moving DOWN), 225-315 means dy<0 (moving UP)
        if 45 <= angle_deg < 135:
            return FlowDirection.DOWNWARD
        elif 135 <= angle_deg < 225:
            return FlowDirection.LEFTWARD
        elif 225 <= angle_deg < 315:
            return FlowDirection.UPWARD
        return FlowDirection.RIGHTWARD
        
    def _calculate_flow_confidence(self, angles: np.ndarray) -> float:
        if len(angles) < 3:
            return 0.0
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        return float(np.sqrt(sin_mean**2 + cos_mean**2))
        
    def contains_point(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.bounds
        return x1 <= x < x2 and y1 <= y < y2


# =====================================================================================
# VEHICLE TRACKER
# =====================================================================================

class PrecisionVehicleTracker:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.kalman: Optional[KalmanTracker] = None
        self.raw_positions: deque = deque(maxlen=config.MAX_POSITION_HISTORY)
        self.filtered_positions: deque = deque(maxlen=config.MAX_POSITION_HISTORY)
        self.velocities: deque = deque(maxlen=30)
        self.bbox_aspect_ratios: deque = deque(maxlen=20)
        
        self.current_velocity: Optional[np.ndarray] = None
        self.current_speed: float = 0.0
        self.current_angle: Optional[float] = None
        self.smoothed_trajectory: Optional[np.ndarray] = None
        self.trajectory_direction: Optional[FlowDirection] = None
        self.direction_confidence: float = 0.0
        
        # Combined state
        self.is_wrong_way: bool = False
        self.is_confirmed_violator: bool = False
        self.violation_type: Optional[str] = None
        self.violation_evidence: deque = deque(maxlen=config.VIOLATION_HISTORY_WINDOW)
        self.consecutive_violations: int = 0
        
        self.total_frames: int = 0
        self.first_frame: int = 0
        self.last_frame: int = 0
        self.vehicle_class: str = ""
        self.confirmed_frame: Optional[int] = None
        
        # Segmentation logic
        self.lane_side: str = 'unknown'
        self.on_sidewalk: bool = False
        
    def update(self, position: np.ndarray, bbox: np.ndarray, frame_num: int, vehicle_class: str = ""):
        self.raw_positions.append(position)
        self.last_frame = frame_num
        
        x1, y1, x2, y2 = bbox
        aspect_ratio = (x2 - x1) / ((y2 - y1) + 1e-6)
        self.bbox_aspect_ratios.append(aspect_ratio)
        
        if self.kalman is None:
            self.kalman = KalmanTracker(position)
            
        self.kalman.predict()
        filtered_pos = self.kalman.update(position)
        self.filtered_positions.append(filtered_pos)
        
        velocity = self.kalman.get_velocity()
        self.velocities.append(velocity)
        self.current_velocity = velocity
        self.current_speed = np.linalg.norm(velocity)
        
        if self.current_speed > 1e-6:
            self.current_angle = np.arctan2(velocity[1], velocity[0])
        
        if self.total_frames == 0:
            self.first_frame = frame_num
        self.total_frames += 1
        self.vehicle_class = vehicle_class
        
        if len(self.filtered_positions) >= 15:
            self._analyze_trajectory()
            
    def _analyze_trajectory(self):
        if len(self.filtered_positions) < 10:
            return
        positions = np.array(list(self.filtered_positions))
        smoothed_x = gaussian_filter1d(positions[:, 0], sigma=config.TRAJECTORY_SMOOTHING_SIGMA)
        smoothed_y = gaussian_filter1d(positions[:, 1], sigma=config.TRAJECTORY_SMOOTHING_SIGMA)
        self.smoothed_trajectory = np.column_stack([smoothed_x, smoothed_y])
        
        start = self.smoothed_trajectory[0]
        end = self.smoothed_trajectory[-1]
        overall_vector = end - start
        
        norm = np.linalg.norm(overall_vector)
        if norm > 5.0:
            angle = np.arctan2(overall_vector[1], overall_vector[0])
            angle_deg = (np.degrees(angle) + 360) % 360
            
            # Image coords: Y-axis points DOWN
            if 45 <= angle_deg < 135:
                self.trajectory_direction = FlowDirection.DOWNWARD
            elif 135 <= angle_deg < 225:
                self.trajectory_direction = FlowDirection.LEFTWARD
            elif 225 <= angle_deg < 315:
                self.trajectory_direction = FlowDirection.UPWARD
            else:
                self.trajectory_direction = FlowDirection.RIGHTWARD
                
            path_length = np.sum(np.linalg.norm(np.diff(self.smoothed_trajectory, axis=0), axis=1))
            self.direction_confidence = norm / path_length if path_length > 1e-6 else 0.0

    def is_opposite_direction(self, zone: SpatialZone) -> Tuple[bool, float]:
        if not zone.is_reliable or self.current_velocity is None:
            return False, 0.0
        if self.current_speed < config.MIN_SPEED_THRESHOLD:
            return False, 0.0
            
        vehicle_angle = np.arctan2(self.current_velocity[1], self.current_velocity[0])
        diff = vehicle_angle - zone.dominant_angle
        while diff > np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        angle_diff_deg = np.degrees(abs(diff))
        
        is_opposite = angle_diff_deg > config.OPPOSITE_ANGLE_THRESHOLD
        confidence = min(angle_diff_deg / 180.0, 1.0)
        
        return is_opposite, confidence
        
    def update_violation_status(self, is_violating: bool, confidence: float, violation_type: str = None):
        self.violation_evidence.append((is_violating, confidence))
        
        if is_violating:
            self.consecutive_violations += 1
            self.is_wrong_way = True
            if violation_type:
                self.violation_type = violation_type
        else:
            # Soft reset: decrement instead of hard reset to tolerate YOLO jitter
            self.consecutive_violations = max(0, self.consecutive_violations - 2)
            if self.consecutive_violations == 0:
                self.is_wrong_way = False
            
        if not self.is_confirmed_violator:
            # Immediate confirmation for severe violations (sidewalk, wrong side)
            if violation_type in ["Driving on sidewalk", "Wrong side of road"]:
                if self.consecutive_violations >= 3: # Fast confirmation
                    self.is_confirmed_violator = True
                    self.confirmed_frame = self.total_frames
            # Normal threshold for flow violations
            elif self.consecutive_violations >= config.VIOLATION_CONSECUTIVE_FRAMES:
                self.is_confirmed_violator = True
                self.confirmed_frame = self.total_frames
                
    def is_valid_for_detection(self, current_frame: int, frame_width: int, frame_height: int) -> bool:
        if self.total_frames < config.MIN_TRACKING_FRAMES:
            return False
        if (current_frame - self.first_frame) < config.ENTRY_GRACE_PERIOD:
            return False
        if self.current_speed < config.MIN_SPEED_THRESHOLD:
            return False
        if len(self.raw_positions) > 0:
            x, y = self.raw_positions[-1]
            margin = config.BOUNDARY_MARGIN
            if x < margin or x > frame_width - margin or y < margin or y > frame_height - margin:
                return False
        return True


# =====================================================================================
# MAIN DETECTOR
# =====================================================================================

class PrecisionWrongWayDetector:
    def __init__(self):
        # Model is NOT loaded here — passed via process_frame(r0=...) or set_model()
        self.model: Optional[YOLO] = None
        
        self.current_frame: int = 0
        self.is_learning: bool = True
        self.zones: List[SpatialZone] = []
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.trackers: Dict[int, PrecisionVehicleTracker] = {}
        
        # Lane Segmentation Manager
        self.lane_manager = LaneSegmentationManager()
        
        self.total_vehicles: int = 0
        self.total_violations: int = 0
        self.violations_by_class: Dict[str, int] = defaultdict(int)
        
        self.fps: float = 0.0
        self.prev_time: float = time.time()

    def set_model(self, model: YOLO):
        """Set the YOLO model (for standalone mode)."""
        self.model = model
        
    def reset(self):
        """Reset tất cả state — cho web integration khi đổi video."""
        self.model = None
        self.current_frame = 0
        self.is_learning = True
        self.zones = []
        self.frame_width = 0
        self.frame_height = 0
        self.trackers = {}
        self.lane_manager = LaneSegmentationManager()
        self.total_vehicles = 0
        self.total_violations = 0
        self.violations_by_class = defaultdict(int)
        self.fps = 0.0
        self.prev_time = time.time()
        
    def initialize_zones(self, frame_shape: Tuple[int, int]):
        self.frame_height, self.frame_width = frame_shape[:2]
        lane_width = self.frame_width / config.NUM_VERTICAL_LANES
        zone_height = self.frame_height / config.NUM_HORIZONTAL_ZONES
        
        for v in range(config.NUM_VERTICAL_LANES):
            for h in range(config.NUM_HORIZONTAL_ZONES):
                x1 = int(v * lane_width)
                x2 = int((v + 1) * lane_width)
                y1 = int(h * zone_height)
                y2 = int((h + 1) * zone_height)
                self.zones.append(SpatialZone((v, h), (x1, y1, x2, y2)))
                
    def get_zone(self, x: float, y: float) -> Optional[SpatialZone]:
        for zone in self.zones:
            if zone.contains_point(x, y):
                return zone
        return None
        
    def process_frame(
        self,
        frame: np.ndarray,
        r0=None,
        model: YOLO = None,
        conf: float = 0.25,
        debug: bool = False
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process frame with combined logic: Lane Segmentation + Spatial Flow Learning.
        
        Args:
            frame: Input frame (BGR)
            r0: YOLO results (từ UnifiedDetector), None nếu chạy standalone
            model: YOLO model (nếu r0 là None và self.model chưa set)
            conf: Confidence threshold
            debug: Bật debug overlay
            
        Returns:
            (annotated_frame, list_of_violations)
        """
        if frame is None:
            return frame, []

        self.current_frame += 1
        violations = []
        
        if self.current_frame == 1:
            self.initialize_zones(frame.shape)

        # Resolve model: parameter > instance > None
        active_model = model or self.model
            
        # Run model if r0 not provided (standalone mode)
        if r0 is None:
            if active_model is None:
                return frame, []

            lane_classes = (config.SOLID_YELLOW_LINE + config.DASHED_YELLOW_LINE + 
                           config.SOLID_WHITE_LINE + config.DASHED_WHITE_LINE + 
                           config.SIDEWALK_CLASS)
            
            results = active_model.track(
                frame, imgsz=config.IMG_SIZE, conf=conf,
                iou=config.IOU_THRESHOLD, persist=True,
                classes=config.VEHICLE_CLASSES + lane_classes, 
                verbose=False, retina_masks=True
            )
            r0 = results[0]
        
        # Copy frame before drawing
        frame_vis = frame.copy()
        frame_shape = frame.shape[:2]  # (height, width)
        
        # Update Lane Segmentation
        if r0.masks is not None:
             self.lane_manager.update_from_results(r0)
        
        if r0.boxes is not None and r0.boxes.id is not None:
            boxes = r0.boxes.xyxy.cpu().numpy()
            track_ids = r0.boxes.id.cpu().numpy().astype(int)
            classes = r0.boxes.cls.cpu().numpy().astype(int)
            
            if self.is_learning:
                self._learning_phase(boxes, track_ids, classes)
            else:
                new_violations = self._detection_phase(boxes, track_ids, classes, frame, frame_shape)
                violations.extend(new_violations)
                
        if self.is_learning and self.current_frame >= config.LEARNING_DURATION_FRAMES:
            self._finalize_learning()
            
        # Draw UI on the copy
        if debug:
            frame_vis = self.lane_manager.draw_debug(frame_vis)
        self._draw_ui(frame_vis)
        
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        
        return frame_vis, violations
        
    def _learning_phase(self, boxes, track_ids, classes):
        for box, tid, cls in zip(boxes, track_ids, classes):
            if cls not in config.VEHICLE_CLASSES:
                continue
                
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            position = np.array([cx, cy])
            
            if tid not in self.trackers:
                self.trackers[tid] = PrecisionVehicleTracker(tid)
                
            tracker = self.trackers[tid]
            vehicle_class = config.VEHICLE_CLASS_NAMES.get(cls, "unknown")
            tracker.update(position, box, self.current_frame, vehicle_class)
            
            if tracker.current_velocity is not None and tracker.current_speed > config.MIN_SPEED_THRESHOLD:
                zone = self.get_zone(cx, cy)
                if zone:
                    zone.add_sample(tracker.current_velocity)
                    
    def _detection_phase(self, boxes, track_ids, classes, frame, frame_shape) -> List[dict]:
        new_violations = []
        current_ids = set()
        
        for box, tid, cls in zip(boxes, track_ids, classes):
            if cls not in config.VEHICLE_CLASSES:
                continue
                
            current_ids.add(tid)
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            position = np.array([cx, cy])
            
            if tid not in self.trackers:
                self.trackers[tid] = PrecisionVehicleTracker(tid)
                self.total_vehicles += 1
                
            tracker = self.trackers[tid]
            vehicle_class = config.VEHICLE_CLASS_NAMES.get(cls, "unknown")
            tracker.update(position, box, self.current_frame, vehicle_class)
            
            # --- VIOLATION LOGIC ---
            is_violating = False
            violation_type = None
            confidence = 0.0

            # Only check sidewalk/lane after enough tracking frames
            if tracker.total_frames >= config.MIN_TRACKING_FRAMES:
                # 1. Sidewalk Check (High Priority)
                if self.lane_manager.is_on_sidewalk(position, frame_shape):
                    is_violating = True
                    violation_type = "Driving on sidewalk"
                    confidence = 1.0
                    tracker.on_sidewalk = True
                
                # 2. Yellow Line Check (Medium Priority)
                elif not is_violating:
                    lane_side = self.lane_manager.get_lane_direction_at_point(position, frame_shape)
                    tracker.lane_side = lane_side
                    if lane_side == 'left_side':
                        is_violating = True
                        violation_type = "Wrong side of road"
                        confidence = 1.0
            
            # 3. Spatial Flow Check (Low Priority - requires reliable zone)
            if not is_violating and tracker.is_valid_for_detection(self.current_frame, self.frame_width, self.frame_height):
                zone = self.get_zone(cx, cy)
                if zone and zone.is_reliable:
                    is_opposite, conf = tracker.is_opposite_direction(zone)
                    if is_opposite:
                        is_violating = True
                        violation_type = "Wrong way (Flow)"
                        confidence = conf

            # Update Tracker Status
            tracker.update_violation_status(is_violating, confidence, violation_type)
            
            # Handle Confirmed Violation
            if tracker.is_confirmed_violator and tracker.confirmed_frame == tracker.total_frames:
                self.total_violations += 1
                self.violations_by_class[vehicle_class] += 1
                
                # Snapshot
                save_violation_snapshot(frame, "wrong_way", tid, box, vehicle_class=vehicle_class)
                print(f"🚨 VIOLATION: ID {tid} ({tracker.violation_type})")
                
                new_violations.append({
                    'type': 'wrong_way',
                    'id': tid,
                    'label': tracker.violation_type or violation_type or 'Wrong Way'
                })
                    
            self._draw_vehicle(frame, tracker, box)
            
        # Cleanup stale trackers using last_frame
        self._cleanup_stale_trackers(current_ids)
        
        return new_violations

    def _cleanup_stale_trackers(self, current_ids: set = None):
        """Xóa tracker đã mất quá lâu để tránh phình state."""
        stale_ids = []
        for tid, tracker in self.trackers.items():
            # If current_ids provided, only check those not currently visible
            if current_ids is not None and tid in current_ids:
                continue
            if (self.current_frame - tracker.last_frame) > config.STALE_TRACKER_FRAMES:
                stale_ids.append(tid)
        for tid in stale_ids:
            del self.trackers[tid]
        
    def _finalize_learning(self):
        print("\n" + "="*60)
        print("📚 LEARNING COMPLETE")
        print("="*60)
        
        reliable = 0
        for zone in self.zones:
            if zone.analyze_flow():
                reliable += 1
                print(f"✓ Zone {zone.zone_id}: {zone.sample_count} samples | Flow: {zone.dominant_flow.value}")
                
        print(f"\n📊 {reliable}/{len(self.zones)} zones learned")
        print("="*60 + "\n")
        
        self.is_learning = False
        
    def _draw_vehicle(self, frame, tracker, box):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if tracker.is_confirmed_violator:
            color = config.COLOR_VIOLATION
            label = f"VIOLATION: {tracker.violation_type}"
        elif tracker.is_wrong_way:
            color = config.COLOR_WARNING
            label = f"WARNING #{tracker.track_id}"
        else:
            color = config.COLOR_SAFE
            label = f"Safe #{tracker.track_id}"
        
        # Use unified bbox style
        draw_bbox_with_label(frame, (x1, y1, x2, y2), label, color)
        
        if tracker.current_velocity is not None:
            scale = 5.0
            end = (int(cx + tracker.current_velocity[0]*scale),
                   int(cy + tracker.current_velocity[1]*scale))
            cv2.arrowedLine(frame, (int(cx), int(cy)), end, color, 3, tipLength=0.3)
            
    def _draw_ui(self, frame):
        if self.is_learning:
            progress = (self.current_frame / config.LEARNING_DURATION_FRAMES) * 100
            draw_calibration_hud(frame, progress, config.LEARNING_DURATION_FRAMES / 30.0)
        else:
            hud_lines = [
                (f"FPS: {self.fps:.1f} | Frame: {self.current_frame}", config.HUD_TEXT_COLOR),
                (f"Vehicles: {len(self.trackers)} | Violations: {self.total_violations}", config.HUD_TEXT_COLOR),
                ("ACTIVE DETECTION", config.COLOR_SAFE),
            ]
            draw_info_hud(frame, hud_lines, title="WRONG-WAY DETECTION", title_color=config.COLOR_WARNING, width=450)

    def get_stats(self) -> dict:
        """Trả về stats cho web — đồng bộ với SignViolationDetector."""
        return {
            'violations': self.total_violations,
            'vehicles': len(self.trackers),
            'learning': self.is_learning,
            'total': self.total_violations
        }


# =====================================================================================
# STANDALONE
# =====================================================================================

def main():
    print("\n" + "="*60)
    print("    WRONG-WAY DETECTION SYSTEM (MERGED)")
    print("="*60 + "\n")
    
    detector = PrecisionWrongWayDetector()
    
    # Load model only for standalone mode
    print(f"🚀 Loading model: {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)
    detector.set_model(model)
    
    cap = cv2.VideoCapture(config.DEFAULT_VIDEO)
    if not cap.isOpened():
        print(f"❌ Cannot open: {config.DEFAULT_VIDEO}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS")
    print(f"⏳ Learning: {config.LEARNING_DURATION_FRAMES} frames")
    print(f"⌨️  Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed, _ = detector.process_frame(frame, conf=config.CONF_DETECTION, debug=True)
            cv2.imshow('Wrong-Way Detection', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("📊 FINAL STATS")
        print("="*60)
        print(f"Frames: {detector.current_frame}")
        print(f"Violations: {detector.total_violations}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()