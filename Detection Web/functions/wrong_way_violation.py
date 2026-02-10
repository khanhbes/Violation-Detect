"""
PRECISION WRONG-WAY DETECTION SYSTEM
Uses shared Config class from config/config.py
"""

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
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, save_violation_snapshot


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
        angle_deg = np.degrees(angle)
        angle_deg = (angle_deg + 360) % 360
        if 45 <= angle_deg < 135:
            return FlowDirection.UPWARD
        elif 135 <= angle_deg < 225:
            return FlowDirection.LEFTWARD
        elif 225 <= angle_deg < 315:
            return FlowDirection.DOWNWARD
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
        
        self.is_wrong_way: bool = False
        self.is_confirmed_violator: bool = False
        self.violation_evidence: deque = deque(maxlen=config.VIOLATION_HISTORY_WINDOW)
        self.consecutive_violations: int = 0
        
        self.total_frames: int = 0
        self.first_frame: int = 0
        self.vehicle_class: str = ""
        self.confirmed_frame: Optional[int] = None
        
    def update(self, position: np.ndarray, bbox: np.ndarray, frame_num: int, vehicle_class: str = ""):
        self.raw_positions.append(position)
        
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
            
            if 45 <= angle_deg < 135:
                self.trajectory_direction = FlowDirection.UPWARD
            elif 135 <= angle_deg < 225:
                self.trajectory_direction = FlowDirection.LEFTWARD
            elif 225 <= angle_deg < 315:
                self.trajectory_direction = FlowDirection.DOWNWARD
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
        
    def update_violation_status(self, is_violating: bool, confidence: float):
        self.violation_evidence.append((is_violating, confidence))
        
        if is_violating:
            self.consecutive_violations += 1
            self.is_wrong_way = True
        else:
            self.consecutive_violations = 0
            self.is_wrong_way = False
            
        if not self.is_confirmed_violator:
            if self.consecutive_violations >= config.VIOLATION_CONSECUTIVE_FRAMES:
                self.is_confirmed_violator = True
                self.confirmed_frame = self.total_frames
                
    def is_valid_for_detection(self, current_frame: int, frame_width: int, frame_height: int) -> bool:
        if self.total_frames < config.MIN_TRACKING_FRAMES:
            return False
        if (current_frame - self.first_frame) < config.ENTRY_GRACE_PERIOD:
            return False
        if self.current_speed < config.MIN_SPEED_THRESHOLD:
            return False
        if self.current_speed > config.MAX_SPEED_THRESHOLD:
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
        print(f"🚀 Loading model: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)
        
        self.current_frame: int = 0
        self.is_learning: bool = True
        self.zones: List[SpatialZone] = []
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.trackers: Dict[int, PrecisionVehicleTracker] = {}
        
        self.total_vehicles: int = 0
        self.total_violations: int = 0
        self.violations_by_class: Dict[str, int] = defaultdict(int)
        
        self.fps: float = 0.0
        self.prev_time: float = time.time()
        
        print("✅ System initialized")
        
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
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.current_frame += 1
        
        if self.current_frame == 1:
            self.initialize_zones(frame.shape)
            
        results = self.model.track(
            frame, imgsz=config.IMG_SIZE, conf=config.CONF_DETECTION,
            iou=config.IOU_THRESHOLD, persist=True,
            classes=config.VEHICLE_CLASSES, verbose=False
        )
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            if self.is_learning:
                self._learning_phase(boxes, track_ids, classes)
            else:
                self._detection_phase(boxes, track_ids, classes, frame)
                
        if self.is_learning and self.current_frame >= config.LEARNING_DURATION_FRAMES:
            self._finalize_learning()
            
        self._draw_ui(frame)
        
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        
        return frame
        
    def _learning_phase(self, boxes, track_ids, classes):
        for box, tid, cls in zip(boxes, track_ids, classes):
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
                    
    def _detection_phase(self, boxes, track_ids, classes, frame):
        for box, tid, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            position = np.array([cx, cy])
            
            if tid not in self.trackers:
                self.trackers[tid] = PrecisionVehicleTracker(tid)
                self.total_vehicles += 1
                
            tracker = self.trackers[tid]
            vehicle_class = config.VEHICLE_CLASS_NAMES.get(cls, "unknown")
            tracker.update(position, box, self.current_frame, vehicle_class)
            
            if not tracker.is_valid_for_detection(self.current_frame, self.frame_width, self.frame_height):
                continue
                
            zone = self.get_zone(cx, cy)
            if zone and zone.is_reliable:
                is_opposite, confidence = tracker.is_opposite_direction(zone)
                tracker.update_violation_status(is_opposite, confidence)
                
                if tracker.is_confirmed_violator and tracker.confirmed_frame == tracker.total_frames:
                    self.total_violations += 1
                    self.violations_by_class[vehicle_class] += 1
                    # Chụp screenshot ngay khi xác nhận violation
                    save_violation_snapshot(frame, "wrong_way", tid, box)
                    print(f"🚨 VIOLATION: ID {tid} ({vehicle_class})")
                    
            self._draw_vehicle(frame, tracker, box)
            
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
            label = f"VIOLATION #{tracker.track_id} (WRONG WAY)"
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
        y = 70
        if self.is_learning:
            progress = (self.current_frame / config.LEARNING_DURATION_FRAMES) * 100
            hud_lines = [
                (f"FPS: {self.fps:.1f} | Frame: {self.current_frame}", config.HUD_TEXT_COLOR),
                (f"Vehicles: {len(self.trackers)} | Violations: {self.total_violations}", config.HUD_TEXT_COLOR),
                (f"LEARNING: {progress:.0f}%", config.COLOR_WARNING),
            ]
            draw_info_hud(frame, hud_lines, title="WRONG-WAY DETECTION", title_color=config.COLOR_WARNING, width=450)
        else:
            hud_lines = [
                (f"FPS: {self.fps:.1f} | Frame: {self.current_frame}", config.HUD_TEXT_COLOR),
                (f"Vehicles: {len(self.trackers)} | Violations: {self.total_violations}", config.HUD_TEXT_COLOR),
                ("ACTIVE DETECTION", config.COLOR_SAFE),
            ]
            draw_info_hud(frame, hud_lines, title="WRONG-WAY DETECTION", title_color=config.COLOR_WARNING, width=450)


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    print("\n" + "="*60)
    print("    WRONG-WAY DETECTION SYSTEM")
    print("="*60 + "\n")
    
    detector = PrecisionWrongWayDetector()
    
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
            processed = detector.process_frame(frame)
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