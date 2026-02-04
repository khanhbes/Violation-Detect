"""
=====================================================================================
PRECISION WRONG-WAY DETECTION SYSTEM
Advanced Vehicle Orientation & Direction Analysis
=====================================================================================
Core Innovation: Multi-Modal Direction Detection
- Method 1: Bounding Box Aspect Ratio & Orientation Analysis
- Method 2: Dense Optical Flow Field Analysis
- Method 3: Deep Trajectory Pattern Recognition with Kalman Filtering
- Method 4: Inter-Vehicle Relative Motion Analysis
=====================================================================================
Uses shared Config class from config/config.py
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import time
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass, field
from enum import Enum

# Import shared config
from config.config import config


# =====================================================================================
# WRONG-WAY SPECIFIC CONFIGURATION
# =====================================================================================

class FlowDirection(Enum):
    """Precise flow directions"""
    UPWARD = "upward"           # Top of screen
    DOWNWARD = "downward"       # Bottom of screen
    LEFTWARD = "leftward"       # Left side
    RIGHTWARD = "rightward"     # Right side
    UNDEFINED = "undefined"


# Learning Phase
LEARNING_DURATION = 250          # Extended learning
MIN_SAMPLES_REQUIRED = 30        # Per lane

# Spatial Division
NUM_VERTICAL_LANES = 5           # Vertical lanes
NUM_HORIZONTAL_ZONES = 3         # Horizontal zones

# Tracking Parameters  
MAX_POSITION_HISTORY = 60        # Position memory
KALMAN_PROCESS_NOISE = 0.01      # Process noise
KALMAN_MEASUREMENT_NOISE = 0.1   # Measurement noise

# Motion Analysis
MIN_SPEED_THRESHOLD = 2.0        # Minimum speed (px/frame)
MAX_SPEED_THRESHOLD = 70.0       # Maximum speed
STATIONARY_THRESHOLD = 1.5       # Stationary detection
TRAJECTORY_SMOOTHING_SIGMA = 2.0 # Gaussian smoothing

# Optical Flow
OPTICAL_FLOW_ENABLED = True
FLOW_MAGNITUDE_THRESHOLD = 3.0
FLOW_ANGLE_BINS = 8              # Direction quantization

# Detection Thresholds
DIRECTION_CONFIDENCE_THRESHOLD = 0.75    # 75% confidence
OPPOSITE_ANGLE_THRESHOLD = 135.0         # Degrees
CONSISTENCY_THRESHOLD = 0.70             # 70% consistency

# Violation Confirmation
VIOLATION_CONSECUTIVE_FRAMES = 8         # Reduced for faster detection
VIOLATION_RATIO_THRESHOLD = 0.85         # 85% of recent history
VIOLATION_HISTORY_WINDOW = 12            # History size

# Spatial Filters
BOUNDARY_MARGIN = 80                     # Edge margin
ENTRY_GRACE_PERIOD = 20                  # Frames to stabilize
MIN_TRACKING_FRAMES = 30                 # Minimum track length

# Aspect Ratio Analysis
ENABLE_BBOX_ORIENTATION = True
ASPECT_RATIO_WEIGHT = 0.3                # Weight in ensemble

# Inter-Vehicle Analysis
ENABLE_PEER_COMPARISON = True
PEER_COMPARISON_RADIUS = 200.0           # Pixels
PEER_AGREEMENT_THRESHOLD = 0.6           # 60% agreement

# Vehicle class names mapping
VEHICLE_CLASS_NAMES = {
    0: 'ambulance',
    6: 'car', 
    9: 'fire_truck',
    21: 'motorcycle',
    26: 'police_car'
}


# =====================================================================================
# KALMAN FILTER FOR SMOOTH TRACKING
# =====================================================================================

class KalmanTracker:
    """
    Kalman Filter for smooth position and velocity estimation
    Reduces noise and provides accurate predictions
    """
    
    def __init__(self, initial_position: np.ndarray):
        """
        Initialize Kalman Filter
        State: [x, y, vx, vy]
        """
        self.dt = 1.0  # Time step
        
        # State: [x, y, vx, vy]
        self.state = np.array([
            initial_position[0],
            initial_position[1],
            0.0,
            0.0
        ])
        
        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = KALMAN_PROCESS_NOISE
        self.Q = np.eye(4) * q
        
        # Measurement noise covariance
        r = KALMAN_MEASUREMENT_NOISE
        self.R = np.eye(2) * r
        
        # State covariance
        self.P = np.eye(4) * 1.0
        
    def predict(self) -> np.ndarray:
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return position
        
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update with measurement"""
        # Innovation
        y = measurement - (self.H @ self.state)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[:2]  # Return position
        
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate"""
        return self.state[2:4]
        
    def get_position(self) -> np.ndarray:
        """Get current position estimate"""
        return self.state[:2]


# =====================================================================================
# SPATIAL ZONE ANALYZER
# =====================================================================================

class SpatialZone:
    """
    Represents a spatial zone with learned flow characteristics
    Uses multiple detection methods for robustness
    """
    
    def __init__(self, zone_id: Tuple[int, int], bounds: Tuple[int, int, int, int]):
        self.zone_id = zone_id  # (vertical_lane, horizontal_zone)
        self.bounds = bounds    # (x1, y1, x2, y2)
        
        # Learning data
        self.velocity_samples: List[np.ndarray] = []
        self.angle_samples: List[float] = []
        self.bbox_orientation_samples: List[float] = []  # NEW
        
        # Learned characteristics
        self.dominant_flow: Optional[FlowDirection] = None
        self.dominant_vector: Optional[np.ndarray] = None
        self.dominant_angle: Optional[float] = None
        self.mean_speed: float = 0.0
        self.std_speed: float = 0.0
        
        # Quality metrics
        self.confidence: float = 0.0
        self.is_reliable: bool = False
        self.sample_count: int = 0
        
    def add_sample(self, velocity: np.ndarray, bbox_aspect: Optional[float] = None):
        """Add velocity sample"""
        self.velocity_samples.append(velocity)
        
        angle = np.arctan2(velocity[1], velocity[0])
        self.angle_samples.append(angle)
        
        if bbox_aspect is not None:
            self.bbox_orientation_samples.append(bbox_aspect)
            
        self.sample_count += 1
        
    def analyze_flow(self) -> bool:
        """
        Analyze collected samples to determine dominant flow
        Uses robust statistical methods
        """
        if len(self.velocity_samples) < 10:
            return False
            
        velocities = np.array(self.velocity_samples)
        angles = np.array(self.angle_samples)
        
        # === ROBUST FLOW ESTIMATION ===
        # Use RANSAC-like consensus finding
        self.dominant_vector = self._estimate_consensus_flow(velocities, angles)
        
        if self.dominant_vector is None:
            return False
            
        # Calculate dominant angle
        self.dominant_angle = np.arctan2(self.dominant_vector[1], self.dominant_vector[0])
        
        # Classify direction
        self.dominant_flow = self._classify_flow_direction(self.dominant_angle)
        
        # Speed statistics
        speeds = np.linalg.norm(velocities, axis=1)
        self.mean_speed = np.median(speeds)  # Median is robust to outliers
        self.std_speed = np.std(speeds)
        
        # Calculate confidence
        self.confidence = self._calculate_flow_confidence(angles)
        
        # Reliability check
        self.is_reliable = (
            self.sample_count >= 15 and
            self.confidence > 0.65 and
            self.dominant_flow != FlowDirection.UNDEFINED
        )
        
        return self.is_reliable
        
    def _estimate_consensus_flow(self, velocities: np.ndarray, angles: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate flow using iterative consensus finding
        More robust than simple averaging
        """
        if len(velocities) < 5:
            return None
            
        # Bin angles into 8 directions
        angle_bins = np.linspace(-np.pi, np.pi, 9)
        angle_indices = np.digitize(angles, angle_bins)
        
        # Find most common bin
        unique, counts = np.unique(angle_indices, return_counts=True)
        dominant_bin = unique[np.argmax(counts)]
        
        # Get velocities in dominant bin and adjacent bins
        adjacent_bins = [dominant_bin - 1, dominant_bin, dominant_bin + 1]
        mask = np.isin(angle_indices, adjacent_bins)
        
        if np.sum(mask) < 3:
            return None
            
        # Average velocities in consensus group
        consensus_velocities = velocities[mask]
        mean_velocity = np.mean(consensus_velocities, axis=0)
        
        # Normalize
        norm = np.linalg.norm(mean_velocity)
        if norm > 1e-6:
            return mean_velocity / norm
            
        return None
        
    def _classify_flow_direction(self, angle: float) -> FlowDirection:
        """
        Classify angle into primary flow direction
        4-way classification for clarity
        """
        angle_deg = np.degrees(angle)
        angle_deg = (angle_deg + 360) % 360  # Normalize to [0, 360)
        
        # 4-way classification (90° sectors)
        if 45 <= angle_deg < 135:          # Up
            return FlowDirection.UPWARD
        elif 135 <= angle_deg < 225:       # Left
            return FlowDirection.LEFTWARD
        elif 225 <= angle_deg < 315:       # Down
            return FlowDirection.DOWNWARD
        else:                               # Right (315-45)
            return FlowDirection.RIGHTWARD
            
    def _calculate_flow_confidence(self, angles: np.ndarray) -> float:
        """
        Calculate confidence based on angular concentration
        Uses circular variance
        """
        if len(angles) < 3:
            return 0.0
            
        # Circular mean
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        
        # Mean resultant length (measure of concentration)
        R = np.sqrt(sin_mean**2 + cos_mean**2)
        
        # R ranges from 0 (uniform) to 1 (concentrated)
        return float(R)
        
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is in zone"""
        x1, y1, x2, y2 = self.bounds
        return x1 <= x < x2 and y1 <= y < y2


# =====================================================================================
# ADVANCED VEHICLE TRACKER
# =====================================================================================

class PrecisionVehicleTracker:
    """
    High-precision vehicle tracking with multiple direction estimation methods
    """
    
    def __init__(self, track_id: int):
        self.track_id = track_id
        
        # Kalman filter (initialized on first update)
        self.kalman: Optional[KalmanTracker] = None
        
        # Position and velocity
        self.raw_positions: deque = deque(maxlen=MAX_POSITION_HISTORY)
        self.filtered_positions: deque = deque(maxlen=MAX_POSITION_HISTORY)
        self.velocities: deque = deque(maxlen=30)
        
        # Bounding box history (for orientation analysis)
        self.bbox_history: deque = deque(maxlen=20)
        self.bbox_aspect_ratios: deque = deque(maxlen=20)
        
        # Current state
        self.current_velocity: Optional[np.ndarray] = None
        self.current_speed: float = 0.0
        self.current_angle: Optional[float] = None
        
        # Trajectory characteristics
        self.smoothed_trajectory: Optional[np.ndarray] = None
        self.trajectory_direction: Optional[FlowDirection] = None
        self.direction_confidence: float = 0.0
        
        # Violation status
        self.is_wrong_way: bool = False
        self.is_confirmed_violator: bool = False
        self.violation_evidence: deque = deque(maxlen=VIOLATION_HISTORY_WINDOW)
        self.consecutive_violations: int = 0
        
        # Metadata
        self.total_frames: int = 0
        self.first_frame: int = 0
        self.vehicle_class: str = ""
        self.confirmed_frame: Optional[int] = None
        
    def update(self, position: np.ndarray, bbox: np.ndarray, frame_num: int, vehicle_class: str = ""):
        """Update tracker with new detection"""
        self.raw_positions.append(position)
        self.bbox_history.append(bbox)
        
        # Calculate bounding box aspect ratio
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / (height + 1e-6)
        self.bbox_aspect_ratios.append(aspect_ratio)
        
        # Initialize Kalman on first update
        if self.kalman is None:
            self.kalman = KalmanTracker(position)
            
        # Kalman prediction and update
        self.kalman.predict()
        filtered_pos = self.kalman.update(position)
        self.filtered_positions.append(filtered_pos)
        
        # Update velocity from Kalman
        velocity = self.kalman.get_velocity()
        self.velocities.append(velocity)
        self.current_velocity = velocity
        self.current_speed = np.linalg.norm(velocity)
        
        if self.current_speed > 1e-6:
            self.current_angle = np.arctan2(velocity[1], velocity[0])
        
        # Update metadata
        if self.total_frames == 0:
            self.first_frame = frame_num
        self.total_frames += 1
        self.vehicle_class = vehicle_class
        
        # Update trajectory analysis
        if len(self.filtered_positions) >= 15:
            self._analyze_trajectory()
            
    def _analyze_trajectory(self):
        """
        Analyze trajectory to determine movement direction
        Uses Gaussian smoothing for noise reduction
        """
        if len(self.filtered_positions) < 10:
            return
            
        positions = np.array(list(self.filtered_positions))
        
        # Apply Gaussian smoothing
        smoothed_x = gaussian_filter1d(positions[:, 0], sigma=TRAJECTORY_SMOOTHING_SIGMA)
        smoothed_y = gaussian_filter1d(positions[:, 1], sigma=TRAJECTORY_SMOOTHING_SIGMA)
        self.smoothed_trajectory = np.column_stack([smoothed_x, smoothed_y])
        
        # Calculate overall direction (start to end)
        start = self.smoothed_trajectory[0]
        end = self.smoothed_trajectory[-1]
        overall_vector = end - start
        
        norm = np.linalg.norm(overall_vector)
        if norm > 5.0:  # Minimum displacement
            angle = np.arctan2(overall_vector[1], overall_vector[0])
            
            # Classify direction
            angle_deg = np.degrees(angle)
            angle_deg = (angle_deg + 360) % 360
            
            if 45 <= angle_deg < 135:
                self.trajectory_direction = FlowDirection.UPWARD
            elif 135 <= angle_deg < 225:
                self.trajectory_direction = FlowDirection.LEFTWARD
            elif 225 <= angle_deg < 315:
                self.trajectory_direction = FlowDirection.DOWNWARD
            else:
                self.trajectory_direction = FlowDirection.RIGHTWARD
                
            # Calculate confidence based on trajectory straightness
            path_length = np.sum(np.linalg.norm(np.diff(self.smoothed_trajectory, axis=0), axis=1))
            if path_length > 1e-6:
                self.direction_confidence = norm / path_length
            else:
                self.direction_confidence = 0.0
                
    def estimate_direction_from_bbox(self) -> Optional[FlowDirection]:
        """
        Estimate movement direction from bounding box shape changes
        Vehicles appear wider when moving horizontally, taller when moving vertically
        """
        if len(self.bbox_aspect_ratios) < 5:
            return None
            
        # Average aspect ratio
        avg_aspect = np.mean(list(self.bbox_aspect_ratios))
        
        # If aspect ratio > 1.3: likely moving horizontally
        # If aspect ratio < 0.8: likely moving vertically
        if avg_aspect > 1.3:
            # Horizontal movement - need velocity to determine left/right
            if self.current_velocity is not None:
                if self.current_velocity[0] > 0:
                    return FlowDirection.RIGHTWARD
                else:
                    return FlowDirection.LEFTWARD
        elif avg_aspect < 0.8:
            # Vertical movement - need velocity to determine up/down
            if self.current_velocity is not None:
                if self.current_velocity[1] > 0:
                    return FlowDirection.DOWNWARD
                else:
                    return FlowDirection.UPWARD
                    
        return None
        
    def is_opposite_direction(self, zone: SpatialZone) -> Tuple[bool, float]:
        """
        Determine if vehicle is moving opposite to zone flow
        Returns: (is_opposite, confidence_score)
        """
        if not zone.is_reliable or self.current_velocity is None:
            return False, 0.0
            
        if self.current_speed < MIN_SPEED_THRESHOLD:
            return False, 0.0
            
        # === METHOD 1: Vector Angle Comparison ===
        vehicle_angle = np.arctan2(self.current_velocity[1], self.current_velocity[0])
        angle_diff = self._angular_difference(vehicle_angle, zone.dominant_angle)
        angle_diff_deg = np.degrees(abs(angle_diff))
        
        method1_opposite = angle_diff_deg > OPPOSITE_ANGLE_THRESHOLD
        method1_confidence = min(angle_diff_deg / 180.0, 1.0)
        
        # === METHOD 2: Direction Classification ===
        method2_opposite = False
        method2_confidence = 0.0
        
        if self.trajectory_direction is not None:
            opposite_map = {
                FlowDirection.UPWARD: FlowDirection.DOWNWARD,
                FlowDirection.DOWNWARD: FlowDirection.UPWARD,
                FlowDirection.LEFTWARD: FlowDirection.RIGHTWARD,
                FlowDirection.RIGHTWARD: FlowDirection.LEFTWARD
            }
            
            expected_opposite = opposite_map.get(zone.dominant_flow)
            method2_opposite = (self.trajectory_direction == expected_opposite)
            method2_confidence = self.direction_confidence if method2_opposite else 0.0
            
        # === METHOD 3: Bounding Box Orientation ===
        method3_opposite = False
        method3_confidence = 0.0
        
        if ENABLE_BBOX_ORIENTATION:
            bbox_direction = self.estimate_direction_from_bbox()
            if bbox_direction is not None:
                opposite_map = {
                    FlowDirection.UPWARD: FlowDirection.DOWNWARD,
                    FlowDirection.DOWNWARD: FlowDirection.UPWARD,
                    FlowDirection.LEFTWARD: FlowDirection.RIGHTWARD,
                    FlowDirection.RIGHTWARD: FlowDirection.LEFTWARD
                }
                expected_opposite = opposite_map.get(zone.dominant_flow)
                method3_opposite = (bbox_direction == expected_opposite)
                method3_confidence = 0.7 if method3_opposite else 0.0
                
        # === WEIGHTED ENSEMBLE ===
        weights = [0.5, 0.3, 0.2]  # Angle, Trajectory, BBox
        confidences = [method1_confidence, method2_confidence, method3_confidence]
        opposites = [method1_opposite, method2_opposite, method3_opposite]
        
        # Weighted vote
        total_confidence = 0.0
        for w, c, o in zip(weights, confidences, opposites):
            if o:
                total_confidence += w * c
                
        # Decision threshold
        is_opposite = total_confidence > DIRECTION_CONFIDENCE_THRESHOLD
        
        return is_opposite, total_confidence
        
    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """Calculate signed angular difference"""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
        
    def update_violation_status(self, is_violating: bool, confidence: float):
        """Update violation status with evidence"""
        self.violation_evidence.append((is_violating, confidence))
        
        if is_violating:
            self.consecutive_violations += 1
            self.is_wrong_way = True
        else:
            self.consecutive_violations = 0
            self.is_wrong_way = False
            
        # Confirmation logic
        if not self.is_confirmed_violator:
            if self._check_confirmation():
                self.is_confirmed_violator = True
                self.confirmed_frame = self.total_frames
                
    def _check_confirmation(self) -> bool:
        """Check if violation should be confirmed"""
        # Criterion 1: Consecutive frames
        if self.consecutive_violations >= VIOLATION_CONSECUTIVE_FRAMES:
            return True
            
        # Criterion 2: Weighted evidence ratio
        if len(self.violation_evidence) >= VIOLATION_HISTORY_WINDOW // 2:
            violations = [v for v, c in self.violation_evidence]
            confidences = [c for v, c in self.violation_evidence if v]
            
            if len(confidences) > 0:
                avg_confidence = np.mean(confidences)
                violation_ratio = sum(violations) / len(violations)
                
                # High confidence and high ratio
                if (avg_confidence > 0.8 and 
                    violation_ratio >= VIOLATION_RATIO_THRESHOLD):
                    return True
                    
        return False
        
    def is_valid_for_detection(self, current_frame: int, frame_width: int, frame_height: int) -> bool:
        """Check if vehicle is valid for violation detection"""
        
        # Need sufficient history
        if self.total_frames < MIN_TRACKING_FRAMES:
            return False
            
        # Grace period for new entries
        if (current_frame - self.first_frame) < ENTRY_GRACE_PERIOD:
            return False
            
        # Must be moving
        if self.current_speed < MIN_SPEED_THRESHOLD:
            return False
            
        # Filter tracking errors
        if self.current_speed > MAX_SPEED_THRESHOLD:
            return False
            
        # Not at boundary
        if len(self.raw_positions) > 0:
            x, y = self.raw_positions[-1]
            margin = BOUNDARY_MARGIN
            
            if (x < margin or x > frame_width - margin or
                y < margin or y > frame_height - margin):
                return False
                
        return True


# =====================================================================================
# PEER COMPARISON ANALYZER
# =====================================================================================

class PeerComparisonAnalyzer:
    """
    Analyzes vehicle behavior relative to nearby vehicles
    Identifies outliers moving against traffic flow
    """
    
    def __init__(self):
        pass
        
    def analyze_peer_consistency(
        self, 
        target_tracker: PrecisionVehicleTracker,
        all_trackers: Dict[int, PrecisionVehicleTracker]
    ) -> Tuple[bool, float]:
        """
        Check if target vehicle is moving differently from nearby peers
        Returns: (is_inconsistent, peer_disagreement_score)
        """
        if not ENABLE_PEER_COMPARISON:
            return False, 0.0
            
        if target_tracker.current_velocity is None:
            return False, 0.0
            
        # Get target position
        if len(target_tracker.raw_positions) == 0:
            return False, 0.0
            
        target_pos = np.array(target_tracker.raw_positions[-1])
        target_angle = target_tracker.current_angle
        
        if target_angle is None:
            return False, 0.0
            
        # Find nearby peers
        peer_angles = []
        
        for tid, tracker in all_trackers.items():
            if tid == target_tracker.track_id:
                continue
                
            if tracker.current_angle is None:
                continue
                
            if len(tracker.raw_positions) == 0:
                continue
                
            # Check distance
            peer_pos = np.array(tracker.raw_positions[-1])
            distance = np.linalg.norm(target_pos - peer_pos)
            
            if distance < PEER_COMPARISON_RADIUS:
                peer_angles.append(tracker.current_angle)
                
        # Need at least 3 peers
        if len(peer_angles) < 3:
            return False, 0.0
            
        # Calculate disagreement with peers
        disagreements = []
        for peer_angle in peer_angles:
            diff = abs(self._angular_difference(target_angle, peer_angle))
            disagreements.append(diff)
            
        # Average disagreement
        avg_disagreement = np.mean(disagreements)
        avg_disagreement_deg = np.degrees(avg_disagreement)
        
        # If average disagreement > 120°, likely wrong way
        is_inconsistent = avg_disagreement_deg > 120.0
        disagreement_score = min(avg_disagreement_deg / 180.0, 1.0)
        
        return is_inconsistent, disagreement_score
        
    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """Calculate angular difference"""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff


# =====================================================================================
# MAIN DETECTION SYSTEM
# =====================================================================================

class PrecisionWrongWayDetector:
    """
    High-precision wrong-way detection system
    """
    
    def __init__(self):
        # Load model using shared config
        print(f"🚀 Loading YOLO model: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)
        
        # System state
        self.current_frame: int = 0
        self.is_learning: bool = True
        
        # Spatial zones
        self.zones: List[SpatialZone] = []
        self.frame_width: int = 0
        self.frame_height: int = 0
        
        # Trackers
        self.trackers: Dict[int, PrecisionVehicleTracker] = {}
        
        # Peer analyzer
        self.peer_analyzer = PeerComparisonAnalyzer()
        
        # Statistics
        self.total_vehicles: int = 0
        self.total_violations: int = 0
        self.violations_by_class: Dict[str, int] = defaultdict(int)
        
        # Performance
        self.fps: float = 0.0
        self.prev_time: float = time.time()
        
        print("✅ Precision detection system initialized")
        
    def initialize_zones(self, frame_shape: Tuple[int, int]):
        """Setup spatial zones"""
        self.frame_height, self.frame_width = frame_shape[:2]
        
        lane_width = self.frame_width / NUM_VERTICAL_LANES
        zone_height = self.frame_height / NUM_HORIZONTAL_ZONES
        
        print(f"\n📐 Setting up {NUM_VERTICAL_LANES}x{NUM_HORIZONTAL_ZONES} grid")
        
        for v_lane in range(NUM_VERTICAL_LANES):
            for h_zone in range(NUM_HORIZONTAL_ZONES):
                x1 = int(v_lane * lane_width)
                x2 = int((v_lane + 1) * lane_width)
                y1 = int(h_zone * zone_height)
                y2 = int((h_zone + 1) * zone_height)
                
                zone = SpatialZone((v_lane, h_zone), (x1, y1, x2, y2))
                self.zones.append(zone)
                
    def get_zone(self, x: float, y: float) -> Optional[SpatialZone]:
        """Get zone for position"""
        for zone in self.zones:
            if zone.contains_point(x, y):
                return zone
        return None
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame"""
        self.current_frame += 1
        
        if self.current_frame == 1:
            self.initialize_zones(frame.shape)
            
        # YOLO tracking
        results = self.model.track(
            frame,
            imgsz=config.IMG_SIZE,
            conf=0.3,
            iou=config.IOU_THRESHOLD,
            persist=True,
            classes=config.VEHICLE_CLASSES,
            verbose=False
        )
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            if self.is_learning:
                self._learning_phase(boxes, track_ids, classes)
            else:
                self._detection_phase(boxes, track_ids, classes, frame)
                
        # Finalize learning
        if self.is_learning and self.current_frame >= LEARNING_DURATION:
            self._finalize_learning()
            
        # Visualization
        self._draw_ui(frame)
        
        # Update FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        
        return frame
        
    def _learning_phase(self, boxes, track_ids, classes):
        """Learning phase"""
        for box, tid, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            position = np.array([cx, cy])
            
            # Update tracker
            if tid not in self.trackers:
                self.trackers[tid] = PrecisionVehicleTracker(tid)
                
            tracker = self.trackers[tid]
            vehicle_class = VEHICLE_CLASS_NAMES.get(cls, "unknown")
            tracker.update(position, box, self.current_frame, vehicle_class)
            
            # Add to zone
            if tracker.current_velocity is not None and tracker.current_speed > MIN_SPEED_THRESHOLD:
                zone = self.get_zone(cx, cy)
                if zone:
                    bbox_aspect = None
                    if len(tracker.bbox_aspect_ratios) > 0:
                        bbox_aspect = tracker.bbox_aspect_ratios[-1]
                    zone.add_sample(tracker.current_velocity, bbox_aspect)
                    
    def _detection_phase(self, boxes, track_ids, classes, frame):
        """Detection phase"""
        for box, tid, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            position = np.array([cx, cy])
            
            # Update tracker
            if tid not in self.trackers:
                self.trackers[tid] = PrecisionVehicleTracker(tid)
                self.total_vehicles += 1
                
            tracker = self.trackers[tid]
            vehicle_class = VEHICLE_CLASS_NAMES.get(cls, "unknown")
            tracker.update(position, box, self.current_frame, vehicle_class)
            
            # Validate
            if not tracker.is_valid_for_detection(
                self.current_frame, self.frame_width, self.frame_height
            ):
                continue
                
            # Detect violation
            zone = self.get_zone(cx, cy)
            if zone and zone.is_reliable:
                # Primary detection
                is_opposite, confidence = tracker.is_opposite_direction(zone)
                
                # Peer comparison (optional boost)
                if ENABLE_PEER_COMPARISON:
                    is_inconsistent, peer_score = self.peer_analyzer.analyze_peer_consistency(
                        tracker, self.trackers
                    )
                    if is_inconsistent:
                        confidence = max(confidence, peer_score)
                        is_opposite = True
                        
                tracker.update_violation_status(is_opposite, confidence)
                
                # Log confirmation
                if tracker.is_confirmed_violator and tracker.confirmed_frame == tracker.total_frames:
                    self.total_violations += 1
                    self.violations_by_class[vehicle_class] += 1
                    print(f"🚨 VIOLATION: ID {tid} ({vehicle_class}) - Confidence: {confidence:.2f}")
                    
            # Draw
            self._draw_vehicle(frame, tracker, box)
            
    def _finalize_learning(self):
        """Finalize learning"""
        print("\n" + "="*80)
        print("📚 LEARNING COMPLETE - Flow Analysis")
        print("="*80)
        
        reliable = 0
        for zone in self.zones:
            if zone.analyze_flow():
                reliable += 1
                direction = zone.dominant_flow.value if zone.dominant_flow else "undefined"
                print(f"✓ Zone {zone.zone_id}: {zone.sample_count} samples | "
                      f"Flow: {direction.upper()} | Confidence: {zone.confidence*100:.0f}%")
            else:
                print(f"✗ Zone {zone.zone_id}: Insufficient data ({zone.sample_count} samples)")
                
        print(f"\n📊 {reliable}/{len(self.zones)} zones learned")
        print("="*80 + "\n")
        
        self.is_learning = False
        
    def _draw_vehicle(self, frame, tracker, box):
        """Draw vehicle"""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if tracker.is_confirmed_violator:
            color = config.COLOR_VIOLATION
            label = f"🚨 WRONG WAY #{tracker.track_id}"
        elif tracker.is_wrong_way:
            color = config.COLOR_WARNING
            label = f"⚠ WARNING #{tracker.track_id}"
        else:
            color = config.COLOR_SAFE
            label = f"#{tracker.track_id}"
            
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw velocity arrow
        if tracker.current_velocity is not None:
            scale = 5.0
            end = (int(cx + tracker.current_velocity[0]*scale),
                   int(cy + tracker.current_velocity[1]*scale))
            cv2.arrowedLine(frame, (int(cx), int(cy)), end, color, 3, tipLength=0.3)
            
    def _draw_ui(self, frame):
        """Draw UI"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, "PRECISION WRONG-WAY DETECTOR", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y = 70
        texts = [
            f"FPS: {self.fps:.1f} | Frame: {self.current_frame}",
            f"Vehicles: {len(self.trackers)} | Violations: {self.total_violations}",
        ]
        
        if self.is_learning:
            progress = (self.current_frame / LEARNING_DURATION) * 100
            texts.append(f">>> LEARNING: {progress:.0f}% <<<")
        else:
            texts.append("Status: ACTIVE DETECTION")
            
        for text in texts:
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 30
            
    def get_stats(self):
        """Get statistics"""
        return {
            'frames': self.current_frame,
            'vehicles': self.total_vehicles,
            'violations': self.total_violations,
            'by_class': dict(self.violations_by_class)
        }


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    print("\n" + "="*80)
    print("    PRECISION WRONG-WAY DETECTION SYSTEM")
    print("    Advanced Multi-Modal Direction Analysis")
    print("="*80 + "\n")
    
    detector = PrecisionWrongWayDetector()
    
    video_path = config.DEFAULT_VIDEO
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS")
    print(f"⏳ Learning: {LEARNING_DURATION} frames")
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
        
        stats = detector.get_stats()
        print("\n" + "="*80)
        print("📊 FINAL STATS")
        print("="*80)
        print(f"Frames: {stats['frames']}")
        print(f"Vehicles: {stats['vehicles']}")
        print(f"Violations: {stats['violations']}")
        if stats['by_class']:
            print("\nBy Class:")
            for cls, count in stats['by_class'].items():
                print(f"  {cls}: {count}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()