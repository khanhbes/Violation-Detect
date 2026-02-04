"""
Traffic Sign Violation Detection System
Detects vehicles violating traffic sign rules (turn restrictions, directional signs)

Model: YOLOv12s-seg (Instance Segmentation)
Uses shared Config class from config/config.py
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import shared config
from config.config import config


# =====================================================================================
# SIGN-SPECIFIC CONFIGURATION
# =====================================================================================

# Sign class IDs (specific to this detection type)
SIGN_CLASSES = {
    1: 'arrow_left',
    2: 'arrow_right',
    3: 'arrow_straight',
    4: 'arrow_straight_left',
    5: 'arrow_straight_right',
}

# Calibration parameters
CALIBRATION_FRAMES = 100
SIGN_CONFIDENCE_THRESHOLD = 0.5

# Tracking parameters
MAX_HISTORY = 50
MIN_TRACK_LENGTH = 20
MIN_SPEED = 2.0
VIOLATION_CONSECUTIVE_FRAMES = 10


class SignType(Enum):
    """Traffic sign types"""
    NO_LEFT_TURN = "no_left_turn"
    NO_RIGHT_TURN = "no_right_turn"
    NO_STRAIGHT = "no_straight"
    TURN_LEFT_ONLY = "turn_left_only"
    TURN_RIGHT_ONLY = "turn_right_only"
    STRAIGHT_ONLY = "straight_only"


# =====================================================================================
# VEHICLE TRACKER
# =====================================================================================

@dataclass
class VehicleTracker:
    """Track individual vehicle"""
    track_id: int
    vehicle_class: str
    positions: deque
    first_frame: int
    last_frame: int
    
    # Violation tracking
    is_violating: bool = False
    is_confirmed: bool = False
    violation_frames: int = 0
    violation_type: Optional[str] = None
    
    def update_position(self, pos: Tuple[float, float], frame_num: int):
        """Update vehicle position"""
        self.positions.append(pos)
        self.last_frame = frame_num
        
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory as numpy array"""
        if len(self.positions) < 2:
            return None
        return np.array(list(self.positions))
    
    def get_direction(self) -> Optional[Tuple[float, float]]:
        """Calculate movement direction"""
        if len(self.positions) < 10:
            return None
            
        # Use first and last points for overall direction
        start = np.array(self.positions[0])
        end = np.array(self.positions[-1])
        
        direction = end - start
        norm = np.linalg.norm(direction)
        
        if norm < 5.0:  # Too small movement
            return None
            
        return tuple(direction / norm)
    
    def update_violation(self, is_violating: bool, violation_type: str = None):
        """Update violation status"""
        if is_violating:
            self.violation_frames += 1
            self.is_violating = True
            if violation_type:
                self.violation_type = violation_type
        else:
            self.violation_frames = 0
            self.is_violating = False


# =====================================================================================
# TRAFFIC SIGN DETECTOR
# =====================================================================================

class TrafficSignDetector:
    """Detect and track traffic signs"""
    
    def __init__(self):
        # Sign memory (learned during calibration)
        self.detected_signs: Dict[int, List[Tuple]] = defaultdict(list)  # sign_class: [(bbox, conf)]
        self.active_signs: List[Dict] = []  # Finalized signs with zones
        
        # Calibration
        self.is_calibrating = True
        self.calibration_count = 0
        
    def update_calibration(self, detections: List[Tuple]):
        """Collect sign detections during calibration"""
        if not self.is_calibrating:
            return
            
        for class_id, confidence, bbox in detections:
            if class_id in SIGN_CLASSES:
                if confidence > SIGN_CONFIDENCE_THRESHOLD:
                    self.detected_signs[class_id].append((bbox, confidence))
        
        self.calibration_count += 1
        
        # Finalize calibration
        if self.calibration_count >= CALIBRATION_FRAMES:
            self._finalize_calibration()
            
    def _finalize_calibration(self):
        """Finalize calibration and create active sign zones"""
        print("\n" + "="*80)
        print("📊 CALIBRATION COMPLETE - Traffic Sign Analysis")
        print("="*80)
        
        for sign_class, detections in self.detected_signs.items():
            if len(detections) < 5:  # Need minimum detections
                continue
                
            # Cluster detections (simple averaging)
            bboxes = [bbox for bbox, conf in detections]
            avg_bbox = np.mean(bboxes, axis=0)
            
            # Create sign zone
            sign_info = {
                'class_id': sign_class,
                'sign_type': SIGN_CLASSES[sign_class],
                'bbox': tuple(avg_bbox),
                'detection_count': len(detections),
                'enforcement_zone': self._create_enforcement_zone(avg_bbox)
            }
            
            self.active_signs.append(sign_info)
            
            print(f"✓ Detected: {sign_info['sign_type']} ({len(detections)} samples)")
        
        print(f"\n📊 Total active signs: {len(self.active_signs)}")
        print("="*80 + "\n")
        
        self.is_calibrating = False
        
    def _create_enforcement_zone(self, sign_bbox: np.ndarray) -> Tuple:
        """Create enforcement zone based on sign position"""
        x1, y1, x2, y2 = sign_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Create zone extending from sign (simplified)
        # In real scenario, this would be more sophisticated
        zone_x1 = max(0, cx - 200)
        zone_y1 = max(0, cy - 100)
        zone_x2 = cx + 200
        zone_y2 = cy + 200
        
        return (zone_x1, zone_y1, zone_x2, zone_y2)
    
    def check_violation(self, tracker: VehicleTracker) -> Optional[str]:
        """Check if vehicle violates any sign"""
        if self.is_calibrating:
            return None
            
        direction = tracker.get_direction()
        if direction is None:
            return None
            
        current_pos = tracker.positions[-1]
        
        for sign in self.active_signs:
            # Check if vehicle in enforcement zone
            if not self._in_zone(current_pos, sign['enforcement_zone']):
                continue
                
            # Check direction vs sign rule
            violation = self._check_direction_violation(
                direction, sign['sign_type']
            )
            
            if violation:
                return sign['sign_type']
                
        return None
    
    def _in_zone(self, pos: Tuple[float, float], zone: Tuple) -> bool:
        """Check if position is in zone"""
        x, y = pos
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _check_direction_violation(self, direction: Tuple, sign_type: str) -> bool:
        """Check if direction violates sign rule"""
        dx, dy = direction
        
        # Classify movement direction
        angle = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle)
        angle_deg = (angle_deg + 360) % 360
        
        # Determine movement type
        if 45 <= angle_deg < 135:  # Up
            movement = "straight"
        elif 135 <= angle_deg < 225:  # Left
            movement = "left"
        elif 225 <= angle_deg < 315:  # Down
            movement = "down"
        else:  # Right
            movement = "right"
        
        # Check against sign rules
        if sign_type == "no_left_turn" and movement == "left":
            return True
        elif sign_type == "no_right_turn" and movement == "right":
            return True
        elif sign_type == "no_straight" and movement == "straight":
            return True
        elif sign_type == "turn_left_only" and movement != "left":
            return True
        elif sign_type == "turn_right_only" and movement != "right":
            return True
        elif sign_type == "straight_only" and movement != "straight":
            return True
            
        return False


# =====================================================================================
# MAIN DETECTION SYSTEM
# =====================================================================================

class TrafficSignViolationDetector:
    """Main detection system"""
    
    def __init__(self):
        # Load model
        print(f"🚀 Loading model: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)
        
        # Components
        self.sign_detector = TrafficSignDetector()
        self.trackers: Dict[int, VehicleTracker] = {}
        
        # Vehicle class names mapping from config
        self.vehicle_class_names = {
            0: 'ambulance',
            6: 'car',
            9: 'fire_truck',
            21: 'motorcycle',
            26: 'police_car'
        }
        
        # State
        self.current_frame = 0
        self.total_violations = 0
        self.violations_by_type = defaultdict(int)
        
        # Performance
        self.fps = 0.0
        self.prev_time = time.time()
        
        print("✅ System initialized")
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame"""
        self.current_frame += 1
        
        # Run YOLO tracking
        results = self.model.track(
            frame,
            imgsz=config.IMG_SIZE,
            conf=0.3,
            iou=config.IOU_THRESHOLD,
            persist=True,
            verbose=False
        )
        
        # Extract detections
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confs):
                detections.append((cls, conf, tuple(box)))
        
        # Update calibration
        if self.sign_detector.is_calibrating:
            self.sign_detector.update_calibration(detections)
            self._draw_calibration(frame)
        else:
            # Process vehicles
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, tid, cls in zip(boxes, track_ids, classes):
                    if cls not in config.VEHICLE_CLASSES:
                        continue
                        
                    # Update tracker
                    if tid not in self.trackers:
                        self.trackers[tid] = VehicleTracker(
                            track_id=tid,
                            vehicle_class=self.vehicle_class_names.get(cls, 'vehicle'),
                            positions=deque(maxlen=MAX_HISTORY),
                            first_frame=self.current_frame,
                            last_frame=self.current_frame
                        )
                    
                    tracker = self.trackers[tid]
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    tracker.update_position((cx, cy), self.current_frame)
                    
                    # Check violation
                    if len(tracker.positions) >= MIN_TRACK_LENGTH:
                        violation_type = self.sign_detector.check_violation(tracker)
                        tracker.update_violation(violation_type is not None, violation_type)
                        
                        # Confirm violation
                        if (not tracker.is_confirmed and 
                            tracker.violation_frames >= VIOLATION_CONSECUTIVE_FRAMES):
                            tracker.is_confirmed = True
                            self.total_violations += 1
                            if violation_type:
                                self.violations_by_type[violation_type] += 1
                            print(f"🚨 VIOLATION: ID {tid} - {violation_type}")
                    
                    # Draw
                    self._draw_vehicle(frame, tracker, box)
            
            # Draw signs
            self._draw_signs(frame)
        
        # Draw UI
        self._draw_ui(frame)
        
        # Update FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        
        return frame
    
    def _draw_calibration(self, frame):
        """Draw calibration progress"""
        progress = (self.sign_detector.calibration_count / CALIBRATION_FRAMES) * 100
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "CALIBRATION MODE", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Progress: {progress:.0f}%", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_vehicle(self, frame, tracker, box):
        """Draw vehicle"""
        x1, y1, x2, y2 = box
        
        if tracker.is_confirmed:
            color = config.COLOR_VIOLATION
            label = f"🚨 VIOLATION #{tracker.track_id}"
        elif tracker.is_violating:
            color = config.COLOR_WARNING
            label = f"⚠ #{tracker.track_id}"
        else:
            color = config.COLOR_SAFE
            label = f"#{tracker.track_id}"
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw trajectory
        trajectory = tracker.get_trajectory()
        if trajectory is not None and len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                pt1 = tuple(trajectory[i-1].astype(int))
                pt2 = tuple(trajectory[i].astype(int))
                cv2.line(frame, pt1, pt2, color, 2)
    
    def _draw_signs(self, frame):
        """Draw detected signs"""
        for sign in self.sign_detector.active_signs:
            x1, y1, x2, y2 = [int(v) for v in sign['bbox']]
            
            # Draw sign box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, sign['sign_type'], (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Draw enforcement zone
            zx1, zy1, zx2, zy2 = [int(v) for v in sign['enforcement_zone']]
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 255), 1)
    
    def _draw_ui(self, frame):
        """Draw UI"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "TRAFFIC SIGN VIOLATION DETECTION", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y = 65
        texts = [
            f"FPS: {self.fps:.1f} | Frame: {self.current_frame}",
            f"Vehicles: {len(self.trackers)} | Violations: {self.total_violations}"
        ]
        
        if self.sign_detector.is_calibrating:
            progress = (self.sign_detector.calibration_count / CALIBRATION_FRAMES) * 100
            texts.append(f">>> CALIBRATING: {progress:.0f}% <<<")
        else:
            texts.append(f"Active Signs: {len(self.sign_detector.active_signs)}")
        
        for text in texts:
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y += 25


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    print("\n" + "="*80)
    print("    TRAFFIC SIGN VIOLATION DETECTION SYSTEM")
    print("="*80 + "\n")
    
    detector = TrafficSignViolationDetector()
    
    video_path = config.DEFAULT_VIDEO
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS")
    print(f"⏳ Calibration: {CALIBRATION_FRAMES} frames")
    print(f"⌨️  Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = detector.process_frame(frame)
            cv2.imshow('Traffic Sign Violation Detection', processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*80)
        print("📊 FINAL STATISTICS")
        print("="*80)
        print(f"Frames: {detector.current_frame}")
        print(f"Total Violations: {detector.total_violations}")
        
        if detector.violations_by_type:
            print("\nViolations by Type:")
            for vtype, count in detector.violations_by_type.items():
                print(f"  {vtype}: {count}")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
