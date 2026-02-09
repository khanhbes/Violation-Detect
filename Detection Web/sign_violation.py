"""
Traffic Sign Violation Detection System
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

# Import shared config and draw utilities
from config.config import config
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, draw_calibration_hud, save_violation_snapshot


class SignType(Enum):
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
    track_id: int
    vehicle_class: str
    positions: deque
    first_frame: int
    last_frame: int
    is_violating: bool = False
    is_confirmed: bool = False
    violation_frames: int = 0
    violation_type: Optional[str] = None
    
    def update_position(self, pos: Tuple[float, float], frame_num: int):
        self.positions.append(pos)
        self.last_frame = frame_num
        
    def get_trajectory(self) -> np.ndarray:
        if len(self.positions) < 2:
            return None
        return np.array(list(self.positions))
    
    def get_direction(self) -> Optional[Tuple[float, float]]:
        if len(self.positions) < 10:
            return None
        start = np.array(self.positions[0])
        end = np.array(self.positions[-1])
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm < 5.0:
            return None
        return tuple(direction / norm)
    
    def update_violation(self, is_violating: bool, violation_type: str = None):
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
    def __init__(self):
        self.detected_signs: Dict[int, List[Tuple]] = defaultdict(list)
        self.active_signs: List[Dict] = []
        self.is_calibrating = True
        self.calibration_count = 0
        self.calibration_frames = 100
        self.sign_confidence = 0.5
        
    def update_calibration(self, detections: List[Tuple]):
        if not self.is_calibrating:
            return
            
        for class_id, confidence, bbox in detections:
            if class_id in config.SIGN_CLASSES:
                if confidence > self.sign_confidence:
                    self.detected_signs[class_id].append((bbox, confidence))
        
        self.calibration_count += 1
        
        if self.calibration_count >= self.calibration_frames:
            self._finalize_calibration()
            
    def _finalize_calibration(self):
        print("\n" + "="*60)
        print("📊 CALIBRATION COMPLETE - Traffic Sign Analysis")
        print("="*60)
        
        for sign_class, detections in self.detected_signs.items():
            if len(detections) < 5:
                continue
                
            bboxes = [bbox for bbox, conf in detections]
            avg_bbox = np.mean(bboxes, axis=0)
            
            sign_info = {
                'class_id': sign_class,
                'sign_type': config.SIGN_CLASSES[sign_class],
                'bbox': tuple(avg_bbox),
                'detection_count': len(detections),
                'enforcement_zone': self._create_enforcement_zone(avg_bbox)
            }
            
            self.active_signs.append(sign_info)
            print(f"✓ Detected: {sign_info['sign_type']} ({len(detections)} samples)")
        
        print(f"\n📊 Total active signs: {len(self.active_signs)}")
        print("="*60 + "\n")
        
        self.is_calibrating = False
        
    def _create_enforcement_zone(self, sign_bbox: np.ndarray) -> Tuple:
        x1, y1, x2, y2 = sign_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (max(0, cx - 200), max(0, cy - 100), cx + 200, cy + 200)
    
    def check_violation(self, tracker: VehicleTracker) -> Optional[str]:
        if self.is_calibrating:
            return None
            
        direction = tracker.get_direction()
        if direction is None:
            return None
            
        current_pos = tracker.positions[-1]
        
        for sign in self.active_signs:
            if not self._in_zone(current_pos, sign['enforcement_zone']):
                continue
            if self._check_direction_violation(direction, sign['sign_type']):
                return sign['sign_type']
        return None
    
    def _in_zone(self, pos: Tuple[float, float], zone: Tuple) -> bool:
        x, y = pos
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _check_direction_violation(self, direction: Tuple, sign_type: str) -> bool:
        dx, dy = direction
        angle = np.arctan2(dy, dx)
        angle_deg = (np.degrees(angle) + 360) % 360
        
        if 45 <= angle_deg < 135:
            movement = "straight"
        elif 135 <= angle_deg < 225:
            movement = "left"
        elif 225 <= angle_deg < 315:
            movement = "down"
        else:
            movement = "right"
        
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
# MAIN SYSTEM
# =====================================================================================

class TrafficSignViolationDetector:
    def __init__(self):
        print(f"🚀 Loading model: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)
        
        self.sign_detector = TrafficSignDetector()
        self.trackers: Dict[int, VehicleTracker] = {}
        
        self.current_frame = 0
        self.total_violations = 0
        self.violations_by_type = defaultdict(int)
        
        self.fps = 0.0
        self.prev_time = time.time()
        
        print("✅ System initialized")
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.current_frame += 1
        
        results = self.model.track(
            frame, imgsz=config.IMG_SIZE, conf=config.CONF_DETECTION,
            iou=config.IOU_THRESHOLD, persist=True, verbose=False
        )
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confs):
                detections.append((cls, conf, tuple(box)))
        
        if self.sign_detector.is_calibrating:
            self.sign_detector.update_calibration(detections)
            self._draw_calibration(frame)
        else:
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, tid, cls in zip(boxes, track_ids, classes):
                    if cls not in config.VEHICLE_CLASSES:
                        continue
                        
                    if tid not in self.trackers:
                        self.trackers[tid] = VehicleTracker(
                            track_id=tid,
                            vehicle_class=config.VEHICLE_CLASS_NAMES.get(cls, 'vehicle'),
                            positions=deque(maxlen=config.MAX_POSITION_HISTORY),
                            first_frame=self.current_frame,
                            last_frame=self.current_frame
                        )
                    
                    tracker = self.trackers[tid]
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    tracker.update_position((cx, cy), self.current_frame)
                    
                    if len(tracker.positions) >= config.MIN_TRACK_LENGTH:
                        violation_type = self.sign_detector.check_violation(tracker)
                        tracker.update_violation(violation_type is not None, violation_type)
                        
                        if (not tracker.is_confirmed and 
                            tracker.violation_frames >= config.VIOLATION_CONSECUTIVE_FRAMES):
                            tracker.is_confirmed = True
                            self.total_violations += 1
                            if violation_type:
                                self.violations_by_type[violation_type] += 1
                            # Chụp screenshot khi xác nhận violation
                            save_violation_snapshot(frame, f"sign_{violation_type or 'unknown'}", tid, box)
                            print(f"🚨 VIOLATION: ID {tid} - {violation_type}")
                    
                    self._draw_vehicle(frame, tracker, box)
            
            self._draw_signs(frame)
        
        self._draw_ui(frame)
        
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        
        return frame
    
    def _draw_calibration(self, frame):
        progress = (self.sign_detector.calibration_count / self.sign_detector.calibration_frames) * 100
        # Sử dụng draw_calibration_hud thống nhất
        draw_calibration_hud(frame, progress, self.sign_detector.calibration_frames / 30.0)  # ~30 fps
    
    def _draw_vehicle(self, frame, tracker, box):
        x1, y1, x2, y2 = box
        
        if tracker.is_confirmed:
            color = config.COLOR_VIOLATION
            label = f"VIOLATION #{tracker.track_id}"
        elif tracker.is_violating:
            color = config.COLOR_WARNING
            label = f"WARNING #{tracker.track_id}"
        else:
            color = config.COLOR_SAFE
            label = f"Safe #{tracker.track_id}"
        
        # Sử dụng draw_bbox_with_label thống nhất
        draw_bbox_with_label(frame, (x1, y1, x2, y2), label, color)
        
        trajectory = tracker.get_trajectory()
        if trajectory is not None and len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                pt1 = tuple(trajectory[i-1].astype(int))
                pt2 = tuple(trajectory[i].astype(int))
                cv2.line(frame, pt1, pt2, color, 2)
    
    def _draw_signs(self, frame):
        for sign in self.sign_detector.active_signs:
            x1, y1, x2, y2 = [int(v) for v in sign['bbox']]
            # Sử dụng draw_bbox_with_label cho signs
            draw_bbox_with_label(frame, (x1, y1, x2, y2), sign['sign_type'], config.COLOR_WARNING)
            
            # Vẽ enforcement zone (dạng viền mỏng)
            zx1, zy1, zx2, zy2 = [int(v) for v in sign['enforcement_zone']]
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), config.COLOR_WARNING, 1)
    
    def _draw_ui(self, frame):
        # Sử dụng draw_info_hud thống nhất
        status = "CALIBRATING" if self.sign_detector.is_calibrating else f"Signs: {len(self.sign_detector.active_signs)}"
        
        hud_lines = [
            (f"FPS: {self.fps:.1f} | Frame: {self.current_frame}", config.HUD_TEXT_COLOR),
            (f"Vehicles: {len(self.trackers)} | Violations: {self.total_violations}", config.HUD_TEXT_COLOR),
            (status, config.COLOR_WARNING if self.sign_detector.is_calibrating else config.COLOR_SAFE),
        ]
        draw_info_hud(frame, hud_lines, title="SIGN VIOLATION DETECTION", title_color=config.COLOR_WARNING)


def main():
    print("\n" + "="*60)
    print("    TRAFFIC SIGN VIOLATION DETECTION")
    print("="*60 + "\n")
    
    detector = TrafficSignViolationDetector()
    
    cap = cv2.VideoCapture(config.DEFAULT_VIDEO)
    if not cap.isOpened():
        print(f"❌ Cannot open: {config.DEFAULT_VIDEO}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS")
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
        
        print("\n" + "="*60)
        print("📊 FINAL STATISTICS")
        print("="*60)
        print(f"Frames: {detector.current_frame}")
        print(f"Violations: {detector.total_violations}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
