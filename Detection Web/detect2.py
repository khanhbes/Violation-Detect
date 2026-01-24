import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

# ==================== ENUMS & DATA CLASSES ====================

class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"

class VehicleState(Enum):
    BEFORE_STOPLINE = "before"
    CROSSED_STOPLINE = "crossed"

@dataclass
class Vehicle:
    id: int
    class_name: str
    state: VehicleState = VehicleState.BEFORE_STOPLINE
    violation_type: Optional[str] = None
    frames_since_last_seen: int = 0
    bbox_history: deque = None
    
    def __post_init__(self):
        if self.bbox_history is None:
            # Store slightly more history to ensure we catch transitions
            self.bbox_history = deque(maxlen=30)

# ==================== BYTETRACK IMPLEMENTATION ====================

class ByteTracker:
    """Simple ByteTrack implementation for vehicle tracking"""
    
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracked_vehicles: Dict[int, Vehicle] = {}
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections, class_names):
        """
        detections: List of [x1, y1, x2, y2, conf, class_id]
        """
        self.frame_count += 1
        
        # Mark all existing tracks as not seen
        for vehicle in self.tracked_vehicles.values():
            vehicle.frames_since_last_seen += 1
        
        # Handle empty detections
        if len(detections) == 0:
            self._remove_lost_tracks()
            return self.tracked_vehicles
        
        # Match detections to existing tracks
        matched, unmatched_dets = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, det_idx in matched:
            det = detections[det_idx]
            vehicle = self.tracked_vehicles[track_id]
            vehicle.bbox_history.append(det[:4])
            vehicle.frames_since_last_seen = 0
            # Safe class name update
            cls_id = int(det[5])
            if isinstance(class_names, dict):
                vehicle.class_name = class_names.get(cls_id, str(cls_id))
            else:
                vehicle.class_name = class_names[cls_id]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det[4] > self.track_thresh:  # confidence threshold
                cls_id = int(det[5])
                cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
                
                new_vehicle = Vehicle(
                    id=self.next_id,
                    class_name=cls_name
                )
                new_vehicle.bbox_history.append(det[:4])
                self.tracked_vehicles[self.next_id] = new_vehicle
                self.next_id += 1
        
        # Remove lost tracks
        self._remove_lost_tracks()
        
        return self.tracked_vehicles
    
    def _match_detections(self, detections):
        """Match detections to existing tracks using IoU"""
        if len(self.tracked_vehicles) == 0:
            return [], list(range(len(detections)))
        
        # Get last bboxes of tracked vehicles
        track_ids = list(self.tracked_vehicles.keys())
        track_bboxes = np.array([
            list(self.tracked_vehicles[tid].bbox_history[-1]) 
            for tid in track_ids
        ])
        
        det_bboxes = np.array([det[:4] for det in detections])
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)
        
        # Simple greedy matching
        matched = []
        unmatched_dets = list(range(len(detections)))
        
        for track_idx, track_id in enumerate(track_ids):
            if len(unmatched_dets) == 0:
                break
            
            # Look only at unmatched detections
            ious = iou_matrix[track_idx, unmatched_dets]
            
            if len(ious) > 0 and ious.max() > self.match_thresh:
                # Find index in the subset, then map back to original index
                subset_idx = ious.argmax()
                best_det_idx = unmatched_dets[subset_idx]
                
                matched.append((track_id, best_det_idx))
                unmatched_dets.pop(subset_idx) # Remove by index in list
        
        return matched, unmatched_dets
    
    def _compute_iou_matrix(self, bboxes1, bboxes2):
        """Compute IoU between two sets of bounding boxes"""
        ious = np.zeros((len(bboxes1), len(bboxes2)))
        
        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                ious[i, j] = self._compute_iou(bbox1, bbox2)
        
        return ious
    
    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """Compute IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _remove_lost_tracks(self):
        """Remove tracks that haven't been seen for too long"""
        lost_ids = [
            tid for tid, vehicle in self.tracked_vehicles.items()
            if vehicle.frames_since_last_seen > self.track_buffer
        ]
        for tid in lost_ids:
            del self.tracked_vehicles[tid]

# ==================== TRAFFIC VIOLATION DETECTOR ====================

class TrafficViolationDetector:
    """Main class for traffic violation detection"""
    
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
            
        self.model = YOLO(model_path)
        self.tracker = ByteTracker(track_thresh=0.4, track_buffer=30, match_thresh=0.7)
        
        # Class definitions (ENSURE THESE MATCH YOUR MODEL)
        self.vehicle_classes = ['car', 'motorcycle', 'ambulance', 'fire_truck', 'police_car', 'bus', 'truck']
        self.traffic_light_classes = [
            'light_left_green', 'light_left_red', 'light_left_yellow',
            'light_right_green', 
            'light_straight_arrow_green', 'light_straight_arrow_red', 'light_straight_arrow_yellow',
            'light_straight_circle_green', 'light_straight_circle_red', 'light_straight_circle_yellow',
            'traffic_light_red', 'traffic_light_green', 'traffic_light_yellow' # Added generic fallback
        ]
        
        # Stopline detection
        self.stopline_y = None
        self.stopline_detection_start = None
        self.stopline_candidates = []
        self.stopline_locked = False
        self.stopline_detection_duration = 5.0  # seconds
        
        # Statistics
        self.total_vehicles = 0
        self.total_violations = 0
        self.violation_vehicles = set()
        
        # FPS calculation
        self.fps_history = deque(maxlen=30)
        self.prev_time = time.time()
        
    def detect_stopline(self, results, frame_height):
        """Detect and lock the best stopline position"""
        if self.stopline_locked:
            return
        
        # Initialize detection timer
        if self.stopline_detection_start is None:
            self.stopline_detection_start = time.time()
        
        # Find stopline detections
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Check for various common stopline class names
                if ('stop' in class_name and 'line' in class_name) and box.conf[0] > 0.4:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Use bottom of the stopline bbox
                    stopline_y = int(y2)
                    self.stopline_candidates.append({
                        'y': stopline_y,
                        'conf': float(box.conf[0]),
                        'time': time.time()
                    })
        
        # Check if detection period is over
        elapsed = time.time() - self.stopline_detection_start
        
        if elapsed >= self.stopline_detection_duration and len(self.stopline_candidates) > 0:
            # Select best stopline (highest confidence, most recent)
            self.stopline_candidates.sort(key=lambda x: (x['conf'], -x['time']), reverse=True)
            self.stopline_y = self.stopline_candidates[0]['y']
            self.stopline_locked = True
            print(f"✓ Stopline locked at y={self.stopline_y}")
        elif elapsed >= self.stopline_detection_duration:
            # Reset timer to continue looking if nothing found
            # print("⚠ No stopline detected, continuing search...") # Reduced spam
            self.stopline_detection_start = time.time()
    
    def get_traffic_light_state(self, results) -> TrafficLightState:
        """Determine current traffic light state"""
        red_lights = []
        yellow_lights = []
        green_lights = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                conf = float(box.conf[0])
                
                # Simple keyword matching to be more robust
                if class_name in self.traffic_light_classes or 'light' in class_name:
                    if 'red' in class_name:
                        red_lights.append(conf)
                    elif 'yellow' in class_name:
                        yellow_lights.append(conf)
                    elif 'green' in class_name:
                        green_lights.append(conf)
        
        # Priority: Red > Yellow > Green
        if red_lights:
            return TrafficLightState.RED
        elif yellow_lights:
            return TrafficLightState.YELLOW
        elif green_lights:
            return TrafficLightState.GREEN
        
        return TrafficLightState.UNKNOWN
    
    def check_vehicle_violation(self, vehicle: Vehicle, current_bbox, traffic_light_state: TrafficLightState):
        """Check if vehicle violates traffic rules using crossing logic"""
        if self.stopline_y is None:
            return
        
        _, _, _, y2_curr = current_bbox
        
        # We need at least 2 frames of history to detect a "crossing"
        if len(vehicle.bbox_history) < 2:
            return

        # Get the previous frame's Y coordinate (bottom of the box)
        # history[-1] is current, history[-2] is previous
        _, _, _, y2_prev = vehicle.bbox_history[-2]
        
        # LOGIC FIX: Check for crossing the line (Transition)
        # Assumes Y increases downwards (Top=0, Bottom=Height)
        # Previous was BEFORE (less than) stopline, Current is AFTER (greater or equal)
        
        has_crossed = (y2_prev < self.stopline_y) and (y2_curr >= self.stopline_y)
        
        if vehicle.state == VehicleState.BEFORE_STOPLINE and has_crossed:
            vehicle.state = VehicleState.CROSSED_STOPLINE
            
            # Check violation at the moment of crossing
            if traffic_light_state == TrafficLightState.RED:
                vehicle.violation_type = "Violation"
                if vehicle.id not in self.violation_vehicles:
                    self.violation_vehicles.add(vehicle.id)
                    self.total_violations += 1
                    print(f"!!! Violation Detected: Vehicle ID {vehicle.id}")
            elif traffic_light_state == TrafficLightState.YELLOW:
                vehicle.violation_type = "Warning"
            else:
                vehicle.violation_type = None
    
    def get_bbox_color_and_label(self, vehicle: Vehicle, class_name: str) -> Tuple[Tuple[int, int, int], str]:
        """Get bounding box color and label based on vehicle state"""
        # Default color
        color = (0, 255, 0) # Green
        label = f"ID{vehicle.id} {class_name}"

        if vehicle.violation_type == "Violation":
            color = (0, 0, 255) # Red
            label = f"ID{vehicle.id} VIOLATION" 
        elif vehicle.violation_type == "Warning":
            color = (0, 255, 255) # Yellow
            label = f"ID{vehicle.id} WARNING"
            
        return color, label
    
    def process_frame(self, frame):
        """Process a single frame"""
        current_time = time.time()
        time_diff = current_time - self.prev_time
        fps = 1 / time_diff if time_diff > 0.001 else 0 # Avoid div by zero
        self.fps_history.append(fps)
        self.prev_time = current_time
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Detect stopline if not locked
        if not self.stopline_locked:
            self.detect_stopline(results, frame.shape[0])
        
        # Get traffic light state
        traffic_light_state = self.get_traffic_light_state(results)
        
        # Prepare detections for tracking (only vehicles)
        vehicle_detections = []
        all_detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                detection = [x1, y1, x2, y2, conf, class_id]
                all_detections.append((detection, class_name))
                
                if class_name in self.vehicle_classes and conf > 0.4:
                    vehicle_detections.append(detection)
        
        # Update tracker with vehicle detections
        if len(vehicle_detections) > 0:
            vehicle_detections_array = np.array(vehicle_detections)
            tracked_vehicles = self.tracker.update(vehicle_detections_array, self.model.names)
        else:
            # Pass empty array of correct shape if no detections
            tracked_vehicles = self.tracker.update([], self.model.names)
        
        # Update total vehicle count
        self.total_vehicles = len(tracked_vehicles)
        
        # Check violations and draw tracked vehicles
        annotated_frame = frame.copy()
        
        for vehicle_id, vehicle in tracked_vehicles.items():
            if len(vehicle.bbox_history) > 0:
                bbox = vehicle.bbox_history[-1]
                self.check_vehicle_violation(vehicle, bbox, traffic_light_state)
                
                color, label = self.get_bbox_color_and_label(vehicle, vehicle.class_name)
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                # Ensure label doesn't go off screen
                y_label = max(y1, label_size[1] + 10)
                
                cv2.rectangle(annotated_frame, (x1, y_label - label_size[1] - 10), 
                            (x1 + label_size[0], y_label), color, -1)
                cv2.putText(annotated_frame, label, (x1, y_label - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw non-vehicle detections (with class name only)
        for detection, class_name in all_detections:
            if class_name not in self.vehicle_classes:
                x1, y1, x2, y2, conf, class_id = detection
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                if 'stop_line' in class_name or 'line' in class_name:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 1)
                    cv2.putText(annotated_frame, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        # Draw stopline if locked
        if self.stopline_locked and self.stopline_y is not None:
            cv2.line(annotated_frame, (0, self.stopline_y), 
                    (frame.shape[1], self.stopline_y), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"STOP LINE (y={self.stopline_y})", (10, self.stopline_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw statistics overlay
        self._draw_stats(annotated_frame, traffic_light_state)
        
        return annotated_frame
    
    def _draw_stats(self, frame, traffic_light_state):
        avg_fps = np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
        
        # Create info panel
        info_bg = np.zeros((130, 260, 3), dtype=np.uint8)
        info_bg[:] = (40, 40, 40)
        
        cv2.putText(info_bg, f"Vehicles: {self.total_vehicles}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info_bg, f"Violations: {self.total_violations}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(info_bg, f"FPS: {avg_fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        status_text = "Unknown"
        light_color = (100, 100, 100)
        if traffic_light_state == TrafficLightState.RED:
            status_text = "RED"
            light_color = (0, 0, 255)
        elif traffic_light_state == TrafficLightState.YELLOW:
            status_text = "YELLOW"
            light_color = (0, 255, 255)
        elif traffic_light_state == TrafficLightState.GREEN:
            status_text = "GREEN"
            light_color = (0, 255, 0)
            
        cv2.putText(info_bg, f"Signal: {status_text}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)
        
        # Overlay info panel
        h, w, _ = info_bg.shape
        frame[10:10+h, 10:10+w] = cv2.addWeighted(
            frame[10:10+h, 10:10+w], 0.3, info_bg, 0.7, 0
        )

# ==================== MAIN EXECUTION ====================

def main():
    # Configuration
    # NOTE: Ensure these paths are correct relative to your script
    MODEL_PATH = "Detection Web/assets/best_yolo12s_seg.pt" 
    VIDEO_PATH = "Detection Web/assets/test_2.mp4"
    OUTPUT_PATH = "output_violation_detection.mp4"
    
    # Check paths
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at: {os.path.abspath(MODEL_PATH)}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found at: {os.path.abspath(VIDEO_PATH)}")
        return

    # Initialize detector
    detector = TrafficViolationDetector(MODEL_PATH)
    
    # Print model classes to debug mismatches
    print("Model Classes Loaded:", detector.model.names)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    # Try 'mp4v', fallback to 'avc1' or 'XVID' if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    print(f"🚀 Starting processing on {width}x{height} video...")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame = detector.process_frame(frame)
            
            # Write output
            out.write(annotated_frame)
            
            # Display (wrap in try-catch for headless environments)
            try:
                cv2.imshow('Traffic Violation Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                pass # Ignore display errors in headless mode
            
            if frame_count % 30 == 0:
                print(f"\rProcessed {frame_count} frames | Violations: {detector.total_violations}", end="")
    
    except KeyboardInterrupt:
        print("\nStopping...")
        
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processing complete!")
        print(f"   - Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()