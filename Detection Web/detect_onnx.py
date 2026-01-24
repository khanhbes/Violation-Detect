import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict, deque
import time
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
    TOUCHING_STOPLINE = "touching"
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
            self.bbox_history = deque(maxlen=10)

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
        
        for vehicle in self.tracked_vehicles.values():
            vehicle.frames_since_last_seen += 1
        
        if len(detections) == 0:
            self._remove_lost_tracks()
            return self.tracked_vehicles
        
        matched, unmatched_dets = self._match_detections(detections)
        
        for track_id, det_idx in matched:
            det = detections[det_idx]
            vehicle = self.tracked_vehicles[track_id]
            vehicle.bbox_history.append(det[:4])
            vehicle.frames_since_last_seen = 0
            vehicle.class_name = class_names[int(det[5])]
        
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det[4] > self.track_thresh:
                new_vehicle = Vehicle(
                    id=self.next_id,
                    class_name=class_names[int(det[5])]
                )
                new_vehicle.bbox_history.append(det[:4])
                self.tracked_vehicles[self.next_id] = new_vehicle
                self.next_id += 1
        
        self._remove_lost_tracks()
        return self.tracked_vehicles
    
    def _match_detections(self, detections):
        if len(self.tracked_vehicles) == 0:
            return [], list(range(len(detections)))
        
        track_ids = list(self.tracked_vehicles.keys())
        track_bboxes = np.array([
            list(self.tracked_vehicles[tid].bbox_history[-1]) 
            for tid in track_ids
        ])
        
        det_bboxes = np.array([det[:4] for det in detections])
        iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)
        
        matched = []
        unmatched_dets = list(range(len(detections)))
        
        for track_idx, track_id in enumerate(track_ids):
            if len(unmatched_dets) == 0:
                break
            
            ious = iou_matrix[track_idx, unmatched_dets]
            if len(ious) > 0 and ious.max() > self.match_thresh:
                best_det_idx = unmatched_dets[ious.argmax()]
                matched.append((track_id, best_det_idx))
                unmatched_dets.remove(best_det_idx)
        
        return matched, unmatched_dets
    
    def _compute_iou_matrix(self, bboxes1, bboxes2):
        ious = np.zeros((len(bboxes1), len(bboxes2)))
        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                ious[i, j] = self._compute_iou(bbox1, bbox2)
        return ious
    
    @staticmethod
    def _compute_iou(bbox1, bbox2):
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
        lost_ids = [
            tid for tid, vehicle in self.tracked_vehicles.items()
            if vehicle.frames_since_last_seen > self.track_buffer
        ]
        for tid in lost_ids:
            del self.tracked_vehicles[tid]

# ==================== ONNX MODEL WRAPPER ====================

class ONNXModel:
    """Wrapper for ONNX YOLOv12 model"""
    
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Class names from your dataset
        self.names = {
            0: 'ambulance', 1: 'arrow_left', 2: 'arrow_right', 3: 'arrow_straight',
            4: 'arrow_straight_and_left', 5: 'arrow_straight_and_right', 6: 'car',
            7: 'dashed_white_line', 8: 'dashed_yellow_line', 9: 'fire_truck',
            10: 'light_left_green', 11: 'light_left_red', 12: 'light_left_yellow',
            13: 'light_right_green', 14: 'light_straight_arrow_green',
            15: 'light_straight_arrow_red', 16: 'light_straight_arrow_yellow',
            17: 'light_straight_circle_green', 18: 'light_straight_circle_red',
            19: 'light_straight_circle_yellow', 20: 'median', 21: 'motorcycle',
            22: 'pedestrian_crossing', 23: 'person', 24: 'person_no_helmet',
            25: 'person_with_helmet', 26: 'police_car', 27: 'sidewalk',
            28: 'sign_no_car', 29: 'sign_no_entry', 30: 'sign_no_left_and_return',
            31: 'sign_no_left_turn', 32: 'sign_no_parking', 33: 'sign_no_return',
            34: 'sign_no_right_and_return', 35: 'sign_no_right_turn',
            36: 'sign_no_stopping', 37: 'solid_white_line', 38: 'solid_yellow_line',
            39: 'stop_line'
        }
        
    def preprocess(self, img):
        """Preprocess image for ONNX model"""
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, 0)  # Add batch dimension
        img = img.astype(np.float32) / 255.0  # Normalize
        return img
    
    def __call__(self, frame, verbose=False):
        """Run inference"""
        original_shape = frame.shape[:2]
        img = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: img})
        
        # Parse outputs (assuming YOLO format)
        return self.postprocess(outputs, original_shape)
    
    def postprocess(self, outputs, original_shape):
        """Convert ONNX output to YOLOResults-like format"""
        # This is a simplified version - adjust based on your actual ONNX output format
        class Result:
            def __init__(self, boxes, names):
                self.boxes = boxes
                self.names = names
        
        class Boxes:
            def __init__(self):
                self.xyxy = []
                self.conf = []
                self.cls = []
            
            def __iter__(self):
                for i in range(len(self.xyxy)):
                    yield type('Box', (), {
                        'xyxy': [self.xyxy[i]],
                        'conf': [self.conf[i]],
                        'cls': [self.cls[i]]
                    })()
        
        boxes = Boxes()
        
        # Parse ONNX output (format depends on export settings)
        # This is a placeholder - adjust based on your actual output
        if len(outputs) > 0:
            output = outputs[0][0]  # First output, first batch
            
            for detection in output:
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, cls = detection[:6]
                    if conf > 0.25:  # Confidence threshold
                        # Scale to original image
                        h, w = original_shape
                        x1 = int(x1 * w / 640)
                        y1 = int(y1 * h / 640)
                        x2 = int(x2 * w / 640)
                        y2 = int(y2 * h / 640)
                        
                        boxes.xyxy.append(np.array([x1, y1, x2, y2]))
                        boxes.conf.append(conf)
                        boxes.cls.append(int(cls))
        
        return [Result(boxes, self.names)]

# ==================== TRAFFIC VIOLATION DETECTOR ====================

class TrafficViolationDetector:
    """Main class for traffic violation detection"""
    
    def __init__(self, model_path: str, use_onnx=False):
        if use_onnx:
            self.model = ONNXModel(model_path)
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        
        self.tracker = ByteTracker(track_thresh=0.4, track_buffer=30, match_thresh=0.7)
        
        self.vehicle_classes = ['car', 'motorcycle', 'ambulance', 'fire_truck', 'police_car']
        self.traffic_light_classes = [
            'light_left_green', 'light_left_red', 'light_left_yellow',
            'light_right_green', 
            'light_straight_arrow_green', 'light_straight_arrow_red', 'light_straight_arrow_yellow',
            'light_straight_circle_green', 'light_straight_circle_red', 'light_straight_circle_yellow'
        ]
        
        self.stopline_y = None
        self.stopline_detection_start = None
        self.stopline_candidates = []
        self.stopline_locked = False
        self.stopline_detection_duration = 5.0
        
        self.total_vehicles = 0
        self.total_violations = 0
        self.violation_vehicles = set()
        
        self.fps_history = deque(maxlen=30)
        self.prev_time = time.time()
        
    def detect_stopline(self, results, frame_height):
        if self.stopline_locked:
            return
        
        if self.stopline_detection_start is None:
            self.stopline_detection_start = time.time()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if class_name == 'stop_line' and box.conf[0] > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0]
                    if hasattr(x1, 'cpu'):
                        x1, y1, x2, y2 = x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy()
                    stopline_y = int(y2)
                    self.stopline_candidates.append({
                        'y': stopline_y,
                        'conf': float(box.conf[0]),
                        'time': time.time()
                    })
        
        elapsed = time.time() - self.stopline_detection_start
        
        if elapsed >= self.stopline_detection_duration and len(self.stopline_candidates) > 0:
            self.stopline_candidates.sort(key=lambda x: (x['conf'], -x['time']), reverse=True)
            self.stopline_y = self.stopline_candidates[0]['y']
            self.stopline_locked = True
            print(f"✓ Stopline locked at y={self.stopline_y}")
        elif elapsed >= self.stopline_detection_duration:
            print("⚠ No stopline detected, continuing search...")
            self.stopline_detection_start = time.time()
    
    def get_traffic_light_state(self, results) -> TrafficLightState:
        red_lights = []
        yellow_lights = []
        green_lights = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                conf = float(box.conf[0])
                
                if class_name in self.traffic_light_classes and conf > 0.5:
                    if 'red' in class_name:
                        red_lights.append(conf)
                    elif 'yellow' in class_name:
                        yellow_lights.append(conf)
                    elif 'green' in class_name:
                        green_lights.append(conf)
        
        if red_lights:
            return TrafficLightState.RED
        elif yellow_lights:
            return TrafficLightState.YELLOW
        elif green_lights:
            return TrafficLightState.GREEN
        
        return TrafficLightState.UNKNOWN
    
    def check_vehicle_violation(self, vehicle: Vehicle, current_bbox, traffic_light_state: TrafficLightState):
        if self.stopline_y is None:
            return
        
        x1, y1, x2, y2 = current_bbox
        vehicle_bottom = y2
        vehicle_center_y = (y1 + y2) / 2
        
        threshold = 10
        
        if vehicle.state == VehicleState.BEFORE_STOPLINE:
            if vehicle_bottom < self.stopline_y - threshold:
                pass
            elif abs(vehicle_bottom - self.stopline_y) <= threshold:
                vehicle.state = VehicleState.TOUCHING_STOPLINE
        
        elif vehicle.state == VehicleState.TOUCHING_STOPLINE:
            if vehicle_center_y < self.stopline_y:
                vehicle.state = VehicleState.CROSSED_STOPLINE
                
                if traffic_light_state == TrafficLightState.RED:
                    vehicle.violation_type = "Violation"
                    if vehicle.id not in self.violation_vehicles:
                        self.violation_vehicles.add(vehicle.id)
                        self.total_violations += 1
                elif traffic_light_state == TrafficLightState.YELLOW:
                    vehicle.violation_type = "Warning"
                else:
                    vehicle.violation_type = None
    
    def get_bbox_color_and_label(self, vehicle: Vehicle, class_name: str) -> Tuple[Tuple[int, int, int], str]:
        if class_name in self.vehicle_classes:
            if vehicle.violation_type == "Violation":
                return (0, 0, 255), f"ID{vehicle.id} Violation"
            elif vehicle.violation_type == "Warning":
                return (0, 255, 255), f"ID{vehicle.id} Warning"
            else:
                return (0, 255, 0), f"ID{vehicle.id} {class_name}"
        else:
            return (255, 255, 255), class_name
    
    def process_frame(self, frame):
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
        self.fps_history.append(fps)
        self.prev_time = current_time
        
        results = self.model(frame, verbose=False)
        
        if not self.stopline_locked:
            self.detect_stopline(results, frame.shape[0])
        
        traffic_light_state = self.get_traffic_light_state(results)
        
        vehicle_detections = []
        all_detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                xyxy = box.xyxy[0]
                if hasattr(xyxy, 'cpu'):
                    xyxy = xyxy.cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                conf = float(box.conf[0])
                
                detection = [x1, y1, x2, y2, conf, class_id]
                all_detections.append((detection, class_name))
                
                if class_name in self.vehicle_classes and conf > 0.4:
                    vehicle_detections.append(detection)
        
        if vehicle_detections:
            vehicle_detections_array = np.array(vehicle_detections)
            tracked_vehicles = self.tracker.update(vehicle_detections_array, self.model.names)
        else:
            tracked_vehicles = self.tracker.tracked_vehicles
        
        self.total_vehicles = len(tracked_vehicles)
        
        annotated_frame = frame.copy()
        
        for vehicle_id, vehicle in tracked_vehicles.items():
            if len(vehicle.bbox_history) > 0:
                bbox = vehicle.bbox_history[-1]
                self.check_vehicle_violation(vehicle, bbox, traffic_light_state)
                
                color, label = self.get_bbox_color_and_label(vehicle, vehicle.class_name)
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        drawn_boxes = []
        for detection, class_name in all_detections:
            if class_name not in self.vehicle_classes:
                x1, y1, x2, y2, conf, class_id = detection
                
                current_box = [x1, y1, x2, y2]
                overlap = False
                for drawn_box in drawn_boxes:
                    if self._boxes_overlap(current_box, drawn_box):
                        overlap = True
                        break
                
                if not overlap and conf > 0.5:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    if class_name in self.traffic_light_classes:
                        color = (0, 255, 255) if 'red' in class_name else (0, 255, 0)
                        label = class_name
                    elif 'arrow' in class_name or 'line' in class_name or class_name == 'stop_line':
                        color = (255, 200, 0)
                        label = class_name
                    else:
                        color = (200, 200, 200)
                        label = f"C{class_id}"
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    drawn_boxes.append(current_box)
        
        if self.stopline_locked and self.stopline_y is not None:
            cv2.line(annotated_frame, (0, self.stopline_y), 
                    (frame.shape[1], self.stopline_y), (255, 0, 0), 3)
            cv2.putText(annotated_frame, "STOP LINE", (10, self.stopline_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        avg_fps = np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
        
        info_bg = np.zeros((120, 250, 3), dtype=np.uint8)
        info_bg[:] = (40, 40, 40)
        
        cv2.putText(info_bg, f"Vehicles: {self.total_vehicles}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info_bg, f"Violations: {self.total_violations}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(info_bg, f"FPS: {avg_fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        light_colors = {
            TrafficLightState.RED: (0, 0, 255),
            TrafficLightState.YELLOW: (0, 255, 255),
            TrafficLightState.GREEN: (0, 255, 0),
            TrafficLightState.UNKNOWN: (128, 128, 128)
        }
        light_color = light_colors[traffic_light_state]
        cv2.circle(info_bg, (220, 30), 15, light_color, -1)
        
        annotated_frame[10:130, 10:260] = cv2.addWeighted(
            annotated_frame[10:130, 10:260], 0.3, info_bg, 0.7, 0
        )
        
        return annotated_frame
    
    @staticmethod
    def _boxes_overlap(box1, box2, threshold=0.3):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        intersection = x_overlap * y_overlap
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = intersection / min(area1, area2) if min(area1, area2) > 0 else 0
        return iou > threshold

# ==================== MAIN EXECUTION ====================

def main():
    # Configuration
    MODEL_PATH = "Detection Web/assets/best_yolo12s_seg.onnx"  # Hoặc "best.onnx" nếu đã export
    VIDEO_PATH = "Detection Web/assets/test_2.mp4"
    OUTPUT_PATH = "output_violation_detection1.mp4"
    USE_ONNX = MODEL_PATH.endswith('.onnx')  # Auto-detect ONNX
    
    # Initialize detector
    detector = TrafficViolationDetector(MODEL_PATH, use_onnx=USE_ONNX)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    print("🚀 Starting traffic violation detection...")
    print("⏳ Detecting stopline in first 5 seconds...")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            annotated_frame = detector.process_frame(frame)
            out.write(annotated_frame)
            
            cv2.imshow('Traffic Violation Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames | "
                      f"Vehicles: {detector.total_vehicles} | "
                      f"Violations: {detector.total_violations}")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Final Statistics:")
        print(f"   - Total Frames: {frame_count}")
        print(f"   - Total Violations: {detector.total_violations}")
        print(f"   - Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
