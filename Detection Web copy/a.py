"""
Red Light Violation Detection System
=====================================

System for detecting red light violations at intersections.

Phases:
1. Stopline Calibration (0-5s): Fit straight line from mask points
2. Object Detection: Detect vehicles with proper NMS (no loss)
3. Violation Logic: Trigger warning when crossing on red

Author: Research Team
Version: 1.0
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Red Light Violation Detection"""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = str(BASE_DIR / 'assets/model/best_yolo12s_seg.pt')
    INPUT_VIDEO = str(BASE_DIR / 'assets/video/test_2.mp4')
    OUTPUT_VIDEO = str(BASE_DIR / 'output/redlight_violations.mp4')
    
    # Model parameters
    IMG_SIZE = 1280
    IOU_THRESHOLD = 0.45
    CONF_THRESHOLD = 0.50
    
    # Calibration
    CALIBRATION_DURATION = 5.0  # seconds
    MIN_STOPLINE_POINTS = 100
    
    # Class IDs
    STOPLINE_CLASS = [39]
    VEHICLE_CLASSES = [0, 6, 9, 21, 26]  # ambulance, car, fire_truck, motorcycle, police_car
    
    # Colors (BGR)
    COLOR_SAFE = (0, 255, 0)        # Green
    COLOR_VIOLATION = (0, 0, 255)   # Red
    COLOR_STOPLINE = (0, 165, 255)  # Orange
    
    # Display
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720


config = Config()


# ============================================================================
# PHASE 1: STOPLINE CALIBRATION
# ============================================================================

class StoplineCalibrator:
    """
    Phase 1: Stopline Calibration
    
    Process:
    1. Accumulate mask points for first 5 seconds
    2. Apply Linear Regression to fit best-fit straight line
    3. Freeze/Fix line coordinates for rest of video
    """
    
    def __init__(self):
        self.accumulated_points = []
        self.is_calibrated = False
        self.line_equation = None  # (m, b) for y = mx + b
        self.line_endpoints = None  # [(x1, y1), (x2, y2)]
        self.calibration_start_time = None
        
    def accumulate_mask_points(self, masks, boxes):
        """
        Accumulate all stopline mask points during calibration phase
        
        Args:
            masks: Segmentation masks from YOLO
            boxes: Bounding boxes from YOLO
        """
        if masks is None:
            return
        
        for i, cls_id in enumerate(boxes.cls):
            if int(cls_id) in config.STOPLINE_CLASS:
                mask_points = masks.xy[i]
                if len(mask_points) > 0:
                    # Add all points from this mask
                    self.accumulated_points.extend(mask_points.tolist())
    
    def calibrate_stopline(self, frame_width):
        """
        Apply Linear Regression to fit best-fit straight line
        
        Strategy:
        1. Get all accumulated points
        2. Extract bottom edge (lowest Y for each X)
        3. Fit straight line using least squares (polyfit)
        4. Calculate endpoints
        5. Freeze coordinates
        
        Returns:
            bool: True if calibration successful, False otherwise
        """
        if len(self.accumulated_points) < config.MIN_STOPLINE_POINTS:
            print(f"[CALIBRATION FAILED] Not enough points: "
                  f"{len(self.accumulated_points)} < {config.MIN_STOPLINE_POINTS}")
            return False
        
        print(f"[CALIBRATION] Processing {len(self.accumulated_points)} stopline points...")
        
        # Convert to numpy array
        points = np.array(self.accumulated_points, dtype=np.float32)
        
        # Get bottom edge: for each X coordinate, take the maximum Y (lowest point in image)
        points_int = points.astype(np.int32)
        unique_xs = np.unique(points_int[:, 0])
        
        bottom_edge_points = []
        for x in unique_xs:
            # Get all Y values for this X
            y_values = points_int[points_int[:, 0] == x, 1]
            # Take maximum Y (lowest point in image coordinates)
            max_y = np.max(y_values)
            bottom_edge_points.append([x, max_y])
        
        if len(bottom_edge_points) < 2:
            print("[CALIBRATION FAILED] Not enough bottom edge points")
            return False
        
        # Apply Linear Regression: fit y = mx + b
        bottom_points = np.array(bottom_edge_points, dtype=np.float32)
        xs = bottom_points[:, 0]
        ys = bottom_points[:, 1]
        
        # Polynomial fit degree 1 (straight line)
        coefficients = np.polyfit(xs, ys, 1)
        m, b = coefficients[0], coefficients[1]
        
        # Calculate endpoints
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min = int(m * x_min + b)
        y_max = int(m * x_max + b)
        
        # Freeze coordinates
        self.line_equation = (m, b)
        self.line_endpoints = [(x_min, y_min), (x_max, y_max)]
        self.is_calibrated = True
        
        print(f"[CALIBRATION SUCCESS]")
        print(f"  Equation: y = {m:.4f}x + {b:.2f}")
        print(f"  Endpoints: {self.line_endpoints[0]} -> {self.line_endpoints[1]}")
        
        return True
    
    def is_point_below_line(self, point):
        """
        Check if point is below (before) the stopline
        
        Args:
            point: (x, y) tuple
            
        Returns:
            bool: True if below line, False if above line
        """
        if not self.is_calibrated:
            return False
        
        x, y = point
        m, b = self.line_equation
        
        # Calculate Y of stopline at this X
        line_y = m * x + b
        
        # In image coordinates: larger Y = lower in image
        # If point Y > line Y: point is below (before) the line
        return y > line_y
    
    def has_crossed_line(self, point):
        """
        Check if point has crossed (is above/after) the stopline
        
        Args:
            point: (x, y) tuple
            
        Returns:
            bool: True if crossed line, False otherwise
        """
        return not self.is_point_below_line(point)
    
    def draw_stopline(self, frame):
        """
        Draw the fixed stopline on frame
        
        Args:
            frame: Image frame to draw on
        """
        if self.line_endpoints is not None:
            cv2.line(frame, 
                    self.line_endpoints[0], 
                    self.line_endpoints[1], 
                    config.COLOR_STOPLINE, 
                    thickness=4)


# ============================================================================
# PHASE 2: OBJECT DETECTION WITH NMS
# ============================================================================

class VehicleDetector:
    """
    Phase 2: Object Detection Logic
    
    Features:
    - Detect vehicles
    - Apply proper NMS (keep best box per object, no loss)
    - Track bottom-center point
    - Clean visualization (no trajectories)
    """
    
    @staticmethod
    def apply_nms_per_class(boxes, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression per class to avoid losing distinct objects
        
        Strategy:
        - Group boxes by class
        - Apply NMS within each class separately
        - This ensures we don't lose distinct objects of different classes
        
        Args:
            boxes: YOLO boxes object
            iou_threshold: IoU threshold for considering boxes as duplicates
            
        Returns:
            Filtered boxes with only best boxes per object
        """
        if boxes is None or len(boxes) == 0:
            return boxes
        
        # Extract data
        boxes_np = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        
        # Get unique classes
        unique_classes = np.unique(classes)
        
        # Keep track of indices to keep
        keep_indices = []
        
        # Apply NMS per class
        for cls in unique_classes:
            # Get boxes for this class
            cls_mask = classes == cls
            cls_boxes = boxes_np[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            if len(cls_boxes) == 0:
                continue
            
            # Apply NMS for this class
            nms_indices = cv2.dnn.NMSBoxes(
                bboxes=cls_boxes.tolist(),
                scores=cls_scores.tolist(),
                score_threshold=0.0,
                nms_threshold=iou_threshold
            )
            
            if len(nms_indices) > 0:
                nms_indices = nms_indices.flatten()
                # Map back to original indices
                keep_indices.extend(cls_indices[nms_indices].tolist())
        
        if len(keep_indices) == 0:
            return boxes
        
        # Sort to maintain order
        keep_indices = sorted(keep_indices)
        
        # Filter boxes
        boxes.data = boxes.data[keep_indices]
        
        return boxes
    
    @staticmethod
    def get_bottom_center(bbox):
        """
        Get bottom-center point of bounding box (tracking point)
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            (x, y) tuple of bottom-center point
        """
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        bottom_y = int(y2)
        return (center_x, bottom_y)


# ============================================================================
# PHASE 3: VIOLATION LOGIC
# ============================================================================

class RedLightViolationDetector:
    """
    Phase 3: Violation Logic
    
    Rule: IF (traffic_light_state == 'Red') AND (Vehicle crosses stopline)
          THEN trigger "Warning: Red Light Violation"
    """
    
    def __init__(self, stopline_calibrator):
        self.stopline = stopline_calibrator
        
        # Tracking state
        self.vehicle_last_position = defaultdict(lambda: None)  # "BELOW" or "ABOVE"
        self.vehicle_status = defaultdict(lambda: "Safe")
        
        # Statistics
        self.violation_count = 0
        
    def check_violation(self, track_id, bottom_center, traffic_light_state):
        """
        Check if vehicle violated red light
        
        Logic:
        1. Determine if vehicle is below or above stopline
        2. Detect crossing event (BELOW -> ABOVE)
        3. Check traffic light state at crossing moment
        4. Trigger violation if crossing on red
        
        Args:
            track_id: Vehicle tracking ID
            bottom_center: (x, y) bottom-center point of vehicle
            traffic_light_state: 'Red', 'Yellow', or 'Green'
            
        Returns:
            bool: True if violation detected, False otherwise
        """
        if not self.stopline.is_calibrated:
            return False
        
        # Check current position relative to stopline
        is_below = self.stopline.is_point_below_line(bottom_center)
        current_position = "BELOW" if is_below else "ABOVE"
        
        # Get last known position
        last_position = self.vehicle_last_position[track_id]
        
        # Detect crossing event: BELOW -> ABOVE
        if last_position == "BELOW" and current_position == "ABOVE":
            # Vehicle just crossed the stopline
            
            if traffic_light_state == 'Red':
                # VIOLATION: Crossed on red light
                self.vehicle_status[track_id] = "Violation"
                self.violation_count += 1
                
                print(f"[RED LIGHT VIOLATION] Vehicle ID {track_id} crossed stopline on RED light!")
                
                # Update position
                self.vehicle_last_position[track_id] = current_position
                return True
            
            else:
                # Crossed on green/yellow - safe
                self.vehicle_status[track_id] = "Safe"
        
        # Handle first detection (vehicle appears already above line)
        elif last_position is None and current_position == "ABOVE":
            # Vehicle first detected above line - assume safe
            self.vehicle_status[track_id] = "Safe"
        
        # Update position
        self.vehicle_last_position[track_id] = current_position
        
        return False
    
    def get_vehicle_status(self, track_id):
        """Get current status of vehicle"""
        return self.vehicle_status[track_id]
    
    def reset_statistics(self):
        """Reset violation counter"""
        self.violation_count = 0


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Handle all visualization tasks"""
    
    @staticmethod
    def draw_vehicle_box(frame, bbox, track_id, status, bottom_center):
        """
        Draw vehicle bounding box with status
        
        Args:
            frame: Image frame
            bbox: [x1, y1, x2, y2]
            track_id: Vehicle ID
            status: "Safe" or "Violation"
            bottom_center: (x, y) tracking point
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on status
        if status == "Violation":
            color = config.COLOR_VIOLATION
            label = f"ID:{track_id} - RED LIGHT VIOLATION"
        else:
            color = config.COLOR_SAFE
            label = f"ID:{track_id}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Draw bottom-center tracking point
        cv2.circle(frame, bottom_center, 5, color, -1)
        cv2.circle(frame, bottom_center, 7, (255, 255, 255), 2)
    
    @staticmethod
    def draw_info_panel(frame, fps, traffic_light_state, vehicle_count, violation_count, is_calibrated):
        """
        Draw information panel
        
        Args:
            frame: Image frame
            fps: Current FPS
            traffic_light_state: Current traffic light state
            vehicle_count: Number of vehicles detected
            violation_count: Number of violations detected
            is_calibrated: Whether stopline is calibrated
        """
        # Panel background
        cv2.rectangle(frame, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 180), (255, 255, 255), 2)
        
        # Traffic light color
        light_colors = {
            'Red': (0, 0, 255),
            'Yellow': (0, 255, 255),
            'Green': (0, 255, 0)
        }
        light_color = light_colors.get(traffic_light_state, (128, 128, 128))
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 40
        
        cv2.putText(frame, f"FPS: {int(fps)}", (25, y), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Traffic Light: {traffic_light_state}", (25, y + 35), font, 0.7, light_color, 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (25, y + 70), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Violations: {violation_count}", (25, y + 105), font, 0.7, (0, 0, 255), 2)
        
        # Calibration status
        if not is_calibrated:
            cv2.putText(frame, "CALIBRATING STOPLINE...", (25, frame.shape[0] - 20), 
                       font, 0.8, (0, 255, 255), 2)


# ============================================================================
# TRAFFIC LIGHT STATE (MOCK - REPLACE WITH ACTUAL DETECTION)
# ============================================================================

class TrafficLightStateMock:
    """
    Mock traffic light state provider
    In production, replace this with actual traffic light detection
    """
    
    def __init__(self):
        self.states = ['Green', 'Yellow', 'Red']
        self.current_index = 0
        self.frame_counter = 0
        self.frames_per_state = 90  # Change state every 90 frames (~3 seconds at 30fps)
        
    def get_state(self):
        """
        Get current traffic light state
        
        Returns:
            str: 'Red', 'Yellow', or 'Green'
        """
        self.frame_counter += 1
        
        if self.frame_counter >= self.frames_per_state:
            self.frame_counter = 0
            self.current_index = (self.current_index + 1) % len(self.states)
        
        return self.states[self.current_index]


# ============================================================================
# MAIN DETECTION SYSTEM
# ============================================================================

class RedLightViolationSystem:
    """
    Main Red Light Violation Detection System
    
    Integrates all three phases:
    1. Stopline Calibration
    2. Object Detection with NMS
    3. Violation Logic
    """
    
    def __init__(self, model_path, traffic_light_provider=None):
        """
        Initialize the system
        
        Args:
            model_path: Path to YOLO model
            traffic_light_provider: Object that provides get_state() method
                                   If None, uses mock provider
        """
        print("[INIT] Loading YOLO model...")
        self.model = YOLO(model_path)
        
        print("[INIT] Initializing components...")
        self.stopline_calibrator = StoplineCalibrator()
        self.vehicle_detector = VehicleDetector()
        self.violation_detector = RedLightViolationDetector(self.stopline_calibrator)
        self.visualizer = Visualizer()
        
        # Traffic light state provider
        self.traffic_light = traffic_light_provider or TrafficLightStateMock()
        
        print("[INIT] System ready!")
    
    def process_video(self, input_video, output_video):
        """
        Process video for red light violation detection
        
        Args:
            input_video: Path to input video
            output_video: Path to output video
        """
        # Open video
        cap = cv2.VideoCapture(input_video)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {input_video}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[VIDEO INFO]")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps_video}")
        print(f"  Total Frames: {total_frames}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps_video, (frame_width, frame_height))
        
        # Initialize
        frame_count = 0
        start_time = time.time()
        self.stopline_calibrator.calibration_start_time = start_time
        
        print(f"\n[PROCESSING] Starting video processing...")
        print(f"Phase 1: Calibrating stopline (0-{config.CALIBRATION_DURATION}s)...")
        print("Press 'q' to quit\n")
        
        # Processing loop
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # ================================================================
            # YOLO INFERENCE
            # ================================================================
            results = self.model.track(
                frame,
                persist=True,
                conf=config.CONF_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                imgsz=config.IMG_SIZE,
                verbose=False
            )
            result = results[0]
            
            # ================================================================
            # APPLY NMS PER CLASS (Avoid losing distinct objects)
            # ================================================================
            if result.boxes is not None and len(result.boxes) > 0:
                result.boxes = self.vehicle_detector.apply_nms_per_class(
                    result.boxes, 
                    iou_threshold=0.5
                )
            
            # ================================================================
            # PHASE 1: STOPLINE CALIBRATION (First 5 seconds)
            # ================================================================
            if elapsed_time <= config.CALIBRATION_DURATION:
                if not self.stopline_calibrator.is_calibrated:
                    self.stopline_calibrator.accumulate_mask_points(
                        result.masks, 
                        result.boxes
                    )
            
            # Calibrate after calibration period
            if elapsed_time > config.CALIBRATION_DURATION and not self.stopline_calibrator.is_calibrated:
                success = self.stopline_calibrator.calibrate_stopline(frame_width)
                if success:
                    print(f"\nPhase 2: Object detection and violation checking...")
            
            # ================================================================
            # DRAW STOPLINE
            # ================================================================
            self.stopline_calibrator.draw_stopline(frame)
            
            # ================================================================
            # PHASE 2 & 3: OBJECT DETECTION + VIOLATION CHECKING
            # ================================================================
            vehicle_count = 0
            
            if result.boxes is not None and result.boxes.id is not None:
                # Get traffic light state
                traffic_light_state = self.traffic_light.get_state()
                
                # Process each vehicle
                boxes_np = result.boxes.xyxy.cpu().numpy()
                ids_np = result.boxes.id.cpu().numpy()
                classes_np = result.boxes.cls.cpu().numpy()
                
                for bbox, track_id, cls_id in zip(boxes_np, ids_np, classes_np):
                    track_id = int(track_id)
                    cls_id = int(cls_id)
                    
                    # Filter only vehicles
                    if cls_id not in config.VEHICLE_CLASSES:
                        continue
                    
                    vehicle_count += 1
                    
                    # Get bottom-center tracking point
                    bottom_center = self.vehicle_detector.get_bottom_center(bbox)
                    
                    # Check for violation
                    self.violation_detector.check_violation(
                        track_id,
                        bottom_center,
                        traffic_light_state
                    )
                    
                    # Get vehicle status
                    status = self.violation_detector.get_vehicle_status(track_id)
                    
                    # Draw vehicle box (no trajectory)
                    self.visualizer.draw_vehicle_box(
                        frame,
                        bbox,
                        track_id,
                        status,
                        bottom_center
                    )
            else:
                traffic_light_state = self.traffic_light.get_state()
            
            # ================================================================
            # CALCULATE FPS
            # ================================================================
            frame_time = time.time() - current_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            
            # ================================================================
            # DRAW INFO PANEL
            # ================================================================
            self.visualizer.draw_info_panel(
                frame,
                fps,
                traffic_light_state,
                vehicle_count,
                self.violation_detector.violation_count,
                self.stopline_calibrator.is_calibrated
            )
            
            # ================================================================
            # DISPLAY AND WRITE
            # ================================================================
            display_frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            cv2.imshow("Red Light Violation Detection", display_frame)
            out.write(frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"[PROGRESS] Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"Violations: {self.violation_detector.violation_count} | "
                      f"FPS: {fps:.1f}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INTERRUPTED] Processing stopped by user")
                break
        
        # ================================================================
        # CLEANUP AND SUMMARY
        # ================================================================
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average FPS: {frame_count/total_time:.2f}")
        print(f"Red Light Violations Detected: {self.violation_detector.violation_count}")
        print(f"Output Video: {output_video}")
        print("="*70)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    print("="*70)
    print("RED LIGHT VIOLATION DETECTION SYSTEM")
    print("="*70)
    print()
    
    # Create output directory
    output_dir = Path(config.OUTPUT_VIDEO).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize system
    system = RedLightViolationSystem(
        model_path=config.MODEL_PATH,
        traffic_light_provider=None  # Uses mock provider
    )
    
    # Process video
    system.process_video(
        input_video=config.INPUT_VIDEO,
        output_video=config.OUTPUT_VIDEO
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()