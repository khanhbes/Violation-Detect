"""
Lane Violation Detection System
Using YOLOv12-seg for traffic video processing
Author: Senior Computer Vision Engineer
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
from collections import defaultdict, deque
import math

# ============================================================================
# CONFIGURATION - CLASS MAPPING
# ============================================================================

# Lane Rules (Arrows)
ARROW_LEFT = 1
ARROW_RIGHT = 2
ARROW_STRAIGHT = 3
ARROW_STRAIGHT_AND_LEFT = 4
ARROW_STRAIGHT_AND_RIGHT = 5

# Lane Markings
DASHED_WHITE_LINE = 7
DASHED_YELLOW_LINE = 8
SOLID_WHITE_LINE = 37
SOLID_YELLOW_LINE = 38
STOP_LINE = 39

# Vehicles
CAR = 6
MOTORCYCLE = 21
AMBULANCE = 0
FIRE_TRUCK = 9
POLICE_CAR = 26

# Priority vehicles (exempt from violations)
PRIORITY_VEHICLES = {AMBULANCE, FIRE_TRUCK, POLICE_CAR}

# Color definitions for visualization
LINE_COLORS = {
    SOLID_WHITE_LINE: (0, 0, 255),      # Red
    SOLID_YELLOW_LINE: (0, 0, 255),     # Red
    DASHED_WHITE_LINE: (0, 255, 255),   # Yellow
    DASHED_YELLOW_LINE: (0, 255, 255),  # Yellow
    STOP_LINE: (255, 0, 0)              # Blue
}

ARROW_NAMES = {
    ARROW_LEFT: "MUST_LEFT",
    ARROW_RIGHT: "MUST_RIGHT",
    ARROW_STRAIGHT: "ONLY_STRAIGHT",
    ARROW_STRAIGHT_AND_LEFT: "STRAIGHT_OR_LEFT",
    ARROW_STRAIGHT_AND_RIGHT: "STRAIGHT_OR_RIGHT"
}

# ============================================================================
# GEOMETRIC PROCESSING FUNCTIONS
# ============================================================================

def cluster_line_points(points, eps=30, min_samples=50):
    """
    Cluster points into separate line groups using DBSCAN
    This separates multiple lines of the same class (e.g., 2 solid white lines)
    
    Args:
        points: List of (x, y) tuples
        eps: Maximum distance between points in same cluster
        min_samples: Minimum points to form a cluster
    
    Returns:
        List of point arrays, each representing one line
    """
    if len(points) < min_samples:
        return [np.array(points)] if len(points) > 10 else []
    
    from sklearn.cluster import DBSCAN
    
    points_array = np.array(points)
    
    # Use DBSCAN clustering based on x-coordinate primarily
    # Since lane lines are roughly vertical, cluster by x position
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
    
    labels = clustering.labels_
    unique_labels = set(labels)
    
    # Remove noise label (-1)
    unique_labels.discard(-1)
    
    clusters = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = points_array[mask]
        if len(cluster_points) >= 10:
            clusters.append(cluster_points)
    
    # Sort clusters by their mean x-position (left to right)
    clusters.sort(key=lambda c: np.mean(c[:, 0]))
    
    return clusters

def fit_line_ransac(points):
    """
    Fit a line through points using RANSAC algorithm
    Returns: slope (a), intercept (b) for y = ax + b, and y_min, y_max
    """
    if len(points) < 10:
        return None
    
    points = np.array(points)
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    
    try:
        ransac = RANSACRegressor(random_state=42, min_samples=2, 
                                 residual_threshold=5.0, max_trials=100)
        ransac.fit(x, y)
        
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        
        y_min = np.min(y)
        y_max = np.max(y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'y_min': y_min,
            'y_max': y_max
        }
    except:
        return None


def draw_fitted_line(frame, line_params, color, thickness=3):
    """
    Draw a fitted line segment on the frame
    """
    if line_params is None:
        return
    
    slope = line_params['slope']
    intercept = line_params['intercept']
    y_min = int(line_params['y_min'])
    y_max = int(line_params['y_max'])
    
    # Calculate x coordinates for y_min and y_max
    if abs(slope) < 0.001:  # Nearly horizontal line
        return
    
    x_min = int((y_min - intercept) / slope)
    x_max = int((y_max - intercept) / slope)
    
    # Clip to frame boundaries
    h, w = frame.shape[:2]
    x_min = max(0, min(w-1, x_min))
    x_max = max(0, min(w-1, x_max))
    y_min = max(0, min(h-1, y_min))
    y_max = max(0, min(h-1, y_max))
    
    cv2.line(frame, (x_min, y_min), (x_max, y_max), color, thickness)


def point_to_line_distance(point, line_params):
    """
    Calculate perpendicular distance from point to line
    """
    if line_params is None:
        return float('inf')
    
    x0, y0 = point
    a = line_params['slope']
    b = line_params['intercept']
    
    # Line equation: ax - y + b = 0
    distance = abs(a * x0 - y0 + b) / math.sqrt(a**2 + 1)
    return distance


def check_point_in_lane(point, left_line, right_line, tolerance=10):
    """
    Check if a point is within a lane defined by two lines
    Returns: True if point is between the two lines
    """
    if left_line is None or right_line is None:
        return False
    
    x, y = point
    
    # Calculate x positions on both lines at given y
    x_left = (y - left_line['intercept']) / left_line['slope'] if abs(left_line['slope']) > 0.001 else x
    x_right = (y - right_line['intercept']) / right_line['slope'] if abs(right_line['slope']) > 0.001 else x
    
    # Ensure left is actually left
    if x_left > x_right:
        x_left, x_right = x_right, x_left
    
    return (x_left - tolerance) <= x <= (x_right + tolerance)


def get_centroid(mask):
    """
    Calculate centroid of a binary mask
    """
    points = np.column_stack(np.where(mask > 0))
    if len(points) == 0:
        return None
    return (int(np.mean(points[:, 1])), int(np.mean(points[:, 0])))


def calculate_action_from_trajectory(trajectory, threshold_angle=15):
    """
    Determine vehicle action (LEFT, RIGHT, STRAIGHT) from trajectory
    trajectory: list of (x, y) points in chronological order
    """
    if len(trajectory) < 5:
        return "STRAIGHT"
    
    # Use first and last points to determine direction
    start_point = np.array(trajectory[0])
    end_point = np.array(trajectory[-1])
    
    direction_vector = end_point - start_point
    
    # Calculate angle relative to vertical (forward direction)
    angle = math.degrees(math.atan2(direction_vector[0], -direction_vector[1]))
    
    if angle < -threshold_angle:
        return "ACTION_LEFT"
    elif angle > threshold_angle:
        return "ACTION_RIGHT"
    else:
        return "ACTION_STRAIGHT"


def check_cross_stopline(prev_point, curr_point, stopline_params):
    """
    Check if vehicle crossed the stop line
    Returns: True if crossed from bottom to top
    """
    if stopline_params is None:
        return False
    
    # Calculate y position of stopline at vehicle x position
    x_curr = curr_point[0]
    y_stop = stopline_params['slope'] * x_curr + stopline_params['intercept']
    
    # Check if vehicle crossed from below (higher y) to above (lower y)
    prev_y = prev_point[1]
    curr_y = curr_point[1]
    
    # In image coordinates, y increases downward
    # So crossing from bottom to top means: prev_y > y_stop and curr_y <= y_stop
    if prev_y > y_stop and curr_y <= y_stop:
        return True
    
    return False


# ============================================================================
# MAIN SYSTEM CLASS
# ============================================================================

class LaneViolationDetector:
    def __init__(self, model_path, video_path, output_path=None):
        """
        Initialize the Lane Violation Detection System
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_path = output_path
        
        # Stage 1: Accumulation phase (0-5 seconds)
        self.initialization_time = 5  # seconds
        self.is_initialized = False
        self.frame_count = 0
        self.fps = 30  # Will be updated from video
        
        # Accumulated masks for line and arrow detection
        self.accumulated_line_points = defaultdict(list)
        self.accumulated_arrow_masks = defaultdict(list)
        
        # Stage 2: Static map (frozen after initialization)
        self.fitted_lines = {}  # {class_id: line_params}
        self.lane_rules = {}    # {lane_id: rule_class_id}
        self.lanes = []         # List of (left_line_id, right_line_id, rule)
        
        # Tracking data
        self.tracked_vehicles = {}  # {track_id: {'trajectory': deque, 'class': int}}
        self.violation_log = []
        
        # Current frame masks for visualization during initialization
        self.current_frame_masks = {}  # {class_id: mask_array}
        
        print("✓ Lane Violation Detector initialized")
    
    
    def accumulate_masks(self, results):
        """
        Stage 1: Accumulate mask points during initialization phase
        Also stores current frame masks for visualization
        """
        # Clear previous frame masks
        self.current_frame_masks = {}
        
        for result in results:
            if result.masks is None:
                continue
            
            boxes = result.boxes
            masks = result.masks
            
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                class_id = int(box.cls[0])
                mask_data = mask.data[0].cpu().numpy()
                
                # Resize mask to frame size
                mask_resized = cv2.resize(mask_data, 
                                         (result.orig_shape[1], result.orig_shape[0]))
                
                # Store current mask for visualization (lane lines and arrows)
                if class_id in [DASHED_WHITE_LINE, DASHED_YELLOW_LINE, 
                               SOLID_WHITE_LINE, SOLID_YELLOW_LINE, STOP_LINE,
                               ARROW_LEFT, ARROW_RIGHT, ARROW_STRAIGHT,
                               ARROW_STRAIGHT_AND_LEFT, ARROW_STRAIGHT_AND_RIGHT]:
                    if class_id not in self.current_frame_masks:
                        self.current_frame_masks[class_id] = mask_resized.copy()
                    else:
                        # Combine masks of same class
                        self.current_frame_masks[class_id] = np.maximum(
                            self.current_frame_masks[class_id], mask_resized)
                
                # Accumulate line points
                if class_id in [DASHED_WHITE_LINE, DASHED_YELLOW_LINE, 
                               SOLID_WHITE_LINE, SOLID_YELLOW_LINE, STOP_LINE]:
                    points = np.column_stack(np.where(mask_resized > 0.5))
                    if len(points) > 0:
                        # Store as (x, y) format
                        self.accumulated_line_points[class_id].extend(
                            [(p[1], p[0]) for p in points]
                        )
                
                # Accumulate arrow masks
                if class_id in [ARROW_LEFT, ARROW_RIGHT, ARROW_STRAIGHT,
                               ARROW_STRAIGHT_AND_LEFT, ARROW_STRAIGHT_AND_RIGHT]:
                    self.accumulated_arrow_masks[class_id].append(mask_resized)
    
    
    def freeze_static_map(self):
        """
        Stage 1 Complete: Process accumulated data and create static map
        Uses clustering to separate multiple lines of same class
        """
        print("\n" + "="*70)
        print("FREEZING STATIC MAP - Processing accumulated data...")
        print("="*70)
        
        # Step A: Fit lines from accumulated points using clustering
        # fitted_lines now stores a list of lines per class_id
        self.fitted_lines = {}  # Reset - will store {class_id: [line1, line2, ...]}
        
        line_names = {
            DASHED_WHITE_LINE: "Dashed White",
            DASHED_YELLOW_LINE: "Dashed Yellow",
            SOLID_WHITE_LINE: "Solid White",
            SOLID_YELLOW_LINE: "Solid Yellow",
            STOP_LINE: "Stop Line"
        }
        
        all_vertical_lines = []  # Store all individual lines for lane detection
        
        for class_id, points in self.accumulated_line_points.items():
            if len(points) < 50:
                continue
            
            line_name = line_names.get(class_id, f"Line {class_id}")
            
            # Use clustering to separate multiple lines of same class
            clusters = cluster_line_points(points, eps=40, min_samples=30)
            
            if not clusters:
                # Fallback: try fitting single line
                line_params = fit_line_ransac(points)
                if line_params is not None:
                    self.fitted_lines[class_id] = [line_params]
                    print(f"✓ Fitted {line_name}: 1 line, slope={line_params['slope']:.2f}")
                    if class_id != STOP_LINE:
                        all_vertical_lines.append((class_id, line_params))
            else:
                fitted_lines_list = []
                for i, cluster_points in enumerate(clusters):
                    line_params = fit_line_ransac(cluster_points.tolist())
                    if line_params is not None:
                        fitted_lines_list.append(line_params)
                        print(f"✓ Fitted {line_name} #{i+1}: slope={line_params['slope']:.2f}")
                        if class_id != STOP_LINE:
                            all_vertical_lines.append((class_id, line_params))
                
                if fitted_lines_list:
                    self.fitted_lines[class_id] = fitted_lines_list
        
        print(f"\n📊 Total vertical lines found: {len(all_vertical_lines)}")
        
        # Step B: Identify lanes and assign rules
        # Sort all vertical lines by their x-position at middle of frame
        if len(all_vertical_lines) >= 2:
            y_mid = 500  # Middle height reference
            
            def get_x_at_y(line_params, y):
                """Calculate x position at given y"""
                if abs(line_params['slope']) < 0.001:
                    return float('inf')
                return (y - line_params['intercept']) / line_params['slope']
            
            all_vertical_lines.sort(key=lambda x: get_x_at_y(x[1], y_mid))
            
            # Create lanes between consecutive lines
            for i in range(len(all_vertical_lines) - 1):
                left_line_id, left_line = all_vertical_lines[i]
                right_line_id, right_line = all_vertical_lines[i+1]
                
                # Check if lines are close enough to form a lane (not too wide)
                left_x = get_x_at_y(left_line, y_mid)
                right_x = get_x_at_y(right_line, y_mid)
                lane_width = right_x - left_x
                
                if lane_width < 50 or lane_width > 400:  # Skip if too narrow or too wide
                    continue
                
                # Find arrow rule for this lane
                lane_rule = self.find_lane_rule(left_line, right_line)
                
                self.lanes.append({
                    'left_line_id': left_line_id,
                    'right_line_id': right_line_id,
                    'left_line': left_line,
                    'right_line': right_line,
                    'rule': lane_rule,
                    'lane_id': len(self.lanes)
                })
                
                rule_name = ARROW_NAMES.get(lane_rule, 'No Rule') if lane_rule else 'No Rule'
                print(f"✓ Lane {len(self.lanes)}: width={lane_width:.0f}px, Rule = {rule_name}")
        
        self.is_initialized = True
        print("="*70)
        print(f"✓ STATIC MAP FROZEN - {len(self.lanes)} lanes detected")
        print("✓ Violation detection now ACTIVE")
        print("="*70 + "\n")
    
    
    def find_lane_rule(self, left_line, right_line):
        """
        Find which arrow rule applies to a lane by checking arrow centroids
        """
        for arrow_class, masks in self.accumulated_arrow_masks.items():
            if len(masks) == 0:
                continue
            
            # Combine all masks for this arrow type
            combined_mask = np.sum(masks, axis=0)
            combined_mask = (combined_mask > 0).astype(np.uint8)
            
            # Get centroid
            centroid = get_centroid(combined_mask)
            if centroid is None:
                continue
            
            # Check if centroid is in this lane
            if check_point_in_lane(centroid, left_line, right_line, tolerance=50):
                return arrow_class
        
        return None
    
    
    def detect_violations(self, frame, results):
        """
        Stage 2: Real-time violation detection
        """
        violations = []
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return violations
        
        boxes = results[0].boxes
        
        for box in boxes:
            class_id = int(box.cls[0])
            
            # Only process vehicles
            if class_id not in [CAR, MOTORCYCLE, AMBULANCE, FIRE_TRUCK, POLICE_CAR]:
                continue
            
            # Get tracking ID
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is None:
                continue
            
            # Calculate bottom-center point (wheel position)
            xyxy = box.xyxy[0].cpu().numpy()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_bottom = xyxy[3]
            bottom_center = (int(x_center), int(y_bottom))
            
            # Initialize or update trajectory
            if track_id not in self.tracked_vehicles:
                self.tracked_vehicles[track_id] = {
                    'trajectory': deque(maxlen=30),
                    'class': class_id,
                    'prev_point': None,
                    'crossed_stopline': False,
                    'lane_at_cross': None
                }
            
            vehicle_data = self.tracked_vehicles[track_id]
            vehicle_data['trajectory'].append(bottom_center)
            
            # Check if crossing stop line
            if STOP_LINE in self.fitted_lines and len(self.fitted_lines[STOP_LINE]) > 0:
                stopline = self.fitted_lines[STOP_LINE][0]  # Use first stopline
                
                if vehicle_data['prev_point'] is not None:
                    if check_cross_stopline(vehicle_data['prev_point'], 
                                          bottom_center, stopline):
                        
                        if not vehicle_data['crossed_stopline']:
                            vehicle_data['crossed_stopline'] = True
                            
                            # Determine which lane vehicle is in
                            current_lane = None
                            for lane in self.lanes:
                                if check_point_in_lane(bottom_center, 
                                                      lane['left_line'], 
                                                      lane['right_line']):
                                    current_lane = lane
                                    break
                            
                            vehicle_data['lane_at_cross'] = current_lane
                            
                            # Analyze action after sufficient trajectory
                            if len(vehicle_data['trajectory']) >= 10:
                                action = calculate_action_from_trajectory(
                                    list(vehicle_data['trajectory'])
                                )
                                
                                # Check for violation
                                if current_lane and class_id not in PRIORITY_VEHICLES:
                                    is_violation = self.check_lane_violation(
                                        action, current_lane['rule']
                                    )
                                    
                                    if is_violation:
                                        violations.append({
                                            'track_id': track_id,
                                            'bbox': xyxy,
                                            'action': action,
                                            'required': ARROW_NAMES.get(
                                                current_lane['rule'], 'Unknown'
                                            ),
                                            'lane_id': current_lane['lane_id']
                                        })
                                        
                                        self.violation_log.append({
                                            'frame': self.frame_count,
                                            'track_id': track_id,
                                            'action': action,
                                            'lane_rule': current_lane['rule']
                                        })
                                        
                                        print(f"⚠️  VIOLATION DETECTED: Vehicle {track_id} "
                                              f"- Action: {action}, Required: "
                                              f"{ARROW_NAMES.get(current_lane['rule'], 'Unknown')}")
            
            vehicle_data['prev_point'] = bottom_center
        
        return violations
    
    
    def check_lane_violation(self, action, lane_rule):
        """
        Check if vehicle action violates lane rule
        """
        if lane_rule == ARROW_LEFT:
            return action != "ACTION_LEFT"
        elif lane_rule == ARROW_RIGHT:
            return action != "ACTION_RIGHT"
        elif lane_rule == ARROW_STRAIGHT:
            return action != "ACTION_STRAIGHT"
        elif lane_rule == ARROW_STRAIGHT_AND_LEFT:
            return action == "ACTION_RIGHT"
        elif lane_rule == ARROW_STRAIGHT_AND_RIGHT:
            return action == "ACTION_LEFT"
        
        return False
    
    
    def visualize_frame(self, frame, results, violations):
        """
        Draw visualization on frame
        """
        vis_frame = frame.copy()
        
        # During initialization: Draw mask overlays for lane lines and arrows
        if not self.is_initialized:
            # Define colors for each class (BGR format)
            mask_colors = {
                SOLID_WHITE_LINE: (0, 0, 255),      # Red
                SOLID_YELLOW_LINE: (0, 0, 255),    # Red
                DASHED_WHITE_LINE: (0, 255, 255),  # Yellow
                DASHED_YELLOW_LINE: (0, 255, 255), # Yellow
                STOP_LINE: (255, 0, 0),            # Blue
                ARROW_LEFT: (0, 255, 0),           # Green
                ARROW_RIGHT: (0, 255, 0),          # Green
                ARROW_STRAIGHT: (0, 255, 0),       # Green
                ARROW_STRAIGHT_AND_LEFT: (0, 255, 0),
                ARROW_STRAIGHT_AND_RIGHT: (0, 255, 0)
            }
            
            # Create overlay for masks
            overlay = vis_frame.copy()
            
            for class_id, mask in self.current_frame_masks.items():
                if class_id in mask_colors:
                    color = mask_colors[class_id]
                    # Create colored mask
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, 
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, color, -1)
                    
                    # Draw contour outline
                    cv2.drawContours(vis_frame, contours, -1, color, 2)
            
            # Blend overlay with original frame
            alpha = 0.3  # Transparency
            vis_frame = cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0)
            
            # Show calibration progress bar
            time_elapsed = self.frame_count / self.fps
            progress = min(time_elapsed / self.initialization_time, 1.0)
            bar_width = int(400 * progress)
            cv2.rectangle(vis_frame, (10, 120), (410, 140), (100, 100, 100), -1)
            cv2.rectangle(vis_frame, (10, 120), (10 + bar_width, 140), (0, 255, 0), -1)
            cv2.putText(vis_frame, f"Calibrating: {progress*100:.0f}%", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.is_initialized:
            # Draw fitted lines (each class may have multiple lines)
            for class_id, lines_list in self.fitted_lines.items():
                color = LINE_COLORS.get(class_id, (255, 255, 255))
                thickness = 4 if class_id == STOP_LINE else 3
                # lines_list is now a list of line_params
                for line_params in lines_list:
                    draw_fitted_line(vis_frame, line_params, color, thickness)
            
            # Draw lane rules
            for lane in self.lanes:
                # Calculate middle position of lane for text
                y_mid = int((lane['left_line']['y_min'] + 
                            lane['left_line']['y_max']) / 2)
                x_left = int((y_mid - lane['left_line']['intercept']) / 
                            lane['left_line']['slope'])
                x_right = int((y_mid - lane['right_line']['intercept']) / 
                             lane['right_line']['slope'])
                x_mid = (x_left + x_right) // 2
                
                rule_text = ARROW_NAMES.get(lane['rule'], "Unknown")
                cv2.putText(vis_frame, rule_text, (x_mid - 80, y_mid),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw violations
            for violation in violations:
                bbox = violation['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Red box for violating vehicle
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # "WRONG LANE" text
                cv2.putText(vis_frame, "WRONG LANE", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Add status text
        status = "MONITORING" if self.is_initialized else "INITIALIZING"
        color = (0, 255, 0) if self.is_initialized else (0, 165, 255)
        cv2.putText(vis_frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        time_elapsed = self.frame_count / self.fps
        cv2.putText(vis_frame, f"Time: {time_elapsed:.1f}s", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.is_initialized:
            cv2.putText(vis_frame, f"Violations: {len(self.violation_log)}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return vis_frame
    
    
    def process_video(self):
        """
        Main processing loop
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open video file")
            return
        
        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video Info: {width}x{height} @ {self.fps} FPS")
        print(f"📊 Total Frames: {total_frames}")
        print(f"⏱️  Duration: {total_frames/self.fps:.1f} seconds\n")
        
        # Setup video writer if output path specified
        writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, 
                                    (width, height))
        
        initialization_frames = self.initialization_time * self.fps
        
        print("🚀 Starting processing...\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Run YOLO detection with tracking
                results = self.model.track(frame, persist=True, verbose=False,
                                          imgsz=640, conf=0.3)
                
                # Stage 1: Accumulation phase
                if not self.is_initialized:
                    self.accumulate_masks(results)
                    
                    # Check if initialization period is complete
                    if self.frame_count >= initialization_frames:
                        self.freeze_static_map()
                
                # Stage 2: Violation detection
                violations = []
                if self.is_initialized:
                    violations = self.detect_violations(frame, results)
                
                # Visualization
                vis_frame = self.visualize_frame(frame, results, violations)
                
                # Display
                cv2.imshow('Lane Violation Detection', vis_frame)
                
                # Write to output
                if writer:
                    writer.write(vis_frame)
                
                # Progress indicator
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Frame {self.frame_count}/{total_frames}")
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Processing interrupted by user")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print summary
            print("\n" + "="*70)
            print("PROCESSING COMPLETE")
            print("="*70)
            print(f"📊 Total Frames Processed: {self.frame_count}")
            print(f"⚠️  Total Violations Detected: {len(self.violation_log)}")
            if self.output_path:
                print(f"💾 Output saved to: {self.output_path}")
            print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    from config.config import Config
    
    # Configuration from Config
    MODEL_PATH = Config.MODEL_PATH
    VIDEO_PATH = Config.DEFAULT_VIDEO
    OUTPUT_PATH = Config.OUTPUT_VIDEO
    
    print("\n" + "="*70)
    print("LANE VIOLATION DETECTION SYSTEM")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = LaneViolationDetector(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH
    )
    
    # Process video
    detector.process_video()


if __name__ == "__main__":
    main()