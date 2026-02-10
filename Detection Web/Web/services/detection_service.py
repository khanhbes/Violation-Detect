"""
Unified Detection Service
==========================
Import và wrap per-frame logic từ functions/ folder.
Mỗi detector chạy YOLO riêng, giữ nguyên logic gốc.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import sys

# Add parent directories to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO
from config.config import config
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, save_violation_snapshot

# Import components from functions/
from functions.wrong_way_violation import PrecisionWrongWayDetector
from functions.wrong_lane_violation import WrongLaneDetector
from functions.redlight_violation import (
    StoplineCalibrator, LightMemory, TrackState as RedlightTrackState,
    detect_traffic_lights, check_violation as redlight_check_violation,
    get_bottom_center, get_signed_distance, dedup_boxes,
    draw_traffic_lights, line_to_segment
)
from functions.helmet_violation import (
    TrackState as HelmetTrackState,
    associate_rider_to_motorcycle,
    CLS_MOTORCYCLE, CLS_PERSON, CLS_PERSON_NO_HELMET, CLS_PERSON_WITH_HELMET,
    CONF_MOTORCYCLE, CONF_RIDER_NO_HELMET, CONF_RIDER_WITH_HELMET, CONF_RIDER_FALLBACK,
    C_GREEN, C_RED, C_ORANGE, C_HELMET, C_NO_HELMET, C_PERSON
)
from functions.sidewalk_violation import (
    process_mask_to_polygons_from_mask,
    check_violation_with_points,
    draw_forbidden_zones,
    CALIBRATION_DURATION, CONF_THRESHOLD_CALIBRATION, CONF_THRESHOLD_TRACKING
)

# =============================================================================
# PATH CONFIG (from config)
# =============================================================================

SNAPSHOT_DIR = config.SNAPSHOT_DIR
for vtype in ['helmet', 'redlight', 'sidewalk', 'wrong_way', 'wrong_lane']:
    (SNAPSHOT_DIR / vtype).mkdir(exist_ok=True)


# =============================================================================
# HELMET DETECTOR WRAPPER
# =============================================================================

class HelmetDetectorWrapper:
    """Wrap helmet detection logic from functions/helmet_violation.py"""
    
    def __init__(self):
        self.moto_states: Dict[int, HelmetTrackState] = defaultdict(HelmetTrackState)
        self.total_violations = 0
        self.total_safe = 0
        self.fps_smooth = 0.0
        self.prev_time = time.time()
    
    def reset(self):
        self.moto_states = defaultdict(HelmetTrackState)
        self.total_violations = 0
        self.total_safe = 0
    
    def process_frame(self, frame: np.ndarray, r0) -> Tuple[np.ndarray, List[dict]]:
        """Process frame with helmet detection logic"""
        h, w = frame.shape[:2]
        violations = []
        frame_vis = frame  # Draw on same frame
        
        motos = []
        riders = []
        
        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
            boxes_cls = r0.boxes.cls.cpu().numpy().astype(int)
            boxes_conf = r0.boxes.conf.cpu().numpy()
            boxes_id = None
            if hasattr(r0.boxes, "id") and r0.boxes.id is not None:
                boxes_id = r0.boxes.id.cpu().numpy().astype(int)
            
            for i in range(len(boxes_xyxy)):
                bbox = tuple(boxes_xyxy[i].tolist())
                cls_id = int(boxes_cls[i])
                conf = float(boxes_conf[i])
                
                if cls_id == CLS_MOTORCYCLE and conf >= CONF_MOTORCYCLE:
                    track_id = int(boxes_id[i]) if boxes_id is not None else i
                    motos.append((bbox, track_id, conf))
                elif cls_id == CLS_PERSON_NO_HELMET and conf >= CONF_RIDER_NO_HELMET:
                    riders.append((bbox, cls_id, conf))
                elif cls_id == CLS_PERSON_WITH_HELMET and conf >= CONF_RIDER_WITH_HELMET:
                    riders.append((bbox, cls_id, conf))
                elif cls_id == CLS_PERSON and conf >= CONF_RIDER_FALLBACK:
                    riders.append((bbox, cls_id, conf))
        
        # Draw riders
        for rb, rcls, rconf in riders:
            if rcls == CLS_PERSON_WITH_HELMET:
                draw_bbox_with_label(frame_vis, rb, f"Helmet {rconf:.2f}", C_HELMET)
            elif rcls == CLS_PERSON_NO_HELMET:
                draw_bbox_with_label(frame_vis, rb, f"NoHelmet {rconf:.2f}", C_NO_HELMET)
            else:
                draw_bbox_with_label(frame_vis, rb, f"Person {rconf:.2f}", C_PERSON)
        
        now = time.time()
        live_violations = 0
        live_safe = 0
        
        for moto_bbox, moto_id, _ in motos:
            state = self.moto_states[moto_id]
            best_rider = associate_rider_to_motorcycle(moto_bbox, riders, w, h)
            
            rider_cls = None
            if best_rider is not None:
                rb, rider_cls, rc = best_rider
                state.last_rider_bbox = rb
                state.last_rider_cls = rider_cls
                
                if rider_cls == CLS_PERSON_WITH_HELMET and not state.safe_latched:
                    state.safe_latched = True
                    self.total_safe += 1
                
                if rider_cls == CLS_PERSON_NO_HELMET and not state.safe_latched:
                    if (now - state.last_snapshot_time) >= 2.0:
                        save_violation_snapshot(frame, "no_helmet", moto_id, moto_bbox, vehicle_class="motorcycle")
                        state.last_snapshot_time = now
                        self.total_violations += 1
                        violations.append({
                            'type': 'helmet',
                            'id': moto_id,
                            'label': 'No Helmet'
                        })
            
            if state.safe_latched:
                live_safe += 1
                draw_bbox_with_label(frame_vis, moto_bbox, f"MC:{moto_id} Helmet", C_GREEN)
            else:
                if rider_cls == CLS_PERSON_NO_HELMET:
                    live_violations += 1
                    draw_bbox_with_label(frame_vis, moto_bbox, f"MC:{moto_id} NO HELMET", C_RED)
                else:
                    draw_bbox_with_label(frame_vis, moto_bbox, f"MC:{moto_id}", C_ORANGE)
        
        return frame_vis, violations
    
    def get_stats(self):
        return {
            'violations': self.total_violations,
            'safe': self.total_safe
        }


# =============================================================================
# SIDEWALK DETECTOR WRAPPER
# =============================================================================

class SidewalkDetectorWrapper:
    """Wrap sidewalk detection logic from functions/sidewalk_violation.py"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.accum_sidewalk = None
        self.accum_median = None
        self.calibration_frame_count = 0
        self.is_calibrated = False
        self.calibration_start_time = None
        self.forbidden_zones = []
        self.zone_labels = []
        self.vehicle_status = defaultdict(lambda: "Safe")
        self.violated_ids = set()
        self.violation_details = {}
        self.frame_count = 0
        self.sidewalk_class = config.SIDEWALK_CLASS[0] if config.SIDEWALK_CLASS else 27
        self.median_class = config.MEDIAN_CLASS[0] if config.MEDIAN_CLASS else 20
        
        # Inject into sidewalk_violation module globals
        import functions.sidewalk_violation as sw
        sw.is_calibrated = False
        sw.forbidden_zones = []
        sw.zone_labels = []
        sw.violated_ids = set()
        sw.vehicle_status = defaultdict(lambda: "Safe")
    
    def process_frame(self, frame: np.ndarray, model, r0=None, conf: float = 0.25) -> Tuple[np.ndarray, List[dict]]:
        """Process frame with sidewalk logic"""
        h, w = frame.shape[:2]
        violations = []
        frame_vis = frame
        self.frame_count += 1
        
        if self.accum_sidewalk is None:
            self.accum_sidewalk = np.zeros((h, w), dtype=np.float32)
            self.accum_median = np.zeros((h, w), dtype=np.float32)
            self.calibration_start_time = time.time()
        
        current_time = time.time()
        
        # Update module globals for check_violation_with_points
        import functions.sidewalk_violation as sw
        sw.is_calibrated = self.is_calibrated
        sw.forbidden_zones = self.forbidden_zones
        sw.zone_labels = self.zone_labels
        
        if not self.is_calibrated:
            elapsed = current_time - self.calibration_start_time
            
            # Use predict for calibration (need masks)
            results = model.predict(frame, imgsz=config.IMG_SIZE, conf=conf, verbose=False)
            result = results[0]
            
            if result.masks is not None:
                for i, cls_id in enumerate(result.boxes.cls):
                    cls_id = int(cls_id)
                    mask_xy = result.masks.xy[i]
                    if len(mask_xy) < 5:
                        continue
                    
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array(mask_xy, dtype=np.int32)
                    cv2.fillPoly(temp_mask, [pts], 255)
                    
                    if cls_id == self.sidewalk_class:
                        self.accum_sidewalk += temp_mask.astype(np.float32) / 255.0
                        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame_vis, contours, -1, (255, 0, 0), 2)
                    elif cls_id == self.median_class:
                        self.accum_median += temp_mask.astype(np.float32) / 255.0
                        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame_vis, contours, -1, (0, 255, 255), 2)
                
                if result.masks is not None and len(result.masks) > 0:
                    self.calibration_frame_count += 1
            
            if elapsed >= CALIBRATION_DURATION:
                # Lock zones
                if np.any(self.accum_sidewalk > 0):
                    threshold = 0.38 * self.calibration_frame_count
                    sidewalk_final = (self.accum_sidewalk >= threshold).astype(np.uint8) * 255
                    polys = process_mask_to_polygons_from_mask(sidewalk_final, "Sidewalk", min_area=10000)
                    for p in polys:
                        self.forbidden_zones.append(p)
                        self.zone_labels.append("SIDEWALK")
                
                if np.any(self.accum_median > 0):
                    threshold = 0.30 * self.calibration_frame_count
                    median_final = (self.accum_median >= threshold).astype(np.uint8) * 255
                    polys = process_mask_to_polygons_from_mask(median_final, "Median", min_area=8000)
                    for p in polys:
                        self.forbidden_zones.append(p)
                        self.zone_labels.append("MEDIAN")
                
                self.is_calibrated = True
                sw.is_calibrated = True
                sw.forbidden_zones = self.forbidden_zones
                sw.zone_labels = self.zone_labels
                print(f"✅ Sidewalk: Locked {len(self.forbidden_zones)} zones")
        
        else:
            # Detection phase - use tracking results
            if r0 is not None and r0.boxes is not None and r0.boxes.id is not None:
                boxes = r0.boxes.xyxy.cpu().numpy()
                track_ids = r0.boxes.id.cpu().numpy().astype(int)
                classes = r0.boxes.cls.cpu().numpy().astype(int)
                confs = r0.boxes.conf.cpu().numpy()
                
                for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                    if cls_id not in config.VEHICLE_CLASSES:
                        continue
                    
                    is_violation, zone_type = check_violation_with_points(box, threshold_points=2, offset_ratio=0.3)
                    
                    if is_violation:
                        box_color = config.COLOR_VIOLATION
                        label = "Violation"
                        
                        if track_id not in self.violated_ids:
                            self.violated_ids.add(track_id)
                            vclass = config.CLASS_NAMES.get(cls_id, "vehicle").lower()
                            save_violation_snapshot(frame, "sidewalk", track_id, box, label, vehicle_class=vclass)
                            violations.append({
                                'type': 'sidewalk',
                                'id': int(track_id),
                                'label': f'Sidewalk: {zone_type}'
                            })
                        
                        self.vehicle_status[track_id] = "Violation"
                    else:
                        box_color = config.COLOR_SAFE
                        vehicle_name = config.CLASS_NAMES.get(cls_id, "Vehicle")
                        label = f"{vehicle_name}:{track_id}"
                    
                    draw_bbox_with_label(frame_vis, box, label, box_color)
            
            # Draw zones
            for polygon, zlabel in zip(self.forbidden_zones, self.zone_labels):
                border_color = (255, 0, 0) if zlabel == "SIDEWALK" else (0, 255, 255)
                cv2.polylines(frame_vis, [polygon], True, border_color, 2)
        
        return frame_vis, violations
    
    def get_stats(self):
        return {
            'violations': len(self.violated_ids),
            'is_calibrated': self.is_calibrated,
            'zones': len(self.forbidden_zones)
        }


# =============================================================================
# REDLIGHT DETECTOR WRAPPER
# =============================================================================

class RedlightDetectorWrapper:
    """Wrap redlight detection logic from functions/redlight_violation.py"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.calibrator = StoplineCalibrator(duration=config.STOPLINE_CALIBRATION_DURATION)
        self.calibrator.start()
        self.light_memory = LightMemory()
        self.tracks: Dict[int, RedlightTrackState] = {}
        self.violations = 0
        self.warnings = 0
        self.frame_idx = 0
    
    def process_frame(self, frame: np.ndarray, r0) -> Tuple[np.ndarray, List[dict]]:
        """Process frame with redlight logic"""
        h, w = frame.shape[:2]
        violations = []
        self.frame_idx += 1
        current_time = time.time()
        
        frame_vis = frame
        
        # Parse detections
        if r0.boxes is None or len(r0.boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls_ids = np.zeros((0,), dtype=np.int32)
            confs = np.zeros((0,), dtype=np.float32)
            track_ids = np.zeros((0,), dtype=np.int32)
        else:
            boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_ids = r0.boxes.cls.cpu().numpy().astype(np.int32)
            confs = r0.boxes.conf.cpu().numpy().astype(np.float32)
            track_ids = r0.boxes.id.cpu().numpy().astype(np.int32) if r0.boxes.id is not None else np.array([-1] * len(cls_ids))
        
        # Traffic light state
        light_state = detect_traffic_lights(cls_ids)
        self.light_memory.update(light_state, current_time)
        light_state = self.light_memory.get(current_time)
        
        # Stopline mask
        stopline_mask = np.zeros((h, w), dtype=np.uint8)
        if r0.masks is not None and r0.masks.data is not None:
            masks_data = r0.masks.data.cpu().numpy()
            orig_cls = r0.boxes.cls.cpu().numpy().astype(np.int32) if r0.boxes is not None else np.array([])
            stopline_idxs = np.where(np.isin(orig_cls, config.STOPLINE_CLASS))[0]
            
            if len(stopline_idxs) > 0 and stopline_idxs.max() < len(masks_data):
                m = masks_data[stopline_idxs].max(axis=0)
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m.astype(np.float32), (w, h))
                stopline_mask = (m > 0.5).astype(np.uint8)
        
        # Draw traffic lights
        draw_traffic_lights(frame_vis, boxes, cls_ids, confs)
        
        # Calibration
        if not self.calibrator.is_calibrated():
            self.calibrator.add_mask_points(stopline_mask)
            self.calibrator.maybe_finish(h, w)
            
            if np.any(stopline_mask > 0):
                overlay = frame_vis.copy()
                overlay[stopline_mask > 0] = [0, 255, 0]
                frame_vis = cv2.addWeighted(overlay, 0.4, frame_vis, 0.6, 0)
                contours, _ = cv2.findContours(stopline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_vis, contours, -1, (0, 255, 0), 2)
        
        # Draw stopline
        if self.calibrator.is_calibrated():
            a, b, c = self.calibrator.line_abc
            if abs(b) > 1e-9:
                y1 = int(-(a * self.calibrator.min_x + c) / b)
                y2 = int(-(a * self.calibrator.max_x + c) / b)
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))
                cv2.line(frame_vis, (self.calibrator.min_x, y1), (self.calibrator.max_x, y2), config.COLOR_STOPLINE, 3)
        
        # Dedup
        boxes, cls_ids, confs, track_ids = dedup_boxes(boxes, cls_ids, confs, track_ids)
        
        # Process vehicles
        for xyxy, cls_id, conf, tid in zip(boxes, cls_ids, confs, track_ids):
            if cls_id not in config.VEHICLE_CLASSES:
                continue
            if conf < config.CONF_THRESHOLD_VEHICLE:
                continue
            if tid < 0:
                continue
            
            tid = int(tid)
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            px, py = get_bottom_center(xyxy)
            
            if tid not in self.tracks:
                self.tracks[tid] = RedlightTrackState()
            
            label = self.tracks[tid].label
            color = self.tracks[tid].color
            
            if self.calibrator.is_calibrated():
                a, b, c = self.calibrator.line_abc
                signed = get_signed_distance(px, py, a, b, c) * self.calibrator.sign_flip
                
                label, color = redlight_check_violation(
                    self.tracks[tid], signed, light_state, self.frame_idx, px, py
                )
                
                if label == "VIOLATION" and self.tracks[tid].last_event_frame == self.frame_idx:
                    self.violations += 1
                    vclass = config.CLASS_NAMES.get(cls_id, "vehicle").lower()
                    save_violation_snapshot(frame, "redlight", tid, (x1, y1, x2, y2), vehicle_class=vclass)
                    violations.append({
                        'type': 'redlight',
                        'id': tid,
                        'label': 'Red Light Violation'
                    })
                elif label == "WARNING" and self.tracks[tid].last_event_frame == self.frame_idx:
                    self.warnings += 1
            
            vehicle_name = config.CLASS_NAMES.get(cls_id, "Vehicle")
            if label == "VIOLATION":
                display_label = "Violation"
            elif label == "WARNING":
                display_label = "Warning"
            else:
                display_label = f"{vehicle_name}:{tid}"
            draw_bbox_with_label(frame_vis, (x1, y1, x2, y2), display_label, color)
            cv2.circle(frame_vis, (int(px), int(py)), 4, color, -1)
        
        return frame_vis, violations
    
    def get_stats(self):
        simple_light = self.light_memory.get(time.time()).get_simple_state()
        return {
            'violations': self.violations,
            'warnings': self.warnings,
            'light': simple_light,
            'calibrated': self.calibrator.is_calibrated()
        }


# =============================================================================
# WRONG WAY DETECTOR WRAPPER (already has class API!)
# =============================================================================

class WrongWayDetectorWrapper:
    """Thin wrapper around PrecisionWrongWayDetector"""
    
    def __init__(self):
        self.detector = None  # Lazy init
    
    def reset(self):
        self.detector = None
    
    def ensure_init(self):
        if self.detector is None:
            self.detector = PrecisionWrongWayDetector()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """process_frame already exists in PrecisionWrongWayDetector"""
        self.ensure_init()
        prev_violations = self.detector.total_violations
        frame_out = self.detector.process_frame(frame)
        
        violations = []
        if self.detector.total_violations > prev_violations:
            violations.append({
                'type': 'wrong_way',
                'id': -1,
                'label': 'Wrong Way'
            })
        
        return frame_out, violations
    
    def get_stats(self):
        if self.detector is None:
            return {'violations': 0, 'learning': True}
        return {
            'violations': self.detector.total_violations,
            'learning': self.detector.is_learning
        }


# =============================================================================
# UNIFIED DETECTOR
# =============================================================================

class UnifiedDetector:
    """
    Unified Detector - load model 1 lần, dispatch tới sub-detectors
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.MODEL_PATH
        print(f"🔄 Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("✅ Model loaded!")
        
        # Sub-detectors
        self.helmet = HelmetDetectorWrapper()
        self.sidewalk = SidewalkDetectorWrapper()
        self.redlight = RedlightDetectorWrapper()
        self.wrong_way = WrongWayDetectorWrapper()
        self.wrong_lane = WrongLaneDetector()
        
        # Tracking
        self.frame_idx = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.prev_time = time.time()
    
    def reset(self):
        """Reset all detectors"""
        self.helmet.reset()
        self.sidewalk.reset()
        self.redlight.reset()
        self.wrong_way.reset()
        self.wrong_lane.reset()
        self.frame_idx = 0
        self.start_time = time.time()
        # Reset tracker state 
        self.model = YOLO(self.model_path)
        print("🔄 All detectors reset")
    
    def process_frame(self, frame: np.ndarray, enabled: List[str], conf: float = 0.25, debug: bool = False) -> Tuple[np.ndarray, List[dict]]:
        """
        Process frame with enabled detectors.
        
        Args:
            frame: Input frame (BGR)
            enabled: List of enabled detectors ['helmet', 'sidewalk', 'redlight', 'wrong_way']
        
        Returns:
            (annotated_frame, list_of_violations)
        """
        self.frame_idx += 1
        all_violations = []
        frame_vis = frame.copy()
        
        # Wrong way runs its own model
        if 'wrong_way' in enabled:
            frame_vis, ww_viols = self.wrong_way.process_frame(frame_vis)
            all_violations.extend(ww_viols)
        
        # Other detectors share YOLO results
        needs_shared_model = bool(set(enabled) & {'helmet', 'sidewalk', 'redlight', 'wrong_lane'})
        
        r0 = None
        if needs_shared_model:
            results = self.model.track(
                frame,
                imgsz=config.IMG_SIZE,
                conf=conf,
                iou=config.IOU_THRESHOLD,
                persist=True,
                verbose=False,
                tracker=config.TRACKER
            )
            r0 = results[0]
        
        if 'helmet' in enabled and r0 is not None:
            frame_vis, h_viols = self.helmet.process_frame(frame_vis, r0)
            all_violations.extend(h_viols)
        
        if 'sidewalk' in enabled:
            frame_vis, sw_viols = self.sidewalk.process_frame(frame_vis, self.model, r0, conf=conf)
            all_violations.extend(sw_viols)
        
        if 'redlight' in enabled and r0 is not None:
            frame_vis, rl_viols = self.redlight.process_frame(frame_vis, r0)
            all_violations.extend(rl_viols)

        if 'wrong_lane' in enabled:
            frame_vis, wl_viols = self.wrong_lane.process_frame(
                frame_vis, r0=r0, model=self.model,
                conf_track=conf, conf_calib=min(conf, 0.25), debug=debug
            )
            all_violations.extend(wl_viols)
        
        # FPS calculation (no HUD overlay - stats sent via WebSocket)
        current_time = time.time()
        dt = max(1e-6, current_time - self.prev_time)
        self.fps = 0.85 * self.fps + 0.15 * (1.0 / dt) if self.fps > 0 else 1.0 / dt
        self.prev_time = current_time
        
        return frame_vis, all_violations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats"""
        elapsed = time.time() - self.start_time
        wrong_way_total = self.wrong_way.detector.total_violations if self.wrong_way.detector else 0
        wrong_lane_stats = self.wrong_lane.get_stats()
        violations = {
            'helmet': self.helmet.total_violations,
            'sidewalk': len(self.sidewalk.violated_ids),
            'redlight': self.redlight.violations,
            'wrong_way': wrong_way_total,
            'wrong_lane': wrong_lane_stats.get('total', 0)
        }
        return {
            'frame_idx': self.frame_idx,
            'fps': round(self.fps, 1),
            'elapsed': round(elapsed, 1),
            'helmet': self.helmet.get_stats(),
            'sidewalk': self.sidewalk.get_stats(),
            'redlight': self.redlight.get_stats(),
            'wrong_way': self.wrong_way.get_stats(),
            'wrong_lane': wrong_lane_stats,
            'violations': violations,
            'total': sum(violations.values())
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing UnifiedDetector import...")
    detector = UnifiedDetector()
    print("✅ UnifiedDetector created successfully!")
    print(f"Stats: {detector.get_stats()}")
