"""
Traffic Sign Violation Detection System
========================================
Phát hiện vi phạm biển báo giao thông sử dụng YOLO Segmentation + Object Tracking

Features:
1. Calibration biển báo (N frame đầu) - xác định biển báo cố định
2. Xác định vùng hiệu lực (enforcement zone) từ vị trí biển báo (scale theo frame)
3. Phát hiện hướng di chuyển xe và so sánh với luật biển báo
4. Snapshot vi phạm tự động

Architecture: Đồng bộ interface với PrecisionWrongWayDetector
- Không load model trong constructor
- process_frame(frame, r0=..., model=..., conf=..., debug=...)
- reset(), get_stats()

Uses shared Config class from config/config.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

# Add parent directory to path so we can import config module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared config and draw utilities
from config.config import config
from utils.draw_utils import draw_bbox_with_label, draw_info_hud, draw_calibration_hud, save_violation_snapshot


# =============================================================================
# VEHICLE TRACKER
# =============================================================================

@dataclass
class VehicleTracker:
    """Theo dõi xe qua các frame, tích lũy quỹ đạo và trạng thái vi phạm."""
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

    def get_trajectory(self) -> Optional[np.ndarray]:
        if len(self.positions) < 2:
            return None
        return np.array(list(self.positions))

    def get_direction(self) -> Optional[Tuple[float, float]]:
        """Tính hướng di chuyển tổng thể (normalized vector)."""
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
        """
        Cập nhật trạng thái vi phạm.
        Soft reset: giảm dần thay vì reset về 0 ngay — tránh mất đếm khi YOLO jitter.
        """
        if is_violating:
            self.violation_frames += 1
            self.is_violating = True
            if violation_type:
                self.violation_type = violation_type
        else:
            # Soft reset: decrement nhanh nhưng không mất hết progress
            self.violation_frames = max(0, self.violation_frames - 2)
            if self.violation_frames == 0:
                self.is_violating = False


# =============================================================================
# TRAFFIC SIGN DETECTOR
# =============================================================================

class TrafficSignDetector:
    """
    Phân tích biển báo cố định qua calibration phase, tạo enforcement zones,
    và kiểm tra vi phạm dựa trên hướng di chuyển xe.
    """
    def __init__(self):
        self.detected_signs: Dict[int, List[Tuple]] = defaultdict(list)
        self.active_signs: List[Dict] = []
        self.is_calibrating = True
        self.calibration_count = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.sign_class_map: Dict[int, str] = dict(config.SIGN_CLASSES)

    def update_sign_classes_from_model(self, names) -> None:
        """
        Đồng bộ sign class từ model.names để tránh lệch mapping với config.
        Chỉ lấy các biển mà detector hiện có rule xử lý.
        """
        if names is None:
            return

        if isinstance(names, list):
            names_dict = {i: n for i, n in enumerate(names)}
        elif isinstance(names, dict):
            names_dict = names
        else:
            return

        # Map label model -> sign_type nội bộ
        name_to_sign_type = {
            "no_left_turn": "no_left_turn",
            "no_right_turn": "no_right_turn",
            "no_straight": "no_straight",
            "turn_left_only": "turn_left_only",
            "turn_right_only": "turn_right_only",
            "straight_only": "straight_only",
            "straight_and_left_turn_only": "straight_and_left_turn_only",
            "straight_and_right_turn_only": "straight_and_right_turn_only",
            # Tên class trong model hiện tại
            "sign_no_left_turn": "no_left_turn",
            "sign_no_right_turn": "no_right_turn",
            "sign_no_left_and_return": "no_left_turn",
            "sign_no_right_and_return": "no_right_turn",
        }

        mapped: Dict[int, str] = {}
        for cls_id, cls_name in names_dict.items():
            key = str(cls_name).strip().lower()
            sign_type = name_to_sign_type.get(key)
            if sign_type is not None:
                mapped[int(cls_id)] = sign_type

        if mapped:
            self.sign_class_map = mapped

    def update_calibration(self, detections: List[Tuple], frame_shape: Tuple[int, int]):
        """Thu thập biển báo trong calibration phase."""
        if not self.is_calibrating:
            return

        self.frame_height, self.frame_width = frame_shape

        for class_id, confidence, bbox in detections:
            if class_id in self.sign_class_map:
                if confidence > config.SIGN_CONFIDENCE:
                    if not self._is_valid_sign_bbox(bbox, frame_shape):
                        continue
                    self.detected_signs[class_id].append((bbox, confidence))

        self.calibration_count += 1

        if self.calibration_count >= config.SIGN_CALIBRATION_FRAMES:
            self._finalize_calibration()

    def _finalize_calibration(self):
        print("\n" + "=" * 60)
        print("📊 CALIBRATION COMPLETE - Traffic Sign Analysis")
        print("=" * 60)

        for sign_class, detections in self.detected_signs.items():
            clusters = self._cluster_sign_detections(detections)
            for cluster in clusters:
                if len(cluster) < config.SIGN_MIN_SAMPLES:
                    continue

                bboxes = [bbox for bbox, conf in cluster]
                avg_bbox = np.mean(bboxes, axis=0)

                sign_info = {
                    'class_id': sign_class,
                    'sign_type': self.sign_class_map[sign_class],
                    'bbox': tuple(avg_bbox),
                    'detection_count': len(cluster),
                    'enforcement_zone': self._create_enforcement_zone(avg_bbox)
                }

                self.active_signs.append(sign_info)
                print(f"✓ Detected: {sign_info['sign_type']} ({len(cluster)} samples)")

        print(f"\n📊 Total active signs: {len(self.active_signs)}")
        print("=" * 60 + "\n")

        self.is_calibrating = False

    def _create_enforcement_zone(self, sign_bbox: np.ndarray) -> Tuple:
        """
        Tạo vùng hiệu lực từ vị trí biển báo, scale theo frame size.
        Zone rộng sang hai bên và xuống dưới nhiều hơn lên trên (vì xe ở dưới biển).
        """
        x1, y1, x2, y2 = sign_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        zone_half_w = self.frame_width * config.SIGN_ENFORCEMENT_ZONE_X
        zone_up = self.frame_height * config.SIGN_ENFORCEMENT_ZONE_UP
        zone_down = self.frame_height * config.SIGN_ENFORCEMENT_ZONE_DOWN

        return (
            max(0, cx - zone_half_w),
            max(0, cy - zone_up),
            min(self.frame_width, cx + zone_half_w),
            min(self.frame_height, cy + zone_down)
        )

    def _is_valid_sign_bbox(self, bbox: Tuple, frame_shape: Tuple[int, int]) -> bool:
        """Lọc bbox biển báo bất thường theo hình học để giảm false positive."""
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        fh, fw = frame_shape
        area_ratio = (bw * bh) / max(1.0, float(fw * fh))
        if area_ratio > config.SIGN_MAX_AREA_RATIO:
            return False

        aspect = max(bw / bh, bh / bw)
        if aspect > config.SIGN_MAX_ASPECT_RATIO:
            return False
        return True

    def _cluster_sign_detections(self, detections: List[Tuple]) -> List[List[Tuple]]:
        """
        Tách các biển cùng class theo cụm vị trí để tránh gộp trung bình sai.
        Mỗi detection: (bbox, confidence).
        """
        clusters: List[List[Tuple]] = []
        centers: List[np.ndarray] = []

        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

            if not centers:
                clusters.append([(bbox, conf)])
                centers.append(center)
                continue

            distances = [np.linalg.norm(center - c) for c in centers]
            nearest_idx = int(np.argmin(distances))
            if distances[nearest_idx] <= config.SIGN_CLUSTER_DISTANCE:
                clusters[nearest_idx].append((bbox, conf))
                # Cập nhật center của cluster
                cluster_centers = []
                for cbbox, _ in clusters[nearest_idx]:
                    ccx = (cbbox[0] + cbbox[2]) / 2
                    ccy = (cbbox[1] + cbbox[3]) / 2
                    cluster_centers.append((ccx, ccy))
                centers[nearest_idx] = np.mean(np.array(cluster_centers), axis=0)
            else:
                clusters.append([(bbox, conf)])
                centers.append(center)

        return clusters

    def check_violation(self, tracker: VehicleTracker) -> Optional[str]:
        """Kiểm tra xe có vi phạm biển báo nào không."""
        if self.is_calibrating:
            return None

        movement = self._classify_movement(tracker)
        if movement is None:
            return None

        current_pos = tracker.positions[-1]

        for sign in self.active_signs:
            if not self._in_zone(current_pos, sign['enforcement_zone']):
                continue
            if self._check_direction_violation(movement, sign['sign_type']):
                return sign['sign_type']
        return None

    def _in_zone(self, pos: Tuple[float, float], zone: Tuple) -> bool:
        x, y = pos
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2

    def _classify_movement(self, tracker: VehicleTracker) -> Optional[str]:
        """
        Phân loại hướng đi từ quỹ đạo gần nhất.
        Trả về: "left" | "right" | "straight" | None
        """
        positions = list(tracker.positions)
        if len(positions) < 10:
            return None

        window = positions[-15:] if len(positions) >= 15 else positions
        start = np.array(window[0], dtype=np.float32)
        end = np.array(window[-1], dtype=np.float32)
        delta = end - start
        dx, dy = float(delta[0]), float(delta[1])

        travel = float(np.linalg.norm(delta))
        if travel < 10.0:
            return None

        # Nếu di chuyển gần như ngang → rõ ràng rẽ
        if abs(dy) < 5.0:
            return "left" if dx < 0 else "right"

        ratio = dx / (abs(dy) + 1e-6)
        if ratio <= -config.SIGN_STRAIGHT_RATIO:
            return "left"
        if ratio >= config.SIGN_STRAIGHT_RATIO:
            return "right"
        return "straight"

    def _check_direction_violation(self, movement: str, sign_type: str) -> bool:
        """Kiểm tra movement có vi phạm sign_type không."""
        rules = {
            "no_left_turn":                 lambda m: m == "left",
            "no_right_turn":                lambda m: m == "right",
            "no_straight":                  lambda m: m == "straight",
            "turn_left_only":               lambda m: m != "left",
            "turn_right_only":              lambda m: m != "right",
            "straight_only":                lambda m: m != "straight",
            "straight_and_left_turn_only":   lambda m: m not in ("straight", "left"),
            "straight_and_right_turn_only":  lambda m: m not in ("straight", "right"),
        }
        check = rules.get(sign_type)
        return check(movement) if check else False


# =============================================================================
# SIGN VIOLATION DETECTOR (Main class - reusable for Web)
# =============================================================================

class SignViolationDetector:
    """
    Reusable detector class cho cả standalone và web integration.
    Interface đồng bộ với PrecisionWrongWayDetector:
    - Không load model trong constructor
    - process_frame(frame, r0=..., model=..., conf=..., debug=...)
    - reset(), get_stats()
    """

    def __init__(self):
        self.sign_detector = TrafficSignDetector()
        self.trackers: Dict[int, VehicleTracker] = {}

        self.current_frame = 0
        self.total_violations = 0
        self.violations_by_type: Dict[str, int] = defaultdict(int)

        self.fps = 0.0
        self.prev_time = time.time()

    def reset(self):
        """Reset tất cả state — cho web integration khi đổi video."""
        self.sign_detector = TrafficSignDetector()
        self.trackers = {}
        self.current_frame = 0
        self.total_violations = 0
        self.violations_by_type = defaultdict(int)
        self.fps = 0.0
        self.prev_time = time.time()

    def process_frame(
        self,
        frame: np.ndarray,
        r0=None,
        model=None,
        conf: float = 0.25,
        debug: bool = False
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process một frame, trả về (annotated_frame, list_of_violations).
        
        Args:
            frame: Input frame (BGR)
            r0: YOLO results (từ UnifiedDetector), None nếu chạy standalone
            model: YOLO model (nếu r0 là None)
            conf: Confidence threshold
            debug: Bật debug overlay
            
        Returns:
            (frame_vis, violations_list)
        """
        if frame is None:
            return frame, []

        self.current_frame += 1
        violations = []

        # Nếu không có results, tự chạy model
        if r0 is None:
            if model is None:
                return frame, []
            results = model.track(
                frame, imgsz=config.IMG_SIZE, conf=conf,
                iou=config.IOU_THRESHOLD, persist=True, verbose=False
            )
            r0 = results[0]

        frame_vis = frame.copy()
        self.sign_detector.update_sign_classes_from_model(getattr(r0, "names", None))

        # Parse detections
        detections = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes = r0.boxes.xyxy.cpu().numpy()
            classes = r0.boxes.cls.cpu().numpy().astype(int)
            confs = r0.boxes.conf.cpu().numpy()

            for box, cls, conf_val in zip(boxes, classes, confs):
                detections.append((cls, conf_val, tuple(box)))

        # PHASE 1: Calibration
        if self.sign_detector.is_calibrating:
            self.sign_detector.update_calibration(detections, frame.shape[:2])

            # Vẽ biển báo đang detect
            for cls_id, conf_val, bbox in detections:
                if cls_id in self.sign_detector.sign_class_map:
                    x1, y1, x2, y2 = map(int, bbox)
                    sign_name = self.sign_detector.sign_class_map[cls_id]
                    draw_bbox_with_label(
                        frame_vis, (x1, y1, x2, y2),
                        f"SIGN: {sign_name} {conf_val:.2f}",
                        config.COLOR_WARNING
                    )

            # Progress HUD
            progress = (self.sign_detector.calibration_count / config.SIGN_CALIBRATION_FRAMES) * 100
            draw_calibration_hud(frame_vis, progress, config.SIGN_CALIBRATION_FRAMES / 30.0)

        # PHASE 2: Detection
        else:
            if r0.boxes is not None and r0.boxes.id is not None:
                boxes = r0.boxes.xyxy.cpu().numpy()
                track_ids = r0.boxes.id.cpu().numpy().astype(int)
                classes = r0.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, track_ids, classes):
                    # Xử lý xe
                    if cls in config.VEHICLE_CLASSES:
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

                        # Check violation khi đủ trajectory
                        if len(tracker.positions) >= config.MIN_TRACK_LENGTH:
                            violation_type = self.sign_detector.check_violation(tracker)
                            tracker.update_violation(violation_type is not None, violation_type)

                            # Confirm violation: dùng tracker.violation_type (đã lưu)
                            # thay vì local violation_type có thể None ở frame này
                            if (not tracker.is_confirmed and
                                    tracker.violation_frames >= config.VIOLATION_CONSECUTIVE_FRAMES):
                                tracker.is_confirmed = True
                                self.total_violations += 1
                                confirmed_type = tracker.violation_type or 'unknown'
                                self.violations_by_type[confirmed_type] += 1

                                # Snapshot
                                save_violation_snapshot(
                                    frame, f"sign_{confirmed_type}",
                                    tid, box, vehicle_class=tracker.vehicle_class
                                )
                                violations.append({
                                    'type': 'sign',
                                    'id': tid,
                                    'label': f'Sign: {confirmed_type}'
                                })
                                print(f"🚨 VIOLATION: ID {tid} - {confirmed_type}")

                        # Vẽ vehicle bbox
                        self._draw_vehicle(frame_vis, tracker, box)

                    # Vẽ biển báo
                    elif cls in self.sign_detector.sign_class_map:
                        x1, y1, x2, y2 = map(int, box)
                        sign_name = self.sign_detector.sign_class_map[cls]
                        draw_bbox_with_label(
                            frame_vis, (x1, y1, x2, y2),
                            sign_name, config.COLOR_WARNING
                        )

            # Vẽ enforcement zones
            if debug:
                self._draw_signs(frame_vis)
            self._cleanup_stale_trackers()

        # FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time

        # HUD
        self._draw_ui(frame_vis)

        return frame_vis, violations

    def _draw_vehicle(self, frame, tracker: VehicleTracker, box):
        """Vẽ vehicle bbox với style thống nhất"""
        x1, y1, x2, y2 = map(int, box)

        if tracker.is_confirmed:
            color = config.COLOR_VIOLATION
            label = f"VIOLATION: {tracker.violation_type}"
        elif tracker.is_violating:
            color = config.COLOR_WARNING
            label = f"WARNING #{tracker.track_id}"
        else:
            color = config.COLOR_SAFE
            label = f"{tracker.vehicle_class}:{tracker.track_id}"

        draw_bbox_with_label(frame, (x1, y1, x2, y2), label, color)

        # Vẽ trajectory
        trajectory = tracker.get_trajectory()
        if trajectory is not None and len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                pt1 = tuple(trajectory[i - 1].astype(int))
                pt2 = tuple(trajectory[i].astype(int))
                cv2.line(frame, pt1, pt2, color, 2)

    def _cleanup_stale_trackers(self):
        """Xóa track đã mất quá lâu để tránh phình state."""
        stale_ids = [
            tid for tid, tracker in self.trackers.items()
            if (self.current_frame - tracker.last_frame) > config.STALE_TRACKER_FRAMES
        ]
        for tid in stale_ids:
            del self.trackers[tid]

    def _draw_signs(self, frame):
        """Vẽ active signs và enforcement zones (debug mode)"""
        for sign in self.sign_detector.active_signs:
            x1, y1, x2, y2 = [int(v) for v in sign['bbox']]
            draw_bbox_with_label(frame, (x1, y1, x2, y2), sign['sign_type'], config.COLOR_WARNING)

            # Enforcement zone (viền mỏng)
            zx1, zy1, zx2, zy2 = [int(v) for v in sign['enforcement_zone']]
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), config.COLOR_WARNING, 1)

    def _draw_ui(self, frame):
        """Vẽ HUD thống nhất"""
        if self.sign_detector.is_calibrating:
            return  # calibration HUD đã vẽ ở trên

        hud_lines = [
            (f"FPS: {self.fps:.1f} | Frame: {self.current_frame}", config.HUD_TEXT_COLOR),
            (f"Vehicles: {len(self.trackers)} | Violations: {self.total_violations}", config.HUD_TEXT_COLOR),
            (f"Signs: {len(self.sign_detector.active_signs)}", config.COLOR_SAFE),
        ]
        draw_info_hud(frame, hud_lines, title="SIGN VIOLATION DETECTION", title_color=config.COLOR_WARNING)

    def get_stats(self) -> dict:
        """Trả về stats cho web — đồng bộ với PrecisionWrongWayDetector."""
        return {
            'violations': self.total_violations,
            'vehicles': len(self.trackers),
            'signs': len(self.sign_detector.active_signs),
            'calibrating': self.sign_detector.is_calibrating,
            'total': self.total_violations
        }


# =============================================================================
# STANDALONE RUN FUNCTION
# =============================================================================

def run(
    video_path: str = None,
    output_path: str = None,
    model_path: str = None,
    show_preview: bool = True
):
    """
    Chạy detection vi phạm biển báo (standalone mode)

    Args:
        video_path: Đường dẫn video input
        output_path: Đường dẫn video output
        model_path: Đường dẫn model YOLO
        show_preview: Có hiển thị preview không
    """
    from datetime import datetime

    video_path = video_path or config.DEFAULT_VIDEO
    model_path = model_path or config.MODEL_PATH

    # Tạo output folder
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"sign_violation_{timestamp}.mp4"
        output_path = str(output_dir / output_filename)
    else:
        output_path = str(output_dir / Path(output_path).name)

    print(f"[INFO] Output will be saved to: {output_path}")

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    # Load model (chỉ cho standalone mode)
    print(f"[INIT] Loading model: {model_path}")
    model = YOLO(model_path)

    # Initialize detector
    detector = SignViolationDetector()

    print("\n[START] Processing video...")
    print("Press 'q' to quit\n")

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            processed, violations = detector.process_frame(
                frame, model=model, conf=config.CONF_DETECTION, debug=True
            )

            writer.write(processed)

            if show_preview:
                cv2.imshow("Sign Violation Detection", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_idx % 100 == 0:
                print(f"[PROGRESS] Frame {frame_idx} | Violations: {detector.total_violations}")
    finally:
        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 50)
    print("DETECTION SUMMARY")
    print("=" * 50)
    print(f"Total Frames: {frame_idx}")
    print(f"Sign Violations: {detector.total_violations}")
    print(f"\nOutput saved: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    run()
