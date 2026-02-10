"""Test import detection_service"""
import sys
import traceback
from pathlib import Path

# Same path setup as app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    print("Step 1: Import detection_service...")
    from services.detection_service import UnifiedDetector
    print("SUCCESS: UnifiedDetector imported!")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
    
    # Try individual imports to find the issue
    print("\n--- Testing individual imports ---")
    
    try:
        from functions.redlight_violation import StoplineCalibrator, LightMemory
        print("OK: redlight components")
    except Exception as e2:
        print(f"FAIL redlight: {e2}")
    
    try:
        from functions.redlight_violation import TrackState as RedlightTrackState
        print("OK: redlight TrackState")
    except Exception as e2:
        print(f"FAIL redlight TrackState: {e2}")
    
    try:
        from functions.redlight_violation import detect_traffic_lights, check_violation
        print("OK: redlight detect funcs")
    except Exception as e2:
        print(f"FAIL redlight detect: {e2}")
    
    try:
        from functions.redlight_violation import get_bottom_center, get_signed_distance, dedup_boxes
        print("OK: redlight geometry")
    except Exception as e2:
        print(f"FAIL redlight geometry: {e2}")
    
    try:
        from functions.redlight_violation import draw_traffic_lights, line_to_segment
        print("OK: redlight drawing")
    except Exception as e2:
        print(f"FAIL redlight draw: {e2}")
    
    try:
        from functions.helmet_violation import TrackState as HelmetTrackState
        print("OK: helmet TrackState")
    except Exception as e2:
        print(f"FAIL helmet TrackState: {e2}")
    
    try:
        from functions.helmet_violation import associate_rider_to_motorcycle
        print("OK: helmet associate")
    except Exception as e2:
        print(f"FAIL helmet associate: {e2}")

    try:
        from functions.helmet_violation import CLS_MOTORCYCLE, CLS_PERSON, CLS_PERSON_NO_HELMET, CLS_PERSON_WITH_HELMET
        print("OK: helmet constants")
    except Exception as e2:
        print(f"FAIL helmet constants: {e2}")
    
    try:
        from functions.helmet_violation import CONF_MOTORCYCLE, CONF_RIDER_NO_HELMET, CONF_RIDER_WITH_HELMET, CONF_RIDER_FALLBACK
        print("OK: helmet conf thresholds")
    except Exception as e2:
        print(f"FAIL helmet conf: {e2}")
    
    try:
        from functions.helmet_violation import C_GREEN, C_RED, C_ORANGE, C_HELMET, C_NO_HELMET, C_PERSON
        print("OK: helmet colors")
    except Exception as e2:
        print(f"FAIL helmet colors: {e2}")

    try:
        from functions.sidewalk_violation import process_mask_to_polygons_from_mask
        print("OK: sidewalk process_mask")
    except Exception as e2:
        print(f"FAIL sidewalk process_mask: {e2}")
    
    try:
        from functions.sidewalk_violation import check_violation_with_points, draw_forbidden_zones
        print("OK: sidewalk check_violation")
    except Exception as e2:
        print(f"FAIL sidewalk check: {e2}")

    try:
        from functions.sidewalk_violation import CALIBRATION_DURATION, CONF_THRESHOLD_CALIBRATION, CONF_THRESHOLD_TRACKING
        print("OK: sidewalk constants")
    except Exception as e2:
        print(f"FAIL sidewalk constants: {e2}")
    
    try:
        from functions.wrong_way_violation import PrecisionWrongWayDetector
        print("OK: wrong_way detector")
    except Exception as e2:
        print(f"FAIL wrong_way: {e2}")
