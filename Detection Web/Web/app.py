"""
Traffic Violation Detection Web Server
=======================================
FastAPI + WebSocket cho real-time detection streaming.
Sử dụng UnifiedDetector import logic gốc từ functions/

Usage:
    cd Web
    python app.py
    
    Then open: http://localhost:8000
"""

import asyncio
import base64
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import sys
import time

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import UnifiedDetector from detection_service
from services.detection_service import UnifiedDetector

# =============================================================================
# PATH CONFIG
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / 'assets'
MODEL_DIR = ASSETS_DIR / 'model'
VIDEO_DIR = ASSETS_DIR / 'video'
OUTPUT_DIR = BASE_DIR / 'output'
SNAPSHOT_DIR = BASE_DIR / 'snapshots'

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)
for vtype in ['helmet', 'redlight', 'sidewalk', 'wrong_way', 'wrong_lane', 'sign']:
    (SNAPSHOT_DIR / vtype).mkdir(exist_ok=True)

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Traffic Violation Detection",
    description="Real-time traffic violation detection with WebSocket streaming",
    version="3.0.0"
)

# Static files & templates
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =============================================================================
# DETECTOR MANAGEMENT
# =============================================================================

# Global detector (lazy loaded)
detector: Optional[UnifiedDetector] = None
current_model: Optional[str] = None


def get_detector(model_path: str) -> UnifiedDetector:
    """Get or create UnifiedDetector with specified model"""
    global detector, current_model
    
    if detector is None or current_model != model_path:
        detector = UnifiedDetector(model_path)
        current_model = model_path
    
    return detector


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/videos")
async def list_videos():
    videos = []
    for ext in ["*.mp4", "*.avi", "*.mkv"]:
        for f in VIDEO_DIR.glob(ext):
            videos.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size
            })
    return JSONResponse(videos)


@app.get("/api/models")
async def list_models():
    models = []
    for f in MODEL_DIR.glob("*.pt"):
        models.append({
            "name": f.name,
            "path": str(f),
            "size": f.stat().st_size
        })
    return JSONResponse(models)


@app.get("/api/detectors")
async def list_detectors():
    """List available detection functions"""
    return JSONResponse([
        {"id": "helmet", "name": "🏍 Helmet Detection", "desc": "Phát hiện vi phạm không đội mũ bảo hiểm"},
        {"id": "sidewalk", "name": "🚶 Sidewalk Detection", "desc": "Phát hiện vi phạm chạy lên vỉa hè (calibration 10s)"},
        {"id": "redlight", "name": "🚦 Red Light Detection", "desc": "Phát hiện vi phạm vượt đèn đỏ (calibration 5s)"},
        {"id": "wrong_way", "name": "↩️ Wrong Way Detection", "desc": "Phát hiện vi phạm chạy ngược chiều (learning phase)"},
        {"id": "wrong_lane", "name": "🛣️ Wrong Lane Detection", "desc": "Phát hiện vi phạm sai làn / chạm vạch (calibration 5s)"},
    ])


@app.get("/api/snapshots")
async def list_snapshots():
    snapshots = []
    for violation_type in ['helmet', 'redlight', 'sidewalk', 'wrong_way', 'wrong_lane', 'sign', 'no_helmet']:
        folder = SNAPSHOT_DIR / violation_type
        if folder.exists():
            for f in folder.glob("*.jpg"):
                snapshots.append({
                    "type": violation_type,
                    "filename": f.name,
                    "path": f"/snapshots/{violation_type}/{f.name}",
                    "timestamp": f.stat().st_mtime
                })
    
    snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
    return JSONResponse(snapshots[:50])


@app.get("/api/outputs")
async def list_outputs():
    outputs = []
    for f in OUTPUT_DIR.glob("*.mp4"):
        outputs.append({
            "name": f.name,
            "path": f"/output/{f.name}",
            "size": f.stat().st_size,
            "timestamp": f.stat().st_mtime
        })
    outputs.sort(key=lambda x: x['timestamp'], reverse=True)
    return JSONResponse(outputs[:20])


# =============================================================================
# WEBSOCKET - REAL-TIME DETECTION
# =============================================================================

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection.
    Uses UnifiedDetector which imports logic from functions/
    
    Client sends:
        {"action": "start", "video": "path", "model": "path", "detectors": ["helmet"]}
        {"action": "stop"}
    
    Server sends:
        {"type": "frame", "image": "base64...", "stats": {...}}
        {"type": "violation", "data": {...}}
    """
    await websocket.accept()
    
    cap = None
    running = False
    det = None
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")
            
            if action == "start":
                # Get parameters
                video_path = msg.get("video") or str(VIDEO_DIR / "test_2.mp4")
                model_path = msg.get("model") or str(MODEL_DIR / "best_yolo12s_seg.pt")
                detectors = msg.get("detectors", ["helmet"])
                conf = float(msg.get("conf", 0.25))
                debug = bool(msg.get("debug", False))
                
                # Validate paths
                if not Path(video_path).exists():
                    await websocket.send_json({"type": "error", "message": f"Video not found: {video_path}"})
                    continue
                    
                if not Path(model_path).exists():
                    await websocket.send_json({"type": "error", "message": f"Model not found: {model_path}"})
                    continue
                
                # Get UnifiedDetector
                det = get_detector(model_path)
                det.reset()
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    await websocket.send_json({"type": "error", "message": f"Cannot open video: {video_path}"})
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_delay = 1.0 / fps
                
                await websocket.send_json({
                    "type": "started",
                    "video": Path(video_path).name,
                    "model": Path(model_path).name,
                    "fps": fps,
                    "total_frames": total_frames,
                    "detectors": detectors,
                    "conf": conf,
                    "debug": debug
                })
                
                running = True
                
                # Process frames
                while running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        await websocket.send_json({"type": "finished", "stats": det.get_stats()})
                        break
                    
                    # Detect using UnifiedDetector (imports logic from functions/)
                    frame_vis, violations = det.process_frame(frame, detectors, conf=conf, debug=debug)
                    
                    # Resize for streaming
                    h, w = frame_vis.shape[:2]
                    if w > 1280:
                        scale = 1280 / w
                        frame_vis = cv2.resize(frame_vis, (1280, int(h * scale)))
                    
                    # Encode to base64
                    _, buffer = cv2.imencode('.jpg', frame_vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame
                    await websocket.send_json({
                        "type": "frame",
                        "image": frame_b64,
                        "stats": det.get_stats(),
                        "progress": det.frame_idx / total_frames * 100
                    })
                    
                    # Send violations if any
                    for v in violations:
                        await websocket.send_json({
                            "type": "violation",
                            "data": v
                        })
                    
                    # Rate limit
                    await asyncio.sleep(frame_delay * 0.3)
                    
                    # Check for stop (non-blocking)
                    try:
                        stop_msg = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=0.001
                        )
                        stop_data = json.loads(stop_msg)
                        if stop_data.get("action") == "stop":
                            running = False
                            if cap:
                                cap.release()
                                cap = None
                            if det:
                                det.reset()
                            await websocket.send_json({"type": "stopped"})
                            break
                    except asyncio.TimeoutError:
                        pass
                
                running = False
                if cap:
                    cap.release()
                    cap = None
                if det:
                    det.reset()
            
            elif action == "stop":
                running = False
                if cap:
                    cap.release()
                    cap = None
                if det:
                    det.reset()
                await websocket.send_json({"type": "stopped"})
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        if cap:
            cap.release()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚦 TRAFFIC VIOLATION DETECTION WEB SERVER v3.0")
    print("   Now using real detection logic from functions/")
    print("=" * 60)
    print(f"📁 Models: {MODEL_DIR}")
    print(f"📁 Videos: {VIDEO_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📁 Functions: {BASE_DIR / 'functions'}")
    print("=" * 60)
    print("🌐 Open: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
