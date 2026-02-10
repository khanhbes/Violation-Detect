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
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import sys
import time

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import UnifiedDetector from detection_service
from services.detection_service import UnifiedDetector
from config.config import config

# =============================================================================
# PATH CONFIG (from config)
# =============================================================================

MODEL_DIR = config.MODEL_DIR
VIDEO_DIR = config.VIDEO_DIR
OUTPUT_DIR = config.OUTPUT_DIR
SNAPSHOT_DIR = config.SNAPSHOT_DIR
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

for vtype in ['helmet', 'redlight', 'sidewalk', 'wrong_way', 'wrong_lane', 'sign']:
    (SNAPSHOT_DIR / vtype).mkdir(exist_ok=True)

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Traffic Violation Detection",
    description="Real-time traffic violation detection with WebSocket streaming",
    version="4.0.0"
)

# Static files & templates
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =============================================================================
# DETECTOR MANAGEMENT
# =============================================================================

# Global detector (lazy loaded)
detector: Optional[UnifiedDetector] = None
current_model: Optional[str] = None

# Video processing tasks
video_tasks: Dict[str, Dict] = {}


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
# IMAGE DETECTION API
# =============================================================================

@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model: str = Form(None),
    conf: float = Form(0.25)
):
    """
    Upload an image and run YOLO detection on it.
    Returns annotated image (base64) + list of detections.
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse({"error": "Cannot decode image"}, status_code=400)
        
        # Get model
        model_path = model or config.MODEL_PATH
        det = get_detector(model_path)
        
        # Run prediction (not tracking, single image)
        from ultralytics import YOLO
        results = det.model.predict(
            img, 
            imgsz=config.IMG_SIZE, 
            conf=conf, 
            verbose=False
        )
        r0 = results[0]
        
        # Extract detections
        detections = []
        frame_vis = img.copy()
        
        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes = r0.boxes.xyxy.cpu().numpy()
            classes = r0.boxes.cls.cpu().numpy().astype(int)
            confs = r0.boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                cls_id = int(classes[i])
                confidence = float(confs[i])
                class_name = config.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2]
                })
                
                # Draw bbox
                color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame_vis, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Encode result image
        _, buffer = cv2.imencode('.jpg', frame_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Class summary
        class_summary = {}
        for d in detections:
            name = d['class_name']
            if name not in class_summary:
                class_summary[name] = {"count": 0, "max_conf": 0, "min_conf": 1}
            class_summary[name]["count"] += 1
            class_summary[name]["max_conf"] = max(class_summary[name]["max_conf"], d["confidence"])
            class_summary[name]["min_conf"] = min(class_summary[name]["min_conf"], d["confidence"])
        
        return JSONResponse({
            "image": img_b64,
            "detections": detections,
            "total": len(detections),
            "class_summary": class_summary
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# VIDEO DETECTION API
# =============================================================================

@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    model: str = Form(None),
    conf: float = Form(0.25)
):
    """
    Upload a video and start processing. Returns task_id to poll status.
    """
    try:
        # Save uploaded video
        task_id = str(uuid.uuid4())[:8]
        input_path = UPLOAD_DIR / f"input_{task_id}.mp4"
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Initialize task
        video_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "input": str(input_path),
            "output": None,
            "total_detections": 0,
            "class_summary": {},
            "error": None
        }
        
        # Process in background
        model_path = model or config.MODEL_PATH
        asyncio.create_task(process_video_task(task_id, str(input_path), model_path, conf))
        
        return JSONResponse({"task_id": task_id, "status": "processing"})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


async def process_video_task(task_id: str, input_path: str, model_path: str, conf: float):
    """Background task to process video with detection."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            video_tasks[task_id]["status"] = "error"
            video_tasks[task_id]["error"] = "Cannot open video"
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_name = f"detected_{task_id}.mp4"
        output_path = OUTPUT_DIR / output_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        class_summary = {}
        total_detections = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Run detection with tracking
            results = model.track(
                frame, imgsz=config.IMG_SIZE, conf=conf,
                iou=config.IOU_THRESHOLD, persist=True,
                verbose=False, tracker=config.TRACKER
            )
            r0 = results[0]
            
            frame_vis = frame.copy()
            
            if r0.boxes is not None and len(r0.boxes) > 0:
                boxes = r0.boxes.xyxy.cpu().numpy()
                classes = r0.boxes.cls.cpu().numpy().astype(int)
                confs = r0.boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                    cls_id = int(classes[i])
                    confidence = float(confs[i])
                    class_name = config.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    
                    total_detections += 1
                    if class_name not in class_summary:
                        class_summary[class_name] = {"count": 0, "max_conf": 0}
                    class_summary[class_name]["count"] += 1
                    class_summary[class_name]["max_conf"] = max(
                        class_summary[class_name]["max_conf"], confidence
                    )
                    
                    # Draw bbox
                    color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame_vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame_vis, label, (x1 + 2, y1 - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            writer.write(frame_vis)
            
            # Update progress
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            video_tasks[task_id]["progress"] = round(progress, 1)
            video_tasks[task_id]["total_detections"] = total_detections
            video_tasks[task_id]["class_summary"] = class_summary
            
            # Yield control periodically
            if frame_idx % 5 == 0:
                await asyncio.sleep(0)
        
        cap.release()
        writer.release()
        
        video_tasks[task_id]["status"] = "done"
        video_tasks[task_id]["progress"] = 100
        video_tasks[task_id]["output"] = f"/output/{output_name}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        video_tasks[task_id]["status"] = "error"
        video_tasks[task_id]["error"] = str(e)


@app.get("/api/detect/video/status/{task_id}")
async def get_video_status(task_id: str):
    """Poll video processing status."""
    if task_id not in video_tasks:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(video_tasks[task_id])


# =============================================================================
# LOOKUP API
# =============================================================================

@app.post("/api/lookup")
async def lookup_violations(
    phone: str = Form(""),
    cccd: str = Form(""),
    license_plate: str = Form("")
):
    """
    Look up violations by phone, CCCD, or license plate.
    For now, search snapshot filenames for matching patterns.
    """
    query = (phone + cccd + license_plate).strip().lower()
    
    if not query:
        return JSONResponse({"results": [], "message": "Vui lòng nhập thông tin tra cứu"})
    
    results = []
    for violation_type in ['helmet', 'redlight', 'sidewalk', 'wrong_way', 'wrong_lane', 'sign']:
        folder = SNAPSHOT_DIR / violation_type
        if folder.exists():
            for f in sorted(folder.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True):
                if query in f.name.lower():
                    results.append({
                        "type": violation_type,
                        "filename": f.name,
                        "path": f"/snapshots/{violation_type}/{f.name}",
                        "timestamp": f.stat().st_mtime
                    })
    
    # If no exact match, return recent violations as "related"
    if not results:
        for violation_type in ['helmet', 'redlight', 'sidewalk', 'wrong_way', 'wrong_lane']:
            folder = SNAPSHOT_DIR / violation_type
            if folder.exists():
                for f in sorted(folder.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    results.append({
                        "type": violation_type,
                        "filename": f.name,
                        "path": f"/snapshots/{violation_type}/{f.name}",
                        "timestamp": f.stat().st_mtime,
                        "related": True
                    })
    
    return JSONResponse({
        "results": results[:20],
        "query": {"phone": phone, "cccd": cccd, "license_plate": license_plate},
        "total": len(results)
    })


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
        {"action": "update_settings", "conf": 0.5, "debug": true}
    
    Server sends:
        {"type": "frame", "image": "base64...", "stats": {...}}
        {"type": "violation", "data": {...}}
    """
    await websocket.accept()
    
    cap = None
    running = False
    det = None
    
    # Mutable settings that can be updated mid-stream
    live_conf = 0.25
    live_debug = False
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")
            
            if action == "start":
                # Get parameters
                video_path = msg.get("video") or config.DEFAULT_VIDEO
                model_path = msg.get("model") or config.MODEL_PATH
                detectors = msg.get("detectors", ["helmet"])
                live_conf = float(msg.get("conf", 0.25))
                live_debug = bool(msg.get("debug", False))
                
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
                    "conf": live_conf,
                    "debug": live_debug
                })
                
                running = True
                
                # Process frames
                while running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        await websocket.send_json({"type": "finished", "stats": det.get_stats()})
                        break
                    
                    # Detect using UnifiedDetector with LIVE settings
                    frame_vis, violations = det.process_frame(
                        frame, detectors, conf=live_conf, debug=live_debug
                    )
                    
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
                    
                    # Check for messages (stop or update_settings) non-blocking
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
                        
                        elif stop_data.get("action") == "update_settings":
                            # Live update confidence and debug
                            if "conf" in stop_data:
                                live_conf = float(stop_data["conf"])
                            if "debug" in stop_data:
                                live_debug = bool(stop_data["debug"])
                            await websocket.send_json({
                                "type": "settings_updated",
                                "conf": live_conf,
                                "debug": live_debug
                            })
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
    print("🚦 TRAFFIC VIOLATION DETECTION WEB SERVER v4.0")
    print("   Now using real detection logic from functions/")
    print("=" * 60)
    print(f"📁 Models: {MODEL_DIR}")
    print(f"📁 Videos: {VIDEO_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📁 Uploads: {UPLOAD_DIR}")
    print(f"📁 Functions: {config.BASE_DIR / 'functions'}")
    print("=" * 60)
    print("🌐 Open: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
