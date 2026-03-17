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
import socket

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import UnifiedDetector from detection_service
from services.detection_service import UnifiedDetector
from services.fcm_service import FCMService
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
    title="Traffic Violation API",
    description="Backend for traffic violation detection.",
    version="4.0"
)

@app.on_event("startup")
async def startup_event():
    try:
        def get_local_ip():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()
            return ip

        local_ip = get_local_ip()
        
        # Write to Firestore if fcm_service works
        if fcm_service and fcm_service._db:
            from firebase_admin import firestore as fb_firestore
            fcm_service._db.collection('server').document('config').set({
                'ip': local_ip,
                'updated_at': fb_firestore.SERVER_TIMESTAMP
            })
            print(f"✅ Bật Firebase Auto-Discovery: Cập nhật IP {local_ip} lên Firestore [server/config]")
    except Exception as e:
        print(f"⚠️ Failed to write Local IP to Firestore: {e}")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# ─── Violation Store (in-memory, for App sync) ─────────────────────────────
violation_store: List[Dict] = []
violation_counter = 0

# ─── App WebSocket Clients (for real-time push) ────────────────────────────
app_clients: set = set()

# ─── FCM Push Notification Service ─────────────────────────────────────────
try:
    fcm_service = FCMService()
except Exception as e:
    print(f"⚠️ FCM service init failed: {e}")
    fcm_service = None

async def broadcast_to_apps(violation: Dict):
    """Push a violation to all connected Flutter app clients."""
    if not app_clients:
        return
    message = json.dumps({
        "type": "new_violation",
        "data": violation
    })
    disconnected = set()
    for ws in app_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    app_clients.difference_update(disconnected)
    if disconnected:
        print(f"📱 Cleaned {len(disconnected)} disconnected app client(s)")
    print(f"📱 Broadcast violation to {len(app_clients)} app client(s)")

VIOLATION_INFO = {
    'helmet':     {'name': 'Không đội mũ bảo hiểm',   'fine': 10000,   'code': 'MBH01', 'law': 'Điều 7, NĐ 100/2019/NĐ-CP'},
    'no_helmet':  {'name': 'Không đội mũ bảo hiểm',   'fine': 10000,   'code': 'MBH01', 'law': 'Điều 7, NĐ 100/2019/NĐ-CP'},
    'redlight':   {'name': 'Vượt đèn đỏ',             'fine': 30000,   'code': 'DD01',  'law': 'Điều 6, NĐ 100/2019/NĐ-CP'},
    'sidewalk':   {'name': 'Chạy lên vỉa hè',         'fine': 15000,   'code': 'VH01',  'law': 'Điều 4, NĐ 100/2019/NĐ-CP'},
    'wrong_way':  {'name': 'Chạy ngược chiều',         'fine': 30000,   'code': 'NC01',  'law': 'Điều 4, NĐ 100/2019/NĐ-CP'},
    'wrong_lane': {'name': 'Đi sai làn đường',         'fine': 20000,   'code': 'LD01',  'law': 'Điều 4, NĐ 100/2019/NĐ-CP'},
    'sign':       {'name': 'Vi phạm biển báo',         'fine': 25000,   'code': 'BB01',  'law': 'Điều 4, NĐ 100/2019/NĐ-CP'},
}

def store_violation(v_type: str, track_id: int, label: str, snapshot_path: str = None):
    """Save a violation to in-memory store, Firestore, and Firebase Storage."""
    global violation_counter
    violation_counter += 1
    info = VIOLATION_INFO.get(v_type, VIOLATION_INFO.get('helmet'))
    now = datetime.now()

    # Try to find the latest snapshot image for this violation
    image_url = None
    local_snapshot = None
    if snapshot_path:
        image_url = snapshot_path
        # Resolve to actual file path for Storage upload
        local_snapshot = Path(snapshot_path)
        if not local_snapshot.is_absolute():
            local_snapshot = SNAPSHOT_DIR / v_type / local_snapshot.name
    else:
        # Search snapshot directory for most recent file matching this type
        snap_dir = SNAPSHOT_DIR / v_type
        if snap_dir.exists():
            files = sorted(snap_dir.glob('*.jpg'), key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                local_snapshot = files[0]
                image_url = f'/snapshots/{v_type}/{files[0].name}'

    # ── Upload snapshot to Firebase Storage ─────────────────────────
    firebase_image_url = image_url  # fallback to local URL
    if fcm_service and fcm_service.is_available:
        try:
            from firebase_admin import storage as fb_storage
            if local_snapshot and local_snapshot.exists():
                bucket = fb_storage.bucket()
                blob_path = f'violations/{v_type}/{local_snapshot.name}'
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(local_snapshot), content_type='image/jpeg')
                blob.make_public()
                firebase_image_url = blob.public_url
                print(f"☁️ Uploaded to Storage: {blob_path}")
            else:
                print(f"📷 No snapshot file to upload (path: {local_snapshot})")
        except Exception as e:
            print(f"⚠️ Storage upload failed (non-fatal): {e}")
            # Continue — Firestore write will use local image URL as fallback

    # ── Write violation to Firestore (INDEPENDENT of Storage) ─────
    firestore_doc_id = None
    if fcm_service and fcm_service.is_available:
        try:
            from firebase_admin import firestore as fb_firestore
            db = fcm_service._db
            if db:
                # Get the most recently active user to link violation (for testing purposes)
                # In production, you'd match by license plate instead
                target_user_id = None
                try:
                    # Fallback 1: Try to get the user who most recently registered an FCM token
                    token_docs = db.collection('user_device_tokens')\
                        .order_by('last_updated', direction=fb_firestore.Query.DESCENDING)\
                        .limit(10).stream()
                    for doc in token_docs:
                        uid = doc.to_dict().get('user_id')
                        if uid and uid != 'default_user':
                            target_user_id = uid
                            break
                except Exception as ue:
                    print(f"⚠️ Could not fetch active user token: {ue}")

                if not target_user_id or target_user_id == 'default_user':
                    # Fallback 2: Pick the most recently registered user
                    try:
                        user_docs = db.collection('users').stream()
                        all_users = [doc for doc in user_docs]
                        if all_users:
                            # Typically the newest user is appended at the end, or we can sort by createdAt
                            all_users.sort(key=lambda d: d.to_dict().get('createdAt').timestamp() if hasattr(d.to_dict().get('createdAt'), 'timestamp') else 0)
                            target_user_id = all_users[-1].id
                    except Exception as ue:
                        print(f"⚠️ Could not fetch users: {ue}")

                raw_doc_ref = db.collection('violations').document()
                doc_ref = db.collection('violations').document(raw_doc_ref.id.upper())
                doc_data = {
                    'type': v_type,
                    'violationType': info['name'],
                    'violationCode': info['code'],
                    'description': f'{info["name"]} - {label}',
                    'fineAmount': info['fine'],
                    'lawReference': info['law'],
                    'timestamp': now.isoformat(),
                    'createdAt': fb_firestore.SERVER_TIMESTAMP,
                    'location': 'Camera giám sát giao thông',
                    'imageUrl': firebase_image_url,
                    'trackId': track_id,
                    'status': 'pending',
                    'licensePlate': 'Đang xác minh',
                }
                # Link violation to user
                if target_user_id:
                    doc_data['userId'] = target_user_id

                doc_ref.set(doc_data)
                firestore_doc_id = doc_ref.id
                print(f"🔥 Firestore: violation saved (ID: {firestore_doc_id})")

                # ── Create notification document for user ──────────────
                if target_user_id:
                    try:
                        db.collection('notifications').add({
                            'userId': target_user_id,
                            'title': f'🚨 {info["name"]}',
                            'body': f'Mức phạt: {info["fine"]:,}₫ - {label}',
                            'type': 'violation',
                            'violationId': firestore_doc_id,
                            'isRead': False,
                            'createdAt': fb_firestore.SERVER_TIMESTAMP,
                        })
                        print(f"🔔 Notification created for user {target_user_id}")
                    except Exception as ne:
                        print(f"⚠️ Notification write failed: {ne}")
            else:
                print(f"⚠️ Firestore DB is None — cannot save violation")
        except Exception as e:
            print(f"⚠️ Firestore write failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ FCM service not available — violation NOT saved to Firestore")

    violation = {
        'id': firestore_doc_id or f'vio_{violation_counter:04d}',
        'type': v_type,
        'violationType': info['name'],
        'violationCode': info['code'],
        'description': f'{info["name"]} - {label}',
        'fineAmount': info['fine'],
        'lawReference': info['law'],
        'timestamp': now.isoformat(),
        'location': 'Camera giám sát giao thông',
        'imageUrl': firebase_image_url,
        'trackId': track_id,
        'status': 'pending',
        'licensePlate': 'Đang xác minh',
    }
    violation_store.append(violation)
    print(f"📱 Violation stored: {info['name']} (ID: {violation['id']})")

    # Broadcast to connected app clients in real-time (WebSocket)
    asyncio.ensure_future(broadcast_to_apps(violation))

    # Send FCM push notification to all registered devices
    if fcm_service and fcm_service.is_available:
        try:
            fcm_service.broadcast_push_notification(
                title=f'🚨 {info["name"]}',
                body=f'Mức phạt: {info["fine"]:,}₫ - {label}',
                data_payload={
                    'route': '/violation-detail',
                    'violation_id': violation['id'],
                    'type': v_type,
                }
            )
        except Exception as e:
            print(f"⚠️ FCM broadcast error: {e}")

    return violation


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

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC_DIR / "favicon.png")


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
# FCM TOKEN API (for push notification registration)
# =============================================================================

from pydantic import BaseModel

class FCMTokenRequest(BaseModel):
    user_id: str = "default_user"
    fcm_token: str
    platform: str = "web"  # android | ios | web
    device_info: str = ""

class FCMTokenRemoveRequest(BaseModel):
    fcm_token: str


@app.post("/api/fcm/register")
async def register_fcm_token(req: FCMTokenRequest):
    """
    Register a device FCM token for push notifications.
    Called by Flutter app and Web client on launch/token refresh.
    """
    if not fcm_service or not fcm_service.is_available:
        return JSONResponse(
            {"status": "warning", "message": "FCM service not available"},
            status_code=503
        )

    result = fcm_service.register_token(
        user_id=req.user_id,
        fcm_token=req.fcm_token,
        platform=req.platform,
        device_info=req.device_info,
    )
    status_code = 200 if result.get("status") == "ok" else 500
    return JSONResponse(result, status_code=status_code)


@app.delete("/api/fcm/unregister")
async def unregister_fcm_token(req: FCMTokenRemoveRequest):
    """
    Remove a device FCM token (on logout or uninstall).
    """
    if not fcm_service or not fcm_service.is_available:
        return JSONResponse(
            {"status": "warning", "message": "FCM service not available"},
            status_code=503
        )

    result = fcm_service.remove_token(fcm_token=req.fcm_token)
    return JSONResponse(result)


# =============================================================================
# SERVER INFO API (show IP for mobile app connection)
# =============================================================================

@app.get("/api/server-info")
async def get_server_info():
    """Return server's local IP addresses and port for mobile app connection."""
    import socket
    ips = []
    try:
        # Get all network interfaces
        hostname = socket.gethostname()
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
        seen = set()
        for info in addr_infos:
            ip = info[4][0]
            if ip not in seen and not ip.startswith('127.'):
                seen.add(ip)
                ips.append(ip)
    except Exception:
        pass

    # Fallback: get primary IP via UDP trick
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
            s.close()
        except Exception:
            ips.append("127.0.0.1")

    return JSONResponse({
        "ips": ips,
        "port": 8000,
        "hostname": socket.gethostname(),
        "ws_path": "/ws/app",
    })


# =============================================================================
# APP SYNC API (for Flutter mobile app)
# =============================================================================

@app.get("/api/app/violations")
async def get_app_violations(since: str = None):
    """
    Get violations for mobile app.
    Optional 'since' param (ISO datetime) to get only new violations.
    """
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
            filtered = [
                v for v in violation_store
                if datetime.fromisoformat(v['timestamp']) > since_dt
            ]
            return JSONResponse({'violations': filtered, 'total': len(filtered)})
        except ValueError:
            pass
    return JSONResponse({'violations': violation_store, 'total': len(violation_store)})


@app.get("/api/app/violations/{violation_id}")
async def get_app_violation_detail(violation_id: str):
    """Get single violation detail for mobile app."""
    for v in violation_store:
        if v['id'] == violation_id:
            return JSONResponse(v)
    return JSONResponse({'error': 'Violation not found'}, status_code=404)


@app.get("/api/app/stats")
async def get_app_stats():
    """Get violation statistics for mobile app dashboard."""
    total = len(violation_store)
    pending = len([v for v in violation_store if v['status'] == 'pending'])
    total_fines = sum(v['fineAmount'] for v in violation_store if v['status'] == 'pending')
    by_type = {}
    for v in violation_store:
        t = v['type']
        by_type[t] = by_type.get(t, 0) + 1
    return JSONResponse({
        'total': total,
        'pending': pending,
        'paid': total - pending,
        'totalFines': total_fines,
        'byType': by_type,
    })


# =============================================================================
# APP OTA UPDATE API
# =============================================================================

# Directory to store APK releases
APK_RELEASE_DIR = Path(__file__).parent / "apk_releases"
APK_RELEASE_DIR.mkdir(exist_ok=True)

# In-memory latest version info (also persisted to JSON file)
VERSION_FILE = APK_RELEASE_DIR / "latest_version.json"

def _load_version_info() -> dict:
    """Load latest version info from JSON file."""
    try:
        if VERSION_FILE.exists():
            with open(VERSION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {
        'version': '1.0.0',
        'buildNumber': 1,
        'downloadUrl': '',
        'changelog': '',
        'forceUpdate': False,
    }

def _save_version_info(info: dict):
    """Save latest version info to JSON file."""
    try:
        with open(VERSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save version info: {e}")

# Load on startup
_latest_version_info = _load_version_info()


@app.get("/api/app/latest-version")
async def get_latest_version():
    """
    Simple endpoint — returns latest version info as JSON.
    Flutter app calls this directly via HTTP to check for updates.
    No Firestore needed!
    
    Response example:
    {
        "version": "1.0.2",
        "buildNumber": 3,
        "downloadUrl": "/api/app/download-apk/app-release-v1.0.2.apk",
        "changelog": "- Sửa lỗi đăng nhập\n- Cải thiện hiệu suất",
        "forceUpdate": false
    }
    """
    return JSONResponse(_latest_version_info)


@app.get("/api/app/check-update")
async def check_app_update():
    """
    Check for the latest app version.
    Reads from Firestore `app_config/latest_version` document.
    Mobile app calls this on startup to compare versions.
    """
    try:
        if fcm_service and fcm_service.is_available:
            db = fcm_service._db
            if db:
                doc = db.collection('app_config').document('latest_version').get()
                if doc.exists:
                    data = doc.to_dict()
                    return JSONResponse({
                        'status': 'ok',
                        'version': data.get('version', '1.0.0'),
                        'buildNumber': data.get('buildNumber', 1),
                        'downloadUrl': data.get('downloadUrl', ''),
                        'changelog': data.get('changelog', ''),
                        'forceUpdate': data.get('forceUpdate', False),
                    })

        return JSONResponse({
            'status': 'ok',
            'version': '1.0.0',
            'buildNumber': 1,
            'downloadUrl': '',
            'changelog': '',
            'forceUpdate': False,
        })
    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)


class UpdateInfoRequest(BaseModel):
    version: str
    build_number: int = 1
    changelog: str = ""
    force_update: bool = False


@app.post("/api/app/upload-apk")
async def upload_apk(
    file: UploadFile = File(...),
    version: str = Form("1.0.0"),
    build_number: int = Form(1),
    changelog: str = Form(""),
    force_update: bool = Form(False),
):
    """
    Upload a new APK release and update Firestore with version info.
    
    Usage:
        curl -X POST http://localhost:8000/api/app/upload-apk \
            -F "file=@app-release.apk" \
            -F "version=1.0.2" \
            -F "build_number=3" \
            -F "changelog=Bug fixes and improvements" \
            -F "force_update=false"
    """
    try:
        # Save APK locally
        apk_filename = f"app-release-v{version}.apk"
        apk_path = APK_RELEASE_DIR / apk_filename
        
        with open(apk_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        file_size = apk_path.stat().st_size
        print(f"📱 APK saved: {apk_filename} ({file_size // 1024} KB)")
        
        # Upload APK to Firebase Storage
        download_url = f"/api/app/download-apk/{apk_filename}"  # fallback local URL
        
        if fcm_service and fcm_service.is_available:
            try:
                from firebase_admin import storage as fb_storage
                bucket = fb_storage.bucket()
                blob_path = f'app_releases/{apk_filename}'
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(apk_path), content_type='application/vnd.android.package-archive')
                blob.make_public()
                download_url = blob.public_url
                print(f"☁️ APK uploaded to Firebase Storage: {blob_path}")
            except Exception as e:
                print(f"⚠️ Firebase Storage upload failed, using local URL: {e}")
        
        # Update Firestore with latest version info
        if fcm_service and fcm_service.is_available:
            try:
                from firebase_admin import firestore as fb_firestore
                db = fcm_service._db
                if db:
                    db.collection('app_config').document('latest_version').set({
                        'version': version,
                        'buildNumber': build_number,
                        'downloadUrl': download_url,
                        'changelog': changelog,
                        'forceUpdate': force_update,
                        'updatedAt': fb_firestore.SERVER_TIMESTAMP,
                        'apkFileName': apk_filename,
                        'apkFileSize': file_size,
                    })
                    print(f"🔥 Firestore updated: latest_version = {version}")
            except Exception as e:
                print(f"⚠️ Firestore update failed: {e}")
        
        # ── Update in-memory version info + persist to JSON ──────
        global _latest_version_info
        _latest_version_info = {
            'version': version,
            'buildNumber': build_number,
            'downloadUrl': download_url,
            'changelog': changelog,
            'forceUpdate': force_update,
        }
        _save_version_info(_latest_version_info)
        print(f"📱 Version info updated: v{version} (build {build_number})")

        return JSONResponse({
            'status': 'ok',
            'version': version,
            'buildNumber': build_number,
            'downloadUrl': download_url,
            'apkFileName': apk_filename,
            'fileSize': file_size,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)


@app.get("/api/app/download-apk/{filename}")
async def download_apk(filename: str):
    """
    Download an APK release file.
    This is a fallback when Firebase Storage URL is not available.
    """
    apk_path = APK_RELEASE_DIR / filename
    if not apk_path.exists():
        return JSONResponse({'error': 'APK not found'}, status_code=404)
    
    import os
    return FileResponse(
        path=str(apk_path),
        filename=filename,
        media_type='application/vnd.android.package-archive',
        stat_result=os.stat(apk_path),
    )


@app.post("/api/app/set-update-info")
async def set_update_info(req: UpdateInfoRequest):
    """
    Manually set update info in Firestore without uploading an APK.
    Useful when APK is already hosted on Firebase Storage.
    """
    try:
        if fcm_service and fcm_service.is_available:
            from firebase_admin import firestore as fb_firestore
            db = fcm_service._db
            if db:
                db.collection('app_config').document('latest_version').set({
                    'version': req.version,
                    'buildNumber': req.build_number,
                    'changelog': req.changelog,
                    'forceUpdate': req.force_update,
                    'updatedAt': fb_firestore.SERVER_TIMESTAMP,
                })
                return JSONResponse({
                    'status': 'ok',
                    'message': f'Update info set: v{req.version}',
                })
        
        return JSONResponse({
            'status': 'error',
            'message': 'Firestore not available',
        }, status_code=503)
    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.get("/api/admin/data")
async def get_admin_data():
    """
    Lấy toàn bộ thông tin từ Firestore: users, vehicles, violations, notifications
    Phục vụ cho chức năng Quản lý dữ liệu trên Web.
    """
    try:
        if not (fcm_service and fcm_service.is_available):
            # Fallback for local testing when Firebase credentials are not found
            return JSONResponse({
                'status': 'ok',
                'data': {
                    'users': [],
                    'vehicles': [],
                    'violations': violation_store,
                    'notifications': [],
                    'complaints': []
                }
            })
            
        from firebase_admin import firestore as fb_firestore
        db = fcm_service._db
        
        # Helper function to get collection data
        def get_collection(col_name):
            docs = db.collection(col_name).stream()
            res = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                # Convert timestamps
                for k, v in data.items():
                    if hasattr(v, 'timestamp'):
                        data[k] = v.timestamp()
                res.append(data)
            return res

        users = get_collection('users')
        vehicles = get_collection('vehicles')
        violations = get_collection('violations')
        notifications = get_collection('notifications')
        complaints = get_collection('complaints')
        
        # Merge settings recursively if needed, but simple dict is returned for now
        
        return JSONResponse({
            'status': 'ok',
            'data': {
                'users': users,
                'vehicles': vehicles,
                'violations': violations,
                'notifications': notifications,
                'complaints': complaints
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: str):
    """
    Xóa toàn bộ dữ liệu của người dùng, bao gồm profile, notifications, complaints, vehicles, và violations
    """
    try:
        if not (fcm_service and fcm_service.is_available):
            return JSONResponse({'status': 'error', 'message': 'Firebase is not initialized'})
            
        from firebase_admin import firestore as fb_firestore
        from firebase_admin import auth
        db = fcm_service._db
        
        # 1. Xóa trong Authentication (Firebase Auth)
        try:
            auth.delete_user(user_id)
        except Exception as e:
            print(f"Lỗi khi xóa Firebase Auth user (có thể đã bị xóa trước đó): {e}")

        # 2. Xóa dữ liệu liên quan trong Firestore (Batch)
        def delete_collection(query):
            batch = db.batch()
            count = 0
            for doc in query.stream():
                batch.delete(doc.reference)
                count += 1
                if count >= 500:
                    batch.commit()
                    batch = db.batch()
                    count = 0
            if count > 0:
                batch.commit()

        # Users collection
        db.collection('users').document(user_id).delete()
        
        # Vehicles
        delete_collection(db.collection('vehicles').where('ownerId', '==', user_id))
        
        # Notifications
        delete_collection(db.collection('notifications').where('userId', '==', user_id))
        
        # Complaints
        delete_collection(db.collection('complaints').where('userId', '==', user_id))
        
        # Violations (optional, but requested to clean up)
        delete_collection(db.collection('violations').where('userId', '==', user_id))
        
        return JSONResponse({'status': 'ok', 'message': 'Dữ liệu người dùng đã được xóa'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.post("/api/webhook/sepay")
async def sepay_webhook(req: Request):
    """
    Webhook Endpoint for Open Banking (SePay / PayOS / Casso).
    Nhận thông báo biến động số dư khi người dùng chuyển tiền nộp phạt.
    Nó trích xuất ID Vi phạm từ nội dung chuyển khoản và tự động cập nhật Database.
    """
    try:
        # Prevent failure if Firebase isn't correctly initiated on local PC
        db = None
        if hasattr(fcm_service, '_db'):
            db = fcm_service._db

        if not db:
            return JSONResponse({'success': False, 'message': 'Firestore không khả dụng'}, status_code=503)
        
        data = await req.json()
        
        # Lấy field "code" từ SePay hoặc content
        code = data.get("code", "")
        
        # PayOS fallback
        if not code and "data" in data and isinstance(data["data"], dict):
            code = data["data"].get("description", "")
            
        if not code and "content" in data:
            code = data.get("content", "")

        if not code:
            return JSONResponse({'success': False, 'message': 'Nội dung CK / Code trống'})
        
        # Fix: Extract violation ID by removing exactly "NP"
        # Bỏ đúng 2 ký tự "NP" ở đầu
        # Nếu dùng webhook SePay, nó trả về param "code": "NPPIG..."
        code_upper = code.upper()
        if code_upper.startswith("NP"):
            violation_id = code_upper[2:]
        else:
            violation_id = code_upper
            
        # Optional: Clean up trailing string like name if `code` was actually the whole content
        # Firestore IDs are precisely 20 alphanumeric chars, and usually appear right at the start
        import re
        id_match = re.search(r'^([A-Z0-9]{20})', violation_id)
        if id_match:
            violation_id = id_match.group(1)

        # Update local memory array too
        def update_local_store(v_id):
            for v in violation_store:
                if v.get('id') == v_id:
                    v['status'] = 'paid'
                    
        # Do ngân hàng (và SePay) viết hoa toàn bộ nội dung chuyển khoản, 
        # Firestore document ID lại case-sensitive (VD: PigZJyn...) nên ta không thể dùng .document().get()
        # Lấy toàn bộ document và so sánh case-insensitive
        found_doc_id = None
        all_docs_list = list(db.collection('violations').stream())
        for d in all_docs_list:
            if d.id.upper() == violation_id:
                found_doc_id = d.id
                break
        
        if found_doc_id:
            doc_ref = db.collection('violations').document(found_doc_id)
            doc_ref.update({'status': 'paid'})
            update_local_store(found_doc_id)
            print(f"[Webhook] Đã tự động cập nhật vi phạm {found_doc_id} thành 'paid'")
            return JSONResponse({'success': True, 'message': f'Cập nhật thành công VP {found_doc_id}'})
        else:
            # Log tất cả ID hiện có để debug
            all_ids = [d.id for d in all_docs_list]
            print(f"[Webhook] Tìm: {violation_id}, Có trong DB: {all_ids}")
            return JSONResponse({'success': False, 'message': f'Không tìm thấy ID vi phạm {violation_id} trên Cloud'})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'message': str(e)}, status_code=500)
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

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
                    
                    # Send violations if any + save to store
                    for v in violations:
                        stored = store_violation(
                            v_type=v.get('type', 'unknown'),
                            track_id=v.get('id', 0),
                            label=v.get('label', ''),
                        )
                        await websocket.send_json({
                            "type": "violation",
                            "data": {**v, "stored": stored}
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
# WEBSOCKET - APP REAL-TIME NOTIFICATIONS
# =============================================================================

@app.websocket("/ws/app")
async def websocket_app(websocket: WebSocket):
    """
    WebSocket endpoint for Flutter app real-time notifications.
    
    App connects here to receive instant violation alerts.
    Server pushes: {"type": "new_violation", "data": {...}}
    App can send:  {"action": "ping"} to keep alive
    """
    await websocket.accept()
    app_clients.add(websocket)
    client_ip = websocket.client.host if websocket.client else "unknown"
    print(f"📱 App client connected: {client_ip} (total: {len(app_clients)})")
    
    # Send current violation count as welcome message
    await websocket.send_text(json.dumps({
        "type": "connected",
        "message": "Connected to violation detection server",
        "pending_violations": len([v for v in violation_store if v['status'] == 'pending']),
        "total_violations": len(violation_store),
    }))
    
    try:
        while True:
            # Keep connection alive — listen for pings or commands
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")
            
            if action == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            elif action == "get_violations":
                # App can request all violations
                await websocket.send_text(json.dumps({
                    "type": "violations_list",
                    "data": violation_store
                }))
            elif action == "get_stats":
                total = len(violation_store)
                pending = len([v for v in violation_store if v['status'] == 'pending'])
                await websocket.send_text(json.dumps({
                    "type": "stats",
                    "data": {
                        "total": total,
                        "pending": pending,
                        "paid": total - pending,
                        "totalFines": sum(v['fineAmount'] for v in violation_store if v['status'] == 'pending'),
                    }
                }))
    except WebSocketDisconnect as e:
        print(f"📱 App client disconnected unexpectedly: {client_ip} (Code: {e.code}, Reason: {e.reason})")
    except Exception as e:
        print(f"📱 App client error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app_clients.discard(websocket)
        print(f"📱 App client session ended: {client_ip} (total: {len(app_clients)})")


# =============================================================================
# MAIN
# =============================================================================

def start_ngrok():
    import subprocess
    import time
    print("🌍 Đang khởi động Ngrok...")
    try:
        # Check if ngrok is available
        subprocess.Popen(['ngrok', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        
        # Start ngrok in the background
        subprocess.Popen(['ngrok', 'http', '8000'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait a bit for ngrok to start
        time.sleep(3)
        
        try:
            import requests
            response = requests.get('http://127.0.0.1:4040/api/tunnels')
            if response.status_code == 200:
                data = response.json()
                tunnels = data.get('tunnels', [])
                if tunnels:
                    public_url = tunnels[0]['public_url']
                    print("\n" + "=" * 60)
                    print(f"🚀 NGROK PUBLIC URL: {public_url}")
                    print(f"🔗 Webhook SePay URL: {public_url}/api/webhook/sepay")
                    print("=" * 60 + "\n")
                else:
                    print("⚠️ Ngrok đang chạy nhưng chưa lấy được tunnel.")
        except Exception as e:
            print(f"⚠️ Đã chạy Ngrok nhưng không lấy được URL qua API http://localhost:4040: {e}")
            
    except FileNotFoundError:
        print("\n⚠️ Không tìm thấy lệnh 'ngrok'. Hãy đảm bảo bạn đã cài đặt Ngrok và thêm vào PATH của Windows.")
    except Exception as e:
        print(f"\n⚠️ Lỗi khởi chạy Ngrok: {e}")


if __name__ == "__main__":
    import threading
    
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
    
    # Run ngrok in a separate thread so it doesn't block the server startup
    threading.Thread(target=start_ngrok, daemon=True).start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
