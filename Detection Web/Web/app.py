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
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import sys
import time
import socket
from urllib.parse import quote, unquote, urlparse, parse_qs

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, RedirectResponse
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
# ─── Admin WebSocket Clients (for realtime Data Management updates) ─────────
admin_clients: set = set()

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

async def broadcast_admin_event(event: str, payload: Optional[Dict[str, Any]] = None):
    """Push data-change events to connected admin dashboard clients."""
    if not admin_clients:
        return
    message = json.dumps({
        "type": "admin_data_changed",
        "event": event,
        "payload": payload or {},
        "timestamp": datetime.utcnow().isoformat(),
    })
    disconnected = set()
    for ws in admin_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    admin_clients.difference_update(disconnected)
    if disconnected:
        print(f"🖥️ Cleaned {len(disconnected)} disconnected admin client(s)")

MAX_LICENSE_POINTS = 12
COMPLAINT_EVIDENCE_KEYS = (
    'evidenceUrl',
    'evidenceURL',
    'evidenceDownloadUrl',
    'evidence_download_url',
    'evidenceImageUrl',
    'evidence_image_url',
    'downloadUrl',
    'downloadURL',
    'evidence',
    'proofUrl',
    'fileUrl',
    'file_url',
    'imageUrl',
    'image_url',
    'evidencePath',
    'evidence_path',
    'storagePath',
)

VIOLATION_INFO = {
    # Demo rule: fine 10,000-30,000 and point deduction 1-3 by severity/vehicle.
    'helmet': {
        'name': 'Không đội mũ bảo hiểm',
        'code': 'MBH01',
        'law': 'Điều 7, NĐ 100/2019/NĐ-CP',
        'severity': 'light',
        'defaultVehicle': 'motorcycle',
        'fines': {'motorcycle': 10000, 'car': 10000},
        'deductions': {'motorcycle': 1, 'car': 1},
    },
    'no_helmet': {
        'name': 'Không đội mũ bảo hiểm',
        'code': 'MBH01',
        'law': 'Điều 7, NĐ 100/2019/NĐ-CP',
        'severity': 'light',
        'defaultVehicle': 'motorcycle',
        'fines': {'motorcycle': 10000, 'car': 10000},
        'deductions': {'motorcycle': 1, 'car': 1},
    },
    'redlight': {
        'name': 'Vượt đèn đỏ',
        'code': 'DD01',
        'law': 'Điều 6, NĐ 100/2019/NĐ-CP',
        'severity': 'heavy',
        'defaultVehicle': 'car',
        'fines': {'motorcycle': 25000, 'car': 30000},
        'deductions': {'motorcycle': 2, 'car': 3},
    },
    'sidewalk': {
        'name': 'Chạy lên vỉa hè',
        'code': 'VH01',
        'law': 'Điều 4, NĐ 100/2019/NĐ-CP',
        'severity': 'medium',
        'defaultVehicle': 'motorcycle',
        'fines': {'motorcycle': 15000, 'car': 20000},
        'deductions': {'motorcycle': 2, 'car': 2},
    },
    'wrong_way': {
        'name': 'Chạy ngược chiều',
        'code': 'NC01',
        'law': 'Điều 4, NĐ 100/2019/NĐ-CP',
        'severity': 'heavy',
        'defaultVehicle': 'car',
        'fines': {'motorcycle': 30000, 'car': 30000},
        'deductions': {'motorcycle': 3, 'car': 3},
    },
    'wrong_lane': {
        'name': 'Đi sai làn đường',
        'code': 'LD01',
        'law': 'Điều 4, NĐ 100/2019/NĐ-CP',
        'severity': 'medium',
        'defaultVehicle': 'car',
        'fines': {'motorcycle': 20000, 'car': 25000},
        'deductions': {'motorcycle': 2, 'car': 2},
    },
    'sign': {
        'name': 'Vi phạm biển báo',
        'code': 'BB01',
        'law': 'Điều 4, NĐ 100/2019/NĐ-CP',
        'severity': 'light',
        'defaultVehicle': 'motorcycle',
        'fines': {'motorcycle': 10000, 'car': 15000},
        'deductions': {'motorcycle': 1, 'car': 1},
    },
}

def _normalize_vehicle_bucket(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip().lower()
    moto_keys = ('motor', 'moto', 'bike', 'xe máy', 'xe may', 'scooter')
    car_keys = ('car', 'ô tô', 'o to', 'auto', 'truck', 'bus', 'van')
    if any(k in text for k in moto_keys):
        return 'motorcycle'
    if any(k in text for k in car_keys):
        return 'car'
    return None

def _is_motorcycle_license(license_item: Dict[str, Any]) -> bool:
    vehicle_type = str(license_item.get('vehicleType', '')).lower()
    cls = str(license_item.get('class', '')).upper().strip()
    return (
        'xe máy' in vehicle_type
        or 'motor' in vehicle_type
        or cls.startswith('A')
    )

def _is_car_license(license_item: Dict[str, Any]) -> bool:
    vehicle_type = str(license_item.get('vehicleType', '')).lower()
    cls = str(license_item.get('class', '')).upper().strip()
    return (
        'ô tô' in vehicle_type
        or 'o to' in vehicle_type
        or 'car' in vehicle_type
        or cls.startswith(('B', 'C', 'D', 'E', 'F'))
    )

def _user_license_flags(user_data: Dict[str, Any]) -> Dict[str, bool]:
    licenses = user_data.get('driverLicenses')
    has_moto = False
    has_car = False
    if isinstance(licenses, list):
        for item in licenses:
            if not isinstance(item, dict):
                continue
            has_moto = has_moto or _is_motorcycle_license(item)
            has_car = has_car or _is_car_license(item)
    if not has_moto and str(user_data.get('motoLicenseClass', '')).strip():
        has_moto = True
    if not has_car and str(user_data.get('carLicenseClass', '')).strip():
        has_car = True
    return {'has_moto': has_moto, 'has_car': has_car}

def _read_license_points(user_data: Dict[str, Any], vehicle_bucket: str) -> int:
    fallback = int(user_data.get('points', MAX_LICENSE_POINTS) or MAX_LICENSE_POINTS)
    keys = (
        ('motoPoints', 'motoLicensePoints', 'points')
        if vehicle_bucket == 'motorcycle'
        else ('carPoints', 'carLicensePoints', 'points')
    )
    for key in keys:
        raw = user_data.get(key)
        if isinstance(raw, (int, float)):
            return int(raw)
    return fallback

def _aggregate_points(has_moto: bool, has_car: bool, moto_points: int, car_points: int) -> int:
    if has_moto and has_car:
        return max(0, min(MAX_LICENSE_POINTS, min(moto_points, car_points)))
    if has_moto:
        return max(0, min(MAX_LICENSE_POINTS, moto_points))
    if has_car:
        return max(0, min(MAX_LICENSE_POINTS, car_points))
    return max(0, min(MAX_LICENSE_POINTS, min(moto_points, car_points)))

def _adjust_user_license_points(
    db,
    user_id: str,
    vehicle_bucket: str,
    delta_points: int,
) -> Dict[str, Any]:
    """Adjust points for the affected license only (motorcycle/car)."""
    if not user_id or not delta_points:
        return {'changed': False}

    user_ref = db.collection('users').document(user_id)
    doc = user_ref.get()
    if not doc.exists:
        return {'changed': False}

    user_data = doc.to_dict() or {}
    flags = _user_license_flags(user_data)
    has_moto = flags['has_moto']
    has_car = flags['has_car']

    moto_points_before = _read_license_points(user_data, 'motorcycle')
    car_points_before = _read_license_points(user_data, 'car')
    moto_points_after = moto_points_before
    car_points_after = car_points_before

    changed = False
    became_disabled = False

    if vehicle_bucket == 'motorcycle' and has_moto:
        moto_points_after = max(0, min(MAX_LICENSE_POINTS, moto_points_before + delta_points))
        changed = moto_points_after != moto_points_before
        became_disabled = moto_points_before > 0 and moto_points_after == 0
    elif vehicle_bucket == 'car' and has_car:
        car_points_after = max(0, min(MAX_LICENSE_POINTS, car_points_before + delta_points))
        changed = car_points_after != car_points_before
        became_disabled = car_points_before > 0 and car_points_after == 0

    if not changed:
        return {'changed': False}

    aggregate_points = _aggregate_points(
        has_moto=has_moto,
        has_car=has_car,
        moto_points=moto_points_after,
        car_points=car_points_after,
    )
    user_ref.set({
        'motoPoints': moto_points_after,
        'carPoints': car_points_after,
        'points': aggregate_points,
        'motoLicenseStatus': 'disabled' if has_moto and moto_points_after == 0 else 'active',
        'carLicenseStatus': 'disabled' if has_car and car_points_after == 0 else 'active',
    }, merge=True)

    return {
        'changed': True,
        'became_disabled': became_disabled,
        'vehicle_bucket': vehicle_bucket,
        'moto_points_before': moto_points_before,
        'moto_points_after': moto_points_after,
        'car_points_before': car_points_before,
        'car_points_after': car_points_after,
        'aggregate_points': aggregate_points,
    }

def _delete_notifications_for_violation(db, violation_id: Optional[str]):
    if not violation_id:
        return 0
    deleted = 0
    docs = db.collection('notifications').where('violationId', '==', violation_id).stream()
    for doc in docs:
        doc.reference.delete()
        deleted += 1
    return deleted

def _remove_violation_from_store(violation_id: Optional[str]):
    if not violation_id:
        return
    global violation_store
    normalized = str(violation_id).upper()
    violation_store = [
        v for v in violation_store
        if str(v.get('id', '')).upper() != normalized
    ]

def _resolve_violation_ref_by_id(db, violation_id: Optional[str]):
    if not violation_id:
        return None

    normalized = str(violation_id).strip()
    if not normalized:
        return None

    direct_ref = db.collection('violations').document(normalized)
    direct_doc = direct_ref.get()
    if direct_doc.exists:
        return direct_ref

    for doc in db.collection('violations').stream():
        if str(doc.id).upper() == normalized.upper():
            return doc.reference
    return None

def _sync_violation_complaint_lock(
    db,
    violation_id: Optional[str],
    *,
    complaint_status: str,
    lock_payment: bool,
    lock_complaint: bool,
    reviewed_at: Optional[str] = None,
) -> bool:
    violation_ref = _resolve_violation_ref_by_id(db, violation_id)
    if not violation_ref:
        return False

    doc = violation_ref.get()
    if not doc.exists:
        return False

    data = doc.to_dict() or {}
    current_status = str(data.get('status') or '').strip().lower()

    payload: Dict[str, Any] = {
        'complaintStatus': complaint_status,
        'paymentLocked': bool(lock_payment),
        'complaintLocked': bool(lock_complaint),
        'updatedAt': datetime.utcnow().isoformat(),
    }

    if reviewed_at:
        payload['complaintReviewedAt'] = reviewed_at

    if not lock_payment and current_status in ('complaint_pending', 'pending_payment', 'pending', ''):
        payload['status'] = 'pending'
    elif lock_payment:
        payload['status'] = 'complaint_pending'

    violation_ref.set(payload, merge=True)
    return True

def _first_non_empty_complaint_evidence(complaint: Dict[str, Any]) -> str:
    if not isinstance(complaint, dict):
        return ''

    for key in COMPLAINT_EVIDENCE_KEYS:
        raw = complaint.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    nested = complaint.get('evidence')
    if isinstance(nested, dict):
        for key in (
            'url',
            'downloadUrl',
            'downloadURL',
            'download_url',
            'path',
            'storagePath',
            'storage_path',
            'fullPath',
            'full_path',
            'gsUrl',
            'gs_url',
        ):
            raw = nested.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return ''

def _extract_storage_blob_path(raw_value: Any) -> Optional[str]:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    lower = value.lower()
    if lower.startswith('gs://'):
        no_scheme = value[5:]
        parts = no_scheme.split('/', 1)
        return parts[1].strip('/') if len(parts) > 1 and parts[1].strip('/') else None

    if value.startswith('complaints/'):
        return value.strip('/')

    parsed = None
    try:
        parsed = urlparse(value)
    except Exception:
        parsed = None

    if parsed:
        query_values = parse_qs(parsed.query or '')
        for key in ('path', 'name', 'object', 'blobPath', 'blob_path', 'file'):
            candidates = query_values.get(key, [])
            for candidate in candidates:
                decoded_query_path = unquote(str(candidate)).strip().strip('/')
                if decoded_query_path.startswith('complaints/'):
                    return decoded_query_path

    if '/o/' in value:
        try:
            marker = '/o/'
            idx = parsed.path.find(marker)
            if idx >= 0:
                encoded_path = parsed.path[idx + len(marker):]
                decoded = unquote(encoded_path).strip('/')
                if decoded:
                    return decoded
        except Exception:
            pass

    decoded_value = unquote(value).strip().strip('/')
    if decoded_value.startswith('complaints/'):
        return decoded_value
    marker_idx = decoded_value.find('complaints/')
    if marker_idx >= 0:
        return decoded_value[marker_idx:].strip('/')

    return None

def _resolve_admin_complaint_evidence_url(complaint_id: str, complaint: Dict[str, Any]) -> str:
    raw = _first_non_empty_complaint_evidence(complaint)
    if not raw:
        return ''

    blob_path = _extract_storage_blob_path(raw)
    if blob_path:
        encoded_path = quote(blob_path, safe='')
        return f'/api/admin/complaints/{complaint_id}/evidence?path={encoded_path}'

    if raw.startswith('http://') or raw.startswith('https://') or raw.startswith('/'):
        return raw

    return ''

def _restore_points_from_violation(db, user_id: Optional[str], violation_data: Dict[str, Any]) -> Dict[str, Any]:
    if not user_id or not violation_data:
        return {'changed': False}
    deducted_points = int(violation_data.get('deductedPoints') or 0)
    if deducted_points <= 0:
        return {'changed': False}
    vehicle_bucket = (
        _normalize_vehicle_bucket(violation_data.get('licenseTarget'))
        or _normalize_vehicle_bucket(violation_data.get('vehicleType'))
        or _normalize_vehicle_bucket(violation_data.get('vehicleClass'))
    )
    if not vehicle_bucket:
        info = VIOLATION_INFO.get(violation_data.get('type'))
        vehicle_bucket = (info or {}).get('defaultVehicle', 'motorcycle')
    return _adjust_user_license_points(
        db=db,
        user_id=user_id,
        vehicle_bucket=vehicle_bucket,
        delta_points=deducted_points,
    )

def _select_violation_vehicle_bucket(
    v_type: str,
    label: str,
    vehicle_class: Optional[str],
    user_data: Dict[str, Any],
) -> str:
    if v_type in ('helmet', 'no_helmet'):
        return 'motorcycle'

    detected = (
        _normalize_vehicle_bucket(vehicle_class)
        or _normalize_vehicle_bucket(label)
    )
    if detected:
        return detected

    flags = _user_license_flags(user_data or {})
    has_moto = flags['has_moto']
    has_car = flags['has_car']
    if has_moto and not has_car:
        return 'motorcycle'
    if has_car and not has_moto:
        return 'car'

    info = VIOLATION_INFO.get(v_type) or {}
    return info.get('defaultVehicle', 'motorcycle')

def _resolve_violation_penalty(info: Dict[str, Any], vehicle_bucket: str) -> Dict[str, Any]:
    fines = info.get('fines') or {}
    deductions = info.get('deductions') or {}
    fine = int(fines.get(vehicle_bucket) or fines.get('motorcycle') or 10000)
    deducted_points = int(
        deductions.get(vehicle_bucket)
        or deductions.get('motorcycle')
        or 1
    )
    return {
        'fine': max(10000, min(30000, fine)),
        'deducted_points': max(1, min(3, deducted_points)),
        'severity': info.get('severity', 'light'),
    }

def create_user_notification(
    db,
    user_id: str,
    title: str,
    body: str,
    notif_type: str,
    violation_id: str = None,
):
    """Create a notification document for a specific user."""
    from firebase_admin import firestore as fb_firestore

    payload = {
        'userId': user_id,
        'title': title,
        'body': body,
        'type': notif_type,
        'isRead': False,
        'createdAt': fb_firestore.SERVER_TIMESTAMP,
    }
    if violation_id:
        payload['violationId'] = violation_id

    db.collection('notifications').add(payload)


def broadcast_app_update_notification(version: str, changelog: str = '') -> int:
    """
    Broadcast update notifications to all users.
    Uses deterministic notification IDs to avoid duplicates per version.
    """
    if not (fcm_service and fcm_service.is_available and fcm_service._db):
        return 0

    from firebase_admin import firestore as fb_firestore

    db = fcm_service._db
    users = list(db.collection('users').stream())
    normalized_version = str(version).replace('.', '_')
    for user_doc in users:
        uid = user_doc.id
        if not uid:
            continue

        notif_id = f'update_{uid}_{normalized_version}'
        db.collection('notifications').document(notif_id).set({
            'userId': uid,
            'title': '📲 Cập nhật ứng dụng',
            'body': f'Đã có phiên bản {version}. Vui lòng cập nhật để nhận tính năng và bản vá mới nhất.',
            'detail': changelog,
            'type': 'update',
            'isRead': False,
            'createdAt': fb_firestore.SERVER_TIMESTAMP,
        }, merge=True)

    return len(users)


def store_violation(
    v_type: str,
    track_id: int,
    label: str,
    snapshot_path: str = None,
    vehicle_class: Optional[str] = None,
):
    """Save a violation to in-memory store, Firestore, and Firebase Storage."""
    global violation_counter
    violation_counter += 1
    info = VIOLATION_INFO.get(v_type, VIOLATION_INFO.get('helmet'))
    now = datetime.now()
    payment_due_date = now + timedelta(days=7)

    # Try to find the latest snapshot image for this violation
    image_url = None
    local_snapshot = None
    if snapshot_path:
        image_url = snapshot_path
        local_snapshot = Path(snapshot_path)
        if not local_snapshot.is_absolute():
            local_snapshot = SNAPSHOT_DIR / v_type / local_snapshot.name
    else:
        snap_dir = SNAPSHOT_DIR / v_type
        if snap_dir.exists():
            files = sorted(snap_dir.glob('*.jpg'), key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                local_snapshot = files[0]
                image_url = f'/snapshots/{v_type}/{files[0].name}'

    # ── Upload snapshot to Firebase Storage ─────────────────────────
    firebase_image_url = image_url
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

    firestore_doc_id = None
    target_user_id = None
    point_update_result: Dict[str, Any] = {'changed': False}

    selected_vehicle_bucket = (
        _normalize_vehicle_bucket(vehicle_class)
        or (info or {}).get('defaultVehicle')
        or 'motorcycle'
    )
    penalty = _resolve_violation_penalty(info, selected_vehicle_bucket)
    fine_amount = penalty['fine']
    deducted_points = penalty['deducted_points']
    severity = penalty['severity']
    vehicle_type_label = 'Xe máy' if selected_vehicle_bucket == 'motorcycle' else 'Ô tô'

    # ── Write violation to Firestore ────────────────────────────────
    if fcm_service and fcm_service.is_available:
        try:
            from firebase_admin import firestore as fb_firestore
            db = fcm_service._db
            if db:
                try:
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
                    try:
                        user_docs = db.collection('users').stream()
                        all_users = [doc for doc in user_docs]
                        if all_users:
                            all_users.sort(
                                key=lambda d: (
                                    d.to_dict().get('createdAt').timestamp()
                                    if hasattr(d.to_dict().get('createdAt'), 'timestamp')
                                    else 0
                                )
                            )
                            target_user_id = all_users[-1].id
                    except Exception as ue:
                        print(f"⚠️ Could not fetch users: {ue}")

                target_user_data: Dict[str, Any] = {}
                if target_user_id:
                    try:
                        user_doc = db.collection('users').document(target_user_id).get()
                        target_user_data = user_doc.to_dict() or {}
                    except Exception as ue:
                        print(f"⚠️ Could not fetch target user profile: {ue}")

                selected_vehicle_bucket = _select_violation_vehicle_bucket(
                    v_type=v_type,
                    label=label,
                    vehicle_class=vehicle_class,
                    user_data=target_user_data,
                )
                penalty = _resolve_violation_penalty(info, selected_vehicle_bucket)
                fine_amount = penalty['fine']
                deducted_points = penalty['deducted_points']
                severity = penalty['severity']
                vehicle_type_label = 'Xe máy' if selected_vehicle_bucket == 'motorcycle' else 'Ô tô'

                raw_doc_ref = db.collection('violations').document()
                doc_ref = db.collection('violations').document(raw_doc_ref.id.upper())
                doc_data = {
                    'type': v_type,
                    'violationType': info['name'],
                    'violationCode': info['code'],
                    'description': f'{info["name"]} - {label}',
                    'fineAmount': fine_amount,
                    'deductedPoints': deducted_points,
                    'severity': severity,
                    'licenseTarget': selected_vehicle_bucket,
                    'vehicleType': vehicle_type_label,
                    'vehicleClass': selected_vehicle_bucket,
                    'lawReference': info['law'],
                    'timestamp': now.isoformat(),
                    'createdAt': fb_firestore.SERVER_TIMESTAMP,
                    'location': 'Camera giám sát giao thông',
                    'imageUrl': firebase_image_url,
                    'trackId': track_id,
                    'status': 'pending',
                    'licensePlate': 'Đang xác minh',
                    'paymentDueDate': payment_due_date.isoformat(),
                }
                if target_user_id:
                    doc_data['userId'] = target_user_id

                doc_ref.set(doc_data)
                firestore_doc_id = doc_ref.id
                print(f"🔥 Firestore: violation saved (ID: {firestore_doc_id})")

                if target_user_id:
                    try:
                        point_update_result = _adjust_user_license_points(
                            db=db,
                            user_id=target_user_id,
                            vehicle_bucket=selected_vehicle_bucket,
                            delta_points=-deducted_points,
                        )
                        due_str = payment_due_date.strftime('%d/%m/%Y')
                        create_user_notification(
                            db=db,
                            user_id=target_user_id,
                            title=f'🚨 {info["name"]}',
                            body=(
                                f'Mức phạt: {fine_amount:,}₫ • '
                                f'Trừ {deducted_points} điểm GPLX {vehicle_type_label.lower()}'
                            ),
                            notif_type='violation',
                            violation_id=firestore_doc_id,
                        )
                        create_user_notification(
                            db=db,
                            user_id=target_user_id,
                            title='⏳ Hạn đóng phạt',
                            body=f'Vi phạm cần được thanh toán trước ngày {due_str}',
                            notif_type='payment_due',
                            violation_id=firestore_doc_id,
                        )
                        if point_update_result.get('became_disabled'):
                            create_user_notification(
                                db=db,
                                user_id=target_user_id,
                                title='🚫 GPLX tạm vô hiệu',
                                body=(
                                    f'GPLX {vehicle_type_label.lower()} đã về 0/12 điểm và tạm vô hiệu. '
                                    f'Chỉ admin web mới khôi phục được điểm.'
                                ),
                                notif_type='license_disabled',
                                violation_id=firestore_doc_id,
                            )
                        print(f"🔔 Notifications created for user {target_user_id}")
                    except Exception as ne:
                        print(f"⚠️ Notification/points update failed: {ne}")
            else:
                print("⚠️ Firestore DB is None — cannot save violation")
        except Exception as e:
            print(f"⚠️ Firestore write failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ FCM service not available — violation NOT saved to Firestore")

    violation = {
        'id': firestore_doc_id or f'vio_{violation_counter:04d}',
        'type': v_type,
        'violationType': info['name'],
        'violationCode': info['code'],
        'description': f'{info["name"]} - {label}',
        'fineAmount': fine_amount,
        'deductedPoints': deducted_points,
        'severity': severity,
        'licenseTarget': selected_vehicle_bucket,
        'vehicleType': vehicle_type_label,
        'vehicleClass': selected_vehicle_bucket,
        'lawReference': info['law'],
        'timestamp': now.isoformat(),
        'location': 'Camera giám sát giao thông',
        'imageUrl': firebase_image_url,
        'trackId': track_id,
        'status': 'pending',
        'licensePlate': 'Đang xác minh',
        'paymentDueDate': payment_due_date.isoformat(),
    }
    if target_user_id:
        violation['userId'] = target_user_id
    violation_store.append(violation)
    print(f"📱 Violation stored: {info['name']} (ID: {violation['id']})")

    asyncio.ensure_future(broadcast_to_apps(violation))
    asyncio.ensure_future(broadcast_admin_event(
        "violation_created",
        {
            "violationId": violation['id'],
            "userId": target_user_id,
            "vehicleType": vehicle_type_label,
            "deductedPoints": deducted_points,
        },
    ))

    if fcm_service and fcm_service.is_available:
        try:
            fcm_service.broadcast_push_notification(
                title=f'🚨 {info["name"]}',
                body=f'Mức phạt: {fine_amount:,}₫ • Trừ {deducted_points} điểm GPLX',
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


@app.get("/firebase-messaging-sw.js", include_in_schema=False)
async def firebase_messaging_sw():
    return FileResponse(STATIC_DIR / "firebase-messaging-sw.js")


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
                    sent = broadcast_app_update_notification(version, changelog)
                    if sent > 0:
                        print(f"🔔 Sent app update notification to {sent} user(s)")
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
                sent = broadcast_app_update_notification(req.version, req.changelog)
                if sent > 0:
                    print(f"🔔 Sent app update notification to {sent} user(s)")
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
                    'complaints': [],
                    'profile_updates': []
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
        for complaint in complaints:
            complaint_id = str(complaint.get('id') or '')
            resolved_evidence_url = _resolve_admin_complaint_evidence_url(
                complaint_id=complaint_id,
                complaint=complaint,
            )
            complaint['evidenceUrlResolved'] = resolved_evidence_url
            complaint['hasEvidence'] = bool(resolved_evidence_url)
        profile_updates = get_collection('profile_update_requests')
        
        # Merge settings recursively if needed, but simple dict is returned for now
        
        return JSONResponse({
            'status': 'ok',
            'data': {
                'users': users,
                'vehicles': vehicles,
                'violations': violations,
                'notifications': notifications,
                'complaints': complaints,
                'profile_updates': profile_updates
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.get("/api/admin/complaints/{complaint_id}/evidence")
async def get_admin_complaint_evidence(
    complaint_id: str,
    path: Optional[str] = None,
):
    """
    Resolve complaint evidence image via Firebase Storage with admin privileges.
    Supports direct Firebase download URL, gs:// URL, or storage path.
    """
    try:
        if not (fcm_service and fcm_service.is_available):
            return JSONResponse(
                {'status': 'error', 'message': 'Firebase not available'},
                status_code=503,
            )

        db = fcm_service._db
        complaint_doc = db.collection('complaints').document(complaint_id).get()
        if not complaint_doc.exists:
            return JSONResponse(
                {'status': 'error', 'message': 'Không tìm thấy khiếu nại'},
                status_code=404,
            )

        complaint = complaint_doc.to_dict() or {}
        raw_evidence = (path or '').strip() or _first_non_empty_complaint_evidence(complaint)
        if not raw_evidence:
            return JSONResponse(
                {'status': 'error', 'message': 'Khiếu nại chưa có ảnh bằng chứng'},
                status_code=404,
            )

        blob_path = _extract_storage_blob_path(raw_evidence)
        if not blob_path:
            if raw_evidence.startswith('http://') or raw_evidence.startswith('https://'):
                return RedirectResponse(url=raw_evidence)
            return JSONResponse(
                {'status': 'error', 'message': 'Không xác định được đường dẫn ảnh bằng chứng'},
                status_code=404,
            )

        from firebase_admin import storage as fb_storage

        bucket = fb_storage.bucket()
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return JSONResponse(
                {'status': 'error', 'message': 'Không tìm thấy file ảnh trên Storage'},
                status_code=404,
            )

        image_bytes = blob.download_as_bytes()
        content_type = blob.content_type or 'image/jpeg'
        return Response(
            content=image_bytes,
            media_type=content_type,
            headers={'Cache-Control': 'private, max-age=120'},
        )

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.post("/api/admin/users/{user_id}/approve_update")
async def approve_profile_update(user_id: str, req: Request):
    try:
        if not (fcm_service and fcm_service.is_available):
            return JSONResponse({'status': 'error', 'message': 'Firebase not available'})
        
        data = await req.json()
        action = data.get("action")  # "approve" or "reject"

        if action not in ("approve", "reject"):
            return JSONResponse(
                {'status': 'error', 'message': 'action không hợp lệ'},
                status_code=400,
            )
        
        db = fcm_service._db
        request_ref = db.collection('profile_update_requests').document(user_id)
        
        if action == "approve":
            doc = request_ref.get()
            if doc.exists:
                update_data = doc.to_dict()
                # Remove metadata from the payload before applying
                for key in [
                    'userId',
                    'status',
                    'createdAt',
                    'updatedAt',
                    'reviewedAt',
                    'requestType',
                    'requestSection',
                    'requestSource',
                    'requestedAt',
                ]:
                    update_data.pop(key, None)
                
                # Apply changes
                db.collection('users').document(user_id).set(update_data, merge=True)
                request_ref.delete()
                await broadcast_admin_event(
                    'profile_update_reviewed',
                    {'userId': user_id, 'action': 'approve'},
                )
                return JSONResponse({'status': 'ok', 'message': 'Đã duyệt thay đổi'})
            else:
                return JSONResponse({'status': 'error', 'message': 'Không tìm thấy yêu cầu'}, status_code=404)
        elif action == "reject":
            request_ref.delete()
            await broadcast_admin_event(
                'profile_update_reviewed',
                {'userId': user_id, 'action': 'reject'},
            )
            return JSONResponse({'status': 'ok', 'message': 'Đã từ chối thay đổi'})

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.post("/api/admin/complaints/{complaint_id}/review")
async def review_complaint(complaint_id: str, req: Request):
    """
    Duyệt hoặc từ chối khiếu nại từ người dùng.
    Body: { "action": "approve" | "reject", "adminNote": "lý do" }
    """
    try:
        if not (fcm_service and fcm_service.is_available):
            return JSONResponse({'status': 'error', 'message': 'Firebase not available'})

        data = await req.json()
        action = str(data.get('action', '')).strip().lower()  # approve | reject
        admin_note = str(data.get('adminNote', '')).strip()

        if action not in ('approve', 'reject'):
            return JSONResponse({'status': 'error', 'message': 'action không hợp lệ'}, status_code=400)

        db = fcm_service._db
        complaint_ref = db.collection('complaints').document(complaint_id)
        complaint_doc = complaint_ref.get()
        if not complaint_doc.exists:
            return JSONResponse({'status': 'error', 'message': 'Không tìm thấy khiếu nại'}, status_code=404)

        complaint = complaint_doc.to_dict() or {}
        user_id = complaint.get('userId')
        violation_id = str(complaint.get('violationId') or '').strip()
        reviewed_at = datetime.utcnow().isoformat()

        if action == 'reject':
            complaint_ref.set(
                {'status': 'rejected', 'adminNote': admin_note, 'reviewedAt': reviewed_at},
                merge=True,
            )
            _sync_violation_complaint_lock(
                db=db,
                violation_id=violation_id,
                complaint_status='rejected',
                lock_payment=False,
                lock_complaint=False,
                reviewed_at=reviewed_at,
            )
            await broadcast_admin_event(
                'complaint_reviewed',
                {'complaintId': complaint_id, 'action': 'reject', 'userId': user_id},
            )
            return JSONResponse({'status': 'ok', 'message': 'Đã từ chối khiếu nại'})

        # approve
        complaint_ref.set(
            {'status': 'approved', 'adminNote': admin_note, 'reviewedAt': reviewed_at},
            merge=True,
        )

        resolved_violation_id = violation_id
        violation_data: Dict[str, Any] = {}

        if violation_id:
            violation_ref = _resolve_violation_ref_by_id(db, violation_id)
            if violation_ref:
                violation_doc = violation_ref.get()
                if violation_doc.exists:
                    resolved_violation_id = violation_doc.id
                    violation_data = violation_doc.to_dict() or {}
                    violation_ref.delete()

        if resolved_violation_id:
            _remove_violation_from_store(resolved_violation_id)
            _delete_notifications_for_violation(db, resolved_violation_id)

        restored_info = {'changed': False}
        if user_id:
            restored_info = _restore_points_from_violation(
                db=db,
                user_id=user_id,
                violation_data=violation_data,
            )

        if user_id:
            violation_name = violation_data.get('violationType') or complaint.get('reason') or 'vi phạm'
            body = (
                f'Khiếu nại của bạn đã được chấp nhận. Vi phạm "{violation_name}" đã được xóa '
                f'và không cần thanh toán.'
            )
            if restored_info.get('changed'):
                body += ' Điểm GPLX đã được hoàn lại.'
            create_user_notification(
                db=db,
                user_id=user_id,
                title='✅ Khiếu nại thành công',
                body=body,
                notif_type='complaint_success',
                violation_id=None,
            )

        await broadcast_admin_event(
            'complaint_reviewed',
            {
                'complaintId': complaint_id,
                'action': 'approve',
                'userId': user_id,
                'violationId': resolved_violation_id or violation_id,
            },
        )
        return JSONResponse({'status': 'ok', 'message': 'Đã chấp nhận khiếu nại và xóa lỗi vi phạm liên quan'})

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.delete("/api/admin/complaints/{complaint_id}")
async def delete_complaint(complaint_id: str):
    """
    Xóa khiếu nại đã xử lý xong (approved/rejected).
    Tránh xóa nhầm khiếu nại đang chờ duyệt.
    """
    try:
        if not (fcm_service and fcm_service.is_available):
            return JSONResponse({'status': 'error', 'message': 'Firebase not available'}, status_code=503)

        normalized_id = str(complaint_id or '').strip()
        if not normalized_id:
            return JSONResponse({'status': 'error', 'message': 'complaint_id không hợp lệ'}, status_code=400)

        db = fcm_service._db
        complaint_ref = db.collection('complaints').document(normalized_id)
        complaint_doc = complaint_ref.get()
        if not complaint_doc.exists:
            return JSONResponse({'status': 'error', 'message': 'Không tìm thấy khiếu nại'}, status_code=404)

        complaint = complaint_doc.to_dict() or {}
        status = str(complaint.get('status') or 'pending').strip().lower()
        if status not in ('approved', 'rejected'):
            return JSONResponse({
                'status': 'error',
                'message': 'Chỉ được xóa khiếu nại đã xử lý (đã duyệt hoặc đã từ chối)',
            }, status_code=400)

        # Best-effort cleanup evidence image from Firebase Storage if it exists.
        try:
            raw_evidence = _first_non_empty_complaint_evidence(complaint)
            blob_path = _extract_storage_blob_path(raw_evidence) if raw_evidence else ''
            if blob_path:
                from firebase_admin import storage as fb_storage
                bucket = fb_storage.bucket()
                blob = bucket.blob(blob_path)
                if blob.exists():
                    blob.delete()
        except Exception as cleanup_err:
            print(f"⚠️ Complaint evidence cleanup skipped for {normalized_id}: {cleanup_err}")

        complaint_ref.delete()
        await broadcast_admin_event(
            'complaint_deleted',
            {
                'complaintId': normalized_id,
                'userId': complaint.get('userId'),
                'status': status,
            },
        )
        return JSONResponse({'status': 'ok', 'message': 'Đã xóa khiếu nại khỏi hệ thống'})
    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.post("/api/admin/users/{user_id}/restore_points")
async def restore_user_points(user_id: str):
    """" 
    Phục hồi đầy đủ (12) điểm giấy phép lái xe cho người dùng. 
    """
    try:
        if not (fcm_service and fcm_service.is_available):
            return JSONResponse({'status': 'error', 'message': 'Firebase not available'})

        db = fcm_service._db
        user_ref = db.collection('users').document(user_id)
        doc = user_ref.get()
        if not doc.exists:
            return JSONResponse({'status': 'error', 'message': 'Không tìm thấy người dùng'}, status_code=404)

        user_ref.set({
            'points': MAX_LICENSE_POINTS,
            'motoPoints': MAX_LICENSE_POINTS,
            'carPoints': MAX_LICENSE_POINTS,
            'motoLicenseStatus': 'active',
            'carLicenseStatus': 'active',
        }, merge=True)
        await broadcast_admin_event(
            'user_points_restored',
            {'userId': user_id},
        )
        return JSONResponse({'status': 'ok', 'message': 'Đã phục hồi điểm GPLX xe máy và ô tô thành 12/12!'})

    except Exception as e:
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
        
        await broadcast_admin_event(
            'user_deleted',
            {'userId': user_id},
        )
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
                    v['paidAt'] = datetime.utcnow().isoformat()
                    
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
            violation_data = doc_ref.get().to_dict() or {}
            paid_at = datetime.utcnow().isoformat()
            doc_ref.update({
                'status': 'paid',
                'paidAt': paid_at,
            })
            update_local_store(found_doc_id)
            target_user_id = violation_data.get('userId')
            if target_user_id:
                try:
                    from firebase_admin import firestore as fb_firestore
                    violation_name = violation_data.get('violationType', 'vi phạm')
                    fine_amount = float(violation_data.get('fineAmount', 0))
                    notif_id = f'payment_paid_{target_user_id}_{found_doc_id}'
                    db.collection('notifications').document(notif_id).set({
                        'userId': target_user_id,
                        'title': '✅ Đã đóng phạt',
                        'body': f'Bạn đã thanh toán {fine_amount:,.0f}₫ cho {violation_name}',
                        'type': 'payment_paid',
                        'violationId': found_doc_id,
                        'isRead': False,
                        'createdAt': fb_firestore.SERVER_TIMESTAMP,
                    }, merge=True)
                except Exception as ne:
                    print(f"[Webhook] Không tạo được notification đã đóng phạt: {ne}")
            await broadcast_admin_event(
                'payment_status_changed',
                {
                    'violationId': found_doc_id,
                    'status': 'paid',
                    'userId': target_user_id,
                },
            )
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
                            vehicle_class=v.get('vehicleClass'),
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


@app.websocket("/ws/admin")
async def websocket_admin(websocket: WebSocket):
    """
    WebSocket endpoint for admin dashboard realtime updates.
    Server pushes admin_data_changed events whenever data mutations happen.
    """
    await websocket.accept()
    admin_clients.add(websocket)
    client_ip = websocket.client.host if websocket.client else "unknown"
    print(f"🖥️ Admin client connected: {client_ip} (total: {len(admin_clients)})")

    await websocket.send_text(json.dumps({
        "type": "connected",
        "channel": "admin",
        "message": "Realtime admin channel connected",
        "timestamp": datetime.utcnow().isoformat(),
    }))

    try:
        while True:
            # Keep-alive commands from dashboard (optional)
            data = await websocket.receive_text()
            msg = json.loads(data) if data else {}
            action = msg.get("action", "")
            if action == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "channel": "admin",
                    "timestamp": datetime.utcnow().isoformat(),
                }))
    except WebSocketDisconnect:
        print(f"🖥️ Admin client disconnected: {client_ip}")
    except Exception as e:
        print(f"🖥️ Admin websocket error: {e}")
    finally:
        admin_clients.discard(websocket)
        print(f"🖥️ Admin client session ended: {client_ip} (total: {len(admin_clients)})")


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
