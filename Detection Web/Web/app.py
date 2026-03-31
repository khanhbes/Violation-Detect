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
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import sys
import time
import socket
from urllib.parse import quote, unquote, urlparse, parse_qs

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form, Header, Query
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from firebase_admin import auth as fb_auth
from google.api_core.exceptions import ResourceExhausted
from google.cloud.firestore_v1.base_query import FieldFilter

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
SNAPSHOT_VIOLATION_TYPES = (
    'helmet',
    'redlight',
    'sidewalk',
    'wrong_way',
    'wrong_lane',
    'sign',
    'no_helmet',
)
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

for vtype in SNAPSHOT_VIOLATION_TYPES:
    (SNAPSHOT_DIR / vtype).mkdir(exist_ok=True)

# =============================================================================
# APP SETUP
# =============================================================================

def _run_startup_tasks() -> None:
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
                'port': 8000,
                'updatedBy': 'backend_startup',
                'updatedAt': fb_firestore.SERVER_TIMESTAMP,
                'updated_at': fb_firestore.SERVER_TIMESTAMP,
            }, merge=True)
            print(
                f"✅ Bật Firebase Auto-Discovery: "
                f"Cập nhật IP {local_ip}:8000 lên Firestore [server/config]"
            )
        else:
            print("⚠️ Firebase auto-discovery bị tắt (FCM/Firestore chưa sẵn sàng)")
    except Exception as e:
        print(f"⚠️ Failed to write Local IP to Firestore: {e}")

    # ── Startup Cleanup: snapshot files + runtime cache + stale sessions + apk retention ──
    try:
        from config.config import config as _cfg

        _snap_dir = _cfg.SNAPSHOT_DIR
        _deleted = 0
        for _pattern in ('*.jpg', '*.jpeg', '*.png'):
            for _img in _snap_dir.rglob(_pattern):
                try:
                    _img.unlink()
                    _deleted += 1
                except Exception:
                    pass

        if _deleted:
            print(f"🧹 Startup cleanup: đã xóa {_deleted} ảnh snapshot cũ từ {_snap_dir}")
        else:
            print("🧹 Startup cleanup: không có ảnh snapshot cũ cần xóa")
    except Exception as _ce:
        print(f"⚠️ Startup snapshot cleanup failed: {_ce}")

    # Reset in-memory runtime caches for a clean session.
    try:
        global violation_store, violation_counter, notified_violations
        violation_store.clear()
        violation_counter = 0
        notified_violations.clear()
        print("🧹 Startup cleanup: runtime caches reset (violation_store/dedupe)")
    except Exception as _ce:
        print(f"⚠️ Startup runtime cleanup failed: {_ce}")

    # Mark stale app sessions inactive so routing only targets live app users.
    try:
        if fcm_service and fcm_service.is_available and fcm_service._db:
            _stale_count = _mark_stale_sessions_inactive(fcm_service._db)
            if _stale_count:
                print(f"🧹 Startup cleanup: marked {_stale_count} stale app session(s) inactive")
    except Exception as _ce:
        print(f"⚠️ Startup session cleanup failed: {_ce}")

    # Keep only the latest few APK releases.
    try:
        _apk_result = _cleanup_old_apk_releases(keep_latest=3)
        if _apk_result.get('deleted_count', 0):
            print(
                f"🧹 Startup cleanup: removed {_apk_result['deleted_count']} old APK release(s), "
                f"kept {_apk_result['kept_count']}"
            )
    except Exception as _ce:
        print(f"⚠️ Startup APK cleanup failed: {_ce}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _run_startup_tasks()
    # Background task: push quota qua WS + tự động save file
    quota_task = asyncio.create_task(_quota_push_loop())
    yield
    # Shutdown: cancel background + flush quota ra file
    quota_task.cancel()
    _save_quota()
    print("📊 Quota saved on shutdown")


async def _quota_push_loop():
    """Mỗi 5s kiểm tra rollover + push quota update qua admin WS channel."""
    global _quota_last_push, _quota_dirty
    _prev_snapshot = ""
    while True:
        try:
            await asyncio.sleep(_QUOTA_PUSH_INTERVAL)
            # Rollover nếu sang ngày mới (dù không có operation nào)
            _quota_rollover()
            # Build snapshot
            snap = json.dumps({
                "reads": _firestore_ops.get("reads", 0),
                "writes": _firestore_ops.get("writes", 0),
                "deletes": _firestore_ops.get("deletes", 0),
            })
            # Chỉ push nếu có thay đổi
            if snap != _prev_snapshot and admin_clients:
                _prev_snapshot = snap
                uptime = time.time() - _firestore_ops.get("started_at", time.time())
                msg = json.dumps({
                    "type": "quota_update",
                    "reads": _firestore_ops.get("reads", 0),
                    "writes": _firestore_ops.get("writes", 0),
                    "deletes": _firestore_ops.get("deletes", 0),
                    "date": _firestore_ops.get("date", _today_str()),
                    "uptime_seconds": round(uptime, 1),
                    "quota_status": _quota_status(),
                    "limits": _DAILY_QUOTA_LIMITS,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                disconnected = set()
                for ws in admin_clients:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        disconnected.add(ws)
                admin_clients.difference_update(disconnected)
            # Debounced file save
            if _quota_dirty and time.time() - _quota_last_save >= _QUOTA_SAVE_INTERVAL:
                _save_quota()
                _quota_dirty = False
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ Quota push loop error: {e}")


app = FastAPI(
    title="Traffic Violation API",
    description="Backend for traffic violation detection.",
    version="4.0",
    lifespan=lifespan,
)
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

# ─── Firestore Quota Tracking (persistent, daily reset 00:00 GMT+7) ─────────
try:
    from zoneinfo import ZoneInfo
    _TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    _TZ_VN = timezone(timedelta(hours=7))  # fallback UTC+7
_QUOTA_FILE = Path(__file__).parent / "quota_data.json"
_quota_dirty = False          # True khi có thay đổi chưa ghi file
_quota_last_save = 0.0        # epoch lần save gần nhất
_QUOTA_SAVE_INTERVAL = 30     # giây — ghi file tối đa mỗi 30s
_quota_last_push = 0.0        # epoch lần push WS gần nhất
_QUOTA_PUSH_INTERVAL = 5      # giây — push WS tối đa mỗi 5s


def _today_str() -> str:
    """Ngày hôm nay theo Asia/Ho_Chi_Minh."""
    return datetime.now(_TZ_VN).strftime("%Y-%m-%d")


def _quota_rollover():
    """Kiểm tra nếu sang ngày mới (00:00 GMT+7) → reset counters, flush file."""
    global _quota_dirty
    today = _today_str()
    if _firestore_ops.get("date") == today:
        return  # cùng ngày, không cần reset
    old_date = _firestore_ops.get("date", "?")
    _firestore_ops["reads"] = 0
    _firestore_ops["writes"] = 0
    _firestore_ops["deletes"] = 0
    _firestore_ops["date"] = today
    _firestore_ops["started_at"] = time.time()
    _quota_dirty = True
    _save_quota()
    _quota_dirty = False
    print(f"📊 Quota daily reset at 00:00 GMT+7 (was {old_date}, now {today})")


def _load_quota() -> dict:
    """Load quota từ file. Reset nếu file hỏng hoặc ngày mới."""
    today = _today_str()
    default = {"reads": 0, "writes": 0, "deletes": 0,
               "date": today, "started_at": time.time()}
    if not _QUOTA_FILE.exists():
        return default
    try:
        with open(_QUOTA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Reset nếu sang ngày mới
        if data.get("date") != today:
            print(f"📊 Quota daily reset on load (was {data.get('date')}, now {today})")
            return default
        data.setdefault("started_at", time.time())
        return data
    except Exception as e:
        print(f"⚠️ Cannot load quota file: {e}")
        return default


def _save_quota():
    """Ghi quota ra file (debounced bởi caller)."""
    global _quota_last_save
    try:
        with open(_QUOTA_FILE, "w", encoding="utf-8") as f:
            json.dump(_firestore_ops, f)
        _quota_last_save = time.time()
    except Exception as e:
        print(f"⚠️ Cannot save quota file: {e}")


_firestore_ops = _load_quota()


def _track_fs(op: str, count: int = 1):
    """Track Firestore operations: op = 'reads' | 'writes' | 'deletes'"""
    global _quota_dirty
    _quota_rollover()  # reset nếu sang ngày mới
    _firestore_ops[op] = _firestore_ops.get(op, 0) + count
    _quota_dirty = True
    # Debounced save
    if time.time() - _quota_last_save >= _QUOTA_SAVE_INTERVAL:
        _save_quota()
        _quota_dirty = False

# ─── Violation Store (in-memory, for App sync) ─────────────────────────────
violation_store: List[Dict] = []
violation_counter = 0

# ─── Notified Violations Cache (to prevent duplicate alerts per vehicle) ────
# Key: (v_type, stable_track_id) — resets when server restarts
# Rule: 1 vehicle + 1 violation type = 1 notification per server session.
notified_violations: set = set()

# ─── Realtime detect session lock (only 1 session at a time) ───────────────
_realtime_busy = False

# ─── App WebSocket Clients (for real-time push) ────────────────────────────
app_clients: Dict[WebSocket, Dict[str, Any]] = {}
# ─── Admin WebSocket Clients (for realtime Data Management updates) ─────────
admin_clients: set = set()

APP_SESSION_COLLECTION = 'app_active_sessions'
APP_SESSION_TTL_SECONDS = 120
APP_SESSION_REQUIRE_AUTH = False
APP_SESSION_CLEANUP_BATCH_LIMIT = 150
APP_SESSION_CLEANUP_TIMEOUT_SECONDS = 20

# ─── FCM Push Notification Service ─────────────────────────────────────────
try:
    fcm_service = FCMService()
except Exception as e:
    print(f"⚠️ FCM service init failed: {e}")
    fcm_service = None

async def broadcast_to_apps(violation: Dict):
    """Push a violation to app clients.
    - If violation has ownerResolution='assigned' AND userId -> only send to that user's WS connections.
    - If violation has ownerResolution='pending_owner' (no assigned userId) -> do NOT push to app clients.
    - Unassigned violations are stored in Firestore only for admin review + manual assignment.
    """
    if not app_clients:
        return
    
    # Check owner resolution: chỉ push nếu đã assign cho user
    owner_resolution = str(violation.get("ownerResolution") or "").strip().lower()
    if owner_resolution != "assigned":
        print(f"📱 Skip app push for {owner_resolution} violation (no active owner assigned)")
        return
    
    target_user_id = str(violation.get("userId") or "").strip()
    if not target_user_id:
        print("📱 Skip app push for violation with no userId (inconsistent state)")
        return

    message = json.dumps({
        "type": "new_violation",
        "data": violation
    })
    disconnected = set()
    delivered = 0
    for ws, meta in list(app_clients.items()):
        client_user_id = str((meta or {}).get("user_id") or "").strip()
        # Strict scoped routing: only to the active signed-in owner.
        if client_user_id != target_user_id:
            continue
        try:
            await ws.send_text(message)
            delivered += 1
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        app_clients.pop(ws, None)
    if disconnected:
        print(f"📱 Cleaned {len(disconnected)} disconnected app client(s)")
    print(f"📱 Broadcast violation to {delivered} app client(s) (user={target_user_id}, resolution={owner_resolution})")

async def broadcast_admin_event(event: str, payload: Optional[Dict[str, Any]] = None):
    """Push data-change events to connected admin dashboard clients."""
    if not admin_clients:
        return
    # Invalidate admin data cache so next fetch hits Firestore
    _admin_data_cache["fetched_at"] = 0.0

    # Determine which scopes changed based on event name
    scope_hint = "all"
    if "violation" in event:
        scope_hint = "violations,notifications"
    elif "complaint" in event:
        scope_hint = "complaints"
    elif "user" in event or "profile" in event:
        scope_hint = "users,profile_updates"

    message = json.dumps({
        "type": "admin_data_changed",
        "event": event,
        "scope": scope_hint,
        "payload": payload or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    _track_fs('reads')
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
    docs = db.collection('notifications').where(
        filter=FieldFilter('violationId', '==', violation_id)
    ).stream()
    for doc in docs:
        doc.reference.delete()
        deleted += 1
    _track_fs('reads', max(deleted, 1))
    _track_fs('deletes', deleted)
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
    _track_fs('reads')
    if direct_doc.exists:
        return direct_ref

    # P0: Removed fallback full-collection scan to avoid N reads on ID mismatch.
    # If direct get() misses, the violation ID format is inconsistent — return None.
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
        'updatedAt': datetime.now(timezone.utc).isoformat(),
    }

    if reviewed_at:
        payload['complaintReviewedAt'] = reviewed_at

    if not lock_payment and current_status in ('complaint_pending', 'pending_payment', 'pending', ''):
        payload['status'] = 'pending'
    elif lock_payment:
        payload['status'] = 'complaint_pending'

    violation_ref.set(payload, merge=True)
    _track_fs('writes')
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
    subtitle: Optional[str] = None,
    detail: Optional[str] = None,
):
    """Create a notification document for a specific user."""
    from firebase_admin import firestore as fb_firestore

    payload = {
        'userId': user_id,
        'title': title,
        'body': body,
        'subtitle': subtitle or body,
        'detail': detail or body,
        'type': notif_type,
        'isRead': False,
        'createdAt': fb_firestore.SERVER_TIMESTAMP,
    }
    if violation_id:
        payload['violationId'] = violation_id

    db.collection('notifications').add(payload)
    _track_fs('writes')


def _list_all_firestore_user_doc_ids(db) -> List[str]:
    """Best-effort list of user document ids in Firestore."""
    user_ids: List[str] = []
    seen_user_ids = set()
    try:
        for user_doc in db.collection('users').stream():
            uid = (user_doc.id or '').strip()
            if not uid or uid in seen_user_ids:
                continue
            seen_user_ids.add(uid)
            user_ids.append(uid)
    except Exception as exc:
        print(f"⚠️ Could not fetch users from Firestore: {exc}")
    return user_ids


def _list_all_auth_user_ids() -> List[str]:
    """
    Prefer Firebase Auth UIDs as canonical account ids.
    This prevents routing by legacy Firestore doc ids (e.g. CCCD strings).
    """
    user_ids: List[str] = []
    seen_user_ids = set()
    try:
        page = fb_auth.list_users()
        while page:
            for user_record in page.users:
                uid = str(getattr(user_record, "uid", "") or "").strip()
                if not uid or uid in seen_user_ids:
                    continue
                seen_user_ids.add(uid)
                user_ids.append(uid)
            page = page.get_next_page()
    except Exception as exc:
        print(f"⚠️ Could not list Firebase Auth users for routing fallback: {exc}")
    return user_ids


def _list_all_user_ids_for_broadcast(db) -> List[str]:
    """
    Unified user listing for broadcast/fallback.
    1) Firebase Auth UIDs (canonical)
    2) Firestore users doc ids (legacy fallback)
    """
    auth_user_ids = _list_all_auth_user_ids()
    if auth_user_ids:
        return auth_user_ids
    return _list_all_firestore_user_doc_ids(db)


def broadcast_app_update_notification(version: str, changelog: str = '') -> int:
    """
    Broadcast update notifications to all users.
    Uses deterministic notification IDs to avoid duplicates per version.
    """
    if not (fcm_service and fcm_service.is_available and fcm_service._db):
        return 0

    from firebase_admin import firestore as fb_firestore

    db = fcm_service._db
    users = _list_all_user_ids_for_broadcast(db)
    normalized_version = str(version).replace('.', '_')
    for uid in users:
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


def _resolve_violation_target_user_ids(db) -> List[str]:
    """
    Route violation notifications to active app sessions.

    Policy (single_active_user):
    - Exactly 1 active user  → assign to them (ownerResolution='assigned').
    - 0 or >1 active users   → return [] so caller creates one pending_owner record.
    """
    ws_user_ids = _get_connected_app_user_ids()
    session_user_ids = _get_active_session_user_ids(db)
    all_active = sorted(set(ws_user_ids) | set(session_user_ids))
    if len(all_active) == 1:
        print(
            "🧭 Violation routing decision: "
            f"ws_active={len(ws_user_ids)} session_active={len(session_user_ids)} "
            f"→ single_active_user={all_active[0]}"
        )
        return all_active
    else:
        print(
            "🧭 Violation routing decision: "
            f"ws_active={len(ws_user_ids)} session_active={len(session_user_ids)} "
            f"→ pending_owner (active_count={len(all_active)})"
        )
        return []  # caller will use [None] to create one unassigned record


def _get_connected_app_user_ids() -> List[str]:
    user_ids = set()
    for meta in list(app_clients.values()):
        uid = str((meta or {}).get("user_id") or "").strip()
        if uid:
            user_ids.add(uid)
    return sorted(user_ids)


def _get_active_app_user_ids(db=None) -> List[str]:
    """
    Unified helper for debugging/routing visibility:
    merge websocket-connected users + fresh active-session users.
    """
    merged_user_ids = set(_get_connected_app_user_ids())
    resolved_db = db
    if resolved_db is None and fcm_service and fcm_service.is_available and fcm_service._db:
        resolved_db = fcm_service._db
    if resolved_db is not None:
        merged_user_ids.update(_get_active_session_user_ids(resolved_db))
    return sorted(merged_user_ids)


def store_violation(
    v_type: str,
    track_id: int,
    label: str,
    snapshot_path: str = None,
    vehicle_class: Optional[str] = None,
    stable_track_id: Optional[int] = None,
    raw_track_id: Optional[int] = None,
):
    """Save a violation to in-memory store, Firestore, and Firebase Storage.

    Deduplication rule per session:
    - 1 vehicle + 1 violation type = 1 notification/record.
    - Vehicle identity uses stable_track_id (fallback to track_id).
    """
    global violation_counter, notified_violations

    try:
        resolved_stable_track_id = int(
            stable_track_id if stable_track_id is not None else track_id
        )
    except Exception:
        resolved_stable_track_id = int(track_id or 0)
    try:
        resolved_raw_track_id = int(
            raw_track_id if raw_track_id is not None else track_id
        )
    except Exception:
        resolved_raw_track_id = int(track_id or 0)

    # ── Deduplication: stable vehicle ID + violation type ─────────────────────
    violation_key = (v_type, resolved_stable_track_id)
    if violation_key in notified_violations:
        return None  # Already processed — do not send duplicate notification
    notified_violations.add(violation_key)
    print(
        f"🆕 New violation [{v_type}] stable_id={resolved_stable_track_id} "
        f"raw_id={resolved_raw_track_id} — processing "
        f"(cache size: {len(notified_violations)})"
    )

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
    persisted_violations: List[Dict[str, Any]] = []

    default_vehicle_bucket = (
        _normalize_vehicle_bucket(vehicle_class)
        or (info or {}).get('defaultVehicle')
        or 'motorcycle'
    )
    default_penalty = _resolve_violation_penalty(info, default_vehicle_bucket)
    default_fine_amount = default_penalty['fine']
    default_deducted_points = default_penalty['deducted_points']
    default_severity = default_penalty['severity']
    default_vehicle_type_label = (
        'Xe máy' if default_vehicle_bucket == 'motorcycle' else 'Ô tô'
    )

    # ── Write violation to Firestore ────────────────────────────────
    if fcm_service and fcm_service.is_available:
        try:
            from firebase_admin import firestore as fb_firestore
            db = fcm_service._db
            if db:
                target_user_ids: List[Optional[str]] = _resolve_violation_target_user_ids(db)
                if target_user_ids:
                    print(
                        f"🎯 Violation routing target users: {len(target_user_ids)} account(s)"
                    )
                else:
                    # Keep one unassigned internal record if nobody is active.
                    target_user_ids = [None]
                    print(
                        "⚠️ No active app session found; creating unassigned internal violation only"
                    )

                for resolved_user_id in target_user_ids:
                    current_user_data: Dict[str, Any] = {}
                    if resolved_user_id:
                        try:
                            user_doc = db.collection('users').document(resolved_user_id).get()
                            _track_fs('reads')
                            current_user_data = user_doc.to_dict() or {}
                        except Exception as ue:
                            print(
                                f"⚠️ Could not fetch target user profile {resolved_user_id}: {ue}"
                            )

                    selected_vehicle_bucket = _select_violation_vehicle_bucket(
                        v_type=v_type,
                        label=label,
                        vehicle_class=vehicle_class,
                        user_data=current_user_data,
                    )
                    penalty = _resolve_violation_penalty(info, selected_vehicle_bucket)
                    fine_amount = penalty['fine']
                    deducted_points = penalty['deducted_points']
                    severity = penalty['severity']
                    vehicle_type_label = (
                        'Xe máy' if selected_vehicle_bucket == 'motorcycle' else 'Ô tô'
                    )

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
                        'trackId': resolved_stable_track_id,
                        'stableTrackId': resolved_stable_track_id,
                        'rawTrackId': resolved_raw_track_id,
                        'status': 'pending',
                        'licensePlate': 'Đang xác minh',
                        'paymentDueDate': payment_due_date.isoformat(),
                        'ownerResolution': 'pending_owner',
                    }
                    if resolved_user_id:
                        doc_data['userId'] = resolved_user_id
                        doc_data['ownerResolution'] = 'assigned'

                    doc_ref.set(doc_data)
                    _track_fs('writes')
                    firestore_doc_id = doc_ref.id
                    if target_user_id is None:
                        target_user_id = resolved_user_id
                    print(f"🔥 Firestore: violation saved (ID: {firestore_doc_id})")

                    violation_payload = {
                        'id': firestore_doc_id,
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
                        'trackId': resolved_stable_track_id,
                        'stableTrackId': resolved_stable_track_id,
                        'rawTrackId': resolved_raw_track_id,
                        'status': 'pending',
                        'licensePlate': 'Đang xác minh',
                        'paymentDueDate': payment_due_date.isoformat(),
                        'ownerResolution': 'pending_owner',  # Default: chưa xác định chủ sở hữu
                    }
                    if resolved_user_id:
                        violation_payload['userId'] = resolved_user_id
                        violation_payload['ownerResolution'] = 'assigned'  # Đã gán owner
                    persisted_violations.append(violation_payload)

                    if resolved_user_id:
                        try:
                            point_update_result = _adjust_user_license_points(
                                db=db,
                                user_id=resolved_user_id,
                                vehicle_bucket=selected_vehicle_bucket,
                                delta_points=-deducted_points,
                            )
                            due_str = payment_due_date.strftime('%d/%m/%Y')
                            create_user_notification(
                                db=db,
                                user_id=resolved_user_id,
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
                                user_id=resolved_user_id,
                                title='⏳ Hạn đóng phạt',
                                body=f'Vi phạm cần được thanh toán trước ngày {due_str}',
                                notif_type='payment_due',
                                violation_id=firestore_doc_id,
                            )
                            if point_update_result.get('became_disabled'):
                                create_user_notification(
                                    db=db,
                                    user_id=resolved_user_id,
                                    title='🚫 GPLX tạm vô hiệu',
                                    body=(
                                        f'GPLX {vehicle_type_label.lower()} đã về 0/12 điểm và tạm vô hiệu. '
                                        f'Chỉ admin web mới khôi phục được điểm.'
                                    ),
                                    notif_type='license_disabled',
                                    violation_id=firestore_doc_id,
                                )
                            print(
                                f"🔔 Notifications created for user {resolved_user_id}"
                            )
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

    if not persisted_violations:
        fallback_violation = {
            'id': firestore_doc_id or f'vio_{violation_counter:04d}',
            'type': v_type,
            'violationType': info['name'],
            'violationCode': info['code'],
            'description': f'{info["name"]} - {label}',
            'fineAmount': default_fine_amount,
            'deductedPoints': default_deducted_points,
            'severity': default_severity,
            'licenseTarget': default_vehicle_bucket,
            'vehicleType': default_vehicle_type_label,
            'vehicleClass': default_vehicle_bucket,
            'lawReference': info['law'],
            'timestamp': now.isoformat(),
            'location': 'Camera giám sát giao thông',
            'imageUrl': firebase_image_url,
            'trackId': resolved_stable_track_id,
            'stableTrackId': resolved_stable_track_id,
            'rawTrackId': resolved_raw_track_id,
            'status': 'pending',
            'licensePlate': 'Đang xác minh',
            'paymentDueDate': payment_due_date.isoformat(),
            'ownerResolution': 'pending_owner',
        }
        if target_user_id:
            fallback_violation['userId'] = target_user_id
            fallback_violation['ownerResolution'] = 'assigned'
        persisted_violations.append(fallback_violation)

    for violation in persisted_violations:
        violation_store.append(violation)
        print(f"📱 Violation stored: {info['name']} (ID: {violation['id']})")
        asyncio.ensure_future(broadcast_to_apps(violation))

    first_violation = persisted_violations[0]
    target_user_ids_for_event = sorted(
        {
            str(v.get('userId')).strip()
            for v in persisted_violations
            if str(v.get('userId') or '').strip()
        }
    )
    asyncio.ensure_future(broadcast_admin_event(
        "violation_created",
        {
            "violationId": first_violation['id'],
            "violationIds": [v['id'] for v in persisted_violations],
            "userId": first_violation.get('userId'),
            "userIds": target_user_ids_for_event,
            "stableTrackId": first_violation.get('stableTrackId'),
            "rawTrackId": first_violation.get('rawTrackId'),
            "vehicleType": first_violation.get('vehicleType'),
            "deductedPoints": first_violation.get('deductedPoints'),
            "recordCount": len(persisted_violations),
        },
    ))

    if fcm_service and fcm_service.is_available:
        try:
            pushed_user_ids = set()
            push_count = 0
            for violation in persisted_violations:
                fine_amount = int(violation.get('fineAmount') or default_fine_amount)
                deducted_points = int(
                    violation.get('deductedPoints') or default_deducted_points
                )
                payload = {
                    'route': '/violation-detail',
                    'violation_id': violation['id'],
                    'type': v_type,
                }
                resolved_user_id = (violation.get('userId') or '').strip()
                if resolved_user_id:
                    if resolved_user_id in pushed_user_ids:
                        continue
                    pushed_user_ids.add(resolved_user_id)
                    fcm_service.send_push_notification(
                        user_id=resolved_user_id,
                        title=f'🚨 {info["name"]}',
                        body=(
                            f'Mức phạt: {fine_amount:,}₫ • '
                            f'Trừ {deducted_points} điểm GPLX'
                        ),
                        data_payload=payload,
                    )
                    push_count += 1
                else:
                    # Active-sessions-only routing policy: no global fallback push.
                    print(f"ℹ️ Skip push for unassigned violation {violation.get('id')}")
            print(
                f"📣 Violation push dispatch complete: pushed_users={push_count} "
                f"records={len(persisted_violations)}"
            )
        except Exception as e:
            print(f"⚠️ FCM broadcast error: {e}")

    return first_violation


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


@app.get("/api/firestore-quota")
async def get_firestore_quota():
    """Return persistent Firestore operation counters (daily reset 00:00 GMT+7)."""
    _quota_rollover()  # đảm bảo fetch luôn thấy số đúng ngày
    uptime = time.time() - _firestore_ops.get("started_at", time.time())
    return JSONResponse({
        "reads": _firestore_ops.get("reads", 0),
        "writes": _firestore_ops.get("writes", 0),
        "deletes": _firestore_ops.get("deletes", 0),
        "date": _firestore_ops.get("date", _today_str()),
        "uptime_seconds": round(uptime, 1),
        "quota_status": _quota_status(),
        "limits": _DAILY_QUOTA_LIMITS,
    })


@app.get("/api/snapshots")
async def list_snapshots():
    snapshots = []
    for violation_type in SNAPSHOT_VIOLATION_TYPES:
        folder = SNAPSHOT_DIR / violation_type
        if folder.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png"):
                for f in folder.glob(pattern):
                    snapshots.append({
                        "type": violation_type,
                        "filename": f.name,
                        "path": f"/snapshots/{violation_type}/{f.name}",
                        "timestamp": f.stat().st_mtime
                    })
    
    snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
    return JSONResponse(snapshots[:50])

@app.delete("/api/snapshots/{violation_type}/{filename}")
async def delete_snapshot(violation_type: str, filename: str):
    """Delete a single snapshot image from the Recent Violations gallery."""
    allowed_types = set(SNAPSHOT_VIOLATION_TYPES)
    if violation_type not in allowed_types:
        return JSONResponse({"error": "Invalid violation type"}, status_code=400)
    # Basic path traversal guard
    safe_name = Path(filename).name
    if safe_name != filename or '/' in filename or '\\' in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    target = SNAPSHOT_DIR / violation_type / safe_name
    if not target.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    try:
        target.unlink()
        print(f"🗑️ Deleted snapshot: {violation_type}/{safe_name}")
        return JSONResponse({"success": True, "deleted": f"{violation_type}/{safe_name}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/snapshots")
async def delete_snapshots(type: str = Query(default="all")):
    """
    Bulk-delete snapshot files.
    Query:
    - type=all (default): delete all snapshot categories
    - type=<violation_type>: delete only one category
    """
    requested_type = (type or "all").strip().lower()
    allowed_types = set(SNAPSHOT_VIOLATION_TYPES)
    if requested_type != "all" and requested_type not in allowed_types:
        return JSONResponse({"error": "Invalid violation type"}, status_code=400)

    target_types = (
        list(SNAPSHOT_VIOLATION_TYPES)
        if requested_type == "all"
        else [requested_type]
    )
    deleted_count = 0

    for violation_type in target_types:
        folder = SNAPSHOT_DIR / violation_type
        if not folder.exists():
            continue
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for image_file in folder.glob(pattern):
                try:
                    image_file.unlink()
                    deleted_count += 1
                except Exception:
                    pass

    print(
        f"🗑️ Bulk snapshot delete: type={requested_type} "
        f"deleted={deleted_count}"
    )
    return JSONResponse(
        {
            "success": True,
            "type": requested_type,
            "deleted_count": deleted_count,
            "deleted_types": target_types,
        }
    )



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


class UserProfileBootstrapRequest(BaseModel):
    full_name: str
    phone: str = ""
    id_card: str = ""
    email: str = ""


class AppProfileUpdateRequest(BaseModel):
    user_id: str = ""
    data: Dict[str, Any] = {}

class FCMTokenRemoveRequest(BaseModel):
    fcm_token: str


class AppSessionUpsertRequest(BaseModel):
    user_id: str
    session_id: str
    device_id: str
    last_seen_at: str = ""


class AppSessionClearRequest(BaseModel):
    session_id: str = ""
    device_id: str = ""
    user_id: str = ""

def _extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    if not authorization_header:
        return None
    raw = authorization_header.strip()
    if not raw:
        return None
    if raw.lower().startswith('bearer '):
        return raw[7:].strip() or None
    return raw


def _normalize_utc_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    dt: Optional[datetime] = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_firestore_quota_error(exc: Exception) -> bool:
    if isinstance(exc, ResourceExhausted):
        return True
    raw = str(exc).lower()
    return (
        "quota exceeded" in raw
        or "resource_exhausted" in raw
        or "status code 429" in raw
        or "code=429" in raw
    )


def _resolve_session_user_id(
    requested_user_id: str,
    bearer_token: Optional[str] = None,
    query_token: Optional[str] = None,
) -> str:
    requested_uid = (requested_user_id or "").strip()
    token = (bearer_token or query_token or "").strip()

    if token:
        decoded = fb_auth.verify_id_token(token)
        token_uid = str((decoded or {}).get("uid") or "").strip()
        if not token_uid:
            raise ValueError("Token không hợp lệ (thiếu uid)")
        if requested_uid and requested_uid != token_uid:
            raise PermissionError("user_id không khớp với token xác thực")
        return token_uid

    if APP_SESSION_REQUIRE_AUTH:
        raise PermissionError("Thiếu token xác thực cho session")
    if not requested_uid:
        raise ValueError("Thiếu user_id")
    return requested_uid


_session_upsert_last: dict[str, float] = {}
_SESSION_UPSERT_THROTTLE_S = 25


def _upsert_app_active_session(
    db,
    *,
    user_id: str,
    session_id: str,
    device_id: str,
    connected_via_ws: bool = False,
    last_seen_at_client: str = "",
    force: bool = False,
) -> None:
    from firebase_admin import firestore as fb_firestore

    normalized_user_id = (user_id or "").strip()
    normalized_session_id = (session_id or "").strip()
    normalized_device_id = (device_id or "").strip()
    if not normalized_user_id or not normalized_session_id or not normalized_device_id:
        return

    # Throttle: skip write if last upsert for this session was < 25s ago
    now_ts = time.time()
    if not force:
        last_ts = _session_upsert_last.get(normalized_session_id, 0)
        if now_ts - last_ts < _SESSION_UPSERT_THROTTLE_S:
            return
    _session_upsert_last[normalized_session_id] = now_ts

    now_iso = datetime.now(timezone.utc).isoformat()
    ref = db.collection(APP_SESSION_COLLECTION).document(normalized_session_id)
    payload = {
        "userId": normalized_user_id,
        "sessionId": normalized_session_id,
        "deviceId": normalized_device_id,
        "isActive": True,
        "connectedViaWs": bool(connected_via_ws),
        "lastSeenAt": now_iso,
        "lastSeenAtClient": (last_seen_at_client or "").strip(),
        "updatedAt": fb_firestore.SERVER_TIMESTAMP,
    }
    # Skip get() before set() to save 1 Firestore read per heartbeat.
    # merge=True preserves createdAt if it already exists.
    payload["createdAt"] = fb_firestore.SERVER_TIMESTAMP
    ref.set(payload, merge=True)
    _track_fs('writes')


def _build_active_session_query(db, *, limit: Optional[int] = None):
    query = db.collection(APP_SESSION_COLLECTION).where(
        filter=FieldFilter("isActive", "==", True)
    )
    if limit and limit > 0:
        query = query.limit(limit)
    return query


def _clear_app_active_sessions(
    db,
    *,
    session_id: str = "",
    device_id: str = "",
    user_id: str = "",
) -> int:
    from firebase_admin import firestore as fb_firestore

    normalized_session_id = (session_id or "").strip()
    normalized_device_id = (device_id or "").strip()
    normalized_user_id = (user_id or "").strip()
    now_iso = datetime.now(timezone.utc).isoformat()

    refs = []
    if normalized_session_id:
        refs = [db.collection(APP_SESSION_COLLECTION).document(normalized_session_id)]
    else:
        query = db.collection(APP_SESSION_COLLECTION)
        if normalized_device_id:
            query = query.where(
                filter=FieldFilter("deviceId", "==", normalized_device_id)
            )
        if normalized_user_id:
            query = query.where(
                filter=FieldFilter("userId", "==", normalized_user_id)
            )
        if not normalized_device_id and not normalized_user_id:
            return 0
        try:
            refs = [
                doc.reference
                for doc in query
                .limit(APP_SESSION_CLEANUP_BATCH_LIMIT)
                .stream(timeout=APP_SESSION_CLEANUP_TIMEOUT_SECONDS)
            ]
        except Exception as exc:
            if _is_firestore_quota_error(exc):
                print("⚠️ Session clear skipped (Firestore quota exceeded).")
            else:
                print(f"⚠️ Could not query app sessions to clear: {exc}")
            return 0

    cleared = 0
    for ref in refs:
        try:
            ref.set(
                {
                    "isActive": False,
                    "connectedViaWs": False,
                    "lastSeenAt": now_iso,
                    "updatedAt": fb_firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            cleared += 1
        except Exception as exc:
            print(f"⚠️ Could not clear app session {ref.id}: {exc}")
    return cleared


def _get_active_session_user_ids(db) -> List[str]:
    threshold = datetime.now(timezone.utc) - timedelta(seconds=APP_SESSION_TTL_SECONDS)
    active_user_ids: List[str] = []
    seen_user_ids = set()
    stale_refs = []

    try:
        docs = _build_active_session_query(db).stream(
            timeout=APP_SESSION_CLEANUP_TIMEOUT_SECONDS
        )
        for doc in docs:
            data = doc.to_dict() or {}
            uid = str(data.get("userId") or "").strip()
            if not uid or uid in seen_user_ids:
                continue

            last_seen = _normalize_utc_datetime(data.get("lastSeenAt"))
            if not last_seen:
                last_seen = _normalize_utc_datetime(data.get("updatedAt"))

            if last_seen and last_seen >= threshold:
                seen_user_ids.add(uid)
                active_user_ids.append(uid)
            else:
                stale_refs.append(doc.reference)
        _track_fs('reads', len(seen_user_ids) + len(stale_refs))
    except Exception as exc:
        if _is_firestore_quota_error(exc):
            print("⚠️ Active session fetch skipped (Firestore quota exceeded).")
        else:
            print(f"⚠️ Could not fetch active app sessions: {exc}")
        return []

    # Best effort cleanup for stale sessions.
    for ref in stale_refs:
        try:
            ref.set({"isActive": False, "connectedViaWs": False}, merge=True)
        except Exception as exc:
            if _is_firestore_quota_error(exc):
                print("⚠️ Stale session cleanup paused (Firestore quota exceeded).")
                break

    return active_user_ids


def _mark_stale_sessions_inactive(db) -> int:
    """
    Force stale app sessions to inactive state.
    Safe startup cleanup only; does not delete Firestore business data.
    """
    threshold = datetime.now(timezone.utc) - timedelta(seconds=APP_SESSION_TTL_SECONDS)
    stale_count = 0
    try:
        docs = _build_active_session_query(
            db, limit=APP_SESSION_CLEANUP_BATCH_LIMIT
        ).stream(
            timeout=APP_SESSION_CLEANUP_TIMEOUT_SECONDS
        )
        for doc in docs:
            data = doc.to_dict() or {}
            last_seen = _normalize_utc_datetime(data.get("lastSeenAt"))
            if not last_seen:
                last_seen = _normalize_utc_datetime(data.get("updatedAt"))
            if last_seen and last_seen >= threshold:
                continue
            try:
                doc.reference.set(
                    {"isActive": False, "connectedViaWs": False},
                    merge=True,
                )
                stale_count += 1
            except Exception as inner_exc:
                if _is_firestore_quota_error(inner_exc):
                    print("⚠️ Stale session cleanup paused (Firestore quota exceeded).")
                    break
    except Exception as exc:
        if _is_firestore_quota_error(exc):
            print("⚠️ Startup session cleanup skipped (Firestore quota exceeded).")
        else:
            print(f"⚠️ Could not mark stale sessions inactive: {exc}")
    return stale_count


def _filter_violation_store_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Return violations for a given user.

    Always include:
    - Violations owned by this user (matched userId).
    - Violations with NO userId (global/unassigned violations that any admin/officer can see).

    If user_id is empty → return all violations.
    """
    normalized_uid = (user_id or "").strip()
    if not normalized_uid:
        return list(violation_store)
    return [
        violation for violation in violation_store
        if not str(violation.get("userId") or "").strip()          # Global (no owner)
        or str(violation.get("userId") or "").strip() == normalized_uid  # Owned by this user
    ]


def _is_internal_login_email(email: str) -> bool:
    normalized = (email or "").strip().lower()
    if not normalized or "@" not in normalized:
        return False
    local, domain = normalized.split("@", 1)
    return domain == "vnetraffic.vn" and local.isdigit()


def _extract_id_card_from_login_email(email: str) -> str:
    normalized = (email or "").strip().lower()
    if not _is_internal_login_email(normalized):
        return ""
    local = normalized.split("@", 1)[0]
    return local if local.isdigit() else ""


def _resolve_profile_email(*candidates: str) -> str:
    for candidate in candidates:
        normalized = (candidate or "").strip().lower()
        if normalized and not _is_internal_login_email(normalized):
            return normalized
    return ""


@app.post("/api/app/register-profile")
async def register_app_profile(
    req: UserProfileBootstrapRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Create/merge user profile in Firestore using Firebase Admin privileges.
    This endpoint is used by mobile app right after Firebase Auth register,
    so profile creation does not depend on client Firestore security rules.
    """
    if not (fcm_service and fcm_service.is_available and fcm_service._db):
        return JSONResponse(
            {
                "status": "error",
                "message": "Firebase Admin/Firestore chưa sẵn sàng trên server",
            },
            status_code=503,
        )

    id_token = _extract_bearer_token(authorization)
    if not id_token:
        return JSONResponse(
            {"status": "error", "message": "Thiếu Authorization Bearer token"},
            status_code=401,
        )

    try:
        from firebase_admin import firestore as fb_firestore

        decoded = fb_auth.verify_id_token(id_token)
        uid = (decoded or {}).get("uid")
        if not uid:
            return JSONResponse(
                {"status": "error", "message": "Token không hợp lệ (thiếu uid)"},
                status_code=401,
            )

        token_email = ((decoded or {}).get("email") or "").strip().lower()
        req_email = (req.email or "").strip().lower()
        token_is_internal = _is_internal_login_email(token_email)
        req_is_internal = _is_internal_login_email(req_email)
        if (
            req_email
            and token_email
            and not token_is_internal
            and not req_is_internal
            and req_email != token_email
        ):
            return JSONResponse(
                {"status": "error", "message": "Email không khớp với token xác thực"},
                status_code=403,
            )

        full_name = (req.full_name or "").strip()
        if not full_name:
            return JSONResponse(
                {"status": "error", "message": "Họ và tên không được để trống"},
                status_code=422,
            )

        id_card = (req.id_card or "").strip()
        if not id_card:
            id_card = _extract_id_card_from_login_email(token_email)

        db = fcm_service._db
        user_ref = db.collection("users").document(uid)
        existing_doc = user_ref.get()
        existing_data = existing_doc.to_dict() if existing_doc.exists else {}

        raw_driver_licenses = existing_data.get("driverLicenses")
        driver_licenses = (
            raw_driver_licenses if isinstance(raw_driver_licenses, list) else []
        )

        try:
            moto_points = int(existing_data.get("motoPoints", MAX_LICENSE_POINTS))
        except Exception:
            moto_points = MAX_LICENSE_POINTS
        try:
            car_points = int(existing_data.get("carPoints", MAX_LICENSE_POINTS))
        except Exception:
            car_points = MAX_LICENSE_POINTS

        existing_email = (existing_data.get("email") or "").strip().lower()
        profile_email = _resolve_profile_email(req_email, token_email, existing_email)

        payload = {
            "fullName": full_name,
            "email": profile_email,
            "phone": (req.phone or "").strip(),
            "avatar": existing_data.get("avatar", ""),
            "idCard": id_card,
            "idCardIssueDate": existing_data.get("idCardIssueDate", ""),
            "idCardExpiryDate": existing_data.get("idCardExpiryDate", ""),
            "dateOfBirth": existing_data.get("dateOfBirth", ""),
            "gender": existing_data.get("gender", ""),
            "nationality": existing_data.get("nationality", ""),
            "placeOfOrigin": existing_data.get("placeOfOrigin", ""),
            "occupation": existing_data.get("occupation", ""),
            "address": existing_data.get("address", ""),
            "driverLicenses": driver_licenses,
            "licenseNumber": existing_data.get("licenseNumber", ""),
            "motoLicenseClass": existing_data.get("motoLicenseClass", ""),
            "carLicenseClass": existing_data.get("carLicenseClass", ""),
            "licenseIssueDate": existing_data.get("licenseIssueDate", ""),
            "licenseExpiryDate": existing_data.get("licenseExpiryDate", ""),
            "licenseIssuedBy": existing_data.get("licenseIssuedBy", ""),
            "motoPoints": max(0, min(MAX_LICENSE_POINTS, moto_points)),
            "carPoints": max(0, min(MAX_LICENSE_POINTS, car_points)),
            "points": max(0, min(MAX_LICENSE_POINTS, min(moto_points, car_points))),
            "updatedAt": fb_firestore.SERVER_TIMESTAMP,
        }
        if not existing_doc.exists:
            payload["createdAt"] = fb_firestore.SERVER_TIMESTAMP

        user_ref.set(payload, merge=True)
        await broadcast_admin_event("user_profile_bootstrapped", {"userId": uid})
        return JSONResponse({"status": "ok", "uid": uid})

    except fb_auth.ExpiredIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token đã hết hạn, vui lòng đăng nhập lại"},
            status_code=401,
        )
    except fb_auth.RevokedIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token đã bị thu hồi, vui lòng đăng nhập lại"},
            status_code=401,
        )
    except fb_auth.InvalidIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token Firebase không hợp lệ"},
            status_code=401,
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"Không thể tạo hồ sơ người dùng: {e}"},
            status_code=500,
        )


@app.post("/api/app/profile-update-request")
async def create_app_profile_update_request(
    req: AppProfileUpdateRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Create or update a profile update request using Firebase Admin privileges.
    This avoids client-side Firestore permission issues for end users.
    """
    if not (fcm_service and fcm_service.is_available and fcm_service._db):
        return JSONResponse(
            {
                "status": "error",
                "message": "Firebase Admin/Firestore chưa sẵn sàng trên server",
            },
            status_code=503,
        )

    id_token = _extract_bearer_token(authorization)
    if not id_token:
        return JSONResponse(
            {"status": "error", "message": "Thiếu Authorization Bearer token"},
            status_code=401,
        )

    try:
        from firebase_admin import firestore as fb_firestore

        decoded = fb_auth.verify_id_token(id_token)
        uid = (decoded or {}).get("uid")
        if not uid:
            return JSONResponse(
                {"status": "error", "message": "Token không hợp lệ (thiếu uid)"},
                status_code=401,
            )

        requested_uid = (req.user_id or "").strip()
        if requested_uid and requested_uid != uid:
            return JSONResponse(
                {"status": "error", "message": "Không thể gửi yêu cầu cho tài khoản khác"},
                status_code=403,
            )

        incoming_data = req.data if isinstance(req.data, dict) else {}
        reserved_keys = {
            "userId",
            "status",
            "createdAt",
            "updatedAt",
            "reviewedAt",
            "reviewedBy",
            "source",
        }

        cleaned_data = {}
        for key, value in incoming_data.items():
            normalized_key = str(key).strip()
            if not normalized_key or normalized_key in reserved_keys:
                continue
            cleaned_data[normalized_key] = value

        if not cleaned_data:
            return JSONResponse(
                {"status": "error", "message": "Không có dữ liệu thay đổi hợp lệ"},
                status_code=422,
            )

        db = fcm_service._db
        request_ref = db.collection("profile_update_requests").document(uid)
        request_doc = request_ref.get()

        payload = {
            **cleaned_data,
            "userId": uid,
            "status": "pending",
            "source": "app",
            "updatedAt": fb_firestore.SERVER_TIMESTAMP,
        }
        if not request_doc.exists:
            payload["createdAt"] = fb_firestore.SERVER_TIMESTAMP

        request_ref.set(payload, merge=True)
        await broadcast_admin_event("profile_update_requested", {"userId": uid})
        return JSONResponse({"status": "ok", "userId": uid})

    except fb_auth.ExpiredIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token đã hết hạn, vui lòng đăng nhập lại"},
            status_code=401,
        )
    except fb_auth.RevokedIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token đã bị thu hồi, vui lòng đăng nhập lại"},
            status_code=401,
        )
    except fb_auth.InvalidIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token Firebase không hợp lệ"},
            status_code=401,
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"Không thể gửi yêu cầu chỉnh sửa: {e}"},
            status_code=500,
        )


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


@app.post("/api/app/session/upsert")
async def upsert_app_session(
    req: AppSessionUpsertRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Upsert app active session heartbeat for violation routing.
    """
    if not (fcm_service and fcm_service.is_available and fcm_service._db):
        return JSONResponse(
            {"status": "error", "message": "Firebase Admin/Firestore chưa sẵn sàng"},
            status_code=503,
        )

    requested_user_id = (req.user_id or "").strip()
    session_id = (req.session_id or "").strip()
    device_id = (req.device_id or "").strip()
    if not requested_user_id or not session_id or not device_id:
        return JSONResponse(
            {
                "status": "error",
                "message": "Thiếu user_id, session_id hoặc device_id",
            },
            status_code=422,
        )

    bearer_token = _extract_bearer_token(authorization)
    try:
        resolved_uid = _resolve_session_user_id(
            requested_user_id=requested_user_id,
            bearer_token=bearer_token,
        )
        _upsert_app_active_session(
            fcm_service._db,
            user_id=resolved_uid,
            session_id=session_id,
            device_id=device_id,
            connected_via_ws=False,
            last_seen_at_client=req.last_seen_at,
        )
        return JSONResponse(
            {
                "status": "ok",
                "user_id": resolved_uid,
                "session_id": session_id,
                "server_time": datetime.now(timezone.utc).isoformat(),
            }
        )
    except fb_auth.ExpiredIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token đã hết hạn, vui lòng đăng nhập lại"},
            status_code=401,
        )
    except fb_auth.RevokedIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token đã bị thu hồi, vui lòng đăng nhập lại"},
            status_code=401,
        )
    except fb_auth.InvalidIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token Firebase không hợp lệ"},
            status_code=401,
        )
    except PermissionError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=403)
    except ValueError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Không thể cập nhật session: {exc}"},
            status_code=500,
        )


@app.post("/api/app/session/clear")
async def clear_app_session(
    req: AppSessionClearRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Clear app session state by session_id or (device_id + user_id).
    """
    if not (fcm_service and fcm_service.is_available and fcm_service._db):
        return JSONResponse(
            {"status": "error", "message": "Firebase Admin/Firestore chưa sẵn sàng"},
            status_code=503,
        )

    session_id = (req.session_id or "").strip()
    device_id = (req.device_id or "").strip()
    requested_user_id = (req.user_id or "").strip()
    if not session_id and not device_id:
        return JSONResponse(
            {"status": "error", "message": "Thiếu session_id hoặc device_id"},
            status_code=422,
        )
    if device_id and not session_id and not requested_user_id:
        return JSONResponse(
            {"status": "error", "message": "Thiếu user_id khi clear theo device_id"},
            status_code=422,
        )

    bearer_token = _extract_bearer_token(authorization)
    try:
        resolved_uid = requested_user_id
        if requested_user_id or bearer_token or APP_SESSION_REQUIRE_AUTH:
            resolved_uid = _resolve_session_user_id(
                requested_user_id=requested_user_id,
                bearer_token=bearer_token,
            )

        cleared = _clear_app_active_sessions(
            fcm_service._db,
            session_id=session_id,
            device_id=device_id,
            user_id=resolved_uid,
        )
        return JSONResponse({"status": "ok", "cleared": cleared})
    except fb_auth.ExpiredIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token đã hết hạn, vui lòng đăng nhập lại"},
            status_code=401,
        )
    except fb_auth.RevokedIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token đã bị thu hồi, vui lòng đăng nhập lại"},
            status_code=401,
        )
    except fb_auth.InvalidIdTokenError:
        return JSONResponse(
            {"status": "error", "message": "Token Firebase không hợp lệ"},
            status_code=401,
        )
    except PermissionError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=403)
    except ValueError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Không thể clear session: {exc}"},
            status_code=500,
        )


# =============================================================================
# SERVER INFO API (show IP for mobile app connection)
# =============================================================================

@app.get("/api/server-info")
async def get_server_info():
    """Return server's local IP addresses and port for mobile app connection."""
    import socket
    ips = []
    preferred_ip = None
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

    # Determine preferred outbound IP via UDP trick.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        preferred_ip = s.getsockname()[0]
        s.close()
    except Exception:
        preferred_ip = None

    # Keep preferred IP first so web display matches backend auto-discovery.
    if preferred_ip and not preferred_ip.startswith("127."):
        if preferred_ip in ips:
            ips = [preferred_ip] + [ip for ip in ips if ip != preferred_ip]
        else:
            ips.insert(0, preferred_ip)

    if not ips:
        ips.append("127.0.0.1")

    return JSONResponse({
        "ips": ips,
        "preferred_ip": preferred_ip or ips[0],
        "port": 8000,
        "hostname": socket.gethostname(),
        "ws_path": "/ws/app",
    })


# =============================================================================
# APP SYNC API (for Flutter mobile app)
# =============================================================================

@app.get("/api/app/violations")
async def get_app_violations(since: str = None, user_id: str = None):
    """
    Get violations for mobile app.
    
    Data source priority:
    1. Runtime cache (violation_store): realtime violations from detection
    2. Firestore fallback: persistent history, used after server restart
    
    Rules:
    - Only return violations with ownerResolution='assigned' and userId
    - If 'since' provided: filter by createdAt > since (incremental sync)
    - If user_id provided: filter by userId = user_id
    """
    try:
        db = fcm_service._db if (fcm_service and fcm_service.is_available) else None
        normalized_uid = (user_id or "").strip()
        
        # 1. Get violations from runtime cache
        scoped_violations = _filter_violation_store_for_user(normalized_uid)
        
        # 2. Filter for assigned violations only (ownerResolution='assigned')
        assigned_violations = [
            v for v in scoped_violations
            if str(v.get('ownerResolution') or '').strip().lower() == 'assigned'
            and str(v.get('userId') or '').strip() == normalized_uid  # Double-check userId match
        ]
        
        # 3. If cache is empty or insufficient, fallback to Firestore
        use_firestore_fallback = len(assigned_violations) == 0 and db is not None
        
        if use_firestore_fallback and normalized_uid:
            try:
                # Query Firestore for violations assigned to this user
                query = db.collection('violations').where(
                    filter=FieldFilter('userId', '==', normalized_uid)
                )
                fs_docs = list(query.stream())
                _track_fs('reads', max(len(fs_docs), 1))
                
                assigned_violations = []
                for doc in fs_docs:
                    v_data = doc.to_dict() or {}
                    owner_res = str(v_data.get('ownerResolution') or '').strip().lower()
                    # Include 'assigned' records AND legacy records (userId matches but field absent)
                    if owner_res == 'assigned' or owner_res == '':
                        v_data['id'] = doc.id
                        assigned_violations.append(v_data)
                print(f"📱 Fallback to Firestore: fetched {len(assigned_violations)} violations for user={normalized_uid}")
            except Exception as fs_err:
                print(f"⚠️ Firestore fallback error: {fs_err}")
                # Continue with cache-only results
        
        # 4. Apply 'since' filter if provided
        if since:
            try:
                since_dt = datetime.fromisoformat(str(since).replace('Z', '+00:00'))
                filtered = [
                    v for v in assigned_violations
                    if datetime.fromisoformat(str(v.get('timestamp') or v.get('createdAt', '')).replace('Z', '+00:00')) > since_dt
                ]
                return JSONResponse({
                    'violations': filtered,
                    'total': len(filtered),
                    'fallback_used': use_firestore_fallback,
                })
            except ValueError:
                pass
        
        return JSONResponse({
            'violations': assigned_violations,
            'total': len(assigned_violations),
            'fallback_used': use_firestore_fallback,
        })
    except Exception as _e:
        return JSONResponse(
            {'status': 'error', 'error_type': 'server_error', 'message': f'Khong the lay danh sach vi pham: {_e}'},
            status_code=500,
        )


@app.post("/api/admin/violations/backfill-unassigned")
async def admin_backfill_unassigned_violations(request: Request):
    """
    Admin API: Backfill unassigned violations from the last N hours.

    Finds Firestore violations where ownerResolution='pending_owner' (or missing)
    and no userId. If exactly 1 active app user, assigns them to that user.

    Request body (optional JSON):
      { "hours": 24, "dry_run": false }

    Returns:
      { "found": N, "assigned": N, "skipped": N, "dry_run": bool,
        "active_user": "uid or null" }
    """
    try:
        db = fcm_service._db if (fcm_service and fcm_service.is_available) else None
        if not db:
            return JSONResponse({'error': 'Firestore not available'}, status_code=503)

        body: Dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            pass

        hours = int(body.get('hours') or 24)
        dry_run = bool(body.get('dry_run') or False)
        since_dt = datetime.now() - timedelta(hours=hours)

        # Determine active user (single_active_user policy)
        active_users = _get_active_app_user_ids(db)
        single_active_user: Optional[str] = active_users[0] if len(active_users) == 1 else None

        # Query unassigned violations
        from firebase_admin import firestore as fb_firestore
        query = db.collection('violations').where(
            filter=FieldFilter('ownerResolution', '==', 'pending_owner')
        )
        docs = list(query.stream())
        _track_fs('reads', max(len(docs), 1))

        found = len(docs)
        assigned_count = 0
        skipped_count = 0

        for doc in docs:
            v_data = doc.to_dict() or {}
            # Skip if already has userId
            if str(v_data.get('userId') or '').strip():
                skipped_count += 1
                continue

            # Check timestamp within window
            ts_raw = str(v_data.get('timestamp') or v_data.get('createdAt') or '')
            try:
                ts = datetime.fromisoformat(ts_raw.replace('Z', '+00:00')).replace(tzinfo=None)
                if ts < since_dt:
                    skipped_count += 1
                    continue
            except Exception:
                skipped_count += 1
                continue

            if single_active_user and not dry_run:
                doc.reference.set(
                    {'userId': single_active_user, 'ownerResolution': 'assigned'},
                    merge=True,
                )
                _track_fs('writes')
                assigned_count += 1
            elif dry_run:
                assigned_count += 1  # Counts how many *would* be assigned
            else:
                skipped_count += 1  # No active user to assign

        print(
            f"🔧 Backfill unassigned: found={found} assigned={assigned_count} "
            f"skipped={skipped_count} dry_run={dry_run} active_user={single_active_user}"
        )
        return JSONResponse({
            'found': found,
            'assigned': assigned_count,
            'skipped': skipped_count,
            'dry_run': dry_run,
            'active_user': single_active_user,
        })
    except Exception as e:
        return JSONResponse(
            {'error': f'Backfill failed: {e}'},
            status_code=500,
        )


@app.post("/api/admin/violations/{violation_id}/assign-user")
async def admin_assign_violation_user(violation_id: str, request: Request):
    """
    Admin API: Assign a violation to a user (convert pending_owner -> assigned).
    
    Request body: {"user_id": "user_uid"}
    
    Behavior:
    - Find violation by ID in Firestore
    - If ownerResolution='pending_owner': set userId, ownerResolution='assigned'
    - Apply point deduction to user license
    - Send notification to user
    - Broadcast to app client if online
    """
    try:
        db = fcm_service._db if (fcm_service and fcm_service.is_available) else None
        if not db:
            return JSONResponse(
                {'error': 'Firestore not available'},
                status_code=503,
            )
        
        body = await request.json()
        target_user_id = str(body.get('user_id') or '').strip()
        
        if not target_user_id:
            return JSONResponse(
                {'error': 'user_id required'},
                status_code=400,
            )
        
        violation_ref = _resolve_violation_ref_by_id(db, violation_id)
        if not violation_ref:
            return JSONResponse(
                {'error': 'Violation not found'},
                status_code=404,
            )
        
        doc = violation_ref.get()
        if not doc.exists:
            return JSONResponse(
                {'error': 'Violation document not found'},
                status_code=404,
            )
        
        violation_data = doc.to_dict() or {}
        current_owner_resolution = str(violation_data.get('ownerResolution') or '').strip().lower()
        
        # Only allow assign if currently pending_owner
        if current_owner_resolution != 'pending_owner':
            return JSONResponse(
                {'error': f'Violation already has ownerResolution={current_owner_resolution}, cannot reassign'},
                status_code=409,
            )
        
        # Update Firestore: set userId and ownerResolution
        from firebase_admin import firestore as fb_firestore
        violation_ref.set({
            'userId': target_user_id,
            'ownerResolution': 'assigned',
            'assignedAt': fb_firestore.SERVER_TIMESTAMP,
            'updatedAt': fb_firestore.SERVER_TIMESTAMP,
        }, merge=True)
        _track_fs('writes')
        
        # Apply points + create notification
        v_type = str(violation_data.get('type') or '').strip()
        deducted_points = int(violation_data.get('deductedPoints') or 0)
        vehicle_bucket = (
            _normalize_vehicle_bucket(violation_data.get('licenseTarget'))
            or _normalize_vehicle_bucket(violation_data.get('vehicleType'))
            or 'motorcycle'
        )
        
        if deducted_points > 0:
            try:
                point_result = _adjust_user_license_points(
                    db=db,
                    user_id=target_user_id,
                    vehicle_bucket=vehicle_bucket,
                    delta_points=-deducted_points,
                )
                print(f"🔔 Points deducted: {point_result}")
            except Exception as pe:
                print(f"⚠️ Point adjustment failed: {pe}")
        
        # Create notifications
        info = VIOLATION_INFO.get(v_type) or {'name': 'Vi phạm giao thông'}
        try:
            create_user_notification(
                db=db,
                user_id=target_user_id,
                title=f'🚨 {info["name"]}',
                body=(
                    f'Mức phạt: {violation_data.get("fineAmount", 0):,}₫ • '
                    f'Trừ {deducted_points} điểm GPLX'
                ),
                notif_type='violation',
                violation_id=violation_id,
            )
        except Exception as ne:
            print(f"⚠️ Notification creation failed: {ne}")
        
        # Update runtime cache if exists
        for i, v in enumerate(violation_store):
            if v.get('id') == violation_id:
                violation_store[i].update({
                    'userId': target_user_id,
                    'ownerResolution': 'assigned',
                })
                break
        
        return JSONResponse({
            'status': 'ok',
            'message': f'Violation assigned to user {target_user_id}',
            'violation_id': violation_id,
            'user_id': target_user_id,
            'ownerResolution': 'assigned',
        })
    except Exception as e:
        print(f"⚠️ Admin assign violation error: {e}")
        return JSONResponse(
            {'error': f'Failed to assign violation: {e}'},
            status_code=500,
        )


@app.get("/api/app/violations/{violation_id}")
async def get_app_violation_detail(violation_id: str):
    """Get single violation detail for mobile app."""
    for v in violation_store:
        if v['id'] == violation_id:
            return JSONResponse(v)
    return JSONResponse({'error': 'Violation not found', 'error_type': 'not_found'}, status_code=404)


@app.get("/api/app/stats")
async def get_app_stats(user_id: str = None):
    """Get violation statistics for mobile app dashboard."""
    try:
        scoped_violations = _filter_violation_store_for_user(user_id or "")
        total = len(scoped_violations)
        pending = len([v for v in scoped_violations if v['status'] == 'pending'])
        total_fines = sum(v['fineAmount'] for v in scoped_violations if v['status'] == 'pending')
        by_type = {}
        for v in scoped_violations:
            t = v['type']
            by_type[t] = by_type.get(t, 0) + 1
        return JSONResponse({
            'total': total,
            'pending': pending,
            'paid': total - pending,
            'totalFines': total_fines,
            'byType': by_type,
        })
    except Exception as _e:
        return JSONResponse(
            {'status': 'error', 'error_type': 'server_error', 'message': f'Khong the lay thong ke: {_e}'},
            status_code=500,
        )


# =============================================================================
# APP OTA UPDATE API
# =============================================================================

# Directory to store APK releases
APK_RELEASE_DIR = Path(__file__).parent / "apk_releases"
APK_RELEASE_DIR.mkdir(exist_ok=True)

# In-memory latest version info (also persisted to JSON file)
VERSION_FILE = APK_RELEASE_DIR / "latest_version.json"


def _cleanup_old_apk_releases(keep_latest: int = 3) -> Dict[str, Any]:
    """
    Safe cleanup for APK releases:
    - keep latest N release APKs by modified time
    - remove older release APKs only
    """
    keep_n = max(1, int(keep_latest or 1))
    apk_files = sorted(
        APK_RELEASE_DIR.glob("app-release-v*.apk"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    to_keep = apk_files[:keep_n]
    to_delete = apk_files[keep_n:]
    deleted_files = []

    for apk_file in to_delete:
        try:
            apk_file.unlink()
            deleted_files.append(apk_file.name)
        except Exception:
            pass

    return {
        "deleted_count": len(deleted_files),
        "deleted_files": deleted_files,
        "kept_count": len(to_keep),
        "kept_files": [p.name for p in to_keep],
    }

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

# ─── Admin Data Cache (P1: 30s TTL to avoid redundant full-collection reads) ──
_admin_data_cache: Dict[str, Any] = {"data": None, "fetched_at": 0.0}
_ADMIN_CACHE_TTL = 30  # seconds

# Mapping from scope aliases to Firestore collection names
_ADMIN_SCOPE_MAP = {
    "users":           "users",
    "vehicles":        "vehicles",
    "violations":      "violations",
    "notifications":   "notifications",
    "complaints":      "complaints",
    "profile_updates": "profile_update_requests",
}
_ALL_ADMIN_SCOPES = list(_ADMIN_SCOPE_MAP.keys())

# ─── Quota Protection Thresholds (P2) ──────────────────────────────────────
_DAILY_QUOTA_LIMITS = {"reads": 50_000, "writes": 20_000, "deletes": 20_000}
_QUOTA_WARN_RATIO    = 0.80   # 80 % → start returning cache more aggressively
_QUOTA_HARD_RATIO    = 0.95   # 95 % → block non-critical reads entirely

def _quota_status() -> str:
    """Return 'ok' | 'warn' | 'critical' based on estimated daily reads."""
    reads = _firestore_ops.get("reads", 0)
    limit = _DAILY_QUOTA_LIMITS["reads"]
    if reads >= limit * _QUOTA_HARD_RATIO:
        return "critical"
    if reads >= limit * _QUOTA_WARN_RATIO:
        return "warn"
    return "ok"


@app.get("/api/admin/data")
async def get_admin_data(
    force: int = Query(default=0),
    scope: Optional[str] = Query(default=None),
    limit: int = Query(default=0),
):
    """
    Lấy thông tin từ Firestore, phục vụ Quản lý dữ liệu trên Web.
    Query params:
      - force=1  → bypass cache
      - scope=violations,complaints  → chỉ fetch collections được liệt kê (CSV)
      - limit=N  → giới hạn số docs mỗi collection (0 = không giới hạn)
    """
    try:
        if not (fcm_service and fcm_service.is_available):
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

        # ── Determine which scopes to fetch ──────────────────────────
        requested_scopes = _ALL_ADMIN_SCOPES
        if scope:
            parts = [s.strip().lower() for s in scope.split(",") if s.strip()]
            valid = [s for s in parts if s in _ADMIN_SCOPE_MAP]
            if valid:
                requested_scopes = valid
        is_partial = set(requested_scopes) != set(_ALL_ADMIN_SCOPES)

        # ── Cache hit? (only for full-scope, non-forced requests) ────
        qstatus = _quota_status()
        now = time.time()
        cache_valid = (
            _admin_data_cache["data"] is not None
            and (now - _admin_data_cache["fetched_at"]) < _ADMIN_CACHE_TTL
        )
        if not is_partial and not force and cache_valid:
            return JSONResponse({'status': 'ok', 'data': _admin_data_cache["data"], 'cached': True})

        # ── Quota protection: critical → always return cache if any ──
        if qstatus == "critical" and _admin_data_cache["data"] is not None:
            print("⚠️ Quota critical — returning stale admin cache")
            return JSONResponse({'status': 'ok', 'data': _admin_data_cache["data"], 'cached': True, 'quota_status': 'critical'})

        from firebase_admin import firestore as fb_firestore
        db = fcm_service._db

        page_limit = max(0, limit)

        def get_collection(col_name, max_docs: int = 0):
            query = db.collection(col_name)
            if max_docs > 0:
                query = query.limit(max_docs)
            docs = query.stream()
            res = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                for k, v in data.items():
                    if hasattr(v, 'timestamp'):
                        data[k] = v.timestamp()
                res.append(data)
            _track_fs('reads', max(len(res), 1))
            return res

        # ── If partial scope, merge into existing cache ──────────────
        if is_partial and _admin_data_cache["data"]:
            merged = dict(_admin_data_cache["data"])
            for s in requested_scopes:
                col = _ADMIN_SCOPE_MAP[s]
                merged[s] = get_collection(col, page_limit)
            # Re-resolve complaint evidence
            if "complaints" in requested_scopes:
                for c in merged.get("complaints", []):
                    cid = str(c.get('id') or '')
                    url = _resolve_admin_complaint_evidence_url(complaint_id=cid, complaint=c)
                    c['evidenceUrlResolved'] = url
                    c['hasEvidence'] = bool(url)
            _admin_data_cache["data"] = merged
            _admin_data_cache["fetched_at"] = time.time()
            return JSONResponse({'status': 'ok', 'data': merged, 'partial': True})

        # ── Full fetch ───────────────────────────────────────────────
        users = get_collection('users', page_limit)
        vehicles = get_collection('vehicles', page_limit)
        violations = get_collection('violations', page_limit)
        notifications = get_collection('notifications', page_limit)
        complaints = get_collection('complaints', page_limit)
        for complaint in complaints:
            complaint_id = str(complaint.get('id') or '')
            resolved_evidence_url = _resolve_admin_complaint_evidence_url(
                complaint_id=complaint_id,
                complaint=complaint,
            )
            complaint['evidenceUrlResolved'] = resolved_evidence_url
            complaint['hasEvidence'] = bool(resolved_evidence_url)
        profile_updates = get_collection('profile_update_requests', page_limit)

        full_data = {
            'users': users,
            'vehicles': vehicles,
            'violations': violations,
            'notifications': notifications,
            'complaints': complaints,
            'profile_updates': profile_updates,
        }
        _admin_data_cache["data"] = full_data
        _admin_data_cache["fetched_at"] = time.time()

        return JSONResponse({'status': 'ok', 'data': full_data})
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
        _track_fs('reads')
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
        from firebase_admin import firestore as fb_firestore
        
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
                request_section = str(update_data.get('requestSection') or '').strip().lower()
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

                update_data['updatedAt'] = fb_firestore.SERVER_TIMESTAMP
                
                # Apply changes
                db.collection('users').document(user_id).set(update_data, merge=True)
                request_ref.delete()
                try:
                    if request_section == 'wallet_gplx':
                        create_user_notification(
                            db=db,
                            user_id=user_id,
                            title='✅ GPLX đã được duyệt',
                            body='Thông tin GPLX mới đã được cập nhật vào Ví giấy tờ.',
                            notif_type='profile_approved',
                        )
                    elif request_section == 'wallet_cccd':
                        create_user_notification(
                            db=db,
                            user_id=user_id,
                            title='✅ CCCD đã được duyệt',
                            body='Thông tin CCCD mới đã được cập nhật.',
                            notif_type='profile_approved',
                        )
                except Exception as notify_err:
                    print(f"⚠️ Could not create profile approval notification: {notify_err}")
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


# ─── App: Submit Complaint (backend-first) ──────────────────────────────────
@app.post("/api/app/complaints/submit")
async def submit_complaint_app(
    req: Request,
    authorization: Optional[str] = Header(default=None),
    violation_id: str = Form(...),
    reason: str = Form(...),
    description: str = Form(...),
    evidence: Optional[UploadFile] = File(default=None),
):
    """
    Submit a complaint from the mobile app.
    Uses Admin SDK for Storage upload + Firestore write so the client
    never needs direct write access to `violations` or Storage.
    """
    # ── Firebase availability ────────────────────────────────────────────
    if not (fcm_service and fcm_service.is_available and fcm_service._db):
        return JSONResponse(
            {'status': 'error', 'message': 'Firebase chưa sẵn sàng trên server'},
            status_code=503,
        )

    # ── Auth: verify Firebase ID token ───────────────────────────────────
    id_token = _extract_bearer_token(authorization)
    if not id_token:
        return JSONResponse(
            {'status': 'error', 'message': 'Thiếu Authorization Bearer token'},
            status_code=401,
        )
    try:
        decoded = fb_auth.verify_id_token(id_token)
        uid = (decoded or {}).get('uid')
        if not uid:
            raise ValueError('Token thiếu uid')
    except Exception as auth_err:
        return JSONResponse(
            {'status': 'error', 'message': f'Token không hợp lệ hoặc hết hạn: {auth_err}'},
            status_code=401,
        )

    db = fcm_service._db

    # ── Validate inputs ──────────────────────────────────────────────────
    violation_id = (violation_id or '').strip()
    reason = (reason or '').strip()
    description = (description or '').strip()
    if not violation_id or not reason or not description:
        return JSONResponse(
            {'status': 'error', 'message': 'Thiếu violation_id, reason hoặc description'},
            status_code=400,
        )

    # ── Verify violation exists & belongs to this user ───────────────────
    violation_ref = _resolve_violation_ref_by_id(db, violation_id)
    if not violation_ref:
        return JSONResponse(
            {'status': 'error', 'message': 'Không tìm thấy vi phạm'},
            status_code=404,
        )
    violation_doc = violation_ref.get()
    _track_fs('reads')
    if not violation_doc.exists:
        return JSONResponse(
            {'status': 'error', 'message': 'Không tìm thấy vi phạm'},
            status_code=404,
        )
    violation_data = violation_doc.to_dict() or {}
    owner_uid = str(violation_data.get('userId') or '').strip()
    if owner_uid and owner_uid != uid:
        return JSONResponse(
            {'status': 'error', 'message': 'Vi phạm không thuộc về bạn'},
            status_code=403,
        )

    # ── Check complaint_pending status — idempotency guard ───────────────
    current_status = str(violation_data.get('status') or '').strip().lower()
    complaint_locked = violation_data.get('complaintLocked', False)
    if current_status == 'complaint_pending' or complaint_locked:
        # Check if there's already a pending complaint for this violation+user
        existing = (
            db.collection('complaints')
            .where('violationId', '==', violation_id)
            .where('userId', '==', uid)
            .where('status', '==', 'pending')
            .limit(1)
            .get()
        )
        _track_fs('reads')
        if existing:
            return JSONResponse({
                'status': 'already_pending',
                'message': 'Bạn đã gửi khiếu nại cho vi phạm này rồi',
                'complaintId': existing[0].id,
                'violationId': violation_id,
            })

    # ── Upload evidence (Admin SDK) ──────────────────────────────────────
    evidence_url = ''
    evidence_path = ''
    blob_to_rollback = None

    if evidence is not None:
        content = await evidence.read()
        if len(content) > 0:
            import uuid as _uuid
            from firebase_admin import storage as fb_storage

            ext = 'jpg'
            ct = evidence.content_type or 'image/jpeg'
            if 'png' in ct:
                ext = 'png'
            elif 'webp' in ct:
                ext = 'webp'

            ts = int(datetime.now().timestamp() * 1000)
            rand = _uuid.uuid4().hex[:8]
            evidence_path = f'complaints/{uid}/{ts}_{rand}.{ext}'

            try:
                bucket = fb_storage.bucket()
                blob = bucket.blob(evidence_path)
                blob.upload_from_string(content, content_type=ct)
                blob.make_public()
                evidence_url = blob.public_url
                blob_to_rollback = blob
                print(f'☁️ Complaint evidence uploaded: {evidence_path} ({len(content)} bytes)')
            except Exception as upload_err:
                print(f'❌ Complaint evidence upload failed: {upload_err}')
                return JSONResponse(
                    {'status': 'error', 'message': f'Upload ảnh thất bại: {upload_err}'},
                    status_code=500,
                )

    # ── Atomic write: complaint + lock violation ─────────────────────────
    try:
        from firebase_admin import firestore as fb_firestore

        now_iso = datetime.now(timezone.utc).isoformat()
        batch = db.batch()

        # 1) Create complaint doc
        complaint_ref = db.collection('complaints').document()
        batch.set(complaint_ref, {
            'userId': uid,
            'violationId': violation_id,
            'reason': reason,
            'description': description,
            'status': 'pending',
            'evidenceUrl': evidence_url,
            'evidencePath': evidence_path,
            'evidence': {
                'downloadUrl': evidence_url,
                'path': evidence_path,
            },
            'adminNote': '',
            'createdAt': fb_firestore.SERVER_TIMESTAMP,
        })

        # 2) Lock violation
        batch.set(violation_ref, {
            'status': 'complaint_pending',
            'complaintStatus': 'pending',
            'paymentLocked': True,
            'complaintLocked': True,
            'complaintSubmittedAt': now_iso,
            'updatedAt': now_iso,
        }, merge=True)

        batch.commit()
        _track_fs('writes', 2)

        print(f'✅ Complaint {complaint_ref.id} created for violation {violation_id} by {uid}')

        await broadcast_admin_event(
            'new_complaint',
            {
                'complaintId': complaint_ref.id,
                'violationId': violation_id,
                'userId': uid,
            },
        )

        return JSONResponse({
            'status': 'ok',
            'message': 'Khiếu nại đã được gửi thành công',
            'complaintId': complaint_ref.id,
            'violationId': violation_id,
        })

    except Exception as write_err:
        print(f'❌ Complaint Firestore write failed: {write_err}')
        # Best-effort rollback: delete uploaded blob
        if blob_to_rollback:
            try:
                blob_to_rollback.delete()
                print(f'🗑️ Rolled back evidence blob: {evidence_path}')
            except Exception:
                pass
        return JSONResponse(
            {'status': 'error', 'message': f'Lỗi ghi dữ liệu: {write_err}'},
            status_code=500,
        )


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
        _track_fs('reads')
        if not complaint_doc.exists:
            return JSONResponse({'status': 'error', 'message': 'Không tìm thấy khiếu nại'}, status_code=404)

        complaint = complaint_doc.to_dict() or {}
        user_id = complaint.get('userId')
        violation_id = str(complaint.get('violationId') or '').strip()
        reviewed_at = datetime.now(timezone.utc).isoformat()

        if action == 'reject':
            complaint_ref.set(
                {'status': 'rejected', 'adminNote': admin_note, 'reviewedAt': reviewed_at},
                merge=True,
            )
            resolved_violation_id = violation_id
            if violation_id:
                violation_ref = _resolve_violation_ref_by_id(db, violation_id)
                if violation_ref:
                    resolved_violation_id = violation_ref.id

            _sync_violation_complaint_lock(
                db=db,
                violation_id=violation_id,
                complaint_status='rejected',
                lock_payment=False,
                lock_complaint=False,
                reviewed_at=reviewed_at,
            )
            if user_id:
                violation_name = complaint.get('reason') or 'vi phạm'
                reject_reason = admin_note or 'Không đủ căn cứ để chấp nhận khiếu nại.'
                create_user_notification(
                    db=db,
                    user_id=user_id,
                    title='❌ Khiếu nại bị từ chối',
                    body=f'Khiếu nại "{violation_name}" đã bị từ chối.',
                    detail=(
                        f'Lý do từ chối: {reject_reason}\n\n'
                        'Bạn có thể tiếp tục nộp phạt hoặc gửi khiếu nại lại với bằng chứng rõ hơn.'
                    ),
                    notif_type='complaint_rejected',
                    violation_id=resolved_violation_id or None,
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
                _track_fs('reads')
                if violation_doc.exists:
                    resolved_violation_id = violation_doc.id
                    violation_data = violation_doc.to_dict() or {}
                    violation_ref.delete()
                    _track_fs('deletes')

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
        _track_fs('reads')
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
        _track_fs('deletes')
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
                _track_fs('deletes')
                if count >= 500:
                    batch.commit()
                    batch = db.batch()
                    count = 0
            if count > 0:
                batch.commit()

        # Users collection
        db.collection('users').document(user_id).delete()
        _track_fs('deletes')
        
        # Vehicles
        delete_collection(
            db.collection('vehicles').where(
                filter=FieldFilter('ownerId', '==', user_id)
            )
        )
        
        # Notifications
        delete_collection(
            db.collection('notifications').where(
                filter=FieldFilter('userId', '==', user_id)
            )
        )
        
        # Complaints
        delete_collection(
            db.collection('complaints').where(
                filter=FieldFilter('userId', '==', user_id)
            )
        )
        
        # Violations (optional, but requested to clean up)
        delete_collection(
            db.collection('violations').where(
                filter=FieldFilter('userId', '==', user_id)
            )
        )
        
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
                    v['paidAt'] = datetime.now(timezone.utc).isoformat()
                    
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
            paid_at = datetime.now(timezone.utc).isoformat()
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
    conf: float = Form(0.25),
    show_masks: bool = Form(True),
    show_labels: bool = Form(True),
    show_boxes: bool = Form(True),
):
    """
    Upload an image and run YOLO detection on it.
    Returns annotated image (base64) + list of detections.
    Supports toggling: masks (segmentation overlay), labels (class names on bbox), boxes (bounding boxes).
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
        
        # Predefined color palette for classes (BGR)
        _PALETTE = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
            (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
            (64, 224, 208), (255, 165, 0), (138, 43, 226), (0, 191, 255),
        ]

        # Extract detections
        detections = []
        frame_vis = img.copy()
        has_masks = r0.masks is not None and len(r0.masks) > 0
        
        # Draw segmentation masks first (so they appear behind boxes)
        if has_masks and show_masks:
            mask_overlay = frame_vis.copy()
            masks_data = r0.masks.data.cpu().numpy()
            classes_arr = r0.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(masks_data)):
                cls_id = int(classes_arr[i])
                color = _PALETTE[cls_id % len(_PALETTE)]
                mask_resized = cv2.resize(masks_data[i], (frame_vis.shape[1], frame_vis.shape[0]))
                mask_bool = mask_resized > 0.5
                mask_overlay[mask_bool] = color
            cv2.addWeighted(mask_overlay, 0.4, frame_vis, 0.6, 0, frame_vis)

        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes = r0.boxes.xyxy.cpu().numpy()
            classes = r0.boxes.cls.cpu().numpy().astype(int)
            confs = r0.boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                cls_id = int(classes[i])
                confidence = float(confs[i])
                class_name = config.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                color = _PALETTE[cls_id % len(_PALETTE)]
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2]
                })
                
                # Draw bbox
                if show_boxes:
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)

                # Draw label
                if show_labels:
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
            "class_summary": class_summary,
            "has_masks": has_masks,
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
            try:
                results = model.track(
                    frame, imgsz=config.IMG_SIZE, conf=conf,
                    iou=config.IOU_THRESHOLD, persist=True,
                    verbose=False, tracker=config.TRACKER
                )
            except AttributeError as e:
                if "fuse_score" in str(e) or "IterableSimpleNamespace" in str(e):
                    video_tasks[task_id]["status"] = "error"
                    video_tasks[task_id]["error"] = (
                        f"Tracker config không tương thích phiên bản ultralytics. "
                        f"Kiểm tra '{config.TRACKER}' có đầy đủ key. Chi tiết: {e}"
                    )
                    return
                raise
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
    
    global _realtime_busy
    cap = None
    running = False
    det = None
    owns_lock = False
    
    # Mutable settings that can be updated mid-stream
    live_conf = 0.25
    live_debug = False
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")
            
            if action == "start":
                # ── Single-session lock ──────────────────────────
                if _realtime_busy:
                    await websocket.send_json({
                        "type": "error",
                        "code": "busy",
                        "message": "Another realtime session is running. Please wait."
                    })
                    continue
                _realtime_busy = True
                owns_lock = True

                # Get parameters
                video_path = msg.get("video") or config.DEFAULT_VIDEO
                model_path = msg.get("model") or config.MODEL_PATH
                detectors = msg.get("detectors", ["helmet"])
                live_conf = float(msg.get("conf", 0.25))
                live_debug = bool(msg.get("debug", False))
                
                # Validate paths
                if not Path(video_path).exists():
                    await websocket.send_json({"type": "error", "message": f"Video not found: {video_path}"})
                    _realtime_busy = False
                    owns_lock = False
                    continue
                    
                if not Path(model_path).exists():
                    await websocket.send_json({"type": "error", "message": f"Model not found: {model_path}"})
                    _realtime_busy = False
                    owns_lock = False
                    continue
                
                # Get UnifiedDetector
                det = get_detector(model_path)
                det.reset()

                # Reset dedupe cache for fresh session
                notified_violations.clear()
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    await websocket.send_json({"type": "error", "message": f"Cannot open video: {video_path}"})
                    _realtime_busy = False
                    owns_lock = False
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
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
                session_frame_idx = 0
                
                # Process frames
                while running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        await websocket.send_json({"type": "finished", "stats": det.get_stats()})
                        break
                    
                    # Detect using UnifiedDetector with LIVE settings
                    try:
                        frame_vis, violations = det.process_frame(
                            frame, detectors, conf=live_conf, debug=live_debug
                        )
                    except RuntimeError as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                        running = False
                        if cap:
                            cap.release()
                            cap = None
                        break
                    
                    session_frame_idx += 1

                    # Resize for streaming
                    h, w = frame_vis.shape[:2]
                    if w > 1280:
                        scale = 1280 / w
                        frame_vis = cv2.resize(frame_vis, (1280, int(h * scale)))
                    
                    # Encode to base64
                    _, buffer = cv2.imencode('.jpg', frame_vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame with session-scoped progress capped at 100%
                    progress = min(session_frame_idx / total_frames * 100, 100)
                    await websocket.send_json({
                        "type": "frame",
                        "image": frame_b64,
                        "stats": det.get_stats(),
                        "progress": progress
                    })
                    
                    # Send violations if any + save to store
                    for v in violations:
                        stored = store_violation(
                            v_type=v.get('type', 'unknown'),
                            track_id=v.get('id', 0),
                            label=v.get('label', ''),
                            vehicle_class=v.get('vehicleClass'),
                            stable_track_id=v.get('stableTrackId'),
                            raw_track_id=v.get('rawTrackId'),
                            snapshot_path=v.get('snapshotPath'),
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
                _realtime_busy = False
                owns_lock = False
            
            elif action == "stop":
                running = False
                if cap:
                    cap.release()
                    cap = None
                if det:
                    det.reset()
                if owns_lock:
                    _realtime_busy = False
                    owns_lock = False
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
        if det:
            det.reset()
        if owns_lock:
            _realtime_busy = False


# =============================================================================
# WEBSOCKET - APP REAL-TIME NOTIFICATIONS
# =============================================================================

@app.websocket("/ws/app")
async def websocket_app(websocket: WebSocket):
    """
    WebSocket endpoint for Flutter app real-time notifications.
    
    App connects here to receive instant violation alerts.
    Server pushes: {"type": "new_violation", "data": {...}}
    App can send: {"action": "ping"} to keep alive.
    """
    query_params = parse_qs((websocket.scope.get("query_string", b"") or b"").decode())
    requested_user_id = (query_params.get("user_id", [""])[0] or "").strip()
    session_id = (query_params.get("session_id", [""])[0] or "").strip()
    device_id = (query_params.get("device_id", [""])[0] or "").strip()
    query_id_token = (query_params.get("id_token", [""])[0] or "").strip()

    resolved_user_id = ""
    auth_error_message = ""
    try:
        resolved_user_id = _resolve_session_user_id(
            requested_user_id=requested_user_id,
            query_token=query_id_token,
        )
    except fb_auth.ExpiredIdTokenError:
        auth_error_message = "Token đã hết hạn, vui lòng đăng nhập lại"
    except fb_auth.RevokedIdTokenError:
        auth_error_message = "Token đã bị thu hồi, vui lòng đăng nhập lại"
    except fb_auth.InvalidIdTokenError:
        auth_error_message = "Token Firebase không hợp lệ"
    except PermissionError as exc:
        auth_error_message = str(exc)
    except ValueError as exc:
        auth_error_message = str(exc)

    await websocket.accept()
    if auth_error_message:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": auth_error_message,
        }))
        await websocket.close(code=1008)
        return

    app_clients[websocket] = {
        "user_id": resolved_user_id,
        "session_id": session_id,
        "device_id": device_id,
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    client_ip = websocket.client.host if websocket.client else "unknown"
    print(
        f"📱 App client connected: {client_ip} user={resolved_user_id or 'anonymous'} "
        f"(total: {len(app_clients)})"
    )

    if (
        fcm_service
        and fcm_service.is_available
        and fcm_service._db
        and resolved_user_id
        and session_id
        and device_id
    ):
        try:
            _upsert_app_active_session(
                fcm_service._db,
                user_id=resolved_user_id,
                session_id=session_id,
                device_id=device_id,
                connected_via_ws=True,
                force=True,
            )
        except Exception as exc:
            print(f"⚠️ Could not upsert websocket session on connect: {exc}")
    
    user_scoped_data = _filter_violation_store_for_user(resolved_user_id)
    # Send current user-scoped violation count as welcome message
    await websocket.send_text(json.dumps({
        "type": "connected",
        "message": "Connected to violation detection server",
        "pending_violations": len([v for v in user_scoped_data if v['status'] == 'pending']),
        "total_violations": len(user_scoped_data),
    }))
    
    try:
        while True:
            # Keep connection alive — listen for pings or commands
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")

            incoming_session_id = str(msg.get("session_id") or "").strip()
            incoming_device_id = str(msg.get("device_id") or "").strip()
            incoming_last_seen = str(msg.get("last_seen_at") or "").strip()
            if incoming_session_id:
                session_id = incoming_session_id
            if incoming_device_id:
                device_id = incoming_device_id
            if websocket in app_clients:
                app_clients[websocket]["session_id"] = session_id
                app_clients[websocket]["device_id"] = device_id
            
            if action == "ping":
                if (
                    fcm_service
                    and fcm_service.is_available
                    and fcm_service._db
                    and resolved_user_id
                    and session_id
                    and device_id
                ):
                    try:
                        _upsert_app_active_session(
                            fcm_service._db,
                            user_id=resolved_user_id,
                            session_id=session_id,
                            device_id=device_id,
                            connected_via_ws=True,
                            last_seen_at_client=incoming_last_seen,
                        )
                    except Exception as exc:
                        print(f"⚠️ Could not upsert websocket session on ping: {exc}")
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            elif action == "get_violations":
                # App can request its own violation list.
                user_scoped_data = _filter_violation_store_for_user(resolved_user_id)
                await websocket.send_text(json.dumps({
                    "type": "violations_list",
                    "data": user_scoped_data
                }))
            elif action == "get_stats":
                user_scoped_data = _filter_violation_store_for_user(resolved_user_id)
                total = len(user_scoped_data)
                pending = len([v for v in user_scoped_data if v['status'] == 'pending'])
                await websocket.send_text(json.dumps({
                    "type": "stats",
                    "data": {
                        "total": total,
                        "pending": pending,
                        "paid": total - pending,
                        "totalFines": sum(
                            v['fineAmount']
                            for v in user_scoped_data
                            if v['status'] == 'pending'
                        ),
                    }
                }))
    except WebSocketDisconnect as e:
        print(f"📱 App client disconnected unexpectedly: {client_ip} (Code: {e.code}, Reason: {e.reason})")
    except Exception as e:
        print(f"📱 App client error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app_clients.pop(websocket, None)
        if (
            fcm_service
            and fcm_service.is_available
            and fcm_service._db
            and (session_id or (device_id and resolved_user_id))
        ):
            try:
                _clear_app_active_sessions(
                    fcm_service._db,
                    session_id=session_id,
                    device_id=device_id,
                    user_id=resolved_user_id,
                )
            except Exception as exc:
                print(f"⚠️ Could not clear websocket session on disconnect: {exc}")
        print(
            f"📱 App client session ended: {client_ip} user={resolved_user_id or 'anonymous'} "
            f"(total: {len(app_clients)})"
        )


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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
