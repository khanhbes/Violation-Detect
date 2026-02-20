"""
Firebase Cloud Messaging (FCM) Service
=======================================
Production-ready FCM integration for sending push notifications
to Android, iOS, and Web clients via Firebase Admin SDK.

Uses Firestore as the token database for multi-device support.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, messaging, firestore

logger = logging.getLogger(__name__)


class FCMService:
    """
    Firebase Cloud Messaging service.
    
    Handles:
    - Firebase Admin SDK initialization
    - Device token registration/removal (Firestore)
    - Push notification sending (single user + broadcast)
    - Stale token cleanup on send failure
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not FCMService._initialized:
            self._db = None
            self._initialize_firebase()
            FCMService._initialized = True

    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK with service account credentials."""
        try:
            # Look for service account key in multiple locations
            possible_paths = [
                Path(__file__).parent.parent / "serviceAccountKey.json",
                Path(__file__).parent / "serviceAccountKey.json",
                Path(os.environ.get("FIREBASE_CREDENTIALS", "")),
            ]

            cred_path = None
            for p in possible_paths:
                if p and p.exists():
                    cred_path = p
                    break

            if cred_path is None:
                logger.warning(
                    "⚠️ Firebase serviceAccountKey.json not found. "
                    "FCM push notifications will be disabled. "
                    "Place the file in: Detection Web/Web/serviceAccountKey.json"
                )
                return

            # Initialize only if not already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(cred_path))
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'traffic-violation-c9270.appspot.com'
                })

            self._db = firestore.client()
            logger.info("✅ Firebase Admin SDK initialized successfully")
            print("✅ Firebase Admin SDK initialized successfully")

        except Exception as e:
            logger.error(f"❌ Firebase initialization error: {e}")
            print(f"❌ Firebase initialization error: {e}")
            self._db = None

    @property
    def is_available(self) -> bool:
        """Check if FCM service is properly initialized."""
        return self._db is not None and bool(firebase_admin._apps)

    # =========================================================================
    # TOKEN MANAGEMENT (Firestore)
    # =========================================================================

    def register_token(
        self,
        user_id: str,
        fcm_token: str,
        platform: str,
        device_info: str = "",
    ) -> Dict[str, Any]:
        """
        Register or update an FCM device token in Firestore.
        
        Uses fcm_token as the natural unique key — if the same token
        already exists, we update it; otherwise create a new document.
        
        Args:
            user_id: User identifier
            fcm_token: FCM registration token from the client
            platform: "android" | "ios" | "web"
            device_info: Optional device description
            
        Returns:
            Dict with status and message
        """
        if not self.is_available:
            return {"status": "error", "message": "FCM service not initialized"}

        try:
            collection = self._db.collection("user_device_tokens")

            # Check if token already exists (upsert logic)
            existing = collection.where("fcm_token", "==", fcm_token).limit(1).get()

            token_data = {
                "user_id": user_id,
                "fcm_token": fcm_token,
                "platform": platform,
                "device_info": device_info,
                "last_updated": firestore.SERVER_TIMESTAMP,
                "is_active": True,
            }

            if existing:
                # Update existing document
                doc = existing[0]
                doc.reference.update(token_data)
                logger.info(f"🔄 Token updated for user={user_id} platform={platform}")
            else:
                # Create new document
                collection.add(token_data)
                logger.info(f"✅ Token registered for user={user_id} platform={platform}")

            print(f"📱 FCM token registered: user={user_id}, platform={platform}")
            return {"status": "ok", "message": "Token registered"}

        except Exception as e:
            logger.error(f"Token registration error: {e}")
            return {"status": "error", "message": str(e)}

    def remove_token(self, fcm_token: str) -> Dict[str, Any]:
        """
        Soft-delete a device token (mark as inactive).
        Called on user logout or app uninstall.
        """
        if not self.is_available:
            return {"status": "error", "message": "FCM service not initialized"}

        try:
            collection = self._db.collection("user_device_tokens")
            docs = collection.where("fcm_token", "==", fcm_token).get()

            for doc in docs:
                doc.reference.update({
                    "is_active": False,
                    "last_updated": firestore.SERVER_TIMESTAMP,
                })

            print(f"📱 FCM token deactivated")
            return {"status": "ok", "message": "Token removed"}

        except Exception as e:
            logger.error(f"Token removal error: {e}")
            return {"status": "error", "message": str(e)}

    def _get_active_tokens(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        Get all active FCM tokens, optionally filtered by user_id.
        
        Returns list of dicts with 'fcm_token', 'platform', 'doc_ref'.
        """
        if not self.is_available:
            return []

        try:
            collection = self._db.collection("user_device_tokens")
            query = collection.where("is_active", "==", True)

            if user_id:
                query = query.where("user_id", "==", user_id)

            docs = query.get()

            tokens = []
            for doc in docs:
                data = doc.to_dict()
                tokens.append({
                    "fcm_token": data.get("fcm_token", ""),
                    "platform": data.get("platform", "unknown"),
                    "doc_ref": doc.reference,
                })
            return tokens

        except Exception as e:
            logger.error(f"Error fetching tokens: {e}")
            return []

    # =========================================================================
    # SEND NOTIFICATIONS
    # =========================================================================

    def send_push_notification(
        self,
        user_id: str,
        title: str,
        body: str,
        data_payload: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Send push notification to all active devices of a specific user.
        
        Args:
            user_id: Target user identifier
            title: Notification title (shown in system tray)
            body: Notification body text
            data_payload: Custom data dict for in-app routing, e.g.:
                          {"route": "/violation-detail", "id": "vio_0001"}
                          
        Returns:
            Dict with success/failure counts
        """
        return self._send_to_tokens(
            tokens=self._get_active_tokens(user_id=user_id),
            title=title,
            body=body,
            data_payload=data_payload,
        )

    def broadcast_push_notification(
        self,
        title: str,
        body: str,
        data_payload: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Broadcast push notification to ALL active registered devices.
        Used when a new violation is detected and all users should be notified.
        """
        return self._send_to_tokens(
            tokens=self._get_active_tokens(user_id=None),
            title=title,
            body=body,
            data_payload=data_payload,
        )

    def _send_to_tokens(
        self,
        tokens: List[Dict],
        title: str,
        body: str,
        data_payload: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Internal: send FCM message to a list of token dicts.
        Handles individual failures and cleans up stale tokens.
        """
        if not self.is_available:
            return {"status": "disabled", "sent": 0, "failed": 0}

        if not tokens:
            return {"status": "no_tokens", "sent": 0, "failed": 0}

        # Ensure all data values are strings (FCM requirement)
        safe_data = {}
        if data_payload:
            for k, v in data_payload.items():
                safe_data[k] = str(v) if v is not None else ""

        sent = 0
        failed = 0
        stale_refs = []

        for token_info in tokens:
            fcm_token = token_info["fcm_token"]
            if not fcm_token:
                continue

            try:
                # Build message with BOTH notification + data payloads
                message = messaging.Message(
                    token=fcm_token,
                    notification=messaging.Notification(
                        title=title,
                        body=body,
                    ),
                    data=safe_data,
                    # Android-specific: high priority + notification channel
                    android=messaging.AndroidConfig(
                        priority="high",
                        notification=messaging.AndroidNotification(
                            channel_id="violations_channel",
                            priority="high",
                            default_sound=True,
                            icon="@mipmap/ic_launcher",
                        ),
                    ),
                    # iOS-specific: badge + sound + content-available
                    apns=messaging.APNSConfig(
                        payload=messaging.APNSPayload(
                            aps=messaging.Aps(
                                badge=1,
                                sound="default",
                                content_available=True,
                            ),
                        ),
                    ),
                    # Web-specific: handled by service worker
                    webpush=messaging.WebpushConfig(
                        notification=messaging.WebpushNotification(
                            title=title,
                            body=body,
                            icon="/static/favicon.png",
                        ),
                    ),
                )

                messaging.send(message)
                sent += 1

            except messaging.UnregisteredError:
                # Token is no longer valid — mark for cleanup
                logger.warning(f"Stale token detected, marking inactive")
                stale_refs.append(token_info.get("doc_ref"))
                failed += 1

            except messaging.SenderIdMismatchError:
                logger.warning(f"Sender ID mismatch for token, removing")
                stale_refs.append(token_info.get("doc_ref"))
                failed += 1

            except Exception as e:
                logger.error(f"FCM send error: {e}")
                failed += 1

        # Clean up stale tokens
        for ref in stale_refs:
            if ref:
                try:
                    ref.update({"is_active": False, "last_updated": firestore.SERVER_TIMESTAMP})
                except Exception:
                    pass

        result = {"status": "ok", "sent": sent, "failed": failed}
        if sent > 0:
            print(f"📨 FCM: Sent to {sent} device(s), {failed} failed")
        return result
