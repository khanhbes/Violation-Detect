/* ==========================================================================
   Firebase Cloud Messaging — Service Worker
   ==========================================================================
   This service worker runs in the browser's background thread and handles
   push notifications when the website tab is CLOSED or INACTIVE.
   
   IMPORTANT: This file MUST be served from the root path (or /static/ with
   proper scope registration). The Firebase config below must match your
   Firebase project.
   ========================================================================== */

// Import Firebase libraries (compat version for service workers)
importScripts('https://www.gstatic.com/firebasejs/10.14.1/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.14.1/firebase-messaging-compat.js');

// ─── Firebase Configuration ────────────────────────────────────────────
// ⚠️ REPLACE these values with your Firebase project config
// Get them from: Firebase Console → Project Settings → General → Your apps → Web app
firebase.initializeApp({
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_PROJECT.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
});

const messaging = firebase.messaging();

// ─── Background Message Handler ────────────────────────────────────────
// Called when a push message arrives and the web app is NOT in the foreground
messaging.onBackgroundMessage((payload) => {
    console.log('[SW] Background message received:', payload);

    // Extract notification data
    const notificationTitle = payload.notification?.title || '🚨 Vi phạm giao thông';
    const notificationOptions = {
        body: payload.notification?.body || 'Có vi phạm mới được phát hiện',
        icon: '/static/favicon.png',
        badge: '/static/favicon.png',
        tag: 'violation-' + (payload.data?.violation_id || Date.now()),
        // Custom data for click handling
        data: payload.data || {},
        // Actions shown on the notification
        actions: [
            { action: 'view', title: '👁️ Xem chi tiết' },
            { action: 'dismiss', title: '✖️ Bỏ qua' }
        ],
        // Vibration pattern
        vibrate: [200, 100, 200],
        // Require interaction (notification stays until clicked)
        requireInteraction: true,
    };

    return self.registration.showNotification(notificationTitle, notificationOptions);
});

// ─── Notification Click Handler ────────────────────────────────────────
self.addEventListener('notificationclick', (event) => {
    console.log('[SW] Notification clicked:', event.action);

    event.notification.close();

    if (event.action === 'dismiss') return;

    // Navigate to the app or specific violation route
    const data = event.notification.data || {};
    const route = data.route || '/';
    const violationId = data.violation_id || '';

    // Build URL: navigate to the main page with hash routing
    const urlToOpen = new URL('/', self.location.origin);
    if (violationId) {
        urlToOpen.hash = `violation-${violationId}`;
    }

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then((windowClients) => {
            // If the app is already open, focus it
            for (const client of windowClients) {
                if (client.url.startsWith(self.location.origin)) {
                    client.focus();
                    client.postMessage({
                        type: 'NOTIFICATION_CLICK',
                        data: data,
                    });
                    return;
                }
            }
            // Otherwise open a new window
            return clients.openWindow(urlToOpen.href);
        })
    );
});
