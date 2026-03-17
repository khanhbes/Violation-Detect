import 'dart:convert';
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:http/http.dart' as http;
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

/// Top-level background message handler (MUST be top-level function).
/// Called when app is in background or terminated and a data message arrives.
@pragma('vm:entry-point')
Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  debugPrint('📩 Background FCM message: ${message.messageId}');
  // The notification payload is automatically shown by the OS.
  // Data-only messages can be processed here if needed.
}

/// Production-ready FCM Push Notification Service.
///
/// Handles:
/// - Permission request (iOS + Android 13+)
/// - Token management (get, refresh, sync with backend)
/// - Foreground messages (show via local notification)
/// - Background messages (tap → navigate to specific screen)
/// - Terminated/cold-start messages (app wake-up → navigate)
class PushNotificationService {
  // ── Singleton ─────────────────────────────────────────────────────
  static final PushNotificationService _instance =
      PushNotificationService._internal();
  factory PushNotificationService() => _instance;
  PushNotificationService._internal();

  final FirebaseMessaging _messaging = FirebaseMessaging.instance;
  final FlutterLocalNotificationsPlugin _localNotifications =
      FlutterLocalNotificationsPlugin();

  bool _initialized = false;
  String? _currentToken;

  // Navigator key for routing on notification tap
  static final GlobalKey<NavigatorState> navigatorKey =
      GlobalKey<NavigatorState>();

  // ── Initialize ────────────────────────────────────────────────────

  /// Call this in main() after Firebase.initializeApp().
  Future<void> initialize() async {
    if (_initialized) return;

    // 1. Set up the background handler
    FirebaseMessaging.onBackgroundMessage(firebaseMessagingBackgroundHandler);

    // 2. Request permission
    await _requestPermission();

    // 3. Set up local notifications (for foreground display)
    await _setupLocalNotifications();

    // 4. Get token and sync with backend
    await _getAndSyncToken();

    // 5. Listen for token refresh
    _messaging.onTokenRefresh.listen((newToken) {
      debugPrint('🔄 FCM token refreshed');
      _currentToken = newToken;
      _syncTokenWithBackend(newToken);
    });

    // 6. Set up message handlers for all 3 states
    _setupMessageHandlers();

    _initialized = true;
    debugPrint('✅ PushNotificationService initialized');
  }

  // ── Permission ────────────────────────────────────────────────────

  Future<void> _requestPermission() async {
    try {
      NotificationSettings settings = await _messaging.requestPermission(
        alert: true,
        badge: true,
        sound: true,
        provisional: false,
        announcement: false,
        carPlay: false,
        criticalAlert: false,
      );

      debugPrint(
          '📱 Notification permission: ${settings.authorizationStatus}');

      if (settings.authorizationStatus == AuthorizationStatus.denied) {
        debugPrint('⚠️ User denied notification permission');
      }
    } catch (e) {
      debugPrint('❌ Permission request error: $e');
    }
  }

  // ── Local Notifications Setup ─────────────────────────────────────

  Future<void> _setupLocalNotifications() async {
    const androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: false, // Already requested via FCM
      requestBadgePermission: false,
      requestSoundPermission: false,
    );

    final settings = const InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _localNotifications.initialize(
      settings: settings,
      onDidReceiveNotificationResponse: (response) {
        // User tapped the local notification (foreground case)
        _handleNotificationTap(response.payload);
      },
    );

    // Create Android notification channel
    const channel = AndroidNotificationChannel(
      'violations_channel',
      'Vi phạm giao thông',
      description: 'Thông báo vi phạm giao thông thời gian thực',
      importance: Importance.high,
      playSound: true,
      enableVibration: true,
    );

    await _localNotifications
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(channel);
  }

  // ── Token Management ──────────────────────────────────────────────

  Future<void> _getAndSyncToken() async {
    try {
      // For web, you need to pass vapidKey (from Firebase Console)
      String? token;
      if (kIsWeb) {
        // Replace with your VAPID key from Firebase Console
        // Project Settings → Cloud Messaging → Web Push certificates
        token = await _messaging.getToken(
          // vapidKey: 'YOUR_VAPID_KEY_HERE',
        );
      } else {
        token = await _messaging.getToken();
      }

      if (token != null) {
        _currentToken = token;
        debugPrint('📱 FCM Token: ${token.substring(0, 20)}...');
        // Don't await — sync in background so it doesn't block initialization
        _syncTokenWithBackend(token);
      } else {
        debugPrint('⚠️ FCM token is null');
      }
    } catch (e) {
      debugPrint('❌ Get FCM token error: $e');
    }
  }

  Future<void> _syncTokenWithBackend(String token) async {
    try {
      String platform = 'unknown';
      if (kIsWeb) {
        platform = 'web';
      } else if (Platform.isAndroid) {
        platform = 'android';
      } else if (Platform.isIOS) {
        platform = 'ios';
      }

      final apiService = ApiService();
      final response = await http.post(
        Uri.parse('${ApiService.baseUrl}/api/fcm/register'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'user_id': AppSettings().uid ?? 'default_user',
          'fcm_token': token,
          'platform': platform,
          'device_info': platform == 'web'
              ? 'Web Browser'
              : '${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
        }),
      ).timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        debugPrint('✅ FCM token synced with backend');
      } else {
        debugPrint('⚠️ FCM token sync failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('⚠️ FCM token sync error (backend may be offline): $e');
    }
  }

  // ── Message Handlers (3 States) ───────────────────────────────────

  void _setupMessageHandlers() {
    // STATE 1: FOREGROUND — App is open and in the foreground
    FirebaseMessaging.onMessage.listen(_handleForegroundMessage);

    // STATE 2: BACKGROUND — App is in background, user taps notification
    FirebaseMessaging.onMessageOpenedApp.listen(_handleMessageOpenedApp);

    // STATE 3: TERMINATED — App was killed, user taps notification to wake up
    _handleInitialMessage();
  }

  /// STATE 1: FOREGROUND
  /// The OS does NOT automatically show a notification when the app
  /// is in the foreground, so we use local_notifications to show one.
  Future<void> _handleForegroundMessage(RemoteMessage message) async {
    debugPrint('📩 Foreground FCM: ${message.notification?.title}');

    final notification = message.notification;
    if (notification == null) return;

    // Show as local notification (appears in system tray)
    final androidDetails = AndroidNotificationDetails(
      'violations_channel',
      'Vi phạm giao thông',
      channelDescription: 'Thông báo vi phạm giao thông thời gian thực',
      importance: Importance.high,
      priority: Priority.high,
      showWhen: true,
      icon: '@mipmap/ic_launcher',
      styleInformation: BigTextStyleInformation(
        notification.body ?? '',
        contentTitle: notification.title,
      ),
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    final details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    // Use data payload as notification payload for routing on tap
    String? routePayload;
    if (message.data.isNotEmpty) {
      routePayload = json.encode(message.data);
    }

    await _localNotifications.show(
      id: notification.hashCode,
      title: notification.title ?? '🚨 Thông báo',
      body: notification.body ?? '',
      notificationDetails: details,
      payload: routePayload,
    );
  }

  /// STATE 2: BACKGROUND → User taps notification → App resumes
  void _handleMessageOpenedApp(RemoteMessage message) {
    debugPrint('📩 Background tap FCM: ${message.data}');
    _navigateFromMessage(message.data);
  }

  /// STATE 3: TERMINATED → User taps notification → App cold starts
  Future<void> _handleInitialMessage() async {
    try {
      RemoteMessage? initialMessage =
          await _messaging.getInitialMessage();

      if (initialMessage != null) {
        debugPrint('📩 Terminated tap FCM: ${initialMessage.data}');
        // Delay slightly to let the app finish building its widget tree
        Future.delayed(const Duration(seconds: 2), () {
          _navigateFromMessage(initialMessage.data);
        });
      }
    } catch (e) {
      debugPrint('❌ Initial message error: $e');
    }
  }

  // ── Navigation from notification data ─────────────────────────────

  void _handleNotificationTap(String? payload) {
    if (payload == null || payload.isEmpty) return;

    try {
      final data = json.decode(payload) as Map<String, dynamic>;
      _navigateFromMessage(data);
    } catch (e) {
      debugPrint('❌ Notification tap parse error: $e');
    }
  }

  void _navigateFromMessage(Map<String, dynamic> data) {
    final route = data['route'] as String?;
    if (route == null || route.isEmpty) return;

    debugPrint('🧭 Navigating to: $route with data: $data');

    // Use the navigator key to navigate from anywhere
    final navigator = navigatorKey.currentState;
    if (navigator != null) {
      navigator.pushNamed(route, arguments: data);
    } else {
      debugPrint('⚠️ Navigator not available yet, saving pending route');
      // The app may not have built the navigator yet (cold start).
      // The initial message handler uses a delay to handle this.
    }
  }

  // ── Public API ────────────────────────────────────────────────────

  /// Get the current FCM token (useful for debugging).
  String? get currentToken => _currentToken;

  /// Manually re-sync the token with the backend.
  Future<void> resyncToken() async {
    if (_currentToken != null) {
      await _syncTokenWithBackend(_currentToken!);
    } else {
      await _getAndSyncToken();
    }
  }

  /// Remove token from backend (call on logout).
  Future<void> clearToken() async {
    if (_currentToken == null) return;

    try {
      await http.delete(
        Uri.parse('${ApiService.baseUrl}/api/fcm/unregister'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'fcm_token': _currentToken}),
      ).timeout(const Duration(seconds: 5));
      debugPrint('✅ FCM token cleared from backend');
    } catch (e) {
      debugPrint('⚠️ FCM token clear error: $e');
    }

    await _messaging.deleteToken();
    _currentToken = null;
  }
}
