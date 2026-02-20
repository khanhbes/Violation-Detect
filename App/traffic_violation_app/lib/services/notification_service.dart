import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:intl/intl.dart';

/// Local push notification service for violation alerts.
class NotificationService {
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;
  NotificationService._internal();

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();
  bool _initialized = false;

  // Callback when user taps a notification
  static void Function(String? violationId)? onNotificationTap;

  Future<void> initialize() async {
    if (_initialized) return;

    const androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    final settings = const InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _plugin.initialize(
      settings: settings,
      onDidReceiveNotificationResponse: (response) {
        onNotificationTap?.call(response.payload);
      },
    );

    _initialized = true;
  }

  Future<void> showViolationNotification(Violation violation) async {
    if (!_initialized) await initialize();

    final currencyFormatter =
        NumberFormat.currency(locale: 'vi_VN', symbol: '₫');
    final dateFormatter = DateFormat('HH:mm dd/MM/yyyy');

    final androidDetails = AndroidNotificationDetails(
      'violations_channel',
      'Vi phạm giao thông',
      channelDescription: 'Thông báo vi phạm giao thông',
      importance: Importance.high,
      priority: Priority.high,
      showWhen: true,
      icon: '@mipmap/ic_launcher',
      largeIcon: const DrawableResourceAndroidBitmap('@mipmap/ic_launcher'),
      styleInformation: BigTextStyleInformation(
        '${violation.violationType}\n'
        'Mức phạt: ${currencyFormatter.format(violation.fineAmount)}\n'
        'Thời gian: ${dateFormatter.format(violation.timestamp)}\n'
        '${violation.description}',
        contentTitle: '🚨 Phát hiện vi phạm mới!',
        summaryText: violation.violationCode,
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

    await _plugin.show(
      id: violation.id.hashCode,
      title: '🚨 ${violation.violationType}',
      body: 'Mức phạt: ${currencyFormatter.format(violation.fineAmount)}',
      notificationDetails: details,
      payload: violation.id,
    );
  }
}
