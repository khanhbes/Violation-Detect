import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/screens/splash_screen.dart';
import 'package:traffic_violation_app/screens/login_screen.dart';
import 'package:traffic_violation_app/screens/home_screen.dart';
import 'package:traffic_violation_app/screens/violations_screen.dart';
import 'package:traffic_violation_app/screens/violation_detail_screen.dart';
import 'package:traffic_violation_app/screens/payment_screen.dart';
import 'package:traffic_violation_app/screens/profile_screen.dart';
import 'package:traffic_violation_app/screens/traffic_laws_screen.dart';
import 'package:traffic_violation_app/screens/notifications_screen.dart';
import 'package:traffic_violation_app/screens/register_screen.dart';
import 'package:traffic_violation_app/screens/complaint_screen.dart';
import 'package:traffic_violation_app/screens/support_screen.dart';
import 'package:traffic_violation_app/services/notification_service.dart';
import 'package:traffic_violation_app/services/push_notification_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Firebase (required before FCM)
  await Firebase.initializeApp();

  // Initialize local notifications (for foreground display)
  await NotificationService().initialize();

  // Initialize FCM push notifications (remote) — non-blocking
  // Do NOT await: if backend is offline, this would block app startup
  // causing the app to freeze on the splash screen with a timeout error.
  PushNotificationService().initialize();

  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final AppSettings _settings = AppSettings();

  @override
  void initState() {
    super.initState();
    _settings.addListener(_onSettingsChanged);
  }

  @override
  void dispose() {
    _settings.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    setState(() {}); // Rebuild MaterialApp when settings change
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: _settings.isVietnamese ? 'Vi Phạm Giao Thông' : 'Traffic Violations',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: _settings.themeMode,
      locale: _settings.locale,
      navigatorKey: PushNotificationService.navigatorKey,
      initialRoute: '/',
      routes: {
        '/': (context) => const SplashScreen(),
        '/login': (context) => const LoginScreen(),
        '/home': (context) => const HomeScreen(),
        '/violations': (context) => const ViolationsScreen(),
        '/violation-detail': (context) => const ViolationDetailScreen(),
        '/payment': (context) => const PaymentScreen(),
        '/profile': (context) => const ProfileScreen(),
        '/traffic-laws': (context) => const TrafficLawsScreen(),
        '/notifications': (context) => const NotificationsScreen(),
        '/register': (context) => const RegisterScreen(),
        '/complaint': (context) => const ComplaintScreen(),
        '/support': (context) => const SupportScreen(),
      },
    );
  }
}
