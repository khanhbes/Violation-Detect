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
import 'package:traffic_violation_app/services/notification_service.dart';
import 'package:traffic_violation_app/services/push_notification_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Firebase (required before FCM)
  await Firebase.initializeApp();

  // Initialize local notifications (for foreground display)
  await NotificationService().initialize();

  // Initialize FCM push notifications (remote)
  await PushNotificationService().initialize();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vi Phạm Giao Thông',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.light,
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
      },
    );
  }
}
