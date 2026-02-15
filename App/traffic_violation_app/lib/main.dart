import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/screens/splash_screen.dart';
import 'package:traffic_violation_app/screens/login_screen.dart';
import 'package:traffic_violation_app/screens/home_screen.dart';
import 'package:traffic_violation_app/screens/violations_screen.dart';
import 'package:traffic_violation_app/screens/violation_detail_screen.dart';
import 'package:traffic_violation_app/screens/payment_screen.dart';
import 'package:traffic_violation_app/screens/profile_screen.dart';
import 'package:traffic_violation_app/screens/traffic_laws_screen.dart';
import 'package:traffic_violation_app/services/notification_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await NotificationService().initialize();
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
      },
    );
  }
}

// ── Notifications Screen ──────────────────────────────────────────
class NotificationsScreen extends StatelessWidget {
  const NotificationsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Thông báo'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildNotifItem(
            icon: Icons.warning_amber,
            color: Colors.red,
            title: 'Vi phạm mới: Không đội mũ bảo hiểm',
            subtitle: 'Vừa phát hiện vi phạm bởi camera giám sát',
            time: '2 phút trước',
          ),
          _buildNotifItem(
            icon: Icons.check_circle,
            color: Colors.green,
            title: 'Nộp phạt thành công',
            subtitle: 'Vi phạm VH01 đã được xử lý',
            time: '1 giờ trước',
          ),
          _buildNotifItem(
            icon: Icons.info_outline,
            color: Colors.blue,
            title: 'Nhắc nhở nộp phạt',
            subtitle: 'Bạn có 3 vi phạm chưa nộp phạt',
            time: '1 ngày trước',
          ),
          _buildNotifItem(
            icon: Icons.campaign_outlined,
            color: Colors.orange,
            title: 'Quy định mới',
            subtitle: 'Cập nhật mức phạt giao thông 2025',
            time: '3 ngày trước',
          ),
        ],
      ),
    );
  }

  Widget _buildNotifItem({
    required IconData icon,
    required Color color,
    required String title,
    required String subtitle,
    required String time,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.04),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.1)),
      ),
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: color),
        ),
        title: Text(
          title,
          style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 4),
            Text(subtitle,
                style: TextStyle(fontSize: 12, color: Colors.grey[600])),
            const SizedBox(height: 4),
            Text(time,
                style: TextStyle(fontSize: 11, color: Colors.grey[400])),
          ],
        ),
      ),
    );
  }
}

// ── Register Screen (placeholder) ─────────────────────────────────
class RegisterScreen extends StatelessWidget {
  const RegisterScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Đăng ký')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.person_add_outlined,
                  size: 64, color: Colors.grey[400]),
              const SizedBox(height: 16),
              const Text(
                'Tính năng đang phát triển',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              Text(
                'Chức năng đăng ký sẽ được cập nhật trong phiên bản tiếp theo.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey[600]),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Quay lại'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
