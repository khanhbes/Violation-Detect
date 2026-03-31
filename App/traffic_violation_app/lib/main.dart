import 'dart:async';
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
import 'package:traffic_violation_app/screens/qr_scan_screen.dart';
import 'package:traffic_violation_app/screens/support_screen.dart';
import 'package:traffic_violation_app/screens/forgot_password_screen.dart';
import 'package:traffic_violation_app/services/notification_service.dart';
import 'package:traffic_violation_app/services/push_notification_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/update_service.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;

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

class _MyAppState extends State<MyApp> with WidgetsBindingObserver {
  final AppSettings _settings = AppSettings();
  final UpdateService _updateService = UpdateService();
  final ApiService _apiService = ApiService();
  Timer? _updateTimer;
  StreamSubscription<ServerAddressConfig>? _serverAddressSub;
  StreamSubscription<fb.User?>? _authStateSub;
  bool _isCheckingUpdate = false;
  bool _isUpdateDialogVisible = false;
  DateTime? _lastUpdateCheckAt;
  String? _lastPromptedUpdateKey;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _settings.addListener(_onSettingsChanged);
    // Start config sync only when user is authenticated
    _authStateSub = fb.FirebaseAuth.instance.authStateChanges().listen((user) {
      if (user != null) {
        _apiService.startServerConfigAutoSync();
      } else {
        _apiService.stopServerConfigAutoSync();
      }
    });
    _startGlobalUpdateWatcher();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _updateTimer?.cancel();
    _serverAddressSub?.cancel();
    _authStateSub?.cancel();
    _apiService.stopServerConfigAutoSync();
    _settings.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    setState(() {}); // Rebuild MaterialApp when settings change
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _checkForUpdateGlobal(reason: 'app_resumed');
    }
  }

  void _startGlobalUpdateWatcher() {
    _serverAddressSub = _apiService.serverAddressStream.listen((_) {
      _checkForUpdateGlobal(force: true, reason: 'server_address_changed');
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _checkForUpdateGlobal(force: true, reason: 'app_started');
    });
    _updateTimer = Timer.periodic(const Duration(minutes: 2), (_) {
      _checkForUpdateGlobal(reason: 'periodic');
    });
  }

  Future<void> _checkForUpdateGlobal({
    bool force = false,
    required String reason,
  }) async {
    if (_isCheckingUpdate || _isUpdateDialogVisible) return;
    final now = DateTime.now();
    if (!force &&
        _lastUpdateCheckAt != null &&
        now.difference(_lastUpdateCheckAt!) < const Duration(seconds: 45)) {
      return;
    }

    if (PushNotificationService.navigatorKey.currentState == null) return;

    // Don't show the update dialog while on the splash screen — wait until
    // the user has navigated to login or home.
    // We detect splash by checking the navigator history for the '/' route.
    final navState = PushNotificationService.navigatorKey.currentState;
    bool isOnSplash = false;
    if (navState != null) {
      navState.popUntil((route) {
        if (route.settings.name == '/') isOnSplash = true;
        return true; // don't actually pop anything
      });
    }
    if (isOnSplash) {
      // Re-schedule after a 3-second delay — by then the user will be on login or home
      Future.delayed(const Duration(seconds: 3), () {
        _checkForUpdateGlobal(force: force, reason: '${reason}_splash_retry');
      });
      return;
    }

    _isCheckingUpdate = true;
    _lastUpdateCheckAt = now;
    try {
      final updateInfo = await _updateService.checkForUpdate();
      if (updateInfo == null) return;

      final version = updateInfo['version']?.toString() ?? '';
      final build = updateInfo['buildNumber']?.toString() ?? '';
      final key = '$version:$build';
      if (!force &&
          _lastPromptedUpdateKey != null &&
          _lastPromptedUpdateKey == key) {
        return;
      }
      _lastPromptedUpdateKey = key;

      _isUpdateDialogVisible = true;
      await _showGlobalUpdateDialog(updateInfo);
    } catch (e) {
      debugPrint('📱 Global update check skipped ($reason): $e');
    } finally {
      _isUpdateDialogVisible = false;
      _isCheckingUpdate = false;
    }
  }

  Future<void> _showGlobalUpdateDialog(Map<String, dynamic> updateInfo) async {
    final navState = PushNotificationService.navigatorKey.currentState;
    final navContext = navState?.overlay?.context ?? navState?.context;
    if (navContext == null) return;
    await UpdateService.showUpdateDialog(
      navContext,
      updateInfo: updateInfo,
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title:
          _settings.isVietnamese ? 'Vi Phạm Giao Thông' : 'Traffic Violations',
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
        '/qr-scan': (context) => const QrScanScreen(),
        '/support': (context) => const SupportScreen(),
        '/forgot-password': (context) => const ForgotPasswordScreen(),
      },
    );
  }
}
