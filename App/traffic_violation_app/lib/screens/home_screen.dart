import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
// mock_data import removed — all user data now comes from Firestore via AppSettings
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/notification_service.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/screens/violations_screen.dart';
import 'package:traffic_violation_app/screens/profile_screen.dart';
import 'package:traffic_violation_app/screens/vehicles_screen.dart'
    as home_vehicles;
import 'package:traffic_violation_app/widgets/app_info_dialogs.dart';
import 'package:traffic_violation_app/widgets/violation_image.dart';
import 'dart:async';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  // 0=Trang chủ, 1=Vi phạm, 2=Ví giấy tờ, 3=Cá nhân
  int _selectedIndex = 0;
  final ApiService _api = ApiService();
  final NotificationService _notif = NotificationService();
  final AppSettings _settings = AppSettings();
  List<Violation> _violations = [];
  bool _isLoading = true;
  bool _isRefreshingCore = false;
  StreamSubscription? _newViolationSub;
  StreamSubscription? _connectionSub;
  StreamSubscription? _firestoreSub;
  StreamSubscription? _notifCountSub; // realtime unread notification count
  StreamSubscription? _userPointsSub; // realtime GPLX points from Firestore
  String? _boundRealtimeUid;

  late AnimationController _fadeController;
  late AnimationController _slideController;

  // Document wallet state
  int _walletPage = 0;
  final PageController _walletPageController = PageController();
  static const List<String> _licenseClassOptions = <String>[
    'A1',
    'A2',
    'A',
    'B1',
    'B2',
    'C',
    'D',
    'E',
    'F',
  ];
  static const List<String> _licenseVehicleOptions = <String>[
    'Xe máy',
    'Ô tô',
    'Ô tô tải',
    'Ô tô khách',
    'Xe đầu kéo',
    'Xe máy chuyên dùng',
  ];

  @override
  void initState() {
    super.initState();

    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _slideController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );

    _settings.addListener(_onSettingsChanged);
    _initServices();
  }

  Future<void> _initServices() async {
    await _notif.initialize();

    NotificationService.onNotificationTap = (payload) {
      if (payload != null && mounted) {
        Navigator.pushNamed(context, '/violation-detail', arguments: payload);
      }
    };

    _newViolationSub = _api.newViolationStream.listen((violation) {
      final activeUid = _resolveUid();
      final violationUid = violation.userId.trim();
      if (violationUid.isNotEmpty &&
          (activeUid == null || activeUid != violationUid)) {
        debugPrint(
          'Skip popup for other user: violation user=$violationUid, active=$activeUid',
        );
        return;
      }
      // Merge into local list (dedupe by id) so stats update immediately
      if (mounted) {
        setState(() {
          final exists = _violations.any((v) => v.id == violation.id);
          if (!exists) _violations = [violation, ..._violations];
        });
      }
      _notif.showViolationNotification(violation);
      _settings.addNotification();
      _showNewViolationDialog(violation);
    });

    _api.connectWebSocket();

    _connectionSub = _api.connectionStream.distinct().listen((isConnected) {
      if (mounted) setState(() {});
    });

    _bindUserRealtimeStreams(force: true);

    _api.testConnection();

    _fadeController.forward();
    _slideController.forward();
  }

  String? _resolveUid() {
    final settingsUid = _settings.uid?.trim();
    if (settingsUid != null && settingsUid.isNotEmpty) return settingsUid;
    final authUid = fb.FirebaseAuth.instance.currentUser?.uid.trim();
    if (authUid == null || authUid.isEmpty) return null;
    return authUid;
  }

  void _bindUserRealtimeStreams({bool force = false}) {
    final uid = _resolveUid();

    if (uid == null) {
      _boundRealtimeUid = null;
      _firestoreSub?.cancel();
      _firestoreSub = null;
      _notifCountSub?.cancel();
      _notifCountSub = null;
      _userPointsSub?.cancel();
      _userPointsSub = null;
      if (_settings.unreadNotifications != 0) {
        _settings.setNotificationCount(0);
      }
      if (mounted) {
        setState(() {
          _violations = [];
          _isLoading = false;
        });
      } else {
        _violations = [];
        _isLoading = false;
      }
      return;
    }

    if (!force &&
        _boundRealtimeUid == uid &&
        _firestoreSub != null &&
        _notifCountSub != null &&
        _userPointsSub != null) {
      return;
    }

    _boundRealtimeUid = uid;
    _firestoreSub?.cancel();
    _notifCountSub?.cancel();
    _userPointsSub?.cancel();

    if (mounted) {
      setState(() => _isLoading = true);
    } else {
      _isLoading = true;
    }

    _firestoreSub = FirestoreService().violationsStream(userId: uid).listen(
      (violations) {
        if (!mounted) return;
        setState(() {
          _violations = violations;
          _isLoading = false;
        });
        if (violations.isEmpty) {
          _syncFromApiFallback(uid);
        }
      },
      onError: (error, stackTrace) {
        final isDenied = FirestoreService.isPermissionDeniedError(error);
        debugPrint(
            '❌ Home violations stream error${isDenied ? ' (PERMISSION_DENIED)' : ''}: $error');
        if (!mounted) return;
        setState(() {
          _violations = [];
          _isLoading = false;
        });
        if (isDenied) _syncFromApiFallback(uid);
      },
    );

    _notifCountSub = FirestoreService().notificationsStream(uid).listen(
      (notifs) {
        final unread = notifs.where((n) => !n.isRead).length;
        _settings.setNotificationCount(unread);
      },
      onError: (error, stackTrace) {
        final isDenied = FirestoreService.isPermissionDeniedError(error);
        debugPrint(
            '❌ Home notifications stream error${isDenied ? ' (PERMISSION_DENIED)' : ''}: $error');
        if (_settings.unreadNotifications != 0) {
          _settings.setNotificationCount(0);
        }
      },
    );

    _userPointsSub = FirebaseFirestore.instance
        .collection('users')
        .doc(uid)
        .snapshots()
        .listen(
      (doc) {
        if (doc.exists && mounted) {
          final data = doc.data();
          if (data != null) {
            _settings.applyRemoteProfileData(data);
          }
        }
      },
      onError: (error, stackTrace) {
        final isDenied = FirestoreService.isPermissionDeniedError(error);
        debugPrint(
            '❌ Home user profile stream error${isDenied ? ' (PERMISSION_DENIED)' : ''}: $error');
      },
    );
  }

  /// Fallback: fetch violations from backend API when Firestore stream is empty.
  /// Merges results into [_violations] by id (deduplication).
  Future<void> _syncFromApiFallback(String uid) async {
    try {
      final apiViolations = await _api.fetchViolations();
      if (!mounted || apiViolations.isEmpty) return;
      setState(() {
        final ids = _violations.map((v) => v.id).toSet();
        final newOnes = apiViolations.where((v) => !ids.contains(v.id)).toList();
        if (newOnes.isNotEmpty) {
          _violations = [..._violations, ...newOnes]
            ..sort((a, b) => b.timestamp.compareTo(a.timestamp));
        }
      });
      debugPrint('🔄 Home fallback fetch: ${apiViolations.length} violations from API');
    } catch (e) {
      debugPrint('⚠️ Home fallback fetch failed: $e');
    }
  }

  Future<void> _refreshCoreData() async {
    if (_isRefreshingCore) return;
    _isRefreshingCore = true;
    try {
      final uid = _resolveUid();
      if (uid == null) {
        return;
      }

      final report = await _api.refreshCoreData(
        uid,
        taskTimeout: const Duration(seconds: 7),
        hardTimeout: const Duration(seconds: 14),
      );

      await _syncFromApiFallback(uid);
      _bindUserRealtimeStreams(force: true);
      if (mounted) setState(() {});

      if (mounted) {
        final failedTasks = report.taskStatus.entries
            .where((entry) => entry.value != 'success')
            .map((entry) => entry.key)
            .toList();
        if (failedTasks.isNotEmpty) {
          final statusText = report.timedOut
              ? _settings.tr(
                  'Làm mới một phần (hết thời gian chờ)',
                  'Partial refresh (timeout)',
                )
              : _settings.tr(
                  'Làm mới một phần',
                  'Partial refresh',
                );
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('$statusText: ${failedTasks.join(', ')}'),
              backgroundColor: AppTheme.warningColor,
              behavior: SnackBarBehavior.floating,
              duration: const Duration(seconds: 3),
            ),
          );
        }
      }
    } finally {
      _isRefreshingCore = false;
    }
  }

  void _showNewViolationDialog(Violation violation) {
    if (!mounted) return;

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) {
        final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');
        final isDark = Theme.of(ctx).brightness == Brightness.dark;
        return AlertDialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
          contentPadding: const EdgeInsets.all(24),
          title: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.dangerColor.withOpacity(0.1),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.warning_amber_rounded,
                    color: AppTheme.dangerColor),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  _settings.tr('Phát hiện vi phạm!', 'New Violation!'),
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 18,
                    color: isDark ? Colors.white : AppTheme.textPrimary,
                  ),
                ),
              ),
            ],
          ),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  violation.violationType,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: isDark ? Colors.white : AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '${_settings.tr('Mức phạt', 'Fine')}: ${formatter.format(violation.fineAmount)}',
                  style: const TextStyle(
                      color: AppTheme.dangerColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 16),
                ),
                const SizedBox(height: 12),
                if (violation.imageUrl.isNotEmpty)
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: SizedBox(
                      width: double.maxFinite,
                      child: ViolationImage(
                        imageUrl: violation.imageUrl,
                        height: 140,
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: Text(_settings.tr('Đóng', 'Close'),
                  style: TextStyle(
                      color: isDark
                          ? const Color(0xFF9E9E9E)
                          : AppTheme.textSecondary)),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.pop(ctx);
                // Also switch to 'Violations' tab just in case
                setState(() => _selectedIndex = 1);
                Navigator.pushNamed(context, '/violation-detail',
                    arguments: violation.id);
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.primaryColor,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10)),
              ),
              child: Text(_settings.tr('Nộp phạt ngay', 'Pay Fine'),
                  style: const TextStyle(
                      color: Colors.white, fontWeight: FontWeight.bold)),
            ),
          ],
        );
      },
    );
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    _walletPageController.dispose();
    _settings.removeListener(_onSettingsChanged);
    _newViolationSub?.cancel();
    _connectionSub?.cancel();
    _firestoreSub?.cancel();
    _notifCountSub?.cancel();
    _userPointsSub?.cancel();
    _boundRealtimeUid = null;
    super.dispose();
  }

  void _onSettingsChanged() {
    _bindUserRealtimeStreams();
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _selectedIndex,
        children: [
          _buildHomePage(),
          const ViolationsScreen(embedded: true),
          _buildWalletFullPage(),
          const ProfileScreen(embedded: true),
        ],
      ),
      bottomNavigationBar: _buildBottomNav(),
      extendBody: true,
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  BOTTOM NAVIGATION BAR — Matches reference design
  //  Trang chủ | Vi phạm | [+ FAB] | Ví giấy tờ | Cá nhân
  // ═══════════════════════════════════════════════════════════════
  Widget _buildBottomNav() {
    final pendingCount = _violations.where((v) => v.canPay).length;

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(24),
          topRight: Radius.circular(24),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.08),
            blurRadius: 24,
            offset: const Offset(0, -6),
          ),
        ],
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(4, 6, 4, 6),
          child: Row(
            children: [
              // Trang chủ
              Expanded(
                  child: _buildNavItem(0, Icons.home_outlined,
                      Icons.home_rounded, _settings.tr('Trang chủ', 'Home'))),
              // Vi phạm
              Expanded(
                  child: _buildNavItem(
                      1,
                      Icons.warning_amber_outlined,
                      Icons.warning_amber_rounded,
                      _settings.tr('Vi phạm', 'Violations'),
                      badge: pendingCount)),
              // Center FAB (+)
              _buildCenterFAB(),
              // Ví giấy tờ
              Expanded(
                  child: _buildNavItem(
                      2,
                      Icons.account_balance_wallet_outlined,
                      Icons.account_balance_wallet_rounded,
                      _settings.tr('Ví giấy tờ', 'Wallet'))),
              // Cá nhân
              Expanded(
                  child: _buildNavItem(
                      3,
                      Icons.person_outline_rounded,
                      Icons.person_rounded,
                      _settings.tr('Cá nhân', 'Profile'))),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(
      int index, IconData icon, IconData activeIcon, String label,
      {int badge = 0}) {
    final isSelected = _selectedIndex == index;

    return GestureDetector(
      onTap: () => setState(() => _selectedIndex = index),
      behavior: HitTestBehavior.opaque,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 250),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
            decoration: BoxDecoration(
              color: isSelected
                  ? AppTheme.primaryColor.withOpacity(0.1)
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Badge(
              isLabelVisible: badge > 0,
              label: Text('$badge',
                  style: const TextStyle(fontSize: 9, color: Colors.white)),
              backgroundColor: AppTheme.primaryColor,
              child: Icon(
                isSelected ? activeIcon : icon,
                color:
                    isSelected ? AppTheme.primaryColor : AppTheme.textSecondary,
                size: 24,
              ),
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              fontSize: 10,
              fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
              color:
                  isSelected ? AppTheme.primaryColor : AppTheme.textSecondary,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }

  /// Center FAB — large red circle with + icon
  Widget _buildCenterFAB() {
    return GestureDetector(
      onTap: _showFunctionsSheet,
      child: Container(
        width: 60,
        height: 60,
        margin: const EdgeInsets.symmetric(horizontal: 8),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFFE53935), Color(0xFFD32F2F)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: AppTheme.primaryColor.withOpacity(0.4),
              blurRadius: 16,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: const Icon(Icons.add_rounded, color: Colors.white, size: 32),
      ),
    );
  }

  void _openNotifications() {
    _settings.clearNotifications();
    Navigator.pushNamed(context, '/notifications');
  }

  /// Bottom sheet when tapping the + FAB
  void _showFunctionsSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        final isDark = Theme.of(ctx).brightness == Brightness.dark;
        final sheetBg = isDark ? const Color(0xFF121A28) : Colors.white;
        final dividerColor =
            isDark ? const Color(0xFF2B3650) : AppTheme.dividerColor;
        final titleColor =
            isDark ? const Color(0xFFE8EEFB) : AppTheme.textPrimary;

        return Container(
          decoration: BoxDecoration(
            color: sheetBg,
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(28),
              topRight: Radius.circular(28),
            ),
            border: Border(
              top: BorderSide(color: dividerColor.withOpacity(0.9)),
            ),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Drag handle
              Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(top: 12),
                decoration: BoxDecoration(
                  color: dividerColor,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFFE53935), Color(0xFFD32F2F)],
                        ),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(Icons.apps_rounded,
                          color: Colors.white, size: 20),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      _settings.tr('Chức năng', 'Functions'),
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                        color: titleColor,
                      ),
                    ),
                  ],
                ),
              ),
              Divider(color: dividerColor),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
                child: GridView.count(
                  crossAxisCount: 4,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  crossAxisSpacing: 8,
                  mainAxisSpacing: 12,
                  childAspectRatio: 0.85,
                  children: [
                    _buildSheetItem(
                        Icons.search_rounded,
                        _settings.tr('Tra cứu\nvi phạm', 'Search\nViolations'),
                        const Color(0xFFE53935), () {
                      Navigator.pop(ctx);
                      _showViolationLookup();
                    }),
                    _buildSheetItem(
                        Icons.payment_rounded,
                        _settings.tr(
                            'Nộp phạt\ntrực tuyến', 'Pay Fines\nOnline'),
                        const Color(0xFFF57C00), () {
                      Navigator.pop(ctx);
                      _showPaymentList();
                    }),
                    _buildSheetItem(
                        Icons.history_rounded,
                        _settings.tr('Lịch sử\nvi phạm', 'Violation\nHistory'),
                        const Color(0xFF1565C0), () {
                      Navigator.pop(ctx);
                      setState(() => _selectedIndex = 1);
                    }),
                    _buildSheetItem(
                        Icons.rate_review_rounded,
                        _settings.tr('Khiếu nại\nvi phạm', 'File\nComplaint'),
                        const Color(0xFF2E7D32), () {
                      Navigator.pop(ctx);
                      Navigator.pushNamed(context, '/complaint');
                    }),
                    _buildSheetItem(
                        Icons.gavel_rounded,
                        _settings.tr('Luật\nGTĐB', 'Traffic\nLaws'),
                        AppTheme.warningColor, () {
                      Navigator.pop(ctx);
                      Navigator.pushNamed(context, '/traffic-laws');
                    }),
                    _buildSheetItem(
                        Icons.qr_code_scanner_rounded,
                        _settings.tr('Quét\nQR Code', 'Scan\nQR Code'),
                        AppTheme.infoColor, () {
                      Navigator.pop(ctx);
                      Navigator.pushNamed(context, '/qr-scan');
                    }),
                    _buildSheetItem(
                        Icons.directions_car_rounded,
                        _settings.tr('Phương\ntiện', 'My\nVehicles'),
                        const Color(0xFF5C6BC0), () {
                      Navigator.pop(ctx);
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) =>
                                const home_vehicles.VehiclesScreen()),
                      );
                    }),
                    _buildSheetItem(
                        Icons.support_agent_rounded,
                        _settings.tr('Hỗ trợ\ntrực tuyến', 'Online\nSupport'),
                        AppTheme.successColor, () {
                      Navigator.pop(ctx);
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Hotline: 1900.xxxx')),
                      );
                    }),
                    _buildSheetItem(
                      Icons.notifications_outlined,
                      _settings.tr('Thông\nbáo', 'Notifi-\ncations'),
                      Colors.deepOrange,
                      () {
                        Navigator.pop(ctx);
                        _openNotifications();
                      },
                      badgeCount: _settings.unreadNotifications,
                    ),
                    _buildSheetItem(
                        Icons.badge_rounded,
                        _settings.tr('Ví\ngiấy tờ', 'Doc\nWallet'),
                        const Color(0xFF1A237E), () {
                      Navigator.pop(ctx);
                      setState(() => _selectedIndex = 2);
                    }),
                    _buildSheetItem(
                        Icons.router_rounded,
                        _settings.tr('Chỉnh\nIP Server', 'Server\nIP'),
                        AppTheme.primaryColor, () {
                      Navigator.pop(ctx);
                      setState(() => _selectedIndex = 3);
                    }),
                    _buildSheetItem(
                        Icons.info_outline_rounded,
                        _settings.tr('Về\nứng dụng', 'About\nApp'),
                        AppTheme.textSecondary, () {
                      Navigator.pop(ctx);
                      AppInfoDialogs.showAboutDialog(context, _settings);
                    }),
                  ],
                ),
              ),
              SizedBox(height: MediaQuery.of(context).padding.bottom + 12),
            ],
          ),
        );
      },
    );
  }

  Widget _buildSheetItem(
      IconData icon, String label, Color color, VoidCallback onTap,
      {int badgeCount = 0}) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final tileTextColor =
        isDark ? const Color(0xFFDCE5F7) : AppTheme.textPrimary;
    final tileBorderColor = color.withOpacity(isDark ? 0.32 : 0.16);
    final tileBg = color.withOpacity(isDark ? 0.22 : 0.1);

    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Stack(
            clipBehavior: Clip.none,
            children: [
              AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  color: tileBg,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: tileBorderColor),
                ),
                child: Icon(icon, color: color, size: 24),
              ),
              if (badgeCount > 0)
                Positioned(
                  right: -4,
                  top: -4,
                  child: Container(
                    constraints: const BoxConstraints(minWidth: 18),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                    decoration: BoxDecoration(
                      color: AppTheme.dangerColor,
                      borderRadius: BorderRadius.circular(99),
                      border: Border.all(
                        color: isDark ? const Color(0xFF121A28) : Colors.white,
                        width: 1.2,
                      ),
                    ),
                    child: Text(
                      badgeCount > 99 ? '99+' : '$badgeCount',
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 9,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            label,
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: tileTextColor,
              height: 1.2,
            ),
          ),
        ],
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  HOME PAGE
  // ═══════════════════════════════════════════════════════════════
  Widget _buildHomePage() {
    return RefreshIndicator(
      onRefresh: _refreshCoreData,
      color: AppTheme.primaryColor,
      child: CustomScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        slivers: [
          _buildHeader(),
          SliverToBoxAdapter(child: _buildConnectionBanner()),
          SliverToBoxAdapter(
            child: FadeTransition(
              opacity: _fadeController,
              child: _buildDocumentWallet(),
            ),
          ),
          SliverToBoxAdapter(
            child: FadeTransition(
              opacity: _fadeController,
              child: _buildQuickMenu(),
            ),
          ),
          SliverToBoxAdapter(
            child: FadeTransition(
              opacity: _fadeController,
              child: _buildFineOverview(),
            ),
          ),
          // Bottom padding so content is never hidden behind the navigation bar
          const SliverToBoxAdapter(child: SizedBox(height: 110)),
        ],
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  WALLET FULL PAGE (tab index 2)
  // ═══════════════════════════════════════════════════════════════
  Widget _buildWalletFullPage() {
    final licenses = _walletLicenses();

    return Scaffold(
      backgroundColor: AppTheme.surfaceColor,
      body: RefreshIndicator(
        onRefresh: _refreshCoreData,
        color: AppTheme.primaryColor,
        child: CustomScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          slivers: [
            // Header
            SliverToBoxAdapter(
              child: Container(
                decoration: const BoxDecoration(
                  gradient: AppTheme.headerGradient,
                  borderRadius: BorderRadius.only(
                    bottomLeft: Radius.circular(28),
                    bottomRight: Radius.circular(28),
                  ),
                ),
                child: SafeArea(
                  bottom: false,
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Icon(Icons.wallet_rounded,
                              color: Colors.white, size: 22),
                        ),
                        const SizedBox(width: 12),
                        Text(
                          _settings.tr('Ví giấy tờ', 'Document Wallet'),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            // Cards
            SliverPadding(
              padding: const EdgeInsets.all(16),
              sliver: SliverList(
                delegate: SliverChildListDelegate([
                  _buildSectionLabel(
                      _settings.tr('Căn cước công dân', 'Citizen ID Card'),
                      Icons.credit_card_rounded,
                      const Color(0xFF1A237E)),
                  const SizedBox(height: 10),
                  GestureDetector(
                    onTap: () => _showCccdDetail(null),
                    child: _buildCccdCard(null),
                  ),
                  const SizedBox(height: 20),
                  _buildSectionLabel(
                      _settings.tr('Giấy phép lái xe', 'Driver License'),
                      Icons.badge_rounded,
                      AppTheme.primaryColor),
                  const SizedBox(height: 10),
                  ...licenses.asMap().entries.map((entry) {
                    final index = entry.key;
                    final license = entry.value;
                    final licenseType = _resolveLicenseType(license);
                    final licensePoints = _pointsForLicenseType(licenseType);
                    final isDisabled = licensePoints <= 0;
                    return Padding(
                      padding: EdgeInsets.only(
                          bottom: index == licenses.length - 1 ? 0 : 12),
                      child: GestureDetector(
                        onTap: () => _showLicenseDetail(
                          licenseIndex: index,
                          licenseClass: license['class'] ?? '',
                          vehicleType: license['vehicleType'] ?? '',
                          issueDate: license['issueDate'] ?? '',
                          expiryDate: license['expiryDate'] ?? '',
                          licenseNumber: license['licenseNumber'] ?? '',
                          issuedBy: license['issuedBy'] ?? '',
                          licenseType: licenseType,
                          points: licensePoints,
                          isDisabled: isDisabled,
                        ),
                        child: _buildLicenseCard(
                          licenseClass: license['class'] ?? '',
                          vehicleType: license['vehicleType'] ?? '',
                          issueDate: license['issueDate'] ?? '',
                          expiryDate: license['expiryDate'] ?? '',
                          licenseNumber: license['licenseNumber'] ?? '',
                          licenseType: licenseType,
                          points: licensePoints,
                          isDisabled: isDisabled,
                        ),
                      ),
                    );
                  }),
                  if (licenses.isEmpty) ...[
                    _buildAddLicensePromptCard(),
                    const SizedBox(height: 12),
                  ],
                  const SizedBox(height: 100),
                ]),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionLabel(String title, IconData icon, Color color) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: color, size: 16),
        ),
        const SizedBox(width: 8),
        Text(
          title,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: color,
          ),
        ),
      ],
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  HEADER (Red gradient with user info)
  // ═══════════════════════════════════════════════════════════════
  Widget _buildHeader() {
    final avatarUrl = _settings.userAvatar;
    final displayName = _settings.userName;
    final idCard = _settings.userIdCard;

    return SliverToBoxAdapter(
      child: Container(
        decoration: const BoxDecoration(
          gradient: AppTheme.headerGradient,
          borderRadius: BorderRadius.only(
            bottomLeft: Radius.circular(28),
            bottomRight: Radius.circular(28),
          ),
        ),
        child: SafeArea(
          bottom: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
            child: Column(
              children: [
                // User row
                Row(
                  children: [
                    Container(
                      width: 48,
                      height: 48,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                            color: Colors.white.withOpacity(0.4), width: 2),
                        image: _buildAvatarDecoration(avatarUrl),
                      ),
                      child: (avatarUrl.isEmpty)
                          ? Center(
                              child: Text(
                                displayName.isNotEmpty
                                    ? displayName.substring(0, 1).toUpperCase()
                                    : '?',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            )
                          : null,
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _settings.tr('Xin chào,', 'Hello,'),
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.8),
                              fontSize: 13,
                            ),
                          ),
                          Text(
                            displayName,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: IconButton(
                        icon: Badge(
                          isLabelVisible: _settings.unreadNotifications > 0,
                          label: Text(
                            '${_settings.unreadNotifications}',
                            style: const TextStyle(
                                fontSize: 9,
                                color: Colors.white,
                                fontWeight: FontWeight.w700),
                          ),
                          backgroundColor: AppTheme.accentColor,
                          child: const Icon(Icons.notifications_outlined,
                              color: Colors.white),
                        ),
                        onPressed: _openNotifications,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                // Verification bar
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.verified_user_outlined,
                          color: Colors.white.withOpacity(0.9), size: 20),
                      const SizedBox(width: 10),
                      Text(
                        '${_settings.tr('Xác thực', 'Verified')}: $idCard',
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.9),
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const Spacer(),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: AppTheme.accentColor,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          _settings.tr('Đã xác minh', 'Verified'),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  CONNECTION BANNER
  // ═══════════════════════════════════════════════════════════════
  Widget _buildConnectionBanner() {
    final status = _api.connectionStatus;

    final MaterialColor color;
    final IconData icon;
    final String label;
    final String badge;

    switch (status) {
      case ConnectionStatus.connected:
        color = Colors.green;
        icon = Icons.bolt_rounded;
        label = _settings.tr('Đã kết nối', 'Connected');
        badge = 'LIVE';
      case ConnectionStatus.connecting:
        color = Colors.orange;
        icon = Icons.sync_rounded;
        label = _settings.tr('Đang kết nối...', 'Connecting...');
        badge = '';
      case ConnectionStatus.disconnected:
        color = Colors.red;
        icon = Icons.wifi_off_rounded;
        label = _settings.tr('Mất kết nối máy chủ', 'Server disconnected');
        badge = '';
    }

    return AnimatedContainer(
      duration: const Duration(milliseconds: 500),
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 0),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        border: Border.all(color: color.withOpacity(0.25)),
      ),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, color: color, size: 18),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                color: color[700],
                fontWeight: FontWeight.w600,
                fontSize: 13,
              ),
            ),
          ),
          if (badge.isNotEmpty)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: color.withOpacity(0.15),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                badge,
                style: TextStyle(
                  color: color[800],
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.5,
                ),
              ),
            ),
          if (status == ConnectionStatus.disconnected)
            GestureDetector(
              onTap: () {
                _api.reconnect();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                      content: Text(_settings.tr(
                          'Đang kết nối lại...', 'Reconnecting...'))),
                );
              },
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.refresh, size: 14, color: Colors.red[700]),
                    const SizedBox(width: 4),
                    Text(
                      _settings.tr('Thử lại', 'Retry'),
                      style: TextStyle(
                        color: Colors.red[700],
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  SECTION HEADER — Colored box to visually distinguish sections
  // ═══════════════════════════════════════════════════════════════
  Widget _buildSectionHeader({
    required String title,
    required IconData icon,
    required Color color,
    Widget? trailing,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [color.withOpacity(0.08), color.withOpacity(0.03)],
          begin: Alignment.centerLeft,
          end: Alignment.centerRight,
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.12)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(7),
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(9),
            ),
            child: Icon(icon, color: color, size: 18),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              title,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w800,
                color: color,
              ),
            ),
          ),
          if (trailing != null) trailing,
        ],
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  DOCUMENT WALLET (Ví giấy tờ — CCCD + GPLX + Điểm)
  // ═══════════════════════════════════════════════════════════════
  List<Map<String, String>> _walletLicenses() {
    return _settings.driverLicenses
        .where((l) =>
            (l['class'] ?? '').isNotEmpty &&
            (l['licenseNumber'] ?? '').isNotEmpty)
        .map((l) => Map<String, String>.from(l))
        .toList();
  }

  String _resolveLicenseType(Map<String, String> license) {
    final vehicleType = (license['vehicleType'] ?? '').toLowerCase();
    final cls = (license['class'] ?? '').toUpperCase();
    if (vehicleType.contains('xe máy') ||
        vehicleType.contains('motor') ||
        cls.startsWith('A')) {
      return 'motorcycle';
    }
    if (vehicleType.contains('ô tô') ||
        vehicleType.contains('o to') ||
        vehicleType.contains('car') ||
        cls.startsWith('B') ||
        cls.startsWith('C') ||
        cls.startsWith('D') ||
        cls.startsWith('E') ||
        cls.startsWith('F')) {
      return 'car';
    }
    return 'motorcycle';
  }

  bool _isMotorcycleLicenseClass(String licenseClass) {
    return licenseClass.toUpperCase().trim().startsWith('A');
  }

  String _defaultVehicleTypeForLicenseClass(String licenseClass) {
    return _isMotorcycleLicenseClass(licenseClass) ? 'Xe máy' : 'Ô tô';
  }

  DateTime? _tryParseLicenseDate(String raw) {
    final value = raw.trim();
    if (value.isEmpty) return null;
    final direct = DateTime.tryParse(value);
    if (direct != null) return direct;

    final slash = RegExp(r'^(\d{2})/(\d{2})/(\d{4})$');
    final dash = RegExp(r'^(\d{2})-(\d{2})-(\d{4})$');
    final slashMatch = slash.firstMatch(value);
    if (slashMatch != null) {
      final day = int.tryParse(slashMatch.group(1)!);
      final month = int.tryParse(slashMatch.group(2)!);
      final year = int.tryParse(slashMatch.group(3)!);
      if (day != null && month != null && year != null) {
        return DateTime(year, month, day);
      }
    }
    final dashMatch = dash.firstMatch(value);
    if (dashMatch != null) {
      final day = int.tryParse(dashMatch.group(1)!);
      final month = int.tryParse(dashMatch.group(2)!);
      final year = int.tryParse(dashMatch.group(3)!);
      if (day != null && month != null && year != null) {
        return DateTime(year, month, day);
      }
    }
    return null;
  }

  String _formatLicenseDate(DateTime date) {
    return DateFormat('dd/MM/yyyy').format(date);
  }

  String _autoLicenseExpiry({
    required String licenseClass,
    required String issueDate,
  }) {
    if (_isMotorcycleLicenseClass(licenseClass)) {
      return _settings.tr('Vĩnh viễn', 'Permanent');
    }
    final issued = _tryParseLicenseDate(issueDate) ?? DateTime.now();
    final expiry = DateTime(issued.year + 10, issued.month, issued.day);
    return _formatLicenseDate(expiry);
  }

  int _pointsForLicenseType(String licenseType) {
    if (licenseType == 'car') {
      return _settings.carLicensePoints.clamp(0, 12).toInt();
    }
    return _settings.motoLicensePoints.clamp(0, 12).toInt();
  }

  Map<String, dynamic> _legacyLicenseFieldsFrom(
      List<Map<String, String>> licenses) {
    Map<String, String>? car;
    Map<String, String>? moto;
    for (final l in licenses) {
      final vehicleType = (l['vehicleType'] ?? '').toLowerCase();
      final cls = (l['class'] ?? '').toUpperCase();
      if (car == null &&
          (vehicleType.contains('ô tô') ||
              vehicleType.contains('car') ||
              cls.startsWith('B') ||
              cls.startsWith('C') ||
              cls.startsWith('D') ||
              cls.startsWith('E') ||
              cls == 'FB2')) {
        car = l;
      }
      if (moto == null &&
          (vehicleType.contains('xe máy') ||
              vehicleType.contains('motor') ||
              cls.startsWith('A'))) {
        moto = l;
      }
    }

    final first = licenses.isNotEmpty ? licenses.first : null;
    return {
      'licenseNumber': first?['licenseNumber'] ?? '',
      'licenseIssueDate': first?['issueDate'] ?? '',
      'licenseExpiryDate': first?['expiryDate'] ?? '',
      'licenseIssuedBy': first?['issuedBy'] ?? _settings.userLicenseIssuedBy,
      'carLicenseClass': car?['class'] ?? '',
      'motoLicenseClass': moto?['class'] ?? '',
    };
  }

  Future<void> _submitWalletUpdateRequest(
    Map<String, dynamic> requestData, {
    required String successVi,
    required String successEn,
  }) async {
    final uid = _resolveUid();
    if (uid == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(_settings.tr(
            'Không xác định được tài khoản đang đăng nhập.',
            'Unable to resolve current signed-in account.',
          )),
          backgroundColor: AppTheme.warningColor,
        ),
      );
      return;
    }

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => const Center(child: CircularProgressIndicator()),
    );

    try {
      await FirestoreService().requestProfileUpdate(uid, requestData);
      if (!mounted) return;
      Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(_settings.tr(successVi, successEn)),
          backgroundColor: AppTheme.successColor,
        ),
      );
    } catch (e) {
      if (!mounted) return;
      Navigator.pop(context);
      final raw = e.toString().trim();
      final msg = raw.isEmpty
          ? _settings.tr('Lỗi khi gửi yêu cầu chỉnh sửa giấy tờ.',
              'Failed to send document update request.')
          : raw.replaceFirst('Exception: ', '');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(msg),
          backgroundColor: AppTheme.dangerColor,
        ),
      );
    }
  }

  void _showEditCccdDialog() {
    final fullNameController = TextEditingController(text: _settings.userName);
    final idCardController = TextEditingController(text: _settings.userIdCard);
    final genderController = TextEditingController(text: _settings.userGender);
    final nationalityController = TextEditingController(
      text:
          _settings.userNationality.isNotEmpty ? _settings.userNationality : '',
    );
    final originController = TextEditingController(
      text: _settings.userPlaceOfOrigin,
    );
    final addressController =
        TextEditingController(text: _settings.userAddress);
    final occupationController =
        TextEditingController(text: _settings.userOccupation);
    final issueDateController =
        TextEditingController(text: _settings.userIdCardIssueDate);
    final expiryDateController =
        TextEditingController(text: _settings.userIdCardExpiryDate);
    final dobController =
        TextEditingController(text: _settings.userDateOfBirth);

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(_settings.tr('Chỉnh sửa CCCD', 'Edit Citizen ID')),
        content: SingleChildScrollView(
          child: SizedBox(
            width: 360,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: fullNameController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Họ và tên', 'Full name'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: idCardController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Số CCCD', 'ID Number'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: dobController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Ngày sinh', 'Date of birth'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: genderController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Giới tính', 'Gender'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: nationalityController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Quốc tịch', 'Nationality'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: originController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Quê quán', 'Place of origin'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: addressController,
                  decoration: InputDecoration(
                    labelText:
                        _settings.tr('Nơi thường trú', 'Permanent address'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: issueDateController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Ngày cấp', 'Issue date'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: expiryDateController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Có giá trị đến', 'Valid until'),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: occupationController,
                  decoration: InputDecoration(
                    labelText: _settings.tr('Nghề nghiệp', 'Occupation'),
                  ),
                ),
              ],
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(_settings.tr('Hủy', 'Cancel')),
          ),
          ElevatedButton(
            onPressed: () {
              final fullName = fullNameController.text.trim();
              final idCard = idCardController.text.trim();
              if (fullName.isEmpty || idCard.isEmpty) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(_settings.tr(
                      'Vui lòng nhập đầy đủ Họ tên và Số CCCD',
                      'Please enter both full name and ID number',
                    )),
                    backgroundColor: AppTheme.warningColor,
                  ),
                );
                return;
              }
              Navigator.pop(ctx);
              _submitWalletUpdateRequest(
                {
                  'fullName': fullName,
                  'idCard': idCard,
                  'gender': genderController.text.trim(),
                  'nationality': nationalityController.text.trim(),
                  'placeOfOrigin': originController.text.trim(),
                  'address': addressController.text.trim(),
                  'idCardIssueDate': issueDateController.text.trim(),
                  'idCardExpiryDate': expiryDateController.text.trim(),
                  'dateOfBirth': dobController.text.trim(),
                  'occupation': occupationController.text.trim(),
                  'requestSection': 'wallet_cccd',
                },
                successVi:
                    'Đã gửi yêu cầu cập nhật CCCD. Vui lòng chờ admin duyệt.',
                successEn:
                    'Citizen ID update request sent. Please wait for admin approval.',
              );
            },
            child: Text(_settings.tr('Gửi duyệt', 'Submit for approval')),
          ),
        ],
      ),
    );
  }

  void _showEditLicenseDialog({required int licenseIndex}) {
    final licenses =
        _walletLicenses().map((l) => Map<String, String>.from(l)).toList();
    final isAddMode =
        licenses.isEmpty || licenseIndex < 0 || licenseIndex >= licenses.length;
    final safeIndex = isAddMode
        ? -1
        : (licenseIndex < 0
            ? 0
            : (licenseIndex >= licenses.length
                ? licenses.length - 1
                : licenseIndex));
    final current = isAddMode ? <String, String>{} : licenses[safeIndex];

    final classOptions = _licenseClassOptions.toList(growable: true);
    final currentClass = (current['class'] ?? '').trim().toUpperCase();
    if (currentClass.isNotEmpty && !classOptions.contains(currentClass)) {
      classOptions.add(currentClass);
    }
    String selectedClass =
        currentClass.isNotEmpty ? currentClass : classOptions.first;

    final vehicleOptions = _licenseVehicleOptions.toList(growable: true);
    final currentVehicleType = (current['vehicleType'] ?? '').trim();
    if (currentVehicleType.isNotEmpty &&
        !vehicleOptions.contains(currentVehicleType)) {
      vehicleOptions.add(currentVehicleType);
    }
    String selectedVehicleType = currentVehicleType.isNotEmpty
        ? currentVehicleType
        : _defaultVehicleTypeForLicenseClass(selectedClass);

    final issueDateController =
        TextEditingController(text: current['issueDate'] ?? '');
    final expiryDateController =
        TextEditingController(text: current['expiryDate'] ?? '');
    final numberController =
        TextEditingController(text: current['licenseNumber'] ?? '');
    final issuedByController = TextEditingController(
      text: current['issuedBy'] ?? _settings.userLicenseIssuedBy,
    );
    if (expiryDateController.text.trim().isEmpty) {
      expiryDateController.text = _autoLicenseExpiry(
        licenseClass: selectedClass,
        issueDate: issueDateController.text.trim(),
      );
    }

    showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) {
          Future<void> pickIssueDate() async {
            final now = DateTime.now();
            final initialDate =
                _tryParseLicenseDate(issueDateController.text.trim()) ?? now;
            final picked = await showDatePicker(
              context: ctx,
              initialDate: initialDate,
              firstDate: DateTime(1950),
              lastDate: DateTime(now.year + 30),
            );
            if (picked == null) return;
            setDialogState(() {
              issueDateController.text = _formatLicenseDate(picked);
              expiryDateController.text = _autoLicenseExpiry(
                licenseClass: selectedClass,
                issueDate: issueDateController.text.trim(),
              );
            });
          }

          return AlertDialog(
            title: Text(_settings.tr(
              isAddMode ? 'Thêm GPLX' : 'Chỉnh sửa GPLX',
              isAddMode ? 'Add Driver License' : 'Edit Driver License',
            )),
            content: SingleChildScrollView(
              child: SizedBox(
                width: 360,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    DropdownButtonFormField<String>(
                      value: selectedClass,
                      items: classOptions
                          .map(
                            (value) => DropdownMenuItem<String>(
                              value: value,
                              child: Text(value),
                            ),
                          )
                          .toList(),
                      onChanged: (value) {
                        if (value == null || value.trim().isEmpty) return;
                        setDialogState(() {
                          selectedClass = value.trim().toUpperCase();
                          selectedVehicleType =
                              _defaultVehicleTypeForLicenseClass(selectedClass);
                          expiryDateController.text = _autoLicenseExpiry(
                            licenseClass: selectedClass,
                            issueDate: issueDateController.text.trim(),
                          );
                        });
                      },
                      decoration: InputDecoration(
                        labelText: _settings.tr('Hạng bằng', 'License class'),
                      ),
                    ),
                    const SizedBox(height: 10),
                    DropdownButtonFormField<String>(
                      value: selectedVehicleType,
                      items: vehicleOptions
                          .map(
                            (value) => DropdownMenuItem<String>(
                              value: value,
                              child: Text(value),
                            ),
                          )
                          .toList(),
                      onChanged: (value) {
                        if (value == null || value.trim().isEmpty) return;
                        setDialogState(
                            () => selectedVehicleType = value.trim());
                      },
                      decoration: InputDecoration(
                        labelText: _settings.tr('Loại xe', 'Vehicle type'),
                      ),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: numberController,
                      decoration: InputDecoration(
                        labelText: _settings.tr('Số GPLX', 'License number'),
                      ),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: issueDateController,
                      readOnly: true,
                      onTap: pickIssueDate,
                      decoration: InputDecoration(
                        labelText: _settings.tr('Ngày cấp', 'Issue date'),
                        hintText: 'dd/MM/yyyy',
                        suffixIcon: const Icon(Icons.calendar_today_rounded),
                      ),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: expiryDateController,
                      readOnly: true,
                      decoration: InputDecoration(
                        labelText:
                            _settings.tr('Có giá trị đến', 'Valid until'),
                      ),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: issuedByController,
                      decoration: InputDecoration(
                        labelText:
                            _settings.tr('Nơi cấp GPLX', 'Issued by authority'),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: Text(_settings.tr('Hủy', 'Cancel')),
              ),
              ElevatedButton(
                onPressed: () {
                  if (selectedClass.trim().isEmpty ||
                      numberController.text.trim().isEmpty ||
                      selectedVehicleType.trim().isEmpty) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(_settings.tr(
                          'Vui lòng nhập đủ Hạng bằng, Loại xe và Số GPLX',
                          'Please fill class, vehicle type and license number',
                        )),
                        backgroundColor: AppTheme.warningColor,
                      ),
                    );
                    return;
                  }

                  final normalizedIssueDate = issueDateController.text.trim();
                  final normalizedExpiryDate = _autoLicenseExpiry(
                    licenseClass: selectedClass,
                    issueDate: normalizedIssueDate,
                  );

                  final updated = licenses
                      .map((l) => Map<String, String>.from(l))
                      .toList(growable: true);
                  final editedLicense = <String, String>{
                    'class': selectedClass.trim().toUpperCase(),
                    'vehicleType': selectedVehicleType.trim(),
                    'issueDate': normalizedIssueDate,
                    'expiryDate': normalizedExpiryDate,
                    'licenseNumber': numberController.text.trim(),
                    'issuedBy': issuedByController.text.trim(),
                  };
                  if (isAddMode) {
                    updated.add(editedLicense);
                  } else {
                    updated[safeIndex] = editedLicense;
                  }

                  Navigator.pop(ctx);
                  _submitWalletUpdateRequest(
                    {
                      'driverLicenses': updated,
                      ..._legacyLicenseFieldsFrom(updated),
                      'licenseIssuedBy': issuedByController.text.trim(),
                      'requestSection': 'wallet_gplx',
                    },
                    successVi: isAddMode
                        ? 'Đã gửi yêu cầu thêm GPLX. Vui lòng chờ admin duyệt.'
                        : 'Đã gửi yêu cầu cập nhật GPLX. Vui lòng chờ admin duyệt.',
                    successEn: isAddMode
                        ? 'Driver license add request sent. Please wait for admin approval.'
                        : 'Driver license update request sent. Please wait for admin approval.',
                  );
                },
                child: Text(_settings.tr('Gửi duyệt', 'Submit for approval')),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildDocumentWallet() {
    final licenses = _walletLicenses();
    final hasLicenses = licenses.isNotEmpty;
    final totalPages = hasLicenses ? 1 + licenses.length : 2;
    if (_walletPage >= totalPages) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        final targetPage = totalPages - 1;
        if (_walletPageController.hasClients) {
          _walletPageController.jumpToPage(targetPage);
        }
        setState(() => _walletPage = targetPage);
      });
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionHeader(
            title: _settings.tr('Ví giấy tờ', 'Document Wallet'),
            icon: Icons.wallet_rounded,
            color: const Color(0xFF1A237E),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(
                  totalPages,
                  (i) => AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        margin: const EdgeInsets.symmetric(horizontal: 3),
                        width: _walletPage == i ? 18 : 6,
                        height: 6,
                        decoration: BoxDecoration(
                          color: _walletPage == i
                              ? const Color(0xFF1A237E)
                              : AppTheme.dividerColor,
                          borderRadius: BorderRadius.circular(3),
                        ),
                      )),
            ),
          ),
          const SizedBox(height: 14),
          SizedBox(
            height: 195,
            child: PageView(
              controller: _walletPageController,
              onPageChanged: (i) => setState(() => _walletPage = i),
              children: [
                GestureDetector(
                  onTap: () => _showCccdDetail(null),
                  child: _buildCccdCard(null),
                ),
                if (hasLicenses)
                  ...licenses.asMap().entries.map((entry) {
                    final index = entry.key;
                    final license = entry.value;
                    final licenseType = _resolveLicenseType(license);
                    final licensePoints = _pointsForLicenseType(licenseType);
                    final isDisabled = licensePoints <= 0;
                    return GestureDetector(
                      onTap: () => _showLicenseDetail(
                        licenseIndex: index,
                        licenseClass: license['class'] ?? '',
                        vehicleType: license['vehicleType'] ?? '',
                        issueDate: license['issueDate'] ?? '',
                        expiryDate: license['expiryDate'] ?? '',
                        licenseNumber: license['licenseNumber'] ?? '',
                        issuedBy: license['issuedBy'] ?? '',
                        licenseType: licenseType,
                        points: licensePoints,
                        isDisabled: isDisabled,
                      ),
                      child: _buildLicenseCard(
                        licenseClass: license['class'] ?? '',
                        vehicleType: license['vehicleType'] ?? '',
                        issueDate: license['issueDate'] ?? '',
                        expiryDate: license['expiryDate'] ?? '',
                        licenseNumber: license['licenseNumber'] ?? '',
                        licenseType: licenseType,
                        points: licensePoints,
                        isDisabled: isDisabled,
                      ),
                    );
                  })
                else
                  _buildAddLicensePromptCard(compact: true),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAddLicensePromptCard({bool compact = false}) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 2),
      padding: EdgeInsets.all(compact ? 16 : 18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFFE8F0FE), Color(0xFFF5F9FF)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.primaryColor.withOpacity(0.22)),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _settings.tr('Chưa có GPLX', 'No driver license yet'),
            style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            _settings.tr(
              'Thêm GPLX để theo dõi điểm và xử lý vi phạm theo đúng loại xe.',
              'Add your license to track points and handle violations by vehicle type.',
            ),
            style:
                const TextStyle(fontSize: 12.5, color: AppTheme.textSecondary),
          ),
          const SizedBox(height: 12),
          Align(
            alignment: Alignment.centerLeft,
            child: ElevatedButton.icon(
              onPressed: () => _showEditLicenseDialog(licenseIndex: -1),
              icon: const Icon(Icons.add_rounded, size: 18),
              label: Text(_settings.tr('Thêm GPLX', 'Add license')),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.primaryColor,
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// CCCD Card
  Widget _buildCccdCard(dynamic user) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 2),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF1A237E), Color(0xFF283593)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF1A237E).withOpacity(0.3),
            blurRadius: 14,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Stack(
        children: [
          Positioned(
            right: -20,
            bottom: -20,
            child: Icon(
              Icons.shield_rounded,
              size: 120,
              color: Colors.white.withOpacity(0.04),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(6),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(Icons.credit_card_rounded,
                          color: Colors.amber, size: 18),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        _settings.tr('CĂN CƯỚC CÔNG DÂN', 'CITIZEN ID CARD'),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.5,
                        ),
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: Colors.green.withOpacity(0.3),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.verified_rounded,
                              color: Colors.greenAccent, size: 12),
                          const SizedBox(width: 4),
                          Text(
                            _settings.tr('Hợp lệ', 'Valid'),
                            style: const TextStyle(
                              color: Colors.greenAccent,
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Text(
                  _settings.userIdCard.isNotEmpty
                      ? _settings.userIdCard
                      : '---',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 3,
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _settings.tr('Họ và tên', 'Full name'),
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.5),
                              fontSize: 10,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            _settings.userName.toUpperCase(),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          _settings.tr('Ngày sinh', 'DOB'),
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.5),
                            fontSize: 10,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          _settings.userDateOfBirth.isNotEmpty
                              ? _settings.userDateOfBirth
                              : '—',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Driver License Card
  Widget _buildLicenseCard({
    required String licenseClass,
    required String vehicleType,
    required String issueDate,
    required String expiryDate,
    required String licenseNumber,
    required String licenseType,
    required int points,
    required bool isDisabled,
  }) {
    final disabled = isDisabled || points <= 0;
    final isExpiring = !disabled &&
        expiryDate != 'Không thời hạn' &&
        expiryDate != 'No expiry' &&
        points <= 4;
    final pointColor = disabled ? Colors.white70 : Colors.amber;
    final vehicleIcon = licenseType == 'motorcycle'
        ? Icons.two_wheeler_rounded
        : Icons.directions_car_rounded;
    final gradientColors = disabled
        ? const [Color(0xFF9E9E9E), Color(0xFF616161)]
        : const [Color(0xFFD32F2F), Color(0xFF8B1A1A)];

    return Opacity(
      opacity: disabled ? 0.75 : 1.0,
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 2),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: gradientColors,
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: (disabled ? Colors.black54 : AppTheme.primaryColor)
                  .withOpacity(0.25),
              blurRadius: 14,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: Stack(
          children: [
            Positioned(
              right: -15,
              bottom: -15,
              child: Icon(
                vehicleIcon,
                size: 100,
                color: Colors.white.withOpacity(0.04),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(6),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Icon(Icons.badge_rounded,
                            color: pointColor, size: 18),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          _settings.tr('GIẤY PHÉP LÁI XE', 'DRIVER LICENSE'),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.5,
                          ),
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: pointColor.withOpacity(0.25),
                          borderRadius: BorderRadius.circular(10),
                          border:
                              Border.all(color: pointColor.withOpacity(0.4)),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.star_rounded,
                                color: pointColor, size: 14),
                            const SizedBox(width: 4),
                            Text(
                              '$points/12',
                              style: TextStyle(
                                color: pointColor,
                                fontSize: 13,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 14, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          '${_settings.tr('Hạng', 'Class')} $licenseClass',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          vehicleType,
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.8),
                            fontSize: 13,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(_settings.tr('Ngày cấp', 'Issued'),
                                style: TextStyle(
                                    color: Colors.white.withOpacity(0.5),
                                    fontSize: 10)),
                            const SizedBox(height: 2),
                            Text(issueDate,
                                style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600)),
                          ],
                        ),
                      ),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(_settings.tr('Có giá trị đến', 'Valid until'),
                                style: TextStyle(
                                    color: Colors.white.withOpacity(0.5),
                                    fontSize: 10)),
                            const SizedBox(height: 2),
                            Text(
                              expiryDate,
                              style: TextStyle(
                                color: isExpiring ? Colors.amber : Colors.white,
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Text(_settings.tr('Số GPLX', 'License No.'),
                              style: TextStyle(
                                  color: Colors.white.withOpacity(0.5),
                                  fontSize: 10)),
                          const SizedBox(height: 2),
                          Text(licenseNumber,
                              style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600)),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
            if (disabled)
              Positioned.fill(
                child: Container(
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.25),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Center(
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.5),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        _settings.tr(
                          'GPLX tạm vô hiệu (0/12)\nAdmin web mới khôi phục được',
                          'License disabled (0/12)\nOnly web admin can restore',
                        ),
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          height: 1.35,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // ── Avatar decoration helper ─────────────────────────────────────
  DecorationImage? _buildAvatarDecoration(String avatarUrl) {
    if (avatarUrl.isEmpty) return null;
    if (avatarUrl.startsWith('http')) {
      return DecorationImage(image: NetworkImage(avatarUrl), fit: BoxFit.cover);
    }
    // Local file path (from image_picker) — guard against deleted cache files
    final file = File(avatarUrl);
    if (!file.existsSync()) return null;
    return DecorationImage(image: FileImage(file), fit: BoxFit.cover);
  }

  // ── CCCD Detail Dialog ───────────────────────────────────────────
  void _showCccdDetail(dynamic user) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF1E1E1E) : Colors.white;
    final textPrimary = isDark ? const Color(0xFFE0E0E0) : AppTheme.textPrimary;
    final textSecondary =
        isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary;
    final displayName = _settings.userName;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.78),
        decoration: BoxDecoration(
          color: bgColor,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Drag handle
              Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(top: 12),
                decoration: BoxDecoration(
                    color: AppTheme.dividerColor,
                    borderRadius: BorderRadius.circular(2)),
              ),
              // Header
              Container(
                width: double.infinity,
                margin: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                      colors: [Color(0xFF1A237E), Color(0xFF283593)]),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  children: [
                    const Icon(Icons.shield_rounded,
                        color: Colors.amber, size: 36),
                    const SizedBox(height: 8),
                    Text(
                      _settings.tr('CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM',
                          'SOCIALIST REPUBLIC OF VIETNAM'),
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.5),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _settings.tr(
                          'CĂN CƯỚC CÔNG DÂN', 'CITIZEN IDENTITY CARD'),
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 1.5),
                    ),
                    const SizedBox(height: 14),
                    Text(
                      _settings.userIdCard.isNotEmpty
                          ? _settings.userIdCard
                          : '---',
                      style: const TextStyle(
                          color: Colors.amber,
                          fontSize: 24,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 4),
                    ),
                  ],
                ),
              ),
              // Detail rows
              Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    _buildDetailRow(_settings.tr('Họ và tên', 'Full name'),
                        displayName.toUpperCase(), textPrimary, textSecondary),
                    _buildDetailRow(
                        _settings.tr('Ngày sinh', 'Date of birth'),
                        _settings.userDateOfBirth.isNotEmpty
                            ? _settings.userDateOfBirth
                            : '—',
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(
                        _settings.tr('Giới tính', 'Gender'),
                        _settings.userGender.isNotEmpty
                            ? _settings.userGender
                            : _settings.tr('Chưa cập nhật', 'Not updated'),
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(
                        _settings.tr('Quốc tịch', 'Nationality'),
                        _settings.userNationality.isNotEmpty
                            ? _settings.userNationality
                            : _settings.tr('Việt Nam', 'Vietnam'),
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(
                        _settings.tr('Quê quán', 'Place of origin'),
                        _settings.userPlaceOfOrigin.isNotEmpty
                            ? _settings.userPlaceOfOrigin
                            : _settings.tr('Chưa cập nhật', 'Not updated'),
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(
                        _settings.tr('Nơi thường trú', 'Permanent address'),
                        _settings.profileInitialized &&
                                _settings.userAddress.isNotEmpty
                            ? _settings.userAddress
                            : _settings.tr('Chưa cập nhật', 'Not updated'),
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(
                        _settings.tr('Ngày cấp', 'Issue date'),
                        _settings.userIdCardIssueDate.isNotEmpty
                            ? _settings.userIdCardIssueDate
                            : _settings.tr('Chưa cập nhật', 'Not updated'),
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(
                        _settings.tr('Có giá trị đến', 'Valid until'),
                        _settings.userIdCardExpiryDate.isNotEmpty
                            ? _settings.userIdCardExpiryDate
                            : _settings.tr('Chưa cập nhật', 'Not updated'),
                        textPrimary,
                        textSecondary),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: SizedBox(
                            height: 48,
                            child: OutlinedButton(
                              onPressed: () => Navigator.pop(ctx),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: textSecondary,
                                side: BorderSide(color: AppTheme.dividerColor),
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(14)),
                              ),
                              child: Text(_settings.tr('Đóng', 'Close'),
                                  style: const TextStyle(
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700)),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: SizedBox(
                            height: 48,
                            child: ElevatedButton.icon(
                              onPressed: () {
                                Navigator.pop(ctx);
                                _showEditCccdDialog();
                              },
                              icon: const Icon(Icons.edit_rounded, size: 18),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFF1A237E),
                                foregroundColor: Colors.white,
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(14)),
                                elevation: 0,
                              ),
                              label: Text(_settings.tr('Chỉnh sửa', 'Edit'),
                                  style: const TextStyle(
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700)),
                            ),
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: MediaQuery.of(context).padding.bottom + 8),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── License Detail Dialog ────────────────────────────────────────
  void _showLicenseDetail({
    required int licenseIndex,
    required String licenseClass,
    required String vehicleType,
    required String issueDate,
    required String expiryDate,
    required String licenseNumber,
    required String issuedBy,
    required String licenseType,
    required int points,
    required bool isDisabled,
  }) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF1E1E1E) : Colors.white;
    final textPrimary = isDark ? const Color(0xFFE0E0E0) : AppTheme.textPrimary;
    final textSecondary =
        isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary;
    final displayName = _settings.userName;

    final pointColor =
        (isDisabled || points <= 0) ? Colors.white70 : Colors.amber;
    final pointPercent = (isDisabled || points <= 0) ? 0.0 : (points / 12.0);

    // Points come from Firestore via separate moto/car pools.
    // Show deduction history only if points < 12 as a signal label.
    final deductions = <Map<String, dynamic>>[];
    if (points < 12) {
      deductions.add({
        'type': _settings.tr(
            'điểm đã được trừ theo vi phạm', 'points deducted from violations'),
        'points': 12 - points,
        'date': ''
      });
    }

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.85),
        decoration: BoxDecoration(
          color: bgColor,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(top: 12),
                decoration: BoxDecoration(
                    color: AppTheme.dividerColor,
                    borderRadius: BorderRadius.circular(2)),
              ),
              // Header
              Container(
                width: double.infinity,
                margin: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                      colors: [Color(0xFFD32F2F), Color(0xFF8B1A1A)]),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  children: [
                    const Icon(Icons.badge_rounded,
                        color: Colors.amber, size: 36),
                    const SizedBox(height: 8),
                    Text(
                      _settings.tr('GIẤY PHÉP LÁI XE', 'DRIVER LICENSE'),
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 1.5),
                    ),
                    const SizedBox(height: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 20, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '${_settings.tr('Hạng', 'Class')} $licenseClass',
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.w800),
                      ),
                    ),
                    const SizedBox(height: 14),
                    // Points circle
                    Stack(
                      alignment: Alignment.center,
                      children: [
                        SizedBox(
                          width: 72,
                          height: 72,
                          child: CircularProgressIndicator(
                            value: pointPercent,
                            strokeWidth: 6,
                            backgroundColor: Colors.white.withOpacity(0.15),
                            valueColor: AlwaysStoppedAnimation(pointColor),
                          ),
                        ),
                        Column(
                          children: [
                            Text('$points',
                                style: TextStyle(
                                    color: pointColor,
                                    fontSize: 24,
                                    fontWeight: FontWeight.w800)),
                            Text('/12',
                                style: TextStyle(
                                    color: Colors.white.withOpacity(0.6),
                                    fontSize: 12)),
                          ],
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _settings.tr('Điểm giấy phép', 'License Points'),
                      style: TextStyle(
                          color: Colors.white.withOpacity(0.7), fontSize: 12),
                    ),
                  ],
                ),
              ),
              // Info rows
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildDetailRow(_settings.tr('Họ và tên', 'Full name'),
                        displayName.toUpperCase(), textPrimary, textSecondary),
                    _buildDetailRow(_settings.tr('Số GPLX', 'License No.'),
                        licenseNumber, textPrimary, textSecondary),
                    _buildDetailRow(_settings.tr('Hạng', 'Class'), licenseClass,
                        textPrimary, textSecondary),
                    _buildDetailRow(_settings.tr('Loại xe', 'Vehicle type'),
                        vehicleType, textPrimary, textSecondary),
                    _buildDetailRow(
                        _settings.tr('Nơi cấp', 'Issued by'),
                        issuedBy.isNotEmpty
                            ? issuedBy
                            : (_settings.userLicenseIssuedBy.isNotEmpty
                                ? _settings.userLicenseIssuedBy
                                : '—'),
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(
                        _settings.tr('Nhóm GPLX', 'License target'),
                        licenseType == 'motorcycle'
                            ? _settings.tr('Xe máy (A2)', 'Motorcycle (A2)')
                            : _settings.tr('Ô tô', 'Car'),
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(_settings.tr('Ngày cấp', 'Issue date'),
                        issueDate, textPrimary, textSecondary),
                    _buildDetailRow(
                        _settings.tr('Có giá trị đến', 'Valid until'),
                        expiryDate,
                        textPrimary,
                        textSecondary),
                    _buildDetailRow(
                        _settings.tr('Ngày sinh', 'Date of birth'),
                        _settings.userDateOfBirth.isNotEmpty
                            ? _settings.userDateOfBirth
                            : '—',
                        textPrimary,
                        textSecondary),
                    if (isDisabled || points <= 0) ...[
                      const SizedBox(height: 10),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppTheme.dangerColor.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                            color: AppTheme.dangerColor.withOpacity(0.2),
                          ),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.block_rounded,
                                color: AppTheme.dangerColor, size: 18),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _settings.tr(
                                  'GPLX này đang tạm vô hiệu do 0/12 điểm. Chỉ admin trên web mới khôi phục điểm được.',
                                  'This license is temporarily disabled at 0/12 points. Only web admin can restore points.',
                                ),
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: AppTheme.dangerColor,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                    if (deductions.isNotEmpty) ...[
                      const SizedBox(height: 16),
                      Text(
                        _settings.tr(
                            'Lịch sử trừ điểm', 'Point deduction history'),
                        style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w700,
                            color: AppTheme.dangerColor),
                      ),
                      const SizedBox(height: 8),
                      ...deductions.map((d) => Container(
                            margin: const EdgeInsets.only(bottom: 8),
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: AppTheme.dangerColor
                                  .withOpacity(isDark ? 0.15 : 0.05),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(
                                  color:
                                      AppTheme.dangerColor.withOpacity(0.15)),
                            ),
                            child: Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(6),
                                  decoration: BoxDecoration(
                                    color:
                                        AppTheme.dangerColor.withOpacity(0.15),
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Text('-${d['points']}',
                                      style: const TextStyle(
                                          color: AppTheme.dangerColor,
                                          fontSize: 14,
                                          fontWeight: FontWeight.w800)),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(d['type'],
                                          style: TextStyle(
                                              fontSize: 13,
                                              fontWeight: FontWeight.w600,
                                              color: textPrimary)),
                                      const SizedBox(height: 2),
                                      Text(d['date'],
                                          style: TextStyle(
                                              fontSize: 11,
                                              color: textSecondary)),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          )),
                    ] else ...[
                      const SizedBox(height: 12),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: Colors.green.withOpacity(isDark ? 0.15 : 0.06),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.check_circle_rounded,
                                color: Colors.green, size: 20),
                            const SizedBox(width: 10),
                            Text(
                              _settings.tr(
                                  'Chưa bị trừ điểm', 'No points deducted'),
                              style: const TextStyle(
                                  color: Colors.green,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600),
                            ),
                          ],
                        ),
                      ),
                    ],
                    const SizedBox(height: 20),
                    Row(
                      children: [
                        Expanded(
                          child: SizedBox(
                            height: 48,
                            child: OutlinedButton(
                              onPressed: () => Navigator.pop(ctx),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: textSecondary,
                                side: BorderSide(color: AppTheme.dividerColor),
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(14)),
                              ),
                              child: Text(_settings.tr('Đóng', 'Close'),
                                  style: const TextStyle(
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700)),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: SizedBox(
                            height: 48,
                            child: ElevatedButton.icon(
                              onPressed: () {
                                Navigator.pop(ctx);
                                _showEditLicenseDialog(
                                    licenseIndex: licenseIndex);
                              },
                              icon: const Icon(Icons.edit_rounded, size: 18),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: AppTheme.primaryColor,
                                foregroundColor: Colors.white,
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(14)),
                                elevation: 0,
                              ),
                              label: Text(_settings.tr('Chỉnh sửa', 'Edit'),
                                  style: const TextStyle(
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700)),
                            ),
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: MediaQuery.of(context).padding.bottom + 8),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Detail row helper ───────────────────────────────────────────
  Widget _buildDetailRow(
      String label, String value, Color textPrimary, Color textSecondary) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 130,
            child: Text(label,
                style: TextStyle(
                    fontSize: 13,
                    color: textSecondary,
                    fontWeight: FontWeight.w500)),
          ),
          Expanded(
            child: Text(value,
                style: TextStyle(
                    fontSize: 14,
                    color: textPrimary,
                    fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  QUICK MENU (2x2 Grid)
  // ═══════════════════════════════════════════════════════════════
  Widget _buildQuickMenu() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionHeader(
            title: _settings.tr('Dịch vụ', 'Services'),
            icon: Icons.apps_rounded,
            color: AppTheme.infoColor,
          ),
          const SizedBox(height: 12),
          GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            childAspectRatio: 1.85,
            crossAxisSpacing: 10,
            mainAxisSpacing: 10,
            padding: EdgeInsets.zero,
            children: [
              _buildMenuCard(
                icon: Icons.search_rounded,
                label: _settings.tr('Tra cứu\nvi phạm', 'Search\nViolations'),
                gradientColors: [
                  const Color(0xFFE53935),
                  const Color(0xFFD32F2F)
                ],
                onTap: () => _showViolationLookup(),
              ),
              _buildMenuCard(
                icon: Icons.payment_rounded,
                label:
                    _settings.tr('Nộp phạt\ntrực tuyến', 'Pay Fines\nOnline'),
                gradientColors: [
                  const Color(0xFFF57C00),
                  const Color(0xFFE65100)
                ],
                onTap: () => _showPaymentList(),
              ),
              _buildMenuCard(
                icon: Icons.history_rounded,
                label: _settings.tr('Lịch sử\nvi phạm', 'Violation\nHistory'),
                gradientColors: [
                  const Color(0xFF1565C0),
                  const Color(0xFF0D47A1)
                ],
                onTap: () {
                  setState(() => _selectedIndex = 1);
                },
              ),
              _buildMenuCard(
                icon: Icons.rate_review_rounded,
                label: _settings.tr('Khiếu nại\nvi phạm', 'File\nComplaint'),
                gradientColors: [
                  const Color(0xFF2E7D32),
                  const Color(0xFF1B5E20)
                ],
                onTap: () => Navigator.pushNamed(context, '/complaint'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMenuCard({
    required IconData icon,
    required String label,
    required List<Color> gradientColors,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: gradientColors,
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          boxShadow: [
            BoxShadow(
              color: gradientColors[0].withOpacity(0.3),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: Colors.white, size: 22),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  height: 1.3,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  FINE OVERVIEW SECTION
  // ═══════════════════════════════════════════════════════════════
  Widget _buildFineOverview() {
    final pending = _violations.where((v) => v.canPay).length;
    final paid = _violations.where((v) => v.isPaid).length;
    final totalFine = _violations
        .where((v) => v.canPay)
        .fold<double>(0, (sum, v) => sum + v.fineAmount);
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionHeader(
            title: _settings.tr('Tổng quan phạt nguội', 'Fine Overview'),
            icon: Icons.account_balance_wallet_rounded,
            color: AppTheme.warningColor,
          ),
          const SizedBox(height: 14),
          // Main fine card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFFD32F2F), Color(0xFFB71C1C)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(AppTheme.radiusXL),
              boxShadow: AppTheme.redShadow,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(Icons.account_balance_wallet_rounded,
                          color: Colors.white, size: 20),
                    ),
                    const SizedBox(width: 10),
                    Text(
                      _settings.tr(
                          'Tổng tiền phạt chưa nộp', 'Total unpaid fines'),
                      style:
                          const TextStyle(color: Colors.white70, fontSize: 14),
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                _isLoading
                    ? const SizedBox(
                        height: 30,
                        width: 30,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : TweenAnimationBuilder<double>(
                        tween: Tween(begin: 0, end: totalFine),
                        duration: const Duration(milliseconds: 1200),
                        curve: Curves.easeOutCubic,
                        builder: (context, value, child) {
                          return Text(
                            formatter.format(value),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 30,
                              fontWeight: FontWeight.w800,
                              letterSpacing: -0.5,
                            ),
                          );
                        },
                      ),
                const SizedBox(height: 8),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    _settings.tr(
                        '$pending khoản chưa thanh toán', '$pending unpaid'),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          // Stat cards row
          Row(
            children: [
              Expanded(
                  child: _buildStatCard(
                icon: Icons.pending_actions_rounded,
                label: _settings.tr('Chưa nộp', 'Unpaid'),
                value: pending.toString(),
                color: Colors.orange,
                delay: 0,
              )),
              const SizedBox(width: 10),
              Expanded(
                  child: _buildStatCard(
                icon: Icons.check_circle_outline_rounded,
                label: _settings.tr('Đã nộp', 'Paid'),
                value: paid.toString(),
                color: AppTheme.successColor,
                delay: 1,
              )),
              const SizedBox(width: 10),
              Expanded(
                  child: _buildStatCard(
                icon: Icons.folder_outlined,
                label: _settings.tr('Tổng cộng', 'Total'),
                value: _violations.length.toString(),
                color: AppTheme.infoColor,
                delay: 2,
              )),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatCard({
    required IconData icon,
    required String label,
    required String value,
    required Color color,
    required int delay,
  }) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: Duration(milliseconds: 600 + delay * 200),
      curve: Curves.easeOutBack,
      builder: (context, anim, child) {
        return Transform.scale(scale: anim, child: child);
      },
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 10),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          boxShadow: AppTheme.cardShadow,
        ),
        child: Column(
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(height: 8),
            Text(
              value,
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.w800,
                color: color,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: const TextStyle(
                fontSize: 11,
                color: AppTheme.textSecondary,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  VIOLATION LOOKUP (Tra cứu vi phạm)
  //  Search by license plate, ID, or violation info
  // ═══════════════════════════════════════════════════════════════
  void _showViolationLookup() {
    final searchController = TextEditingController();
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setSheetState) {
            final query = searchController.text.trim().toLowerCase();
            final hasQuery = query.isNotEmpty;
            final results = hasQuery
                ? _violations.where((v) {
                    return v.licensePlate.toLowerCase().contains(query) ||
                        v.violationType.toLowerCase().contains(query) ||
                        v.location.toLowerCase().contains(query) ||
                        v.id.toLowerCase().contains(query);
                  }).toList()
                : <Violation>[];

            return Container(
              height: MediaQuery.of(context).size.height * 0.85,
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
              ),
              child: Column(
                children: [
                  // Handle
                  Container(
                    width: 40,
                    height: 4,
                    margin: const EdgeInsets.only(top: 12),
                    decoration: BoxDecoration(
                      color: AppTheme.dividerColor,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  // Title
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [Color(0xFFE53935), Color(0xFFD32F2F)],
                            ),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.search_rounded,
                              color: Colors.white, size: 20),
                        ),
                        const SizedBox(width: 12),
                        Text(
                          _settings.tr('Tra cứu vi phạm', 'Violation Lookup'),
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w800,
                            color: AppTheme.textPrimary,
                          ),
                        ),
                      ],
                    ),
                  ),
                  // Search bar
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
                    child: TextField(
                      controller: searchController,
                      onChanged: (_) => setSheetState(() {}),
                      decoration: InputDecoration(
                        hintText: _settings.tr(
                          'Nhập biển số xe, loại vi phạm, địa điểm...',
                          'Enter license plate, violation type, location...',
                        ),
                        hintStyle: const TextStyle(
                            color: AppTheme.textHint, fontSize: 14),
                        prefixIcon: const Icon(Icons.search,
                            color: AppTheme.textSecondary),
                        suffixIcon: searchController.text.isNotEmpty
                            ? IconButton(
                                icon: const Icon(Icons.clear,
                                    color: AppTheme.textSecondary),
                                onPressed: () {
                                  searchController.clear();
                                  setSheetState(() {});
                                },
                              )
                            : null,
                        filled: true,
                        fillColor: AppTheme.surfaceColor,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(14),
                          borderSide: BorderSide.none,
                        ),
                      ),
                    ),
                  ),
                  // Results count (only when searching)
                  if (hasQuery)
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 20, vertical: 4),
                      child: Row(
                        children: [
                          Text(
                            _settings.tr(
                              'Tìm thấy ${results.length} kết quả',
                              'Found ${results.length} results',
                            ),
                            style: const TextStyle(
                              fontSize: 13,
                              color: AppTheme.textSecondary,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                  const Divider(color: AppTheme.dividerColor),
                  // Results list
                  Expanded(
                    child: !hasQuery
                        ? Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.manage_search_rounded,
                                    size: 56,
                                    color: AppTheme.textHint.withOpacity(0.35)),
                                const SizedBox(height: 14),
                                Text(
                                  _settings.tr('Nhập thông tin để tra cứu',
                                      'Enter info to search'),
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w600,
                                    color: AppTheme.textSecondary,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Padding(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 40),
                                  child: Text(
                                    _settings.tr(
                                      'Tìm kiếm theo biển số xe, loại vi phạm, hoặc địa điểm để xem lịch sử vi phạm chi tiết',
                                      'Search by license plate, violation type, or location to view detailed violation history',
                                    ),
                                    textAlign: TextAlign.center,
                                    style: TextStyle(
                                      fontSize: 13,
                                      color: AppTheme.textHint.withOpacity(0.7),
                                      height: 1.4,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          )
                        : results.isEmpty
                            ? Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.search_off,
                                        size: 48,
                                        color:
                                            AppTheme.textHint.withOpacity(0.4)),
                                    const SizedBox(height: 12),
                                    Text(
                                      _settings.tr('Không tìm thấy vi phạm',
                                          'No violations found'),
                                      style: const TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.w600,
                                        color: AppTheme.textSecondary,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      _settings.tr('Thử từ khóa khác',
                                          'Try a different keyword'),
                                      style: TextStyle(
                                        fontSize: 13,
                                        color:
                                            AppTheme.textHint.withOpacity(0.6),
                                      ),
                                    ),
                                  ],
                                ),
                              )
                            : ListView.builder(
                                padding:
                                    const EdgeInsets.fromLTRB(16, 4, 16, 16),
                                itemCount: results.length,
                                itemBuilder: (context, index) {
                                  final v = results[index];
                                  final isPaid = v.isPaid;
                                  final statusColor = isPaid
                                      ? AppTheme.successColor
                                      : AppTheme.warningColor;
                                  final statusText = isPaid
                                      ? _settings.tr('Đã thanh toán', 'Paid')
                                      : _settings.tr(
                                          'Chưa thanh toán', 'Unpaid');

                                  return Container(
                                    margin: const EdgeInsets.only(bottom: 10),
                                    decoration: BoxDecoration(
                                      color: Colors.white,
                                      borderRadius: BorderRadius.circular(14),
                                      border: Border.all(
                                          color: AppTheme.dividerColor),
                                      boxShadow: [
                                        BoxShadow(
                                          color: Colors.black.withOpacity(0.04),
                                          blurRadius: 8,
                                          offset: const Offset(0, 2),
                                        ),
                                      ],
                                    ),
                                    child: InkWell(
                                      borderRadius: BorderRadius.circular(14),
                                      onTap: () {
                                        Navigator.pop(ctx);
                                        Navigator.pushNamed(
                                            context, '/violation-detail',
                                            arguments: v);
                                      },
                                      child: Padding(
                                        padding: const EdgeInsets.all(14),
                                        child: Row(
                                          children: [
                                            // Icon
                                            Container(
                                              width: 44,
                                              height: 44,
                                              decoration: BoxDecoration(
                                                color: AppTheme.primaryColor
                                                    .withOpacity(0.1),
                                                borderRadius:
                                                    BorderRadius.circular(12),
                                              ),
                                              child: const Icon(
                                                  Icons.warning_amber_rounded,
                                                  color: AppTheme.primaryColor,
                                                  size: 22),
                                            ),
                                            const SizedBox(width: 12),
                                            // Content
                                            Expanded(
                                              child: Column(
                                                crossAxisAlignment:
                                                    CrossAxisAlignment.start,
                                                children: [
                                                  Text(
                                                    v.violationType,
                                                    style: const TextStyle(
                                                      fontSize: 14,
                                                      fontWeight:
                                                          FontWeight.w600,
                                                      color:
                                                          AppTheme.textPrimary,
                                                    ),
                                                    maxLines: 1,
                                                    overflow:
                                                        TextOverflow.ellipsis,
                                                  ),
                                                  const SizedBox(height: 4),
                                                  Row(
                                                    children: [
                                                      Container(
                                                        padding:
                                                            const EdgeInsets
                                                                .symmetric(
                                                                horizontal: 6,
                                                                vertical: 2),
                                                        decoration:
                                                            BoxDecoration(
                                                          color: AppTheme
                                                              .infoColor
                                                              .withOpacity(0.1),
                                                          borderRadius:
                                                              BorderRadius
                                                                  .circular(4),
                                                        ),
                                                        child: Text(
                                                          v.licensePlate,
                                                          style:
                                                              const TextStyle(
                                                            fontSize: 11,
                                                            fontWeight:
                                                                FontWeight.w700,
                                                            color: AppTheme
                                                                .infoColor,
                                                          ),
                                                        ),
                                                      ),
                                                      const SizedBox(width: 8),
                                                      Text(
                                                        DateFormat('dd/MM/yyyy')
                                                            .format(
                                                                v.timestamp),
                                                        style: const TextStyle(
                                                          fontSize: 11,
                                                          color: AppTheme
                                                              .textSecondary,
                                                        ),
                                                      ),
                                                    ],
                                                  ),
                                                ],
                                              ),
                                            ),
                                            // Right side: fine + status
                                            Column(
                                              crossAxisAlignment:
                                                  CrossAxisAlignment.end,
                                              children: [
                                                Text(
                                                  formatter
                                                      .format(v.fineAmount),
                                                  style: const TextStyle(
                                                    fontSize: 13,
                                                    fontWeight: FontWeight.w700,
                                                    color: AppTheme.dangerColor,
                                                  ),
                                                ),
                                                const SizedBox(height: 4),
                                                Container(
                                                  padding: const EdgeInsets
                                                      .symmetric(
                                                      horizontal: 8,
                                                      vertical: 3),
                                                  decoration: BoxDecoration(
                                                    color: statusColor
                                                        .withOpacity(0.1),
                                                    borderRadius:
                                                        BorderRadius.circular(
                                                            8),
                                                  ),
                                                  child: Text(
                                                    statusText,
                                                    style: TextStyle(
                                                      fontSize: 10,
                                                      fontWeight:
                                                          FontWeight.w600,
                                                      color: statusColor,
                                                    ),
                                                  ),
                                                ),
                                              ],
                                            ),
                                          ],
                                        ),
                                      ),
                                    ),
                                  );
                                },
                              ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  PAYMENT LIST (Nộp phạt trực tuyến)
  //  Shows all unpaid violations for payment
  // ═══════════════════════════════════════════════════════════════
  void _showPaymentList() {
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        final unpaidViolations = _violations.where((v) => v.canPay).toList();
        final totalFine =
            unpaidViolations.fold<double>(0, (sum, v) => sum + v.fineAmount);

        return Container(
          height: MediaQuery.of(context).size.height * 0.85,
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
          ),
          child: Column(
            children: [
              // Handle
              Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(top: 12),
                decoration: BoxDecoration(
                  color: AppTheme.dividerColor,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              // Title
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFFF57C00), Color(0xFFE65100)],
                        ),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(Icons.payment_rounded,
                          color: Colors.white, size: 20),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _settings.tr(
                                'Nộp phạt trực tuyến', 'Online Payment'),
                            style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w800,
                              color: AppTheme.textPrimary,
                            ),
                          ),
                          Text(
                            _settings.tr(
                              '${unpaidViolations.length} khoản chưa thanh toán',
                              '${unpaidViolations.length} unpaid fines',
                            ),
                            style: const TextStyle(
                              fontSize: 12,
                              color: AppTheme.textSecondary,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              // Total summary
              Container(
                margin: const EdgeInsets.fromLTRB(20, 16, 20, 8),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFFD32F2F), Color(0xFFB71C1C)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.account_balance_wallet_rounded,
                        color: Colors.white, size: 24),
                    const SizedBox(width: 12),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _settings.tr('Tổng tiền phạt', 'Total fines'),
                          style: TextStyle(
                              color: Colors.white.withOpacity(0.8),
                              fontSize: 12),
                        ),
                        Text(
                          formatter.format(totalFine),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const Divider(color: AppTheme.dividerColor),
              // Violations list
              Expanded(
                child: unpaidViolations.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Container(
                              width: 64,
                              height: 64,
                              decoration: BoxDecoration(
                                color: AppTheme.successColor.withOpacity(0.1),
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(Icons.check_circle_outline,
                                  size: 36, color: AppTheme.successColor),
                            ),
                            const SizedBox(height: 12),
                            Text(
                              _settings.tr('Không có khoản phạt nào',
                                  'No pending fines'),
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                                color: AppTheme.textSecondary,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              _settings.tr('Tất cả vi phạm đã được thanh toán',
                                  'All violations have been paid'),
                              style: const TextStyle(
                                fontSize: 13,
                                color: AppTheme.textHint,
                              ),
                            ),
                          ],
                        ),
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.fromLTRB(16, 4, 16, 16),
                        itemCount: unpaidViolations.length,
                        itemBuilder: (context, index) {
                          final v = unpaidViolations[index];

                          return Container(
                            margin: const EdgeInsets.only(bottom: 10),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(14),
                              border: Border.all(color: AppTheme.dividerColor),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.04),
                                  blurRadius: 8,
                                  offset: const Offset(0, 2),
                                ),
                              ],
                            ),
                            child: InkWell(
                              borderRadius: BorderRadius.circular(14),
                              onTap: () {
                                Navigator.pop(ctx);
                                Navigator.pushNamed(context, '/payment',
                                    arguments: v);
                              },
                              child: Padding(
                                padding: const EdgeInsets.all(14),
                                child: Row(
                                  children: [
                                    // Icon
                                    Container(
                                      width: 44,
                                      height: 44,
                                      decoration: BoxDecoration(
                                        color: AppTheme.warningColor
                                            .withOpacity(0.1),
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: const Icon(
                                          Icons.pending_actions_rounded,
                                          color: AppTheme.warningColor,
                                          size: 22),
                                    ),
                                    const SizedBox(width: 12),
                                    // Content
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            v.violationType,
                                            style: const TextStyle(
                                              fontSize: 14,
                                              fontWeight: FontWeight.w600,
                                              color: AppTheme.textPrimary,
                                            ),
                                            maxLines: 1,
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                          const SizedBox(height: 4),
                                          Row(
                                            children: [
                                              Container(
                                                padding:
                                                    const EdgeInsets.symmetric(
                                                        horizontal: 6,
                                                        vertical: 2),
                                                decoration: BoxDecoration(
                                                  color: AppTheme.infoColor
                                                      .withOpacity(0.1),
                                                  borderRadius:
                                                      BorderRadius.circular(4),
                                                ),
                                                child: Text(
                                                  v.licensePlate,
                                                  style: const TextStyle(
                                                    fontSize: 11,
                                                    fontWeight: FontWeight.w700,
                                                    color: AppTheme.infoColor,
                                                  ),
                                                ),
                                              ),
                                              const SizedBox(width: 8),
                                              Text(
                                                DateFormat('dd/MM/yyyy')
                                                    .format(v.timestamp),
                                                style: const TextStyle(
                                                  fontSize: 11,
                                                  color: AppTheme.textSecondary,
                                                ),
                                              ),
                                            ],
                                          ),
                                        ],
                                      ),
                                    ),
                                    // Right side: fine + pay button
                                    Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.end,
                                      children: [
                                        Text(
                                          formatter.format(v.fineAmount),
                                          style: const TextStyle(
                                            fontSize: 13,
                                            fontWeight: FontWeight.w700,
                                            color: AppTheme.dangerColor,
                                          ),
                                        ),
                                        const SizedBox(height: 4),
                                        Container(
                                          padding: const EdgeInsets.symmetric(
                                              horizontal: 10, vertical: 4),
                                          decoration: BoxDecoration(
                                            color: const Color(0xFFF57C00),
                                            borderRadius:
                                                BorderRadius.circular(8),
                                          ),
                                          child: Text(
                                            _settings.tr('Nộp phạt', 'Pay'),
                                            style: const TextStyle(
                                              fontSize: 10,
                                              fontWeight: FontWeight.w700,
                                              color: Colors.white,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
        );
      },
    );
  }
}
