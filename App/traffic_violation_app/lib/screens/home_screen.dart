import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:traffic_violation_app/data/mock_data.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/notification_service.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/screens/violations_screen.dart';
import 'package:traffic_violation_app/screens/profile_screen.dart';
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
  List<Violation> _violations = [];
  bool _isLoading = true;
  StreamSubscription? _newViolationSub;
  StreamSubscription? _connectionSub;
  StreamSubscription? _firestoreSub;

  late AnimationController _fadeController;
  late AnimationController _slideController;

  // Document wallet state
  int _walletPage = 0;
  final PageController _walletPageController = PageController();

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

    _initServices();
  }

  Future<void> _initServices() async {
    await _notif.initialize();

    _newViolationSub = _api.newViolationStream.listen((violation) {
      _notif.showViolationNotification(violation);
    });

    _api.connectWebSocket();

    _connectionSub = _api.connectionStream.distinct().listen((isConnected) {
      if (mounted) setState(() {});
    });

    _firestoreSub = FirestoreService().violationsStream().listen((violations) {
      if (mounted) {
        setState(() {
          _violations = violations;
          _isLoading = false;
        });
      }
    });

    _api.testConnection();

    _fadeController.forward();
    _slideController.forward();
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    _walletPageController.dispose();
    _newViolationSub?.cancel();
    _connectionSub?.cancel();
    _firestoreSub?.cancel();
    super.dispose();
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
    final pendingCount = _violations.where((v) => v.isPending).length;

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
              Expanded(child: _buildNavItem(0, Icons.home_outlined, Icons.home_rounded, 'Trang chủ')),
              // Vi phạm
              Expanded(child: _buildNavItem(1, Icons.warning_amber_outlined, Icons.warning_amber_rounded, 'Vi phạm', badge: pendingCount)),
              // Center FAB (+)
              _buildCenterFAB(),
              // Ví giấy tờ
              Expanded(child: _buildNavItem(2, Icons.account_balance_wallet_outlined, Icons.account_balance_wallet_rounded, 'Ví giấy tờ')),
              // Cá nhân
              Expanded(child: _buildNavItem(3, Icons.settings_outlined, Icons.settings_rounded, 'Cài đặt')),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(int index, IconData icon, IconData activeIcon, String label, {int badge = 0}) {
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
              color: isSelected ? AppTheme.primaryColor.withOpacity(0.1) : Colors.transparent,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Badge(
              isLabelVisible: badge > 0,
              label: Text('$badge', style: const TextStyle(fontSize: 9, color: Colors.white)),
              backgroundColor: AppTheme.primaryColor,
              child: Icon(
                isSelected ? activeIcon : icon,
                color: isSelected ? AppTheme.primaryColor : AppTheme.textSecondary,
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
              color: isSelected ? AppTheme.primaryColor : AppTheme.textSecondary,
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

  /// Bottom sheet when tapping the + FAB
  void _showFunctionsSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(28),
            topRight: Radius.circular(28),
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
                color: AppTheme.dividerColor,
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
                    child: const Icon(Icons.apps_rounded, color: Colors.white, size: 20),
                  ),
                  const SizedBox(width: 12),
                  const Text(
                    'Chức năng',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.textPrimary,
                    ),
                  ),
                ],
              ),
            ),
            const Divider(color: AppTheme.dividerColor),
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
                  _buildSheetItem(Icons.search_rounded, 'Tra cứu\nvi phạm', const Color(0xFFE53935), () {
                    Navigator.pop(ctx);
                    Navigator.pushNamed(context, '/violations');
                  }),
                  _buildSheetItem(Icons.payment_rounded, 'Nộp phạt\ntrực tuyến', const Color(0xFFF57C00), () {
                    Navigator.pop(ctx);
                    Navigator.pushNamed(context, '/payment');
                  }),
                  _buildSheetItem(Icons.history_rounded, 'Lịch sử\nvi phạm', const Color(0xFF1565C0), () {
                    Navigator.pop(ctx);
                    setState(() => _selectedIndex = 1);
                  }),
                  _buildSheetItem(Icons.rate_review_rounded, 'Khiếu nại\nvi phạm', const Color(0xFF2E7D32), () {
                    Navigator.pop(ctx);
                    Navigator.pushNamed(context, '/complaint');
                  }),
                  _buildSheetItem(Icons.gavel_rounded, 'Luật\nGTĐB', AppTheme.warningColor, () {
                    Navigator.pop(ctx);
                    Navigator.pushNamed(context, '/traffic-laws');
                  }),
                  _buildSheetItem(Icons.qr_code_scanner_rounded, 'Quét\nQR Code', AppTheme.infoColor, () {
                    Navigator.pop(ctx);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Tính năng đang phát triển')),
                    );
                  }),
                  _buildSheetItem(Icons.directions_car_rounded, 'Phương\ntiện', const Color(0xFF5C6BC0), () {
                    Navigator.pop(ctx);
                    Navigator.pushNamed(context, '/violations');
                  }),
                  _buildSheetItem(Icons.support_agent_rounded, 'Hỗ trợ\ntrực tuyến', AppTheme.successColor, () {
                    Navigator.pop(ctx);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Hotline: 1900.xxxx')),
                    );
                  }),
                  _buildSheetItem(Icons.notifications_outlined, 'Thông\nbáo', Colors.deepOrange, () {
                    Navigator.pop(ctx);
                    Navigator.pushNamed(context, '/notifications');
                  }),
                  _buildSheetItem(Icons.badge_rounded, 'Ví\ngiấy tờ', const Color(0xFF1A237E), () {
                    Navigator.pop(ctx);
                    setState(() => _selectedIndex = 2);
                  }),
                  _buildSheetItem(Icons.router_rounded, 'Chỉnh\nIP Server', AppTheme.primaryColor, () {
                    Navigator.pop(ctx);
                    setState(() => _selectedIndex = 3);
                  }),
                  _buildSheetItem(Icons.info_outline_rounded, 'Về\nứng dụng', AppTheme.textSecondary, () {
                    Navigator.pop(ctx);
                  }),
                ],
              ),
            ),
            SizedBox(height: MediaQuery.of(context).padding.bottom + 12),
          ],
        ),
      ),
    );
  }

  Widget _buildSheetItem(IconData icon, String label, Color color, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 52,
            height: 52,
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, color: color, size: 24),
          ),
          const SizedBox(height: 6),
          Text(
            label,
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: AppTheme.textPrimary,
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
    return CustomScrollView(
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
        const SliverPadding(padding: EdgeInsets.only(bottom: 100)),
      ],
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  WALLET FULL PAGE (tab index 2)
  // ═══════════════════════════════════════════════════════════════
  Widget _buildWalletFullPage() {
    final user = MockData.currentUser;
    int licensePoints = 12;
    for (final v in _violations) {
      licensePoints -= _getPointsDeducted(v.violationType);
    }
    if (licensePoints < 0) licensePoints = 0;

    return Scaffold(
      backgroundColor: AppTheme.surfaceColor,
      body: CustomScrollView(
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
                        child: const Icon(Icons.wallet_rounded, color: Colors.white, size: 22),
                      ),
                      const SizedBox(width: 12),
                      const Text(
                        'Ví giấy tờ',
                        style: TextStyle(
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
                _buildSectionLabel('Căn cước công dân', Icons.credit_card_rounded, const Color(0xFF1A237E)),
                const SizedBox(height: 10),
                _buildCccdCard(user),
                const SizedBox(height: 20),
                _buildSectionLabel('Giấy phép lái xe', Icons.badge_rounded, AppTheme.primaryColor),
                const SizedBox(height: 10),
                _buildLicenseCard(
                  licenseClass: 'B2',
                  vehicleType: 'Ô tô dưới 9 chỗ',
                  issueDate: '15/03/2020',
                  expiryDate: '15/03/2030',
                  licenseNumber: '079201001234',
                  points: licensePoints,
                ),
                const SizedBox(height: 12),
                _buildLicenseCard(
                  licenseClass: 'A1',
                  vehicleType: 'Xe máy dưới 175cc',
                  issueDate: '20/06/2018',
                  expiryDate: 'Không thời hạn',
                  licenseNumber: '079201001234',
                  points: licensePoints,
                ),
                const SizedBox(height: 100),
              ]),
            ),
          ),
        ],
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
    final user = MockData.currentUser;

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
                        border: Border.all(color: Colors.white.withOpacity(0.4), width: 2),
                        image: user.avatar != null
                            ? DecorationImage(
                                image: NetworkImage(user.avatar!),
                                fit: BoxFit.cover,
                              )
                            : null,
                      ),
                      child: user.avatar == null
                          ? Center(
                              child: Text(
                                user.fullName.substring(0, 1).toUpperCase(),
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
                            'Xin chào,',
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.8),
                              fontSize: 13,
                            ),
                          ),
                          Text(
                            user.fullName,
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
                        icon: const Icon(Icons.notifications_outlined, color: Colors.white),
                        onPressed: () => Navigator.pushNamed(context, '/notifications'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                // Verification bar
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.verified_user_outlined, color: Colors.white.withOpacity(0.9), size: 20),
                      const SizedBox(width: 10),
                      Text(
                        'Xác thực: ${user.idCard}',
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.9),
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const Spacer(),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: AppTheme.accentColor,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: const Text(
                          'Đã xác minh',
                          style: TextStyle(
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
    final isConnected = _api.isConnected;
    final isRealtime = _api.isWebSocketConnected;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 500),
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 0),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: isConnected
            ? (isRealtime ? Colors.green.withOpacity(0.08) : Colors.blue.withOpacity(0.08))
            : Colors.red.withOpacity(0.08),
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        border: Border.all(
          color: isConnected
              ? (isRealtime ? Colors.green.withOpacity(0.25) : Colors.blue.withOpacity(0.25))
              : Colors.red.withOpacity(0.25),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              color: (isConnected ? (isRealtime ? Colors.green : Colors.blue) : Colors.red)
                  .withOpacity(0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(
              isConnected
                  ? (isRealtime ? Icons.bolt_rounded : Icons.sync_rounded)
                  : Icons.wifi_off_rounded,
              color: isConnected ? (isRealtime ? Colors.green : Colors.blue) : Colors.red,
              size: 18,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              isConnected
                  ? (isRealtime ? 'Đã kết nối (Real-time)' : 'Đã kết nối (Polling)')
                  : 'Mất kết nối máy chủ',
              style: TextStyle(
                color: isConnected
                    ? (isRealtime ? Colors.green[700] : Colors.blue[700])
                    : Colors.red[700],
                fontWeight: FontWeight.w600,
                fontSize: 13,
              ),
            ),
          ),
          if (isConnected)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: (isRealtime ? Colors.green : Colors.blue).withOpacity(0.15),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                isRealtime ? 'LIVE' : 'SYNC',
                style: TextStyle(
                  color: isRealtime ? Colors.green[800] : Colors.blue[800],
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.5,
                ),
              ),
            ),
          if (!isConnected)
            GestureDetector(
              onTap: () {
                _api.reconnect();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Đang kết nối lại...')),
                );
              },
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
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
                      'Thử lại',
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
  Widget _buildDocumentWallet() {
    final user = MockData.currentUser;

    int licensePoints = 12;
    for (final v in _violations) {
      licensePoints -= _getPointsDeducted(v.violationType);
    }
    if (licensePoints < 0) licensePoints = 0;

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionHeader(
            title: 'Ví giấy tờ',
            icon: Icons.wallet_rounded,
            color: const Color(0xFF1A237E),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) => AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                margin: const EdgeInsets.symmetric(horizontal: 3),
                width: _walletPage == i ? 18 : 6,
                height: 6,
                decoration: BoxDecoration(
                  color: _walletPage == i ? const Color(0xFF1A237E) : AppTheme.dividerColor,
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
                _buildCccdCard(user),
                _buildLicenseCard(
                  licenseClass: 'B2',
                  vehicleType: 'Ô tô dưới 9 chỗ',
                  issueDate: '15/03/2020',
                  expiryDate: '15/03/2030',
                  licenseNumber: '079201001234',
                  points: licensePoints,
                ),
                _buildLicenseCard(
                  licenseClass: 'A1',
                  vehicleType: 'Xe máy dưới 175cc',
                  issueDate: '20/06/2018',
                  expiryDate: 'Không thời hạn',
                  licenseNumber: '079201001234',
                  points: licensePoints,
                ),
              ],
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
                      child: const Icon(Icons.credit_card_rounded, color: Colors.amber, size: 18),
                    ),
                    const SizedBox(width: 10),
                    const Expanded(
                      child: Text(
                        'CĂN CƯỚC CÔNG DÂN',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.5,
                        ),
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: Colors.green.withOpacity(0.3),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.verified_rounded, color: Colors.greenAccent, size: 12),
                          SizedBox(width: 4),
                          Text(
                            'Hợp lệ',
                            style: TextStyle(
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
                  user.idCard,
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
                            'Họ và tên',
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.5),
                              fontSize: 10,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            user.fullName.toUpperCase(),
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
                          'Ngày sinh',
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.5),
                            fontSize: 10,
                          ),
                        ),
                        const SizedBox(height: 2),
                        const Text(
                          '01/01/2001',
                          style: TextStyle(
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
    required int points,
  }) {
    final isExpiring = expiryDate != 'Không thời hạn' && points <= 4;
    final pointColor = points >= 8
        ? Colors.green
        : points >= 4
            ? Colors.orange
            : AppTheme.dangerColor;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 2),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFFD32F2F), Color(0xFF8B1A1A)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: AppTheme.primaryColor.withOpacity(0.3),
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
              Icons.directions_car_rounded,
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
                      child: const Icon(Icons.badge_rounded, color: Colors.amber, size: 18),
                    ),
                    const SizedBox(width: 10),
                    const Expanded(
                      child: Text(
                        'GIẤY PHÉP LÁI XE',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.5,
                        ),
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: pointColor.withOpacity(0.25),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: pointColor.withOpacity(0.4)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.star_rounded, color: pointColor, size: 14),
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
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        'Hạng $licenseClass',
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
                          Text('Ngày cấp', style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 10)),
                          const SizedBox(height: 2),
                          Text(issueDate, style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600)),
                        ],
                      ),
                    ),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Có giá trị đến', style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 10)),
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
                        Text('Số GPLX', style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 10)),
                        const SizedBox(height: 2),
                        Text(licenseNumber, style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600)),
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

  /// Map violation type to points deducted (Vietnamese traffic law)
  int _getPointsDeducted(String violationType) {
    final lower = violationType.toLowerCase();
    if (lower.contains('ngược chiều') || lower.contains('nồng độ cồn')) return 4;
    if (lower.contains('đèn đỏ') || lower.contains('sai làn')) return 2;
    if (lower.contains('mũ bảo hiểm') || lower.contains('vỉa hè')) return 1;
    if (lower.contains('quá tốc độ') || lower.contains('tốc độ')) return 2;
    return 0;
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
            title: 'Dịch vụ',
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
                label: 'Tra cứu\nvi phạm',
                gradientColors: [const Color(0xFFE53935), const Color(0xFFD32F2F)],
                onTap: () => Navigator.pushNamed(context, '/violations'),
              ),
              _buildMenuCard(
                icon: Icons.payment_rounded,
                label: 'Nộp phạt\ntrực tuyến',
                gradientColors: [const Color(0xFFF57C00), const Color(0xFFE65100)],
                onTap: () => Navigator.pushNamed(context, '/payment'),
              ),
              _buildMenuCard(
                icon: Icons.history_rounded,
                label: 'Lịch sử\nvi phạm',
                gradientColors: [const Color(0xFF1565C0), const Color(0xFF0D47A1)],
                onTap: () {
                  setState(() => _selectedIndex = 1);
                },
              ),
              _buildMenuCard(
                icon: Icons.rate_review_rounded,
                label: 'Khiếu nại\nvi phạm',
                gradientColors: [const Color(0xFF2E7D32), const Color(0xFF1B5E20)],
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
    final pending = _violations.where((v) => v.isPending).length;
    final paid = _violations.where((v) => v.isPaid).length;
    final totalFine = _violations
        .where((v) => v.isPending)
        .fold<double>(0, (sum, v) => sum + v.fineAmount);
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionHeader(
            title: 'Tổng quan phạt nguội',
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
                      child: const Icon(Icons.account_balance_wallet_rounded, color: Colors.white, size: 20),
                    ),
                    const SizedBox(width: 10),
                    const Text(
                      'Tổng tiền phạt chưa nộp',
                      style: TextStyle(color: Colors.white70, fontSize: 14),
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
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '$pending khoản chưa thanh toán',
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
              Expanded(child: _buildStatCard(
                icon: Icons.pending_actions_rounded,
                label: 'Chưa nộp',
                value: pending.toString(),
                color: Colors.orange,
                delay: 0,
              )),
              const SizedBox(width: 10),
              Expanded(child: _buildStatCard(
                icon: Icons.check_circle_outline_rounded,
                label: 'Đã nộp',
                value: paid.toString(),
                color: AppTheme.successColor,
                delay: 1,
              )),
              const SizedBox(width: 10),
              Expanded(child: _buildStatCard(
                icon: Icons.folder_outlined,
                label: 'Tổng cộng',
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
}
