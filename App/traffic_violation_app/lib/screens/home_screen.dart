import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:traffic_violation_app/data/mock_data.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/notification_service.dart';
import 'package:traffic_violation_app/screens/violations_screen.dart';
import 'package:traffic_violation_app/screens/profile_screen.dart';
import 'dart:async';

// ─── Vehicles Page (inline) ───────────────────────────────────────
class _VehiclesPage extends StatelessWidget {
  const _VehiclesPage();

  @override
  Widget build(BuildContext context) {
    final vehicles = MockData.vehicles;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Phương tiện'),
        automaticallyImplyLeading: false,
        actions: [
          IconButton(
            icon: const Icon(Icons.add_circle_outline),
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Tính năng đang phát triển')),
              );
            },
          ),
        ],
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: vehicles.length,
        itemBuilder: (context, index) {
          final v = vehicles[index];
          final isMotorcycle = v.vehicleType.contains('máy');
          return Container(
            margin: const EdgeInsets.only(bottom: 16),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: isMotorcycle
                    ? [const Color(0xFF1A237E), const Color(0xFF283593)]
                    : [const Color(0xFF004D40), const Color(0xFF00695C)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: (isMotorcycle ? Colors.indigo : Colors.teal)
                      .withOpacity(0.3),
                  blurRadius: 12,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        isMotorcycle
                            ? Icons.two_wheeler
                            : Icons.directions_car,
                        color: Colors.white70,
                        size: 28,
                      ),
                      const SizedBox(width: 12),
                      Text(
                        v.brand,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      v.licensePlate,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 3,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      _infoChip(Icons.color_lens_outlined, v.color),
                      const SizedBox(width: 12),
                      _infoChip(Icons.category_outlined, v.vehicleType),
                    ],
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _infoChip(IconData icon, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: Colors.white60),
          const SizedBox(width: 4),
          Text(
            label,
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

// ─── Home Screen ──────────────────────────────────────────────────
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  int _selectedIndex = 0;
  final ApiService _api = ApiService();
  final NotificationService _notif = NotificationService();
  List<Violation> _violations = [];
  bool _isLoading = true;
  StreamSubscription? _newViolationSub;
  StreamSubscription? _connectionSub;

  // Animation controllers
  late AnimationController _fadeController;
  late AnimationController _slideController;

  final List<Widget> _pages = [];

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

    _pages.addAll([
      const SizedBox(), // placeholder, replaced by _buildHomePage()
      const ViolationsScreen(embedded: true),
      const _VehiclesPage(),
      const ProfileScreen(embedded: true),
    ]);

    _initServices();
  }

  Future<void> _initServices() async {
    await _notif.initialize();

    // Listen for new violations → push notification
    _newViolationSub = _api.newViolationStream.listen((violation) {
      _notif.showViolationNotification(violation);
    });

    // Connect WebSocket (real-time)
    _api.connectWebSocket();
    
    // Listen to connection status changes
    _connectionSub = _api.connectionStream.listen((isConnected) {
      if (mounted) setState(() {});
    });

    // Try API first, fallback to mock
    final connected = await _api.testConnection();
    if (connected) {
      _api.startPolling(); // Fallback polling
      _api.violationsStream.listen((violations) {
        if (mounted) {
          setState(() {
            _violations = violations;
            _isLoading = false;
          });
        }
      });
    } else {
      if (mounted) {
        setState(() {
          _violations = MockData.violations;
          _isLoading = false;
        });
      }
    }

    _fadeController.forward();
    _slideController.forward();
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    _newViolationSub?.cancel();
    _connectionSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _selectedIndex,
        children: [
          _buildHomePage(),
          _pages[1],
          _pages[2],
          _pages[3],
        ],
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  // ─── Bottom Navigation (fixed: no more pushNamed) ─────────────
  Widget _buildBottomNav() {
    return Container(
      decoration: BoxDecoration(
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 20,
            offset: const Offset(0, -5),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        child: NavigationBar(
          selectedIndex: _selectedIndex,
          onDestinationSelected: (index) {
            setState(() => _selectedIndex = index);
          },
          animationDuration: const Duration(milliseconds: 400),
          destinations: [
            const NavigationDestination(
              icon: Icon(Icons.home_outlined),
              selectedIcon: Icon(Icons.home_rounded),
              label: 'Trang chủ',
            ),
            NavigationDestination(
              icon: Badge(
                isLabelVisible: _violations.where((v) => v.isPending).isNotEmpty,
                label: Text(
                  '${_violations.where((v) => v.isPending).length}',
                  style: const TextStyle(fontSize: 10),
                ),
                child: const Icon(Icons.warning_amber_outlined),
              ),
              selectedIcon: Badge(
                isLabelVisible: _violations.where((v) => v.isPending).isNotEmpty,
                label: Text(
                  '${_violations.where((v) => v.isPending).length}',
                  style: const TextStyle(fontSize: 10),
                ),
                child: const Icon(Icons.warning_amber_rounded),
              ),
              label: 'Vi phạm',
            ),
            const NavigationDestination(
              icon: Icon(Icons.directions_car_outlined),
              selectedIcon: Icon(Icons.directions_car),
              label: 'Phương tiện',
            ),
            const NavigationDestination(
              icon: Icon(Icons.person_outline),
              selectedIcon: Icon(Icons.person),
              label: 'Tài khoản',
            ),
          ],
        ),
      ),
    );
  }

  // ─── Home Page Content ────────────────────────────────────────
  Widget _buildHomePage() {
    return CustomScrollView(
      slivers: [
        _buildAppBar(),
        SliverToBoxAdapter(child: _buildConnectionBanner()),
        SliverToBoxAdapter(
          child: FadeTransition(
            opacity: _fadeController,
            child: _buildStatsSection(),
          ),
        ),
        SliverToBoxAdapter(
          child: SlideTransition(
            position: Tween<Offset>(
              begin: const Offset(0, 0.3),
              end: Offset.zero,
            ).animate(CurvedAnimation(
              parent: _slideController,
              curve: Curves.easeOutCubic,
            )),
            child: _buildRecentViolations(),
          ),
        ),
        const SliverPadding(padding: EdgeInsets.only(bottom: 20)),
      ],
    );
  }

  // ─── App Bar ──────────────────────────────────────────────────
  Widget _buildAppBar() {
    return SliverAppBar(
      expandedHeight: 160,
      floating: false,
      pinned: true,
      automaticallyImplyLeading: false,
      flexibleSpace: FlexibleSpaceBar(
        background: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF1A237E), Color(0xFF0D47A1)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      CircleAvatar(
                        radius: 22,
                        backgroundColor: Colors.white24,
                        child: Text(
                          MockData.currentUser.fullName.substring(0, 1),
                          style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 18,
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Xin chào,',
                              style: TextStyle(
                                color: Colors.white70,
                                fontSize: 13,
                              ),
                            ),
                            Text(
                              MockData.currentUser.fullName,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.notifications_outlined,
                            color: Colors.white),
                        onPressed: () {
                          Navigator.pushNamed(context, '/notifications');
                        },
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ─── Connection Banner ────────────────────────────────────────
  Widget _buildConnectionBanner() {
    final isConnected = _api.isConnected;
    final isRealtime = _api.isWebSocketConnected;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 500),
      margin: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: isConnected
            ? (isRealtime
                ? Colors.green.withOpacity(0.1)
                : Colors.blue.withOpacity(0.1))
            : Colors.red.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isConnected
              ? (isRealtime
                  ? Colors.green.withOpacity(0.3)
                  : Colors.blue.withOpacity(0.3))
              : Colors.red.withOpacity(0.3),
        ),
      ),
      child: Row(
        children: [
          Icon(
            isConnected
                ? (isRealtime ? Icons.bolt_rounded : Icons.sync_rounded)
                : Icons.wifi_off_rounded,
            color: isConnected ? (isRealtime ? Colors.green : Colors.blue) : Colors.red,
            size: 20,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              isConnected
                  ? (isRealtime
                      ? 'Đã kết nối máy chủ (Real-time)'
                      : 'Đã kết nối máy chủ (Polling mode)')
                  : 'Mất kết nối máy chủ',
              style: TextStyle(
                color: isConnected
                    ? (isRealtime ? Colors.green[700] : Colors.blue[700])
                    : Colors.red[700],
                fontWeight: FontWeight.w600,
                fontSize: 14,
              ),
            ),
          ),
          if (isConnected)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: (isRealtime ? Colors.green : Colors.blue).withOpacity(0.2),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                isRealtime ? 'LIVE' : 'SYNC',
                style: TextStyle(
                  color: isRealtime ? Colors.green[800] : Colors.blue[800],
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
        ],
      ),
    );
  }

  // ─── Stats Section ────────────────────────────────────────────
  Widget _buildStatsSection() {
    final pending = _violations.where((v) => v.isPending).length;
    final paid = _violations.where((v) => v.isPaid).length;
    final totalFine = _violations
        .where((v) => v.isPending)
        .fold<double>(0, (sum, v) => sum + v.fineAmount);
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Fine summary card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFFE53935), Color(0xFFFF5252)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.red.withOpacity(0.3),
                  blurRadius: 15,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Tổng tiền phạt chưa nộp',
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 8),
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
                              fontSize: 28,
                              fontWeight: FontWeight.bold,
                            ),
                          );
                        },
                      ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Stat cards row
          Row(
            children: [
              Expanded(
                child: _buildStatCard(
                  icon: Icons.warning_amber_rounded,
                  label: 'Chưa nộp',
                  value: pending.toString(),
                  color: Colors.orange,
                  delay: 0,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildStatCard(
                  icon: Icons.check_circle_outline,
                  label: 'Đã nộp',
                  value: paid.toString(),
                  color: Colors.green,
                  delay: 1,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildStatCard(
                  icon: Icons.summarize_outlined,
                  label: 'Tổng cộng',
                  value: _violations.length.toString(),
                  color: Colors.blue,
                  delay: 2,
                ),
              ),
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
        return Transform.scale(
          scale: anim,
          child: child,
        );
      },
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withOpacity(0.2)),
        ),
        child: Column(
          children: [
            Icon(icon, color: color, size: 28),
            const SizedBox(height: 8),
            Text(
              value,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                color: color.withOpacity(0.8),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ─── Recent Violations ────────────────────────────────────────
  Widget _buildRecentViolations() {
    final recent = _violations.take(5).toList();

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Vi phạm gần đây',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              TextButton(
                onPressed: () => setState(() => _selectedIndex = 1),
                child: const Text('Xem tất cả →'),
              ),
            ],
          ),
          const SizedBox(height: 8),
          if (_isLoading)
            ...List.generate(
              3,
              (i) => Container(
                height: 80,
                margin: const EdgeInsets.only(bottom: 12),
                decoration: BoxDecoration(
                  color: Colors.grey.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            )
          else if (recent.isEmpty)
            Container(
              padding: const EdgeInsets.all(32),
              decoration: BoxDecoration(
                color: Colors.grey.withOpacity(0.05),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Center(
                child: Column(
                  children: [
                    Icon(Icons.check_circle_outline,
                        size: 48, color: Colors.green),
                    SizedBox(height: 12),
                    Text('Không có vi phạm nào',
                        style: TextStyle(color: Colors.grey)),
                  ],
                ),
              ),
            )
          else
            ...recent.asMap().entries.map((entry) {
              return _buildViolationCard(entry.value, entry.key);
            }),
        ],
      ),
    );
  }

  Widget _buildViolationCard(Violation v, int index) {
    final df = DateFormat('HH:mm — dd/MM/yyyy');

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: Duration(milliseconds: 400 + index * 100),
      curve: Curves.easeOutCubic,
      builder: (context, anim, child) {
        return Opacity(
          opacity: anim,
          child: Transform.translate(
            offset: Offset(0, 20 * (1 - anim)),
            child: child,
          ),
        );
      },
      child: Card(
        margin: const EdgeInsets.only(bottom: 12),
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: Colors.grey.withOpacity(0.15)),
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () {
            Navigator.pushNamed(context, '/violation-detail', arguments: v);
          },
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                // Status indicator
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: v.isPending
                        ? Colors.red.withOpacity(0.1)
                        : Colors.green.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    v.isPending
                        ? Icons.error_outline
                        : Icons.check_circle_outline,
                    color: v.isPending ? Colors.red : Colors.green,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        v.violationType,
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 14,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        df.format(v.timestamp),
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      NumberFormat.currency(locale: 'vi_VN', symbol: '₫')
                          .format(v.fineAmount),
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                        color: Color(0xFFE53935),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 2),
                      decoration: BoxDecoration(
                        color: v.isPending
                            ? Colors.orange.withOpacity(0.1)
                            : Colors.green.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        v.isPending ? 'Chưa nộp' : 'Đã nộp',
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                          color: v.isPending ? Colors.orange : Colors.green,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
