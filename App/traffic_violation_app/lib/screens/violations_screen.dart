import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/screens/payment_screen.dart'; // Added this import
import 'dart:async';

class ViolationsScreen extends StatefulWidget {
  final bool embedded;

  const ViolationsScreen({super.key, this.embedded = false});

  @override
  State<ViolationsScreen> createState() => _ViolationsScreenState();
}

class _ViolationsScreenState extends State<ViolationsScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final FirestoreService _firestore = FirestoreService();
  final AppSettings _settings = AppSettings();
  List<Violation> _violations = [];
  bool _isLoading = true;
  StreamSubscription? _sub;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _settings.addListener(_onSettingsChanged);
    _loadData();
  }

  void _loadData() {
    final uid = _settings.uid;
    _sub = _firestore.violationsStream(userId: uid).listen((list) {
      if (mounted)
        setState(() {
          _violations = list;
          _isLoading = false;
        });
    });
  }

  Future<void> _refresh() async {
    setState(() => _isLoading = true);
    final uid = _settings.uid;
    final violations = await _firestore.getViolations(userId: uid);
    if (mounted) {
      setState(() {
        _violations = violations;
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    _sub?.cancel();
    _settings.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    if (mounted) setState(() {});
  }

  List<Violation> _filtered(int tab) {
    switch (tab) {
      case 1:
        return _violations.where((v) => v.isPending).toList();
      case 2:
        return _violations.where((v) => v.isPaid).toList();
      default:
        return _violations;
    }
  }

  @override
  Widget build(BuildContext context) {
    final pending = _violations.where((v) => v.isPending).length;
    final paid = _violations.where((v) => v.isPaid).length;
    final totalPendingFine = _violations
        .where((v) => v.isPending)
        .fold<double>(0, (sum, v) => sum + v.fineAmount);
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    return Scaffold(
      backgroundColor: AppTheme.surfaceColor,
      body: Column(
        children: [
          // ── Header ─────────────────────────────────────────
          Container(
            decoration: const BoxDecoration(
              gradient: AppTheme.headerGradient,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(24),
                bottomRight: Radius.circular(24),
              ),
            ),
            child: SafeArea(
              bottom: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Title row
                    Row(
                      children: [
                        if (!widget.embedded)
                          GestureDetector(
                            onTap: () => Navigator.pop(context),
                            child: Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.15),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: const Icon(Icons.arrow_back_rounded,
                                  color: Colors.white, size: 20),
                            ),
                          ),
                        if (!widget.embedded) const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            _settings.tr(
                                'Vi phạm giao thông', 'Traffic Violations'),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 22,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                        if (pending > 0)
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color: AppTheme.accentColor,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              '$pending ${_settings.tr('chưa nộp', 'unpaid')}',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    // Stats row
                    Row(
                      children: [
                        _buildHeaderStat(
                          icon: Icons.folder_outlined,
                          label: _settings.tr('Tổng', 'Total'),
                          value: _violations.length.toString(),
                        ),
                        const SizedBox(width: 10),
                        _buildHeaderStat(
                          icon: Icons.pending_actions_rounded,
                          label: _settings.tr('Chưa nộp', 'Unpaid'),
                          value: pending.toString(),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          flex: 2,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12, vertical: 10),
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  _settings.tr('Tiền phạt', 'Total fines'),
                                  style: TextStyle(
                                    color: Colors.white.withOpacity(0.7),
                                    fontSize: 11,
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  formatter.format(totalPendingFine),
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 16,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ],
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

          // ── Tab Bar ─────────────────────────────────────────
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.grey.withOpacity(0.1),
                borderRadius: BorderRadius.circular(AppTheme.radiusM),
              ),
              child: TabBar(
                controller: _tabController,
                indicator: BoxDecoration(
                  borderRadius: BorderRadius.circular(AppTheme.radiusM),
                  color: AppTheme.primaryColor,
                ),
                indicatorSize: TabBarIndicatorSize.tab,
                labelColor: Colors.white,
                unselectedLabelColor: AppTheme.textSecondary,
                dividerColor: Colors.transparent,
                tabs: [
                  Tab(
                      text:
                          '${_settings.tr('Tất cả', 'All')} (${_violations.length})'),
                  Tab(text: '${_settings.tr('Chưa nộp', 'Unpaid')} ($pending)'),
                  Tab(text: '${_settings.tr('Đã nộp', 'Paid')} ($paid)'),
                ],
              ),
            ),
          ),

          // ── List ───────────────────────────────────────────
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: List.generate(3, (tab) => _buildList(tab)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeaderStat({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.15),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: TextStyle(
                color: Colors.white.withOpacity(0.7),
                fontSize: 11,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildList(int tab) {
    final list = _filtered(tab);

    if (_isLoading) {
      return ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: 4,
        itemBuilder: (_, __) => _buildShimmerCard(),
      );
    }

    if (list.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                color: (tab == 2 ? AppTheme.infoColor : AppTheme.successColor)
                    .withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                tab == 2
                    ? Icons.hourglass_empty
                    : Icons.check_circle_outline_rounded,
                size: 32,
                color: tab == 2 ? AppTheme.infoColor : AppTheme.successColor,
              ),
            ),
            const SizedBox(height: 14),
            Text(
              tab == 1
                  ? _settings.tr(
                      'Không có vi phạm chưa nộp', 'No unpaid violations')
                  : tab == 2
                      ? _settings.tr(
                          'Không có vi phạm đã nộp', 'No paid violations')
                      : _settings.tr(
                          'Không có vi phạm nào', 'No violations found'),
              style: const TextStyle(
                fontSize: 15,
                color: AppTheme.textSecondary,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _refresh,
      color: AppTheme.primaryColor,
      child: ListView.builder(
        padding: EdgeInsets.only(
          left: 16,
          right: 16,
          top: 16,
          bottom: widget.embedded ? 100 : 16,
        ),
        itemCount: list.length,
        itemBuilder: (context, index) =>
            _buildViolationCard(list[index], index),
      ),
    );
  }

  Widget _buildShimmerCard() {
    return Container(
      height: 100,
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Row(
        children: [
          Container(
            width: 100,
            decoration: BoxDecoration(
              color: Colors.grey.withOpacity(0.08),
              borderRadius:
                  const BorderRadius.horizontal(left: Radius.circular(16)),
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    height: 14,
                    width: 120,
                    decoration: BoxDecoration(
                      color: Colors.grey.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    height: 10,
                    width: 80,
                    decoration: BoxDecoration(
                      color: Colors.grey.withOpacity(0.07),
                      borderRadius: BorderRadius.circular(4),
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

  Widget _buildViolationCard(Violation v, int index) {
    final df = DateFormat('HH:mm — dd/MM/yyyy');
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: Duration(milliseconds: 350 + index * 60),
      curve: Curves.easeOutCubic,
      builder: (context, anim, child) {
        return Opacity(
          opacity: anim,
          child: Transform.translate(
            offset: Offset(20 * (1 - anim), 0),
            child: child,
          ),
        );
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          border: Border.all(
            color: v.isPending
                ? AppTheme.primaryColor.withOpacity(0.1)
                : AppTheme.successColor.withOpacity(0.1),
          ),
          boxShadow: AppTheme.cardShadow,
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          onTap: () =>
              Navigator.pushNamed(context, '/violation-detail', arguments: v),
          child: SizedBox(
            height: 100,
            child: Row(
              children: [
                // Image
                Hero(
                  tag: 'violation_image_${v.id}',
                  child: ClipRRect(
                    borderRadius: const BorderRadius.horizontal(
                        left: Radius.circular(16)),
                    child: SizedBox(
                      width: 100,
                      height: 100,
                      child: Image.network(
                        v.imageUrl,
                        fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) => Container(
                          color: Colors.grey[100],
                          child: Icon(Icons.image_not_supported_rounded,
                              color: Colors.grey[400]),
                        ),
                      ),
                    ),
                  ),
                ),
                // Info
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                v.violationType,
                                style: const TextStyle(
                                  fontWeight: FontWeight.w600,
                                  fontSize: 14,
                                  color: AppTheme.textPrimary,
                                ),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            Builder(
                              builder: (context) {
                                final bool isProcessing = PaymentScreen.isProcessing(v.id);
                                final Color bColor = v.isPaid
                                    ? AppTheme.successColor
                                    : (isProcessing ? AppTheme.infoColor : Colors.orange);
                                final String textV = v.isPaid
                                    ? _settings.tr('Đã nộp', 'Paid')
                                    : (isProcessing ? _settings.tr('Đang nộp', 'Processing') : _settings.tr('Chưa nộp', 'Unpaid'));

                                return Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: bColor.withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Text(
                                    textV,
                                    style: TextStyle(
                                      fontSize: 10,
                                      fontWeight: FontWeight.w600,
                                      color: v.isPending && !isProcessing ? Colors.orange[800] : bColor,
                                    ),
                                  ),
                                );
                              }
                            ),
                          ],
                        ),
                        const SizedBox(height: 6),
                        Row(
                          children: [
                            const Icon(Icons.access_time_rounded,
                                size: 12, color: AppTheme.textSecondary),
                            const SizedBox(width: 4),
                            Text(
                              df.format(v.timestamp),
                              style: const TextStyle(
                                fontSize: 12,
                                color: AppTheme.textSecondary,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Text(
                          formatter.format(v.fineAmount),
                          style: const TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 15,
                            color: AppTheme.primaryColor,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(right: 10),
                  child: Icon(
                    Icons.chevron_right_rounded,
                    color: Colors.grey[400],
                    size: 22,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
