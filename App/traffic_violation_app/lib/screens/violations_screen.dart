import 'dart:async';

import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/screens/payment_screen.dart';
import 'package:traffic_violation_app/screens/zoom_image_viewer_screen.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/widgets/violation_image.dart';

class ViolationsScreen extends StatefulWidget {
  final bool embedded;

  const ViolationsScreen({super.key, this.embedded = false});

  @override
  State<ViolationsScreen> createState() => _ViolationsScreenState();
}

class _ViolationsScreenState extends State<ViolationsScreen>
    with SingleTickerProviderStateMixin {
  static const int _tabUnpaid = 1;
  static const int _tabProcessing = 2;
  static const int _tabComplaint = 3;
  static const int _tabPaid = 4;

  late TabController _tabController;
  final ApiService _api = ApiService();
  final FirestoreService _firestore = FirestoreService();
  final AppSettings _settings = AppSettings();

  List<Violation> _violations = [];
  bool _isLoading = true;
  bool _isFallbackLoading = false;
  String? _deletingViolationId;
  StreamSubscription? _sub;
  String? _boundUid;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 5, vsync: this);
    _settings.addListener(_onSettingsChanged);
    _loadData();
  }

  void _loadData() {
    final uid = _resolveUid();
    if (uid == null) {
      _boundUid = null;
      _sub?.cancel();
      _sub = null;
      if (mounted) {
        setState(() {
          _violations = [];
          _isLoading = false;
        });
      }
      return;
    }

    if (_boundUid == uid && _sub != null) return;
    _boundUid = uid;

    _sub?.cancel();
    _sub = _firestore.violationsStream(userId: uid).listen(
      (list) {
        debugPrint(
            '📱 ViolationsScreen firestore uid=$uid count=${list.length}');
        if (!mounted) return;
        final merged = _mergeById(list, _violations);
        setState(() {
          _violations = merged;
          _isLoading = false;
        });
        if (list.isEmpty) {
          unawaited(_syncFromApiFallback(uid, reason: 'firestore_empty'));
        }
      },
      onError: (error, stackTrace) {
        final isDenied = FirestoreService.isPermissionDeniedError(error);
        debugPrint(
            '❌ Violations stream error${isDenied ? ' (PERMISSION_DENIED)' : ''}: $error');
        if (!mounted) return;
        setState(() {
          _isLoading = false;
        });
        unawaited(_syncFromApiFallback(
            uid, reason: isDenied ? 'permission_denied' : 'firestore_error'));
      },
    );
  }

  void _onSettingsChanged() {
    _loadData();
    if (mounted) setState(() {});
  }

  String? _resolveUid() {
    final settingsUid = _settings.uid?.trim();
    if (settingsUid != null && settingsUid.isNotEmpty) return settingsUid;
    final authUid = fb.FirebaseAuth.instance.currentUser?.uid.trim();
    if (authUid == null || authUid.isEmpty) return null;
    return authUid;
  }

  @override
  void dispose() {
    _tabController.dispose();
    _sub?.cancel();
    _boundUid = null;
    _settings.removeListener(_onSettingsChanged);
    super.dispose();
  }

  List<Violation> _mergeById(
      List<Violation> primary, List<Violation> secondary) {
    final byId = <String, Violation>{};

    void addAll(List<Violation> source) {
      for (final v in source) {
        final key = v.id.trim();
        if (key.isEmpty) continue;
        byId[key] = v;
      }
    }

    addAll(secondary);
    addAll(primary);

    final merged = byId.values.toList()
      ..sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return merged;
  }

  Future<void> _syncFromApiFallback(
    String uid, {
    required String reason,
  }) async {
    if (_isFallbackLoading) return;
    _isFallbackLoading = true;
    try {
      final apiList = await _api.fetchViolations();
      if (!mounted || _boundUid != uid) return;
      final merged = _mergeById(apiList, _violations);
      debugPrint(
        '📱 ViolationsScreen api_fallback reason=$reason uid=$uid api=${apiList.length} merged=${merged.length}',
      );
      setState(() {
        _violations = merged;
        _isLoading = false;
      });
    } catch (e) {
      debugPrint('❌ ViolationsScreen fallback failed ($reason): $e');
    } finally {
      _isFallbackLoading = false;
    }
  }

  Future<void> _refresh() async {
    if (_isLoading) return;
    if (mounted) setState(() => _isLoading = true);

    final uid = _resolveUid();
    if (uid == null) {
      if (mounted) setState(() => _isLoading = false);
      return;
    }

    try {
      await _api.refreshCoreData(
        uid,
        taskTimeout: const Duration(seconds: 7),
        hardTimeout: const Duration(seconds: 14),
      );
      final latest = await _firestore.violationsStream(userId: uid).first;
      final merged = _mergeById(latest, _api.violations);
      if (!mounted) return;
      setState(() => _violations = merged);
    } catch (e) {
      debugPrint('❌ Refresh violations failed: $e');
      await _syncFromApiFallback(uid, reason: 'manual_refresh_error');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  bool _isPaid(Violation v) => v.isPaid;

  bool _isComplaint(Violation v) => v.isComplaintPending;

  bool _isProcessing(Violation v) {
    if (_isPaid(v) || _isComplaint(v)) return false;
    final s = v.status.toLowerCase().trim();
    return s == 'pending_payment' ||
        s == 'processing_payment' ||
        PaymentScreen.isProcessing(v.id);
  }

  bool _isUnpaid(Violation v) {
    if (_isPaid(v) || _isComplaint(v) || _isProcessing(v)) return false;
    final s = v.status.toLowerCase().trim();
    return s == 'pending' || s == 'pending_payment' || v.canPay;
  }

  bool _isComplaintRejected(Violation v) =>
      v.complaintStatus.trim().toLowerCase() == 'rejected';

  List<Violation> _filtered(int tab) {
    switch (tab) {
      case _tabUnpaid:
        return _violations.where(_isUnpaid).toList();
      case _tabProcessing:
        return _violations.where(_isProcessing).toList();
      case _tabComplaint:
        return _violations.where(_isComplaint).toList();
      case _tabPaid:
        return _violations.where(_isPaid).toList();
      default:
        return _violations;
    }
  }

  Future<void> _deletePaidViolation(Violation v) async {
    final accepted = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(_settings.tr('Xóa vi phạm', 'Delete violation')),
        content: Text(
          _settings.tr(
            'Xóa vi phạm đã thanh toán này khỏi lịch sử?',
            'Delete this paid violation from your history?',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(_settings.tr('Hủy', 'Cancel')),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.dangerColor,
              foregroundColor: Colors.white,
            ),
            child: Text(_settings.tr('Xóa', 'Delete')),
          ),
        ],
      ),
    );

    if (accepted != true) return;

    setState(() => _deletingViolationId = v.id);
    try {
      await _firestore.deletePaidViolation(v.id);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(_settings.tr('Đã xóa vi phạm', 'Violation deleted')),
          backgroundColor: AppTheme.successColor,
        ),
      );
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            _settings.tr('Không thể xóa vi phạm', 'Unable to delete violation'),
          ),
          backgroundColor: AppTheme.dangerColor,
        ),
      );
    } finally {
      if (mounted) setState(() => _deletingViolationId = null);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final unpaidCount = _violations.where(_isUnpaid).length;
    final processingCount = _violations.where(_isProcessing).length;
    final complaintCount = _violations.where(_isComplaint).length;
    final paidCount = _violations.where(_isPaid).length;
    final openCount = unpaidCount + processingCount + complaintCount;

    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF0D131F) : AppTheme.surfaceColor,
      body: Column(
        children: [
          _buildHeader(
            isDark: isDark,
            unpaidCount: unpaidCount,
            processingCount: processingCount,
            complaintCount: complaintCount,
            paidCount: paidCount,
            openCount: openCount,
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
            child: Container(
              decoration: BoxDecoration(
                color: isDark
                    ? const Color(0xFF1A2233)
                    : Colors.grey.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: isDark ? const Color(0xFF2B3853) : Colors.transparent,
                ),
              ),
              child: TabBar(
                controller: _tabController,
                isScrollable: true,
                indicator: BoxDecoration(
                  color: AppTheme.primaryColor,
                  borderRadius: BorderRadius.circular(10),
                ),
                indicatorSize: TabBarIndicatorSize.tab,
                dividerColor: Colors.transparent,
                labelColor: Colors.white,
                unselectedLabelColor:
                    isDark ? const Color(0xFF96A3BC) : AppTheme.textSecondary,
                tabs: [
                  Tab(
                      text:
                          '${_settings.tr('Tất cả', 'All')} (${_violations.length})'),
                  Tab(
                      text:
                          '${_settings.tr('Chưa nộp', 'Unpaid')} ($unpaidCount)'),
                  Tab(
                      text:
                          '${_settings.tr('Đang thanh toán', 'Processing')} ($processingCount)'),
                  Tab(
                      text:
                          '${_settings.tr('Đang khiếu nại', 'Complaint')} ($complaintCount)'),
                  Tab(text: '${_settings.tr('Đã nộp', 'Paid')} ($paidCount)'),
                ],
              ),
            ),
          ),
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: List.generate(5, (tab) => _buildList(tab, isDark)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader({
    required bool isDark,
    required int unpaidCount,
    required int processingCount,
    required int complaintCount,
    required int paidCount,
    required int openCount,
  }) {
    return Container(
      decoration: BoxDecoration(
        gradient: isDark
            ? const LinearGradient(
                colors: [Color(0xFF202B42), Color(0xFF111A2B)],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              )
            : AppTheme.headerGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(24),
          bottomRight: Radius.circular(24),
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          child: Column(
            children: [
              Row(
                children: [
                  if (!widget.embedded)
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_rounded,
                          color: Colors.white),
                    ),
                  Expanded(
                    child: Text(
                      _settings.tr('Vi phạm giao thông', 'Traffic Violations'),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.16),
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      '$openCount ${_settings.tr('đang xử lý', 'open')}',
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w700),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  _chip(
                    '${_settings.tr('Chưa nộp', 'Unpaid')}: $unpaidCount',
                    isDark: isDark,
                  ),
                  const SizedBox(width: 8),
                  _chip(
                    '${_settings.tr('Đang thanh toán', 'Processing')}: $processingCount',
                    isDark: isDark,
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  _chip(
                    '${_settings.tr('Đang khiếu nại', 'Complaint')}: $complaintCount',
                    isDark: isDark,
                  ),
                  const SizedBox(width: 8),
                  _chip(
                    '${_settings.tr('Đã nộp', 'Paid')}: $paidCount',
                    isDark: isDark,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _chip(String text, {required bool isDark}) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: isDark
              ? Colors.white.withOpacity(0.1)
              : Colors.white.withOpacity(0.14),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: isDark ? Colors.white.withOpacity(0.08) : Colors.transparent,
          ),
        ),
        child: Text(
          text,
          style: TextStyle(
              color: Colors.white.withOpacity(0.9),
              fontSize: 12,
              fontWeight: FontWeight.w600),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ),
    );
  }

  Widget _buildList(int tab, bool isDark) {
    final list = _filtered(tab);
    if (_isLoading) {
      return const Center(
          child: CircularProgressIndicator(color: AppTheme.primaryColor));
    }
    if (list.isEmpty) {
      return Center(
        child: Text(
          _settings.tr('Không có dữ liệu', 'No data'),
          style: TextStyle(
            color: isDark ? const Color(0xFF8C9BB4) : AppTheme.textSecondary,
          ),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _refresh,
      color: AppTheme.primaryColor,
      child: ListView.builder(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: EdgeInsets.fromLTRB(12, 12, 12, widget.embedded ? 100 : 16),
        itemCount: list.length,
        itemBuilder: (context, index) => _buildCard(list[index], index, isDark),
      ),
    );
  }

  Widget _buildCard(Violation v, int index, bool isDark) {
    final isPaid = _isPaid(v);
    final isProcessing = _isProcessing(v);
    final isComplaint = _isComplaint(v);
    final canComplain =
        !isPaid && !isComplaint && !isProcessing && v.canComplain;
    final deleting = _deletingViolationId == v.id;
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');
    final dateStr = DateFormat('HH:mm — dd/MM/yyyy').format(v.timestamp);

    final statusText = isPaid
        ? _settings.tr('Đã nộp', 'Paid')
        : isComplaint
            ? _settings.tr('Đang khiếu nại', 'Under complaint')
            : isProcessing
                ? _settings.tr('Đang thanh toán', 'Processing payment')
                : _settings.tr('Chưa nộp', 'Unpaid');

    final statusColor = isPaid
        ? AppTheme.successColor
        : isComplaint
            ? AppTheme.warningColor
            : isProcessing
                ? AppTheme.infoColor
                : Colors.orange.shade700;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1A2233) : Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: statusColor.withOpacity(isDark ? 0.35 : 0.25),
        ),
        boxShadow: isDark
            ? [
                BoxShadow(
                  color: Colors.black.withOpacity(0.28),
                  blurRadius: 14,
                  offset: const Offset(0, 6),
                )
              ]
            : AppTheme.cardShadow,
      ),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            GestureDetector(
              onTap: () {
                final tag = 'violation_image_${v.id}';
                Navigator.push(
                  context,
                  MaterialPageRoute<void>(
                    builder: (_) => ZoomImageViewerScreen(
                      imageUrl: v.imageUrl,
                      heroTag: tag,
                    ),
                  ),
                );
              },
              child: Hero(
                tag: 'violation_image_${v.id}',
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: SizedBox(
                    width: 82,
                    height: 104,
                    child: ViolationImage(
                      imageUrl: v.imageUrl,
                      fit: BoxFit.cover,
                      width: 82,
                      height: 104,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: InkWell(
                borderRadius: BorderRadius.circular(10),
                onTap: () => Navigator.pushNamed(context, '/violation-detail',
                    arguments: v),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            v.violationType,
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                              color: isDark
                                  ? const Color(0xFFF3F6FD)
                                  : AppTheme.textPrimary,
                            ),
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: statusColor.withOpacity(0.12),
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Text(
                            statusText,
                            style: TextStyle(
                                color: statusColor,
                                fontSize: 10,
                                fontWeight: FontWeight.w700),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(dateStr,
                        style: TextStyle(
                            fontSize: 12,
                            color: isDark
                                ? const Color(0xFF9AA9C3)
                                : AppTheme.textSecondary)),
                    const SizedBox(height: 2),
                    Text(v.licensePlate,
                        style: TextStyle(
                            fontSize: 12,
                            color: isDark
                                ? const Color(0xFF9AA9C3)
                                : AppTheme.textSecondary,
                            fontWeight: FontWeight.w600)),
                    const SizedBox(height: 4),
                    Text(
                      formatter.format(v.fineAmount),
                      style: TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w800,
                        color: isPaid
                            ? AppTheme.successColor
                            : AppTheme.primaryColor,
                      ),
                    ),
                    if (_isComplaintRejected(v) && !isComplaint)
                      Text(
                        _settings.tr('Khiếu nại trước đã bị từ chối',
                            'Previous complaint rejected'),
                        style: const TextStyle(
                            fontSize: 11,
                            color: AppTheme.warningColor,
                            fontWeight: FontWeight.w600),
                      ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: [
                        _actionButton(
                          icon: Icons.visibility_rounded,
                          label: _settings.tr('Chi tiết', 'Detail'),
                          color: AppTheme.infoColor,
                          isDark: isDark,
                          onTap: () => Navigator.pushNamed(
                              context, '/violation-detail',
                              arguments: v),
                        ),
                        if (!isPaid && !isComplaint)
                          _actionButton(
                            icon: Icons.payment_rounded,
                            label: isProcessing
                                ? _settings.tr('Tiếp tục nộp', 'Continue pay')
                                : _settings.tr('Nộp phạt', 'Pay'),
                            color: AppTheme.primaryColor,
                            isDark: isDark,
                            onTap: () => Navigator.pushNamed(
                                context, '/payment',
                                arguments: v),
                          ),
                        if (canComplain)
                          _actionButton(
                            icon: Icons.rate_review_rounded,
                            label: _settings.tr('Khiếu nại', 'Complain'),
                            color: AppTheme.warningColor,
                            isDark: isDark,
                            onTap: () =>
                                Navigator.pushNamed(context, '/complaint'),
                          ),
                        if (isPaid)
                          _actionButton(
                            icon: deleting
                                ? Icons.hourglass_top_rounded
                                : Icons.delete_outline_rounded,
                            label: deleting
                                ? _settings.tr('Đang xóa', 'Deleting')
                                : _settings.tr('Xóa', 'Delete'),
                            color: AppTheme.dangerColor,
                            isDark: isDark,
                            onTap:
                                deleting ? null : () => _deletePaidViolation(v),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _actionButton({
    required IconData icon,
    required String label,
    required Color color,
    required bool isDark,
    required VoidCallback? onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
        decoration: BoxDecoration(
          color: color.withOpacity(isDark ? 0.2 : 0.12),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: color.withOpacity(isDark ? 0.35 : 0.18),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 13, color: color),
            const SizedBox(width: 4),
            Text(label,
                style: TextStyle(
                    color: color, fontSize: 11, fontWeight: FontWeight.w700)),
          ],
        ),
      ),
    );
  }
}
