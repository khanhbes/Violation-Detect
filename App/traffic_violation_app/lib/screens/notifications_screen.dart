import 'dart:async';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:traffic_violation_app/models/notification.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/services/update_service.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';

class NotificationsScreen extends StatefulWidget {
  const NotificationsScreen({super.key});

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> {
  final AppSettings _settings = AppSettings();

  List<AppNotification> _notifications = [];
  List<AppNotification> _firestoreNotifications = [];
  List<AppNotification> _derivedNotifications = [];
  AppNotification? _updateAvailableNotification;
  AppNotification? _updateDoneNotification;
  final Set<String> _dismissedLocalNotificationIds = <String>{};

  bool _isLoading = true;
  StreamSubscription? _notifSub;
  StreamSubscription? _violationSub;

  @override
  void initState() {
    super.initState();
    _loadNotifications();
    _listenViolationBasedNotifications();
    _loadDismissedLocalNotificationIds();
    _loadUpdateNotifications();
  }

  void _loadNotifications() {
    final uid = _settings.uid;
    if (uid == null) {
      if (mounted) {
        setState(() => _isLoading = false);
      }
      return;
    }

    _notifSub = FirestoreService().notificationsStream(uid).listen((notifs) {
      if (!mounted) return;

      _firestoreNotifications = notifs;
      _isLoading = false;
      _rebuildMergedNotifications();

      final unreadCount =
          _firestoreNotifications.where((n) => !n.isRead).length;
      _settings.setNotificationCount(unreadCount);
    });
  }

  void _listenViolationBasedNotifications() {
    final uid = _settings.uid;
    if (uid == null) return;

    _violationSub =
        FirestoreService().violationsStream(userId: uid).listen((violations) {
      if (!mounted) return;
      _derivedNotifications = _buildDerivedNotifications(violations);
      _rebuildMergedNotifications();
    });
  }

  Future<void> _loadDismissedLocalNotificationIds() async {
    final uid = _settings.uid;
    if (uid == null) return;

    try {
      final settings = await FirestoreService().getUserSettings(uid);
      final rawIds = settings?['dismissedLocalNotificationIds'];
      if (rawIds is List) {
        _dismissedLocalNotificationIds
          ..clear()
          ..addAll(rawIds.map((e) => e.toString()));
        _rebuildMergedNotifications();
      }
    } catch (_) {
      // Optional setting source. Ignore if load fails.
    }
  }

  Future<void> _persistDismissedLocalNotificationIds() async {
    final uid = _settings.uid;
    if (uid == null) return;

    try {
      final settings = await FirestoreService().getUserSettings(uid) ?? {};
      settings['dismissedLocalNotificationIds'] =
          _dismissedLocalNotificationIds.toList();
      await FirestoreService().saveUserSettings(uid, settings);
    } catch (_) {
      // Ignore persistence failure for local dismiss state.
    }
  }

  Future<void> _loadUpdateNotifications() async {
    try {
      await UpdateService().init();

      final currentVersion = UpdateService().currentVersion;
      final currentBuild = UpdateService().currentBuildNumber.toString();
      final uid = _settings.uid;

      if (uid != null) {
        final settings = await FirestoreService().getUserSettings(uid) ?? {};
        final lastVersion = settings['lastOpenedAppVersion']?.toString() ?? '';
        final lastBuild = settings['lastOpenedAppBuild']?.toString() ?? '';
        final isFirstOpen = lastVersion.isEmpty && lastBuild.isEmpty;

        if (!isFirstOpen &&
            (lastVersion != currentVersion || lastBuild != currentBuild)) {
          final previousVersion = lastVersion.isEmpty ? '?' : lastVersion;
          _updateDoneNotification = AppNotification(
            id: 'local_updated_${currentVersion}_$currentBuild',
            userId: uid,
            title: '🎉 Ứng dụng đã cập nhật',
            titleEn: '🎉 App updated',
            subtitle: 'Đã cập nhật từ v$previousVersion lên v$currentVersion',
            subtitleEn: 'Updated from v$previousVersion to v$currentVersion',
            detail:
                'Ứng dụng đã được cập nhật thành công lên phiên bản v$currentVersion (build $currentBuild).',
            detailEn:
                'The app has been updated successfully to version v$currentVersion (build $currentBuild).',
            type: 'update_done',
            timestamp: DateTime.now(),
            isRead: true,
          );
        } else {
          _updateDoneNotification = null;
        }

        settings['lastOpenedAppVersion'] = currentVersion;
        settings['lastOpenedAppBuild'] = currentBuild;
        await FirestoreService().saveUserSettings(uid, settings);
      }

      _rebuildMergedNotifications();

      final info = await UpdateService().checkForUpdate();
      if (!mounted) return;

      if (info == null) {
        _updateAvailableNotification = null;
        _rebuildMergedNotifications();
        return;
      }

      final version = info['version']?.toString() ?? '';
      final build = (info['buildNumber'] ?? '').toString();
      final changelog = info['changelog']?.toString().trim() ?? '';

      _updateAvailableNotification = AppNotification(
        id: 'local_update_${version}_$build',
        userId: _settings.uid ?? '',
        title: '📲 Cập nhật ứng dụng',
        titleEn: '📲 App update available',
        subtitle: 'Phiên bản $version đã sẵn sàng',
        subtitleEn: 'Version $version is ready',
        detail: changelog.isEmpty
            ? 'Ứng dụng có phiên bản mới $version. Vui lòng cập nhật để dùng tính năng và bản vá mới nhất.'
            : 'Ứng dụng có phiên bản mới $version.\n\nNội dung cập nhật:\n$changelog',
        detailEn: changelog.isEmpty
            ? 'A new app version $version is available. Please update for latest features and fixes.'
            : 'A new app version $version is available.\n\nChangelog:\n$changelog',
        type: 'update',
        timestamp: DateTime.now(),
        isRead: true,
      );
      _rebuildMergedNotifications();
    } catch (_) {
      // Optional notification source. Ignore if update check fails.
    }
  }

  List<AppNotification> _buildDerivedNotifications(List<Violation> violations) {
    final now = DateTime.now();
    final result = <AppNotification>[];

    for (final v in violations) {
      if (v.isPending) {
        final dueDate =
            v.paymentDueDate ?? v.timestamp.add(const Duration(days: 7));
        final isOverdue = dueDate.isBefore(now);
        final dueStr = DateFormat('dd/MM/yyyy').format(dueDate);

        result.add(
          AppNotification(
            id: 'local_due_${v.id}',
            userId: _settings.uid ?? '',
            title: '⏳ Hạn đóng phạt',
            titleEn: '⏳ Fine payment deadline',
            subtitle: isOverdue
                ? 'Vi phạm ${v.violationType} đã quá hạn từ $dueStr'
                : 'Vi phạm ${v.violationType} cần đóng trước $dueStr',
            subtitleEn: isOverdue
                ? '${v.violationType} is overdue since $dueStr'
                : '${v.violationType} must be paid before $dueStr',
            detail: isOverdue
                ? 'Khoản phạt ${_money(v.fineAmount)} cho vi phạm "${v.violationType}" đã quá hạn kể từ $dueStr.'
                : 'Khoản phạt ${_money(v.fineAmount)} cho vi phạm "${v.violationType}" có hạn đóng đến ngày $dueStr.',
            detailEn: isOverdue
                ? 'Fine ${_money(v.fineAmount)} for "${v.violationType}" is overdue since $dueStr.'
                : 'Fine ${_money(v.fineAmount)} for "${v.violationType}" is due by $dueStr.',
            type: 'payment_due',
            violationId: v.id,
            timestamp: dueDate,
            isRead: true,
          ),
        );
      }

      if (v.isPaid) {
        final paidAt = v.paidAt ?? v.timestamp;
        final paidStr = DateFormat('dd/MM/yyyy HH:mm').format(paidAt);
        result.add(
          AppNotification(
            id: 'local_paid_${v.id}',
            userId: _settings.uid ?? '',
            title: '✅ Đã đóng phạt',
            titleEn: '✅ Fine paid',
            subtitle: '${v.violationType} đã được thanh toán',
            subtitleEn: '${v.violationType} has been paid',
            detail:
                'Bạn đã thanh toán ${_money(v.fineAmount)} cho vi phạm "${v.violationType}" lúc $paidStr.',
            detailEn:
                'You paid ${_money(v.fineAmount)} for "${v.violationType}" at $paidStr.',
            type: 'payment_paid',
            violationId: v.id,
            timestamp: paidAt,
            isRead: true,
          ),
        );
      }
    }

    return result;
  }

  String _money(double value) {
    final fmt = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');
    return fmt.format(value);
  }

  void _rebuildMergedNotifications() {
    final merged = <AppNotification>[..._firestoreNotifications];

    for (final n in _derivedNotifications) {
      if (_isDismissedLocalNotification(n)) continue;
      if (!_hasEquivalentNotification(merged, n)) {
        merged.add(n);
      }
    }

    if (_updateDoneNotification != null &&
        !_isDismissedLocalNotification(_updateDoneNotification!) &&
        !_hasEquivalentNotification(merged, _updateDoneNotification!)) {
      merged.add(_updateDoneNotification!);
    }

    if (_updateAvailableNotification != null &&
        !_isDismissedLocalNotification(_updateAvailableNotification!) &&
        !_hasEquivalentNotification(merged, _updateAvailableNotification!)) {
      merged.add(_updateAvailableNotification!);
    }

    merged.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    if (mounted) {
      setState(() => _notifications = merged);
    }
  }

  bool _hasEquivalentNotification(
      List<AppNotification> source, AppNotification incoming) {
    final incomingType = _normalizeType(incoming.type);
    final incomingViolation = incoming.violationId ?? '';

    return source.any((n) {
      if (n.id.isNotEmpty && incoming.id.isNotEmpty && n.id == incoming.id) {
        return true;
      }
      final sameType = _normalizeType(n.type) == incomingType;
      final sameViolation = (n.violationId ?? '') == incomingViolation;
      if (incomingViolation.isNotEmpty) return sameType && sameViolation;
      if (incomingType == 'update' || incomingType == 'update_done') {
        return false;
      }
      return false;
    });
  }

  String _normalizeType(String type) {
    final t = type.toLowerCase().trim();
    if (t == 'violation' || t == 'danger' || t == 'warning') {
      return 'violation';
    }
    if (t == 'paid' || t == 'payment_paid' || t == 'success') {
      return 'payment_paid';
    }
    if (t == 'payment_due' || t == 'due' || t == 'deadline') {
      return 'payment_due';
    }
    if (t == 'update' || t == 'app_update' || t == 'version_update') {
      return 'update';
    }
    if (t == 'update_done' || t == 'updated' || t == 'update_success') {
      return 'update_done';
    }
    return t;
  }

  bool _isLocalNotification(AppNotification item) {
    return item.id.startsWith('local_');
  }

  bool _isDismissedLocalNotification(AppNotification item) {
    return _isLocalNotification(item) &&
        _dismissedLocalNotificationIds.contains(item.id);
  }

  @override
  void dispose() {
    _notifSub?.cancel();
    _violationSub?.cancel();
    super.dispose();
  }

  Color _getIconColor(String type) {
    switch (_normalizeType(type)) {
      case 'violation':
        return Colors.deepOrange;
      case 'payment_paid':
        return Colors.green;
      case 'payment_due':
        return const Color(0xFFF59E0B);
      case 'update':
        return Colors.blue;
      case 'update_done':
        return Colors.teal;
      default:
        return Colors.blueGrey;
    }
  }

  IconData _getIconData(String type) {
    switch (_normalizeType(type)) {
      case 'violation':
        return Icons.warning_amber_rounded;
      case 'payment_paid':
        return Icons.check_circle_outline_rounded;
      case 'payment_due':
        return Icons.schedule_rounded;
      case 'update':
        return Icons.system_update_rounded;
      case 'update_done':
        return Icons.verified_rounded;
      default:
        return Icons.info_outline_rounded;
    }
  }

  String _formatTime(DateTime ts) {
    final difference = DateTime.now().difference(ts);
    if (difference.inMinutes < 60) {
      final mins = difference.inMinutes;
      return _settings.tr('${mins > 0 ? mins : 1} phút trước',
          '${mins > 0 ? mins : 1} minutes ago');
    } else if (difference.inHours < 24) {
      return _settings.tr(
          '${difference.inHours} giờ trước', '${difference.inHours} hours ago');
    } else if (difference.inDays < 7) {
      return _settings.tr(
          '${difference.inDays} ngày trước', '${difference.inDays} days ago');
    }
    return DateFormat('dd/MM/yyyy HH:mm').format(ts);
  }

  @override
  Widget build(BuildContext context) {
    final s = _settings;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF121212) : AppTheme.surfaceColor;
    final cardBg = isDark ? const Color(0xFF1E1E1E) : Colors.white;
    final textPrimary = isDark ? const Color(0xFFE0E0E0) : AppTheme.textPrimary;
    final textSecondary =
        isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary;

    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        title: Text(s.tr('Thông báo', 'Notifications')),
        backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
        foregroundColor: textPrimary,
        elevation: 0,
        scrolledUnderElevation: 1,
        actions: [
          if (_firestoreNotifications.isNotEmpty)
            TextButton.icon(
              onPressed: _clearAll,
              icon: const Icon(Icons.checklist_rounded,
                  color: AppTheme.primaryColor, size: 20),
              label: Text(
                s.tr('Đã đọc tất cả', 'Mark all read'),
                style: const TextStyle(
                  color: AppTheme.primaryColor,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _notifications.isEmpty
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.notifications_off_outlined,
                          size: 64, color: textSecondary.withOpacity(0.4)),
                      const SizedBox(height: 16),
                      Text(
                        s.tr('Không có thông báo', 'No notifications'),
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: textSecondary,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        s.tr(
                          'Sẽ hiển thị: cập nhật, vi phạm, đã đóng phạt, hạn đóng phạt',
                          'Will show: updates, violations, paid, due notifications',
                        ),
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 13,
                          color: textSecondary.withOpacity(0.6),
                        ),
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 80),
                  itemCount: _notifications.length,
                  itemBuilder: (context, index) {
                    final item = _notifications[index];
                    return _buildNotifCard(item, index, isDark, cardBg,
                        textPrimary, textSecondary);
                  },
                ),
    );
  }

  Widget _buildNotifCard(AppNotification item, int index, bool isDark,
      Color cardBg, Color textPrimary, Color textSecondary) {
    final s = _settings;
    final title = s.isVietnamese ? item.title : item.titleEn;
    final subtitle = s.isVietnamese ? item.subtitle : item.subtitleEn;
    final timeStr = _formatTime(item.timestamp);
    final color = _getIconColor(item.type);
    final icon = _getIconData(item.type);

    return GestureDetector(
      onTap: () => _openNotificationDetail(
          item, isDark, textPrimary, textSecondary, color, icon, timeStr),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        margin: const EdgeInsets.only(bottom: 12),
        decoration: BoxDecoration(
          color: item.isRead
              ? (isDark ? const Color(0xFF1E1E1E) : Colors.white)
              : color.withOpacity(isDark ? 0.08 : 0.04),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: item.isRead
                ? (isDark ? const Color(0xFF333333) : AppTheme.dividerColor)
                : color.withOpacity(0.15),
          ),
          boxShadow: item.isRead
              ? []
              : [
                  BoxShadow(
                    color: color.withOpacity(0.08),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: color.withOpacity(isDark ? 0.2 : 0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: color, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            title,
                            style: TextStyle(
                              fontWeight: item.isRead
                                  ? FontWeight.w500
                                  : FontWeight.w700,
                              fontSize: 14,
                              color: textPrimary,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: () => _deleteNotification(item),
                          icon: Icon(Icons.delete_outline_rounded,
                              size: 18, color: textSecondary),
                          splashRadius: 18,
                          tooltip: s.tr('Xóa thông báo', 'Delete notification'),
                        ),
                        if (!item.isRead)
                          Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: color,
                              shape: BoxShape.circle,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(subtitle,
                        style: TextStyle(fontSize: 12, color: textSecondary)),
                    const SizedBox(height: 6),
                    Text(
                      timeStr,
                      style: TextStyle(
                          fontSize: 11, color: textSecondary.withOpacity(0.7)),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _openNotificationDetail(
    AppNotification item,
    bool isDark,
    Color textPrimary,
    Color textSecondary,
    Color color,
    IconData icon,
    String timeStr,
  ) async {
    if (!item.isRead && !_isLocalNotification(item)) {
      await FirestoreService().markNotificationRead(item.id);
    }

    Violation? violation;
    final violationId = item.violationId;
    if (violationId != null && violationId.isNotEmpty) {
      violation = await FirestoreService().getViolationById(violationId);
    }

    if (!mounted) return;
    _showNotifDetail(
      item,
      isDark,
      textPrimary,
      textSecondary,
      color,
      icon,
      timeStr,
      violation: violation,
    );
  }

  Future<void> _deleteNotification(AppNotification item) async {
    if (_isLocalNotification(item)) {
      _dismissedLocalNotificationIds.add(item.id);
      await _persistDismissedLocalNotificationIds();
      _rebuildMergedNotifications();
    } else {
      await FirestoreService().deleteNotification(item.id);
    }

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          _settings.tr('Đã xóa thông báo', 'Notification deleted'),
        ),
      ),
    );
  }

  void _clearAll() {
    final uid = _settings.uid;
    if (uid != null) {
      FirestoreService().markAllNotificationsRead(uid);
    }
  }

  void _showNotifDetail(AppNotification item, bool isDark, Color textPrimary,
      Color textSecondary, Color color, IconData icon, String timeStr,
      {Violation? violation}) {
    final s = _settings;
    final title = s.isVietnamese ? item.title : item.titleEn;
    final detail = s.isVietnamese ? item.detail : item.detailEn;
    final hasViolation =
        item.violationId != null && item.violationId!.isNotEmpty;
    final isPaidFromType = _normalizeType(item.type) == 'payment_paid';
    final isPaidViolation = violation?.isPaid ?? false;
    final isPaid = isPaidFromType || isPaidViolation;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.65),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1E1E1E) : Colors.white,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
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
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 48,
                        height: 48,
                        decoration: BoxDecoration(
                          color: color.withOpacity(isDark ? 0.2 : 0.1),
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child: Icon(icon, color: color, size: 26),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              title,
                              style: TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.w700,
                                color: textPrimary,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Row(
                              children: [
                                Icon(Icons.access_time_rounded,
                                    size: 13, color: textSecondary),
                                const SizedBox(width: 4),
                                Text(
                                  timeStr,
                                  style: TextStyle(
                                      fontSize: 12, color: textSecondary),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  Container(
                    height: 1,
                    color: isDark
                        ? const Color(0xFF333333)
                        : AppTheme.dividerColor,
                  ),
                  const SizedBox(height: 20),
                  Text(
                    s.tr('Chi tiết', 'Details'),
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: color,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    detail,
                    style: TextStyle(
                      fontSize: 14,
                      color: textPrimary,
                      height: 1.6,
                    ),
                  ),
                  const SizedBox(height: 24),
                  hasViolation
                      ? Row(
                          children: [
                            Expanded(
                              child: SizedBox(
                                height: 48,
                                child: TextButton(
                                  onPressed: () => Navigator.pop(ctx),
                                  style: TextButton.styleFrom(
                                    foregroundColor: textSecondary,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                  ),
                                  child: Text(
                                    s.tr('Đóng', 'Close'),
                                    style: const TextStyle(
                                        fontSize: 15,
                                        fontWeight: FontWeight.w600),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              flex: 2,
                              child: SizedBox(
                                height: 48,
                                child: ElevatedButton(
                                  onPressed: () {
                                    Navigator.pop(ctx);
                                    if (isPaid) {
                                      ScaffoldMessenger.of(context)
                                          .showSnackBar(
                                        SnackBar(
                                          content: Text(
                                            s.tr(
                                              'Vi phạm này đã nộp phạt.',
                                              'This violation has already been paid.',
                                            ),
                                          ),
                                          backgroundColor: AppTheme.dangerColor,
                                        ),
                                      );
                                      return;
                                    }

                                    if (violation == null) {
                                      ScaffoldMessenger.of(context)
                                          .showSnackBar(
                                        SnackBar(
                                          content: Text(
                                            s.tr(
                                              'Không tìm thấy dữ liệu vi phạm để nộp phạt.',
                                              'Violation data not found for payment.',
                                            ),
                                          ),
                                          backgroundColor: AppTheme.dangerColor,
                                        ),
                                      );
                                      return;
                                    }

                                    Navigator.pushNamed(
                                      context,
                                      '/payment',
                                      arguments: violation,
                                    );
                                  },
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: color,
                                    foregroundColor: Colors.white,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                    elevation: 0,
                                  ),
                                  child: Text(
                                    isPaid
                                        ? s.tr('Xem', 'View')
                                        : s.tr('Nộp phạt', 'Pay fine'),
                                    style: const TextStyle(
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        )
                      : SizedBox(
                          width: double.infinity,
                          height: 48,
                          child: ElevatedButton(
                            onPressed: () => Navigator.pop(ctx),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: color,
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(14)),
                              elevation: 0,
                            ),
                            child: Text(
                              s.tr('Đã hiểu', 'Got it'),
                              style: const TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ),
                        ),
                  SizedBox(height: MediaQuery.of(context).padding.bottom + 8),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
