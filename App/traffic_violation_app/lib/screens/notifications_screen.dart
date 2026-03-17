import 'dart:async';
import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/models/notification.dart';
import 'package:intl/intl.dart';

class NotificationsScreen extends StatefulWidget {
  const NotificationsScreen({super.key});

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> {
  final AppSettings _settings = AppSettings();
  List<AppNotification> _notifications = [];
  bool _isLoading = true;
  StreamSubscription? _notifSub;

  @override
  void initState() {
    super.initState();
    _loadNotifications();
  }

  void _loadNotifications() {
    final uid = _settings.uid;
    if (uid != null) {
      _notifSub = FirestoreService().notificationsStream(uid).listen((notifs) {
        if (mounted) {
          setState(() {
            _notifications = notifs;
            _isLoading = false;
          });
          // Update unread count in settings globally
          final unreadCount = _notifications.where((n) => !n.isRead).length;
          _settings.setNotificationCount(unreadCount);
        }
      });
    } else {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  void dispose() {
    _notifSub?.cancel();
    super.dispose();
  }

  Color _getIconColor(String type) {
    switch (type) {
      case 'warning':
        return Colors.orange;
      case 'danger':
        return Colors.red;
      case 'success':
        return Colors.green;
      default:
        return Colors.blue;
    }
  }

  IconData _getIconData(String type) {
    switch (type) {
      case 'warning':
        return Icons.warning_amber_rounded;
      case 'danger':
        return Icons.error_outline_rounded;
      case 'success':
        return Icons.check_circle_outline_rounded;
      default:
        return Icons.info_outline_rounded;
    }
  }

  String _formatTime(DateTime ts) {
    final difference = DateTime.now().difference(ts);
    if (difference.inMinutes < 60) {
      final mins = difference.inMinutes;
      return _settings.tr('${mins > 0 ? mins : 1} phút trước', '${mins > 0 ? mins : 1} minutes ago');
    } else if (difference.inHours < 24) {
      return _settings.tr('${difference.inHours} giờ trước', '${difference.inHours} hours ago');
    } else if (difference.inDays < 7) {
      return _settings.tr('${difference.inDays} ngày trước', '${difference.inDays} days ago');
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
    final textSecondary = isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary;

    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        title: Text(s.tr('Thông báo', 'Notifications')),
        backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
        foregroundColor: textPrimary,
        elevation: 0,
        scrolledUnderElevation: 1,
        actions: [
          if (_notifications.isNotEmpty)
            TextButton.icon(
              onPressed: _clearAll,
              icon: Icon(Icons.checklist_rounded, color: AppTheme.primaryColor, size: 20),
              label: Text(
                s.tr('Đã đọc tất cả', 'Mark all read'),
                style: const TextStyle(color: AppTheme.primaryColor, fontSize: 13, fontWeight: FontWeight.w600),
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
                  Icon(Icons.notifications_off_outlined, size: 64, color: textSecondary.withOpacity(0.4)),
                  const SizedBox(height: 16),
                  Text(
                    s.tr('Không có thông báo', 'No notifications'),
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: textSecondary),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    s.tr('Bạn sẽ nhận thông báo khi có vi phạm mới', 'You will be notified of new violations'),
                    style: TextStyle(fontSize: 13, color: textSecondary.withOpacity(0.6)),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 80),
              itemCount: _notifications.length,
              itemBuilder: (context, index) {
                final item = _notifications[index];
                return _buildNotifCard(item, index, isDark, cardBg, textPrimary, textSecondary);
              },
            ),
    );
  }

  Widget _buildNotifCard(AppNotification item, int index, bool isDark, Color cardBg,
      Color textPrimary, Color textSecondary) {
    final s = _settings;
    final title = s.isVietnamese ? item.title : item.titleEn;
    final subtitle = s.isVietnamese ? item.subtitle : item.subtitleEn;
    final timeStr = _formatTime(item.timestamp);
    final color = _getIconColor(item.type);
    final icon = _getIconData(item.type);

    return GestureDetector(
      onTap: () {
        if (!item.isRead) {
          FirestoreService().markNotificationRead(item.id);
        }
        _showNotifDetail(item, isDark, textPrimary, textSecondary, color, icon, timeStr);
      },
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
              // Icon
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
                              fontWeight: item.isRead ? FontWeight.w500 : FontWeight.w700,
                              fontSize: 14,
                              color: textPrimary,
                            ),
                          ),
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
                    Text(subtitle, style: TextStyle(fontSize: 12, color: textSecondary)),
                    const SizedBox(height: 6),
                    Text(timeStr, style: TextStyle(fontSize: 11, color: textSecondary.withOpacity(0.7))),
                  ],
                ),
              ),
            ],
          ),
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

  void _showNotifDetail(AppNotification item, bool isDark, Color textPrimary, Color textSecondary, Color color, IconData icon, String timeStr) {
    final s = _settings;
    final title = s.isVietnamese ? item.title : item.titleEn;
    final detail = s.isVietnamese ? item.detail : item.detailEn;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        constraints: BoxConstraints(maxHeight: MediaQuery.of(context).size.height * 0.65),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1E1E1E) : Colors.white,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
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
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Header
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
                            Text(title,
                                style: TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.w700,
                                  color: textPrimary,
                                )),
                            const SizedBox(height: 4),
                            Row(
                              children: [
                                Icon(Icons.access_time_rounded, size: 13, color: textSecondary),
                                const SizedBox(width: 4),
                                Text(timeStr, style: TextStyle(fontSize: 12, color: textSecondary)),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  // Divider
                  Container(height: 1, color: isDark ? const Color(0xFF333333) : AppTheme.dividerColor),
                  const SizedBox(height: 20),
                  // Detail text
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
                  // Action buttons
                  item.violationId != null && item.violationId!.isNotEmpty
                      ? Row(
                          children: [
                            Expanded(
                              child: SizedBox(
                                height: 48,
                                child: TextButton(
                                  onPressed: () => Navigator.pop(ctx),
                                  style: TextButton.styleFrom(
                                    foregroundColor: textSecondary,
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                                  ),
                                  child: Text(s.tr('Đóng', 'Close'), style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600)),
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
                                    Navigator.pushNamed(context, '/violation_detail', arguments: item.violationId);
                                  },
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: color,
                                    foregroundColor: Colors.white,
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                                    elevation: 0,
                                  ),
                                  child: Text(
                                    s.tr('Xem / Nộp phạt', 'View / Pay'),
                                    style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
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
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                              elevation: 0,
                            ),
                            child: Text(
                              s.tr('Đã hiểu', 'Got it'),
                              style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
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
