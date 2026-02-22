import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

class _NotifItem {
  final IconData icon;
  final Color color;
  final String title;
  final String titleEn;
  final String subtitle;
  final String subtitleEn;
  final String detail;
  final String detailEn;
  final String time;
  final String timeEn;
  bool isRead;

  _NotifItem({
    required this.icon,
    required this.color,
    required this.title,
    required this.titleEn,
    required this.subtitle,
    required this.subtitleEn,
    required this.detail,
    required this.detailEn,
    required this.time,
    required this.timeEn,
    this.isRead = false,
  });
}

class NotificationsScreen extends StatefulWidget {
  const NotificationsScreen({super.key});

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> {
  final AppSettings _settings = AppSettings();

  final List<_NotifItem> _notifications = [
    _NotifItem(
      icon: Icons.warning_amber,
      color: Colors.red,
      title: 'Vi phạm mới: Không đội mũ bảo hiểm',
      titleEn: 'New violation: No helmet',
      subtitle: 'Vừa phát hiện vi phạm bởi camera giám sát',
      subtitleEn: 'Violation detected by surveillance camera',
      detail:
          'Camera tại ngã tư Trần Hưng Đạo - Nguyễn Huệ đã ghi nhận phương tiện biển số 79A-123.45 vi phạm lỗi "Không đội mũ bảo hiểm". Thời gian: 14:35 ngày 22/02/2026. Mức phạt dự kiến: 400.000₫ - 600.000₫.',
      detailEn:
          'Camera at Tran Hung Dao - Nguyen Hue intersection detected vehicle plate 79A-123.45 violating "No helmet". Time: 14:35 on 22/02/2026. Estimated fine: 400,000₫ - 600,000₫.',
      time: '2 phút trước',
      timeEn: '2 minutes ago',
    ),
    _NotifItem(
      icon: Icons.check_circle,
      color: Colors.green,
      title: 'Nộp phạt thành công',
      titleEn: 'Fine paid successfully',
      subtitle: 'Vi phạm VH01 đã được xử lý',
      subtitleEn: 'Violation VH01 has been processed',
      detail:
          'Vi phạm mã VH01 - "Vượt đèn đỏ" đã được thanh toán thành công qua cổng VNPay. Số tiền: 800.000₫. Mã giao dịch: TXN20260222001. Biên lai điện tử đã được gửi về email của bạn.',
      detailEn:
          'Violation code VH01 - "Running red light" has been paid via VNPay. Amount: 800,000₫. Transaction ID: TXN20260222001. E-receipt has been sent to your email.',
      time: '1 giờ trước',
      timeEn: '1 hour ago',
    ),
    _NotifItem(
      icon: Icons.info_outline,
      color: Colors.blue,
      title: 'Nhắc nhở nộp phạt',
      titleEn: 'Fine payment reminder',
      subtitle: 'Bạn có 3 vi phạm chưa nộp phạt',
      subtitleEn: 'You have 3 unpaid violations',
      detail:
          'Bạn hiện có 3 vi phạm giao thông chưa nộp phạt với tổng số tiền 2.400.000₫. Vui lòng thanh toán trước ngày 15/03/2026 để tránh bị xử phạt bổ sung. Bạn có thể nộp phạt trực tuyến qua ứng dụng.',
      detailEn:
          'You currently have 3 traffic violations unpaid totaling 2,400,000₫. Please pay before 15/03/2026 to avoid additional penalties. You can pay online through the app.',
      time: '1 ngày trước',
      timeEn: '1 day ago',
    ),
    _NotifItem(
      icon: Icons.campaign_outlined,
      color: Colors.orange,
      title: 'Quy định mới',
      titleEn: 'New regulations',
      subtitle: 'Cập nhật mức phạt giao thông 2025',
      subtitleEn: 'Updated traffic fine rates 2025',
      detail:
          'Nghị định 168/2024/NĐ-CP có hiệu lực từ 01/01/2025 quy định mức phạt mới cho các lỗi vi phạm giao thông đường bộ. Một số thay đổi nổi bật: Lỗi vượt đèn đỏ tăng lên 18-20 triệu (ô tô), 4-6 triệu (xe máy). Không đội mũ bảo hiểm: 400.000-600.000₫.',
      detailEn:
          'Decree 168/2024 effective from 01/01/2025 establishes new fine rates for traffic violations. Key changes: Running red light increased to 18-20 million (cars), 4-6 million (motorcycles). No helmet: 400,000-600,000₫.',
      time: '3 ngày trước',
      timeEn: '3 days ago',
    ),
  ];

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
              icon: Icon(Icons.delete_sweep_rounded, color: AppTheme.dangerColor, size: 20),
              label: Text(
                s.tr('Xóa tất cả', 'Clear all'),
                style: const TextStyle(color: AppTheme.dangerColor, fontSize: 13, fontWeight: FontWeight.w600),
              ),
            ),
        ],
      ),
      body: _notifications.isEmpty
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

  Widget _buildNotifCard(_NotifItem item, int index, bool isDark, Color cardBg,
      Color textPrimary, Color textSecondary) {
    final s = _settings;
    final title = s.isVietnamese ? item.title : item.titleEn;
    final subtitle = s.isVietnamese ? item.subtitle : item.subtitleEn;
    final time = s.isVietnamese ? item.time : item.timeEn;

    return Dismissible(
      key: ValueKey('notif_$index'),
      direction: DismissDirection.endToStart,
      background: Container(
        margin: const EdgeInsets.only(bottom: 12),
        decoration: BoxDecoration(
          color: AppTheme.dangerColor,
          borderRadius: BorderRadius.circular(16),
        ),
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.delete_rounded, color: Colors.white, size: 24),
            const SizedBox(height: 4),
            Text(s.tr('Xóa', 'Delete'),
                style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
      onDismissed: (_) {
        setState(() => _notifications.removeAt(index));
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(s.tr('Đã xóa thông báo', 'Notification deleted')),
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            action: SnackBarAction(
              label: s.tr('Hoàn tác', 'Undo'),
              textColor: AppTheme.accentColor,
              onPressed: () {
                setState(() => _notifications.insert(index, item));
              },
            ),
          ),
        );
      },
      child: GestureDetector(
        onTap: () {
          setState(() => item.isRead = true);
          _showNotifDetail(item, isDark, textPrimary, textSecondary);
        },
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          margin: const EdgeInsets.only(bottom: 12),
          decoration: BoxDecoration(
            color: item.isRead
                ? (isDark ? const Color(0xFF1E1E1E) : Colors.white)
                : item.color.withOpacity(isDark ? 0.08 : 0.04),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: item.isRead
                  ? (isDark ? const Color(0xFF333333) : AppTheme.dividerColor)
                  : item.color.withOpacity(0.15),
            ),
            boxShadow: item.isRead
                ? []
                : [
                    BoxShadow(
                      color: item.color.withOpacity(0.08),
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
                    color: item.color.withOpacity(isDark ? 0.2 : 0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(item.icon, color: item.color, size: 22),
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
                                color: item.color,
                                shape: BoxShape.circle,
                              ),
                            ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(subtitle, style: TextStyle(fontSize: 12, color: textSecondary)),
                      const SizedBox(height: 6),
                      Text(time, style: TextStyle(fontSize: 11, color: textSecondary.withOpacity(0.7))),
                    ],
                  ),
                ),
                // Delete button
                GestureDetector(
                  onTap: () => _deleteSingle(index, item),
                  child: Padding(
                    padding: const EdgeInsets.only(left: 4),
                    child: Icon(Icons.close_rounded, size: 18, color: textSecondary.withOpacity(0.5)),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _deleteSingle(int index, _NotifItem item) {
    final s = _settings;
    setState(() => _notifications.removeAt(index));
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(s.tr('Đã xóa thông báo', 'Notification deleted')),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        action: SnackBarAction(
          label: s.tr('Hoàn tác', 'Undo'),
          textColor: AppTheme.accentColor,
          onPressed: () {
            setState(() => _notifications.insert(index, item));
          },
        ),
      ),
    );
  }

  void _clearAll() {
    final s = _settings;
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: AppTheme.dangerColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.delete_sweep_rounded, color: AppTheme.dangerColor, size: 22),
            ),
            const SizedBox(width: 12),
            Text(s.tr('Xóa tất cả', 'Clear all'), style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
          ],
        ),
        content: Text(s.tr(
          'Bạn có chắc muốn xóa tất cả ${_notifications.length} thông báo?',
          'Are you sure you want to delete all ${_notifications.length} notifications?',
        )),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(s.tr('Hủy', 'Cancel'), style: const TextStyle(color: AppTheme.textSecondary)),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(ctx);
              setState(() => _notifications.clear());
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.dangerColor,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: Text(s.tr('Xóa tất cả', 'Clear all')),
          ),
        ],
      ),
    );
  }

  void _showNotifDetail(_NotifItem item, bool isDark, Color textPrimary, Color textSecondary) {
    final s = _settings;
    final title = s.isVietnamese ? item.title : item.titleEn;
    final detail = s.isVietnamese ? item.detail : item.detailEn;
    final time = s.isVietnamese ? item.time : item.timeEn;

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
                          color: item.color.withOpacity(isDark ? 0.2 : 0.1),
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child: Icon(item.icon, color: item.color, size: 26),
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
                                Text(time, style: TextStyle(fontSize: 12, color: textSecondary)),
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
                      color: item.color,
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
                  // Close button
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: ElevatedButton(
                      onPressed: () => Navigator.pop(ctx),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: item.color,
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
