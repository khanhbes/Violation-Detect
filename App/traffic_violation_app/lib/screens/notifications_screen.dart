import 'package:flutter/material.dart';

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
