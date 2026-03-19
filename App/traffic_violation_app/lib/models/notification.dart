import 'package:cloud_firestore/cloud_firestore.dart';

class AppNotification {
  final String id;
  final String userId;
  final String title;
  final String titleEn;
  final String subtitle;
  final String subtitleEn;
  final String detail;
  final String detailEn;
  final String type; // 'warning', 'success', 'info', 'violation'
  final String? violationId;
  final DateTime timestamp;
  final bool isRead;

  AppNotification({
    required this.id,
    required this.userId,
    required this.title,
    required this.titleEn,
    required this.subtitle,
    required this.subtitleEn,
    required this.detail,
    required this.detailEn,
    required this.type,
    this.violationId,
    required this.timestamp,
    this.isRead = false,
  });

  factory AppNotification.fromJson(Map<String, dynamic> json) {
    final title = json['title'] ?? '';
    final body = json['body'] ?? '';

    // Parse timestamp: support both 'timestamp' and 'createdAt'
    DateTime ts = DateTime.now();
    final rawTs = json['createdAt'] ?? json['timestamp'];
    if (rawTs != null) {
      if (rawTs is Timestamp) {
        ts = rawTs.toDate();
      } else if (rawTs is DateTime) {
        ts = rawTs;
      } else if (rawTs is String) {
        ts = DateTime.tryParse(rawTs) ?? DateTime.now();
      }
    }

    return AppNotification(
      id: json['id'] ?? '',
      userId: json['userId'] ?? '',
      title: title,
      titleEn: json['titleEn'] ?? title,
      subtitle: json['subtitle'] ?? body,
      subtitleEn: json['subtitleEn'] ?? body,
      detail: json['detail'] ?? body,
      detailEn: json['detailEn'] ?? body,
      type: json['type'] ?? 'info',
      violationId: json['violationId'],
      timestamp: ts,
      isRead: json['isRead'] ?? false,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'userId': userId,
      'title': title,
      'titleEn': titleEn,
      'subtitle': subtitle,
      'subtitleEn': subtitleEn,
      'detail': detail,
      'detailEn': detailEn,
      'type': type,
      'violationId': violationId,
      'timestamp': Timestamp.fromDate(timestamp),
      'isRead': isRead,
    };
  }
}
