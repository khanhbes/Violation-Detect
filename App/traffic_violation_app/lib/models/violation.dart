import 'package:cloud_firestore/cloud_firestore.dart';

class Violation {
  final String id;
  final String licensePlate;
  final String violationType;
  final String violationCode;
  final String description;
  final DateTime timestamp;
  final String location;
  final String imageUrl;
  final double fineAmount;
  final String status; // 'pending', 'paid', 'cancelled'
  final String lawReference;

  Violation({
    required this.id,
    required this.licensePlate,
    required this.violationType,
    this.violationCode = '',
    this.description = '',
    required this.timestamp,
    this.location = '',
    required this.imageUrl,
    required this.fineAmount,
    this.status = 'pending',
    this.lawReference = '',
  });

  bool get isPending => status == 'pending';
  bool get isPaid => status == 'paid';
  bool get isCancelled => status == 'cancelled';

  /// Parse timestamp from either Firestore Timestamp or ISO string.
  static DateTime _parseTimestamp(dynamic value) {
    if (value == null) return DateTime.now();
    if (value is Timestamp) return value.toDate();
    if (value is DateTime) return value;
    if (value is String) return DateTime.tryParse(value) ?? DateTime.now();
    return DateTime.now();
  }

  factory Violation.fromJson(Map<String, dynamic> json) {
    // Prefer 'createdAt' (Firestore server timestamp) over 'timestamp' (ISO string)
    final ts = json['createdAt'] ?? json['timestamp'];

    return Violation(
      id: json['id']?.toString() ?? '',
      licensePlate: json['licensePlate'] ?? 'Đang xác minh',
      violationType: json['violationType'] ?? 'Vi phạm',
      violationCode: json['violationCode'] ?? '',
      description: json['description'] ?? '',
      timestamp: _parseTimestamp(ts),
      location: json['location'] ?? '',
      imageUrl: json['imageUrl'] ?? '',
      fineAmount: (json['fineAmount'] ?? 0).toDouble(),
      status: json['status'] ?? 'pending',
      lawReference: json['lawReference'] ?? '',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'licensePlate': licensePlate,
      'violationType': violationType,
      'violationCode': violationCode,
      'description': description,
      'timestamp': timestamp.toIso8601String(),
      'location': location,
      'imageUrl': imageUrl,
      'fineAmount': fineAmount,
      'status': status,
      'lawReference': lawReference,
    };
  }
}
