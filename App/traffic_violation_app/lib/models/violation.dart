import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:traffic_violation_app/widgets/violation_image.dart'
    show normalizeViolationImageUrl;

class Violation {
  final String id;
  final String userId;
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
  final String complaintStatus; // 'none', 'pending', 'approved', 'rejected'
  final bool paymentLocked;
  final bool complaintLocked;
  final DateTime? paymentDueDate;
  final DateTime? paidAt;

  Violation({
    required this.id,
    this.userId = '',
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
    this.complaintStatus = '',
    this.paymentLocked = false,
    this.complaintLocked = false,
    this.paymentDueDate,
    this.paidAt,
  });

  bool get isPending => status == 'pending' || status == 'pending_payment';
  bool get isPaid => status == 'paid';
  bool get isCancelled => status == 'cancelled';
  bool get isComplaintPending {
    final normalized = complaintStatus.trim().toLowerCase();
    if (normalized == 'pending') return true;
    if (status == 'complaint_pending') return true;
    return paymentLocked || complaintLocked;
  }

  bool get canPay => isPending && !isComplaintPending;
  bool get canComplain => isPending && !isComplaintPending;

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

    // Resolve image URL from multiple possible keys, normalize to absolute URL
    final rawImage = (json['imageUrl'] ?? json['image_url'] ?? json['snapshotPath'] ?? '').toString();
    final resolvedImage = normalizeViolationImageUrl(rawImage);

    return Violation(
      id: json['id']?.toString() ?? '',
      userId: json['userId']?.toString() ?? '',
      licensePlate: json['licensePlate'] ?? 'Đang xác minh',
      violationType: json['violationType'] ?? 'Vi phạm',
      violationCode: json['violationCode'] ?? '',
      description: json['description'] ?? '',
      timestamp: _parseTimestamp(ts),
      location: json['location'] ?? '',
      imageUrl: resolvedImage,
      fineAmount: (json['fineAmount'] ?? 0).toDouble(),
      status: json['status']?.toString().toLowerCase() ?? 'pending',
      lawReference: json['lawReference'] ?? '',
      complaintStatus: json['complaintStatus']?.toString().toLowerCase() ?? '',
      paymentLocked: json['paymentLocked'] == true,
      complaintLocked: json['complaintLocked'] == true,
      paymentDueDate: json['paymentDueDate'] != null
          ? _parseTimestamp(json['paymentDueDate'])
          : null,
      paidAt: json['paidAt'] != null ? _parseTimestamp(json['paidAt']) : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'userId': userId,
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
      'complaintStatus': complaintStatus,
      'paymentLocked': paymentLocked,
      'complaintLocked': complaintLocked,
      'paymentDueDate': paymentDueDate?.toIso8601String(),
      'paidAt': paidAt?.toIso8601String(),
    };
  }
}
