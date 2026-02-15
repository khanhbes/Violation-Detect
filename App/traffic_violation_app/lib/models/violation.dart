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

  factory Violation.fromJson(Map<String, dynamic> json) {
    return Violation(
      id: json['id']?.toString() ?? '',
      licensePlate: json['licensePlate'] ?? 'Đang xác minh',
      violationType: json['violationType'] ?? 'Vi phạm',
      violationCode: json['violationCode'] ?? '',
      description: json['description'] ?? '',
      timestamp: DateTime.tryParse(json['timestamp'] ?? '') ?? DateTime.now(),
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
