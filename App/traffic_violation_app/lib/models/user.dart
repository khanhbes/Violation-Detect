class User {
  final String id;
  final String fullName;
  final String email;
  final String phone;
  final String? avatar;
  final String idCard;
  final String address;
  final DateTime createdAt;

  User({
    required this.id,
    required this.fullName,
    required this.email,
    required this.phone,
    this.avatar,
    required this.idCard,
    required this.address,
    required this.createdAt,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    // Handle createdAt from Firestore (Timestamp), String, or null
    DateTime createdAt;
    final raw = json['createdAt'];
    if (raw == null) {
      createdAt = DateTime.now();
    } else if (raw is DateTime) {
      createdAt = raw;
    } else if (raw is String) {
      createdAt = DateTime.tryParse(raw) ?? DateTime.now();
    } else {
      // Firestore Timestamp object — has .toDate() method
      try {
        createdAt = raw.toDate();
      } catch (_) {
        createdAt = DateTime.now();
      }
    }

    return User(
      id: json['id'] ?? '',
      fullName: json['fullName'] ?? '',
      email: json['email'] ?? '',
      phone: json['phone'] ?? '',
      avatar: json['avatar'] as String?,
      idCard: json['idCard'] ?? '',
      address: json['address'] ?? '',
      createdAt: createdAt,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'fullName': fullName,
      'email': email,
      'phone': phone,
      'avatar': avatar,
      'idCard': idCard,
      'address': address,
      'createdAt': createdAt.toIso8601String(),
    };
  }
}
