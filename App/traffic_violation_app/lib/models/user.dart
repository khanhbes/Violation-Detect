class User {
  final String id;
  final String fullName;
  final String email;
  final String phone;
  final String? avatar;
  final String idCard;
  final String address;
  final String? idCardIssueDate;
  final String? idCardExpiryDate;
  final String? occupation;
  final String? dateOfBirth;
  final String? gender;
  final String? nationality;
  final String? placeOfOrigin;
  final List<Map<String, String>> driverLicenses;
  final String? licenseNumber;
  final String? motoLicenseClass;
  final String? carLicenseClass;
  final String? licenseIssueDate;
  final String? licenseExpiryDate;
  final String? licenseIssuedBy;
  final int motoPoints;
  final int carPoints;
  final int points;
  final DateTime createdAt;

  User({
    required this.id,
    required this.fullName,
    required this.email,
    required this.phone,
    this.avatar,
    required this.idCard,
    required this.address,
    this.idCardIssueDate,
    this.idCardExpiryDate,
    this.occupation,
    this.dateOfBirth,
    this.gender,
    this.nationality,
    this.placeOfOrigin,
    this.driverLicenses = const [],
    this.licenseNumber,
    this.motoLicenseClass,
    this.carLicenseClass,
    this.licenseIssueDate,
    this.licenseExpiryDate,
    this.licenseIssuedBy,
    this.motoPoints = 12,
    this.carPoints = 12,
    this.points = 12,
    required this.createdAt,
  });

  static List<Map<String, String>> _normalizeDriverLicenses(dynamic raw) {
    if (raw is! List) return const [];
    return raw.whereType<Map>().map((item) {
      String read(String key) => item[key]?.toString().trim() ?? '';
      return {
        'class': read('class'),
        'vehicleType': read('vehicleType'),
        'issueDate': read('issueDate'),
        'expiryDate': read('expiryDate'),
        'licenseNumber': read('licenseNumber'),
        'issuedBy': read('issuedBy'),
      };
    }).toList();
  }

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
      idCardIssueDate: json['idCardIssueDate'] as String?,
      idCardExpiryDate: json['idCardExpiryDate'] as String?,
      occupation: json['occupation'] as String?,
      dateOfBirth: json['dateOfBirth'] as String?,
      gender: json['gender'] as String?,
      nationality: json['nationality'] as String?,
      placeOfOrigin: json['placeOfOrigin'] as String?,
      driverLicenses: _normalizeDriverLicenses(json['driverLicenses']),
      licenseNumber: json['licenseNumber'] as String?,
      motoLicenseClass: json['motoLicenseClass'] as String?,
      carLicenseClass: json['carLicenseClass'] as String?,
      licenseIssueDate: json['licenseIssueDate'] as String?,
      licenseExpiryDate: json['licenseExpiryDate'] as String?,
      licenseIssuedBy: json['licenseIssuedBy'] as String?,
      motoPoints: (json['motoPoints'] as num?)?.toInt() ??
          (json['motoLicensePoints'] as num?)?.toInt() ??
          ((json['points'] as num?)?.toInt() ?? 12),
      carPoints: (json['carPoints'] as num?)?.toInt() ??
          (json['carLicensePoints'] as num?)?.toInt() ??
          ((json['points'] as num?)?.toInt() ?? 12),
      points: (json['points'] as num?)?.toInt() ?? 12,
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
      'idCardIssueDate': idCardIssueDate,
      'idCardExpiryDate': idCardExpiryDate,
      'occupation': occupation,
      'dateOfBirth': dateOfBirth,
      'gender': gender,
      'nationality': nationality,
      'placeOfOrigin': placeOfOrigin,
      'driverLicenses': driverLicenses,
      'licenseNumber': licenseNumber,
      'motoLicenseClass': motoLicenseClass,
      'carLicenseClass': carLicenseClass,
      'licenseIssueDate': licenseIssueDate,
      'licenseExpiryDate': licenseExpiryDate,
      'licenseIssuedBy': licenseIssuedBy,
      'motoPoints': motoPoints,
      'carPoints': carPoints,
      'points': points,
      'createdAt': createdAt.toIso8601String(),
    };
  }
}
