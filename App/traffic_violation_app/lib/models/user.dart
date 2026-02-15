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
    return User(
      id: json['id'],
      fullName: json['fullName'],
      email: json['email'],
      phone: json['phone'],
      avatar: json['avatar'],
      idCard: json['idCard'],
      address: json['address'],
      createdAt: DateTime.parse(json['createdAt']),
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
