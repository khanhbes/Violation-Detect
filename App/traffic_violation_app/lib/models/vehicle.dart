class Vehicle {
  final String id;
  final String licensePlate;
  final String vehicleType;
  final String brand;
  final String model;
  final String color;
  final String ownerName;
  final String ownerId;
  final DateTime registrationDate;

  Vehicle({
    required this.id,
    required this.licensePlate,
    required this.vehicleType,
    required this.brand,
    required this.model,
    required this.color,
    required this.ownerName,
    required this.ownerId,
    required this.registrationDate,
  });

  factory Vehicle.fromJson(Map<String, dynamic> json) {
    return Vehicle(
      id: json['id'],
      licensePlate: json['licensePlate'],
      vehicleType: json['vehicleType'],
      brand: json['brand'],
      model: json['model'],
      color: json['color'],
      ownerName: json['ownerName'],
      ownerId: json['ownerId'],
      registrationDate: DateTime.parse(json['registrationDate']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'licensePlate': licensePlate,
      'vehicleType': vehicleType,
      'brand': brand,
      'model': model,
      'color': color,
      'ownerName': ownerName,
      'ownerId': ownerId,
      'registrationDate': registrationDate.toIso8601String(),
    };
  }
}
