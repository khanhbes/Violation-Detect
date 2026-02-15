class TrafficLaw {
  final String code;
  final String title;
  final String description;
  final String category;
  final List<FineLevel> fineLevels;
  final String lawReference;
  final DateTime effectiveDate;

  TrafficLaw({
    required this.code,
    required this.title,
    required this.description,
    required this.category,
    required this.fineLevels,
    required this.lawReference,
    required this.effectiveDate,
  });

  factory TrafficLaw.fromJson(Map<String, dynamic> json) {
    return TrafficLaw(
      code: json['code'],
      title: json['title'],
      description: json['description'],
      category: json['category'],
      fineLevels: (json['fineLevels'] as List)
          .map((e) => FineLevel.fromJson(e))
          .toList(),
      lawReference: json['lawReference'],
      effectiveDate: DateTime.parse(json['effectiveDate']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'code': code,
      'title': title,
      'description': description,
      'category': category,
      'fineLevels': fineLevels.map((e) => e.toJson()).toList(),
      'lawReference': lawReference,
      'effectiveDate': effectiveDate.toIso8601String(),
    };
  }
}

class FineLevel {
  final String vehicleType;
  final double minAmount;
  final double maxAmount;
  final String? additionalPenalty;

  FineLevel({
    required this.vehicleType,
    required this.minAmount,
    required this.maxAmount,
    this.additionalPenalty,
  });

  factory FineLevel.fromJson(Map<String, dynamic> json) {
    return FineLevel(
      vehicleType: json['vehicleType'],
      minAmount: json['minAmount'].toDouble(),
      maxAmount: json['maxAmount'].toDouble(),
      additionalPenalty: json['additionalPenalty'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'vehicleType': vehicleType,
      'minAmount': minAmount,
      'maxAmount': maxAmount,
      'additionalPenalty': additionalPenalty,
    };
  }
}
