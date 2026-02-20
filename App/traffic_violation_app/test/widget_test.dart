import 'package:flutter_test/flutter_test.dart';
import 'package:traffic_violation_app/models/user.dart';
import 'package:traffic_violation_app/models/vehicle.dart';
import 'package:traffic_violation_app/models/violation.dart';

void main() {
  group('User model', () {
    test('fromJson handles complete data', () {
      final json = {
        'id': '1',
        'fullName': 'Test User',
        'email': 'test@example.com',
        'phone': '0123456789',
        'avatar': 'https://example.com/avatar.png',
        'idCard': '123456789',
        'address': '123 Street',
        'createdAt': '2025-01-01T00:00:00.000Z',
      };
      final user = User.fromJson(json);
      expect(user.id, '1');
      expect(user.fullName, 'Test User');
      expect(user.avatar, 'https://example.com/avatar.png');
    });

    test('fromJson handles null fields gracefully', () {
      final json = <String, dynamic>{
        'id': null,
        'fullName': null,
        'email': null,
        'phone': null,
        'avatar': null,
        'idCard': null,
        'address': null,
        'createdAt': null,
      };
      final user = User.fromJson(json);
      expect(user.id, '');
      expect(user.avatar, isNull);
      expect(user.createdAt, isA<DateTime>());
    });
  });

  group('Vehicle model', () {
    test('fromJson handles complete data', () {
      final json = {
        'id': 'v1',
        'licensePlate': '59A-12345',
        'vehicleType': 'Xe máy',
        'brand': 'Honda',
        'model': 'Wave',
        'color': 'Đỏ',
        'ownerName': 'Test',
        'ownerId': '1',
        'registrationDate': '2025-01-01T00:00:00.000Z',
      };
      final vehicle = Vehicle.fromJson(json);
      expect(vehicle.licensePlate, '59A-12345');
      expect(vehicle.brand, 'Honda');
    });

    test('fromJson handles null registrationDate gracefully', () {
      final json = <String, dynamic>{
        'id': 'v1',
        'licensePlate': '59A-12345',
        'vehicleType': 'Xe máy',
        'brand': 'Honda',
        'model': 'Wave',
        'color': 'Đỏ',
        'ownerName': null,
        'ownerId': null,
        'registrationDate': null,
      };
      final vehicle = Vehicle.fromJson(json);
      expect(vehicle.ownerName, '');
      expect(vehicle.registrationDate, isA<DateTime>());
    });
  });

  group('Violation model', () {
    test('fromJson creates valid violation', () {
      final json = {
        'id': 'vio1',
        'licensePlate': '59A-12345',
        'violationType': 'Vượt đèn đỏ',
        'description': 'Test violation',
        'timestamp': '2025-01-01T10:30:00.000Z',
        'location': 'Ngã tư test',
        'imageUrl': 'https://example.com/img.jpg',
        'fineAmount': 500000,
        'status': 'pending',
        'violationCode': 'VH01',
        'lawReference': 'Test law',
      };
      final violation = Violation.fromJson(json);
      expect(violation.id, 'vio1');
      expect(violation.violationType, 'Vượt đèn đỏ');
      expect(violation.isPending, isTrue);
      expect(violation.fineAmount, 500000);
    });
  });
}
