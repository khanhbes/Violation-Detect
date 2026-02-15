import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/models/user.dart';
import 'package:traffic_violation_app/models/vehicle.dart';
import 'package:traffic_violation_app/models/traffic_law.dart';

class MockData {
  // ── User ────────────────────────────────────────────────────────
  static final User currentUser = User(
    id: 'usr_001',
    fullName: 'Nguyễn Văn An',
    email: 'nguyenvanan@email.com',
    phone: '0912345678',
    idCard: '079201001234',
    address: '123 Nguyễn Huệ, Quận 1, TP.HCM',
    avatar: 'https://i.pravatar.cc/300?u=nguyenvanan',
    createdAt: DateTime(2024, 1, 15),
  );

  // ── Vehicles ────────────────────────────────────────────────────
  static final List<Vehicle> vehicles = [
    Vehicle(
      id: 'vh_001',
      licensePlate: '59A-123.45',
      vehicleType: 'Xe máy',
      brand: 'Honda',
      model: 'Wave Alpha',
      color: 'Đen',
      registrationDate: DateTime(2022, 6, 20),
      ownerName: 'Nguyễn Văn An',
      ownerId: 'usr_001',
    ),
    Vehicle(
      id: 'vh_002',
      licensePlate: '59B-678.90',
      vehicleType: 'Ô tô',
      brand: 'Toyota',
      model: 'Vios',
      color: 'Trắng',
      registrationDate: DateTime(2023, 3, 10),
      ownerName: 'Nguyễn Văn An',
      ownerId: 'usr_001',
    ),
  ];

  // ── Violations (fallback mock data) ─────────────────────────────
  static final List<Violation> violations = [
    Violation(
      id: 'vio_m001',
      licensePlate: '59A-123.45',
      violationType: 'Không đội mũ bảo hiểm',
      violationCode: 'MBH01',
      description: 'Không đội mũ bảo hiểm khi điều khiển xe mô tô',
      timestamp: DateTime(2025, 2, 10, 8, 30),
      location: 'Ngã tư Điện Biên Phủ - Hai Bà Trưng, Q.3',
      imageUrl:
          'https://images.unsplash.com/photo-1558618666-fcd25c85f82e?w=800',
      fineAmount: 200000,
      status: 'pending',
      lawReference: 'Điều 7, NĐ 100/2019/NĐ-CP',
    ),
    Violation(
      id: 'vio_m002',
      licensePlate: '59B-678.90',
      violationType: 'Vượt đèn đỏ',
      violationCode: 'DD01',
      description: 'Không chấp hành tín hiệu đèn giao thông',
      timestamp: DateTime(2025, 2, 8, 17, 45),
      location: 'Vòng xoay Lý Thái Tổ, Q.10',
      imageUrl:
          'https://images.unsplash.com/photo-1449034446853-66c86144b0ad?w=800',
      fineAmount: 800000,
      status: 'pending',
      lawReference: 'Điều 6, NĐ 100/2019/NĐ-CP',
    ),
    Violation(
      id: 'vio_m003',
      licensePlate: '59A-123.45',
      violationType: 'Chạy lên vỉa hè',
      violationCode: 'VH01',
      description: 'Điều khiển xe chạy trên hè phố',
      timestamp: DateTime(2025, 1, 25, 12, 15),
      location: 'Đường Nguyễn Huệ, Q.1',
      imageUrl:
          'https://images.unsplash.com/photo-1573348722427-f1d6819fdf98?w=800',
      fineAmount: 300000,
      status: 'paid',
      lawReference: 'Điều 4, NĐ 100/2019/NĐ-CP',
    ),
    Violation(
      id: 'vio_m004',
      licensePlate: '59B-678.90',
      violationType: 'Chạy ngược chiều',
      violationCode: 'NC01',
      description: 'Điều khiển xe đi ngược chiều đường',
      timestamp: DateTime(2025, 1, 20, 22, 10),
      location: 'Đường Cách Mạng Tháng 8, Q.3',
      imageUrl:
          'https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=800',
      fineAmount: 1000000,
      status: 'paid',
      lawReference: 'Điều 4, NĐ 100/2019/NĐ-CP',
    ),
    Violation(
      id: 'vio_m005',
      licensePlate: '59A-123.45',
      violationType: 'Đi sai làn đường',
      violationCode: 'LD01',
      description: 'Điều khiển xe không đi đúng phần đường quy định',
      timestamp: DateTime(2025, 1, 15, 7, 50),
      location: 'Đại lộ Võ Văn Kiệt, Q.5',
      imageUrl:
          'https://images.unsplash.com/photo-1606567595334-d39972c85dbe?w=800',
      fineAmount: 1000000,
      status: 'pending',
      lawReference: 'Điều 4, NĐ 100/2019/NĐ-CP',
    ),
  ];

  // ── Traffic Laws ────────────────────────────────────────────────
  static final List<TrafficLaw> trafficLaws = [
    TrafficLaw(
      code: 'MBH01',
      title: 'Mũ bảo hiểm',
      description:
          'Phạt tiền 200.000 - 300.000 đồng đối với người điều khiển xe mô tô, xe gắn máy không đội mũ bảo hiểm.',
      category: 'Mũ bảo hiểm',
      fineLevels: [
        FineLevel(
            vehicleType: 'Xe máy',
            minAmount: 200000,
            maxAmount: 300000,
            additionalPenalty: null),
      ],
      lawReference: 'Điều 7, NĐ 100/2019/NĐ-CP',
      effectiveDate: DateTime(2020, 1, 1),
    ),
    TrafficLaw(
      code: 'DD01',
      title: 'Vượt đèn đỏ',
      description:
          'Phạt tiền 800.000 - 1.000.000 đồng (xe máy) hoặc 4.000.000 - 6.000.000 đồng (ô tô).',
      category: 'Đèn tín hiệu',
      fineLevels: [
        FineLevel(
            vehicleType: 'Xe máy',
            minAmount: 800000,
            maxAmount: 1000000,
            additionalPenalty: null),
        FineLevel(
            vehicleType: 'Ô tô',
            minAmount: 4000000,
            maxAmount: 6000000,
            additionalPenalty: 'Tước GPLX 1-3 tháng'),
      ],
      lawReference: 'Điều 6, NĐ 100/2019/NĐ-CP',
      effectiveDate: DateTime(2020, 1, 1),
    ),
    TrafficLaw(
      code: 'VH01',
      title: 'Chạy lên vỉa hè',
      description:
          'Phạt tiền 300.000 - 400.000 đồng (xe máy) hoặc 2.000.000 - 3.000.000 đồng (ô tô).',
      category: 'Vỉa hè',
      fineLevels: [
        FineLevel(
            vehicleType: 'Xe máy',
            minAmount: 300000,
            maxAmount: 400000,
            additionalPenalty: null),
        FineLevel(
            vehicleType: 'Ô tô',
            minAmount: 2000000,
            maxAmount: 3000000,
            additionalPenalty: null),
      ],
      lawReference: 'Điều 4, NĐ 100/2019/NĐ-CP',
      effectiveDate: DateTime(2020, 1, 1),
    ),
    TrafficLaw(
      code: 'NC01',
      title: 'Chạy ngược chiều',
      description:
          'Phạt tiền 1.000.000 - 2.000.000 đồng (xe máy) hoặc 5.000.000 - 8.000.000 đồng (ô tô). Có thể tước GPLX.',
      category: 'Ngược chiều',
      fineLevels: [
        FineLevel(
            vehicleType: 'Xe máy',
            minAmount: 1000000,
            maxAmount: 2000000,
            additionalPenalty: null),
        FineLevel(
            vehicleType: 'Ô tô',
            minAmount: 5000000,
            maxAmount: 8000000,
            additionalPenalty: 'Tước GPLX 2-4 tháng'),
      ],
      lawReference: 'Điều 4, NĐ 100/2019/NĐ-CP',
      effectiveDate: DateTime(2020, 1, 1),
    ),
    TrafficLaw(
      code: 'LD01',
      title: 'Đi sai làn đường',
      description:
          'Phạt tiền 1.000.000 - 2.000.000 đồng (xe máy) hoặc 4.000.000 - 6.000.000 đồng (ô tô).',
      category: 'Làn đường',
      fineLevels: [
        FineLevel(
            vehicleType: 'Xe máy',
            minAmount: 1000000,
            maxAmount: 2000000,
            additionalPenalty: null),
        FineLevel(
            vehicleType: 'Ô tô',
            minAmount: 4000000,
            maxAmount: 6000000,
            additionalPenalty: null),
      ],
      lawReference: 'Điều 4, NĐ 100/2019/NĐ-CP',
      effectiveDate: DateTime(2020, 1, 1),
    ),
  ];
}
