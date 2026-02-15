# Traffic Violation App - Ứng dụng Phạt Nguội Giao Thông

Ứng dụng mobile hiện đại dành cho người dùng để quản lý vi phạm giao thông, thanh toán phạt nguội trực tuyến và tra cứu luật giao thông Việt Nam.

## 🌟 Tính năng chính

### ✅ Đã hoàn thành
- 🎨 **UI/UX hiện đại**: Thiết kế gradient đẹp mắt, animations mượt mà
- 🏠 **Dashboard**: Tổng quan vi phạm, thống kê chi tiết
- ⚠️ **Quản lý vi phạm**: 
  - Danh sách vi phạm (tất cả/chưa thanh toán/đã thanh toán)
  - Chi tiết vi phạm với ảnh, thời gian, địa điểm
  - Thông tin luật giao thông liên quan
- 💰 **Thanh toán**:
  - Chuyển khoản ngân hàng (VietcomBank, Techcombank, VietinBank)
  - Mã QR thanh toán tự động
  - Ví điện tử (MoMo, VNPay)
- 👤 **Quản lý tài khoản**:
  - Thông tin cá nhân
  - Danh sách phương tiện
  - Cài đặt ứng dụng
- 📚 **Tra cứu luật**:
  - Database đầy đủ luật giao thông VN (Nghị định 100/2019/NĐ-CP)
  - Tìm kiếm và lọc theo danh mục
  - Chi tiết mức phạt cho từng loại xe
- 🔔 **Thông báo**: Push notification khi có vi phạm mới

### 🎯 Data mẫu
- ✅ User profile hoàn chỉnh
- ✅ 2 phương tiện (xe máy, ô tô)
- ✅ 4 vi phạm mẫu (đã thanh toán và chưa thanh toán)
- ✅ 8 luật giao thông phổ biến tại Việt Nam
- ✅ Thông tin ngân hàng và QR code

## 🚀 Cài đặt

### Yêu cầu
- Flutter SDK >= 3.0.0
- Dart >= 3.0.0
- Android Studio / Xcode (cho Android/iOS)

### Bước 1: Clone project
```bash
cd /home/claude/traffic_violation_app
```

### Bước 2: Cài đặt dependencies
```bash
flutter pub get
```

### Bước 3: Chạy ứng dụng
```bash
# Android
flutter run

# iOS
flutter run -d ios

# Web (preview)
flutter run -d chrome
```

## 📱 Screenshots

### Màn hình chính
- ✅ Splash Screen với animation
- ✅ Login/Register
- ✅ Home Dashboard
- ✅ Violations List
- ✅ Violation Detail
- ✅ Payment (QR Code + Bank Transfer)
- ✅ Profile
- ✅ Traffic Laws
- ✅ Notifications

## 🎨 Design System

### Colors
- **Primary**: Indigo (#6366F1)
- **Secondary**: Purple (#8B5CF6)
- **Success**: Green (#10B981)
- **Warning**: Amber (#F59E0B)
- **Danger**: Red (#EF4444)

### Typography
- **Font**: Inter (Google Fonts)
- **Heading**: Bold, 20-32px
- **Body**: Regular, 14-16px

## 📦 Packages sử dụng

### UI & Design
- `google_fonts`: Typography đẹp
- `flutter_svg`: SVG icons
- `animations`: Smooth transitions
- `lottie`: Animation files

### Functionality
- `provider`: State management
- `go_router`: Navigation
- `http` & `dio`: API calls
- `shared_preferences`: Local storage
- `flutter_local_notifications`: Push notifications
- `qr_flutter`: QR code generation
- `intl`: Định dạng tiền tệ, ngày tháng

## 🔧 Cấu trúc project

```
lib/
├── main.dart                 # Entry point
├── theme/
│   └── app_theme.dart       # Theme configuration
├── models/
│   ├── user.dart
│   ├── vehicle.dart
│   ├── violation.dart
│   └── traffic_law.dart
├── screens/
│   ├── splash_screen.dart
│   ├── login_screen.dart
│   ├── home_screen.dart
│   ├── violations_screen.dart
│   ├── violation_detail_screen.dart
│   ├── payment_screen.dart
│   ├── profile_screen.dart
│   └── traffic_laws_screen.dart
├── data/
│   └── mock_data.dart       # Sample data
└── widgets/                  # Reusable widgets
```

## 🔄 Tích hợp Backend (Tương lai)

### API Endpoints cần thiết:
```
POST   /api/auth/login
POST   /api/auth/register
GET    /api/violations
GET    /api/violations/:id
POST   /api/payments
GET    /api/traffic-laws
GET    /api/user/profile
PUT    /api/user/profile
GET    /api/user/vehicles
```

### WebSocket cho thông báo realtime:
```
ws://your-server.com/notifications
```

## 🔐 Security

- Mã hóa thông tin thanh toán
- Authentication token JWT
- SSL/TLS cho API calls
- Biometric authentication (Face ID/Touch ID)

## 📝 TODO

### Phase 2
- [ ] Tích hợp API backend thực tế
- [ ] Push notification thực tế (Firebase Cloud Messaging)
- [ ] Biometric authentication
- [ ] Dark mode toggle
- [ ] Multi-language (English)
- [ ] History & Analytics
- [ ] Kháng cáo vi phạm
- [ ] In-app chat support

### Phase 3
- [ ] AI chatbot hỗ trợ
- [ ] Tích hợp camera để scan biển số
- [ ] Lịch sử hành trình
- [ ] Bản đồ vi phạm

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo Pull Request.

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết

## 📞 Liên hệ

- Email: support@trafficmonitor.vn
- Website: https://trafficmonitor.vn

---

**Phát triển bởi Traffic Monitor Team** 🚦
Phiên bản: 1.0.0
Ngày cập nhật: Tháng 2, 2026
