# 🚀 HƯỚNG DẪN NHANH - Traffic Violation App

## ✨ Tổng quan dự án

Ứng dụng **Traffic Violation App** là hệ thống quản lý vi phạm giao thông hiện đại với 2 thành phần chính:

### 📱 Mobile App (Flutter) - Dành cho người dùng
- Xem danh sách vi phạm
- Thanh toán phạt nguội qua QR Code
- Tra cứu luật giao thông Việt Nam
- Nhận thông báo realtime

### 💻 Web Admin (Đã có sẵn) - Phát hiện vi phạm
- YOLOv26 Segmentation
- Trích xuất biển số xe tự động
- Gửi thông báo đến app user

---

## 📦 Nội dung Package

```
traffic_violation_app/
├── lib/                      # Flutter source code
│   ├── main.dart            # Entry point
│   ├── screens/             # Các màn hình
│   ├── models/              # Data models
│   ├── theme/               # Theme & colors
│   └── data/                # Mock data
├── pubspec.yaml             # Dependencies
├── README.md                # Documentation
├── DOCUMENTATION.md         # Chi tiết kỹ thuật
└── BACKEND_SETUP.md         # Hướng dẫn backend
```

---

## 🎯 Cài đặt & Chạy App

### Bước 1: Cài đặt Flutter
```bash
# Download Flutter SDK từ: https://flutter.dev/docs/get-started/install
# Hoặc sử dụng Flutter Version Manager (FVM)

# Kiểm tra
flutter doctor
```

### Bước 2: Giải nén và Setup
```bash
# Giải nén file
tar -xzf traffic_violation_app.tar.gz
cd traffic_violation_app

# Cài đặt dependencies
flutter pub get
```

### Bước 3: Chạy App
```bash
# Chạy trên Android
flutter run

# Chạy trên iOS (cần macOS)
flutter run -d ios

# Chạy trên Web (xem trước)
flutter run -d chrome
```

---

## 🎨 Tính năng đã triển khai

### ✅ UI/UX
- [x] Splash Screen với animation
- [x] Login/Register screens
- [x] Home Dashboard với thống kê
- [x] Danh sách vi phạm (filter: tất cả/pending/paid)
- [x] Chi tiết vi phạm với thông tin đầy đủ
- [x] Thanh toán (QR Code, VNPay, MoMo)
- [x] Profile & Vehicle Management
- [x] Tra cứu luật giao thông
- [x] Notifications center

### ✅ Data mẫu
- [x] 1 User profile
- [x] 2 Vehicles (xe máy, ô tô)
- [x] 4 Violations (2 pending, 2 paid)
- [x] 8 Traffic Laws (Nghị định 100/2019/NĐ-CP)
- [x] Thông tin ngân hàng (Vietcombank, Techcombank, VietinBank)

### ✅ Tính năng
- [x] Authentication flow
- [x] Violation filtering & sorting
- [x] Payment QR code generation
- [x] Traffic law search
- [x] Responsive UI
- [x] Modern gradient design

---

## 🔧 Tùy chỉnh

### Thay đổi màu sắc
File: `lib/theme/app_theme.dart`
```dart
static const primaryColor = Color(0xFF6366F1);  // Màu chính
static const secondaryColor = Color(0xFF8B5CF6); // Màu phụ
```

### Thêm data mẫu
File: `lib/data/mock_data.dart`
```dart
// Thêm vi phạm mới
static final List<Violation> violations = [
  // ... thêm data ở đây
];
```

### Kết nối API thực
File: `lib/services/api_service.dart` (cần tạo)
```dart
class ApiService {
  static const baseUrl = 'https://your-api.com/api';
  
  static Future<List<Violation>> getViolations() async {
    final response = await http.get('$baseUrl/violations');
    // Parse và return
  }
}
```

---

## 🔌 Tích hợp Backend

### Quick Start Backend (Node.js)
```bash
# Xem file BACKEND_SETUP.md để biết chi tiết

# 1. Tạo database
createdb traffic_violation

# 2. Setup environment
cp .env.example .env
# Chỉnh sửa .env với thông tin của bạn

# 3. Chạy migrations
npm run migrate

# 4. Seed data
npm run seed

# 5. Start server
npm run dev
```

### API Endpoints cần thiết
```
POST   /api/auth/login
GET    /api/violations
GET    /api/violations/:id
POST   /api/payments/initiate
GET    /api/traffic-laws
```

---

## 📱 Build APK/IPA

### Android APK
```bash
flutter build apk --release
# File: build/app/outputs/flutter-apk/app-release.apk
```

### Android App Bundle (Google Play)
```bash
flutter build appbundle --release
# File: build/app/outputs/bundle/release/app-release.aab
```

### iOS (cần macOS + Xcode)
```bash
flutter build ios --release
# Sau đó archive trong Xcode
```

---

## 🎓 Hướng dẫn sử dụng App

### 1. Đăng nhập
- Email: `nguyenvanan@gmail.com` (mock data)
- Password: bất kỳ (6+ ký tự)

### 2. Xem vi phạm
- Tap vào vi phạm để xem chi tiết
- Filter theo trạng thái (pending/paid)

### 3. Thanh toán
- Chọn vi phạm → "Thanh toán ngay"
- Chọn phương thức (Bank/MoMo/VNPay)
- Scan QR hoặc nhập thông tin
- Xác nhận thanh toán

### 4. Tra cứu luật
- Vào màn hình "Traffic Laws"
- Search hoặc filter theo danh mục
- Tap để xem chi tiết mức phạt

---

## 🐛 Troubleshooting

### Flutter doctor issues
```bash
# Android licenses
flutter doctor --android-licenses

# iOS setup (macOS)
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
```

### Pub get errors
```bash
flutter clean
flutter pub get
```

### Build errors
```bash
flutter clean
cd android && ./gradlew clean && cd ..
flutter build apk
```

---

## 🚀 Roadmap

### Phase 2 (Cần Backend thực)
- [ ] Real API integration
- [ ] Firebase Cloud Messaging
- [ ] Biometric authentication
- [ ] Receipt generation (PDF)
- [ ] Dark mode

### Phase 3
- [ ] AI Chatbot
- [ ] License plate scanner
- [ ] Trip history
- [ ] Violation map
- [ ] Multi-language

---

## 📞 Support & Contact

- **Documentation**: Xem file `DOCUMENTATION.md`
- **Backend Setup**: Xem file `BACKEND_SETUP.md`
- **Issues**: Tạo issue trên GitHub

---

## 📄 License

MIT License - Free to use for personal and commercial projects

---

**Developed with ❤️ by Traffic Monitor Team**  
Version: 1.0.0  
Last Updated: February 2026

---

## ⚡ Quick Commands Reference

```bash
# Setup
flutter pub get

# Run
flutter run

# Build
flutter build apk --release

# Clean
flutter clean

# Check
flutter doctor

# Analyze
flutter analyze
```

---

**🎉 Chúc bạn phát triển thành công!**
