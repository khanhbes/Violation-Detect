# Dự Án VNeTraffic: Hệ Thống Phát Hiện Vi Phạm Giao Thông & Ứng Dụng Nộp Phạt

Dự án **VNeTraffic** là một hệ thống toàn diện (End-to-End) bao gồm hệ thống giám sát vi phạm trên nền Web/Server (AI Detection) và một ứng dụng di động (Flutter App) dành cho người dân để tra cứu, theo dõi và xử lý vi phạm giao thông kịp thời. 

---

## 🏗 Cấu Trúc Tổng Quan (Architecture)

Dự án được chia thành **2 khối chính**:
1. **Khối Server / AI Detection (`Detection Web`)**: 
   - Đảm nhiệm việc nhận diện vi phạm giao thông theo thời gian thực (vượt đèn đỏ, lấn làn, không đội mũ bảo hiểm...).
   - **Backend API (FastAPI - Python)**: Máy chủ xử lý dữ liệu, trả kết quả cho App và quản lý kho ứng dụng (lưu trữ file APK cho người dùng cập nhật).
2. **Khối Di Động (`App/traffic_violation_app`)**:
   - Ứng dụng di động được viết bằng **Flutter**.
   - Dành cho công dân tra cứu vi phạm của bản thân, quản lý "Ví giấy tờ" (tích hợp bằng Lái xe, CCCD/CMND), theo dõi điểm bằng lái.
   - Cho phép nộp phạt trực tuyến và nhận thông báo theo thời gian thực.
3. **Khối DevOps & Automation** (Các file `deploy.bat`, `deploy.sh`, `deploy_helper.ps1`):
   - Hệ thống tối ưu hóa quy trình triển khai ứng dụng bằng 1 click.

---

## 💡 Các Tính Năng Nổi Bật (Features)

### 📲 Về Ứng Dụng Di Động (Flutter App)
- **Tra cứu và Thanh toán vi phạm**: Người dùng có thể kiểm tra danh sách vi phạm qua biển số xe hoặc mã số định danh và tiến hành nộp phạt trực tuyến ngay trên thiết bị di động.
- **Ví giấy tờ Điện Tử (Digital Wallet)**: Lưu trữ và số hóa thẻ Căn cước công dân (Đồng bộ số CCCD theo tài khoản đăng nhập), Giấy phép lái xe cùng với hệ thống tính điểm bằng lái.
- **Real-time Notifications (Thông báo thời gian thực)**: 
  - Ứng dụng tích hợp **Firebase Cloud Messaging (FCM)** và **WebSocket** để đẩy thông báo vi phạm ngay khi hệ thống Camera/AI dưới đường phố ghi nhận được.
- **OTA Updates (Tự động cập nhật phiên bản)**:
  - Tích hợp một service cập nhật độc quyền (không cần qua Google Play/App Store).
  - Khi có phiên bản mới, App sẽ tự động tải file `.apk` về (hiển thị % thanh tiến trình tải chuẩn xác) và tự động gọi hệ thống Android kích hoạt quy trình cài đặt (`open_file`).
- **Đồng bộ hóa Cloud (Cloud Sync)**:
  - Settings (Giao diện Dark/Light mode, Ngôn ngữ Anh/Việt), hồ sơ người dùng được đồng bộ dữ liệu thời gian thực lên **Firebase Firestore**. 
  - Trải nghiệm đăng nhập đồng nhất qua Firebase Auth.

### ⚙️ Về Hệ Thống Server (Python Backend)
- API mạnh mẽ tốc độ cao bằng **FastAPI**.
- Chịu trách nhiệm lưu trữ và phục vụ phân phối file `.apk`. Băng thông file lớn được tối ưu hóa bằng Header `Content-Length` cho client di động dễ dàng bắt stream.
- Quản lý metadata phiên bản App cho cơ chế tự động update (OTA).

### 🚀 Quy Trình Tự Động Hóa (1-Click CI/CD Deploy)
- Mọi thao tác deploy phiên bản ứng dụng đều được tự động hóa triệt để qua script `deploy.bat` (Cho Windows) và `deploy.sh` (Cho Mac/Linux).
- Khi người phát triển chạy script, hệ thống tự động:
  1. **Tìm kiếm IP mạng Wi-Fi/LAN** hiện tại của máy tính Host.
  2. Tự động mở mã nguồn Flutter (`api_service.dart`) thay thế địa chỉ kết nối bằng IP mạng cục bộ vừa tìm được.
  3. Quét tệp `pubspec.yaml`, tự động lấy version hiện tại (VD: `1.0.4+1`) và **tăng tự động** Patch & Build number (VD: `1.0.5+2`).
  4. Đóng gói (`flutter build apk`) tối ưu hóa ra file Release.
  5. Upload tệp APK vừa gắn thông số version lên **Server API** để gửi lệnh cảnh báo mọi App trên điện thoại hiện tại tiến hành Update.

---

## 🛠 Công Nghệ Sử Dụng (Tech Stack)

- **Ngôn ngữ & Framework**: 
  - `Flutter` / `Dart` (Mobile App)
  - `Python` / `FastAPI` (Backend Server / AI Detection)
  - `Bash` / `PowerShell` (CI/CD Automations)
- **Cơ sở dữ liệu & Backend Services**:
  - `Firebase Authentication` (Quản lý User Session)
  - `Cloud Firestore` (Lưu thông tin hồ sơ, logs và settings của App)
  - `Firebase Cloud Messaging` (Push Notifications)
  - `Firebase Storage` (Lưu trữ ảnh avatar, media vi phạm nếu cần)
- **Một số thư viện Dart/Flutter chính yếu**:
  - `package_info_plus`: Quản lý & đọc định danh Version linh hoạt.
  - `open_file`: Bypass hệ thống `FileProvider` của Android Nougat+ để ép buộc cài Update APK an toàn.
  - `http`: Giao tiếp Client - API.
  - `qr_flutter`: Tạo mã QR định danh cho ví điện tử/vi phạm.
  - Thư viện State Management (`Provider`/`ChangeNotifier` - Notification, App Settings...).
