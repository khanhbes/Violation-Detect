# 🚦 Hướng Dẫn Chạy Hệ Thống VNeTraffic

## Mục lục
- [1. Tổng quan dự án](#1-tổng-quan-dự-án)
- [2. Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
- [3. Cài đặt & Chạy Backend (Detection Web)](#3-cài-đặt--chạy-backend-detection-web)
- [4. Cài đặt & Chạy Mobile App (Flutter)](#4-cài-đặt--chạy-mobile-app-flutter)
- [5. Deploy tự động (1-Click)](#5-deploy-tự-động-1-click)
- [6. Thanh toán Webhook (Ngrok)](#6-thanh-toán-webhook-ngrok)
- [7. Cấu trúc thư mục](#7-cấu-trúc-thư-mục)
- [8. Các file đã xóa (cleanup)](#8-các-file-đã-xóa-cleanup)
- [9. Lưu ý & Troubleshooting](#9-lưu-ý--troubleshooting)

---

## 1. Tổng quan dự án

**VNeTraffic** gồm 2 thành phần chính:

| Thành phần | Công nghệ | Vị trí |
|------------|-----------|--------|
| **Backend / AI Detection** | Python, FastAPI, YOLOv12, OpenCV | `Detection Web/` |
| **Mobile App** | Flutter / Dart | `App/traffic_violation_app/` |

**Luồng hoạt động:**
```
Camera/Video → YOLOv12 Detection → Vi phạm → Firebase Firestore → Push Notification → App
```

---

## 2. Yêu cầu hệ thống

### Backend (Python)
| Yêu cầu | Phiên bản tối thiểu |
|----------|---------------------|
| Python | 3.10+ |
| CUDA (khuyến nghị) | 11.8+ |
| GPU (khuyến nghị) | NVIDIA với VRAM ≥ 4GB |
| RAM | ≥ 8GB |

### Mobile App (Flutter)
| Yêu cầu | Phiên bản |
|----------|-----------|
| Flutter SDK | 3.41+ |
| Dart SDK | 3.11+ |
| Android Studio | Bản mới nhất |
| Java JDK | 21+ |

### Firebase
- Tạo project Firebase tại https://console.firebase.google.com
- Bật các dịch vụ: **Authentication**, **Cloud Firestore**, **Firebase Storage**, **Cloud Messaging**
- Tải `serviceAccountKey.json` đặt vào `Detection Web/Web/`
- Tải `google-services.json` đặt vào `App/traffic_violation_app/android/app/`
- Tạo `firebase-config.js` từ file mẫu `firebase-config.example.js` trong `Detection Web/Web/static/`

---

## 3. Cài đặt & Chạy Backend (Detection Web)

### Bước 1: Tạo môi trường ảo Python

```powershell
# Mở terminal tại thư mục gốc dự án
cd "Detection Web"
python -m venv .venv
```

### Bước 2: Kích hoạt môi trường ảo

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Mac/Linux
source .venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```powershell
# Quay về thư mục gốc
cd ..
pip install -r requirements.txt

# Cài thêm FastAPI & Uvicorn (nếu chưa có trong requirements.txt)
pip install fastapi uvicorn jinja2 python-multipart pyyaml
```

> **Lưu ý GPU:** Nếu muốn dùng GPU (khuyến nghị cho tốc độ):
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Bước 4: Chuẩn bị Model YOLO

- Đặt file model `.pt` vào thư mục `Detection Web/assets/model/`
- Model mặc định: `yolo26_rbf.pt` (model phát hiện vi phạm 40 class)
- Model segmentation: `yolov26s_seg.pt` hoặc `yolov12s_seg.pt`

### Bước 5: Cấu hình Firebase

1. Copy file `serviceAccountKey.json` vào `Detection Web/Web/`
2. Copy file `firebase-config.example.js` → `firebase-config.js` và điền thông tin Firebase project

### Bước 6: Chạy Server

```powershell
cd "Detection Web\Web"
python app.py
```

Server sẽ chạy tại: **http://localhost:8000**

### Bước 7: Truy cập Web Dashboard

Mở trình duyệt: http://localhost:8000

- Dashboard quản lý phát hiện vi phạm
- Chọn video + model + loại vi phạm → bấm Start Detection
- Xem kết quả realtime qua WebSocket

---

## 4. Cài đặt & Chạy Mobile App (Flutter)

### Bước 1: Cài đặt Flutter SDK

```powershell
# Kiểm tra Flutter đã cài chưa
flutter doctor
```

### Bước 2: Cài dependencies

```powershell
cd "App\traffic_violation_app"
flutter pub get
```

### Bước 3: Cấu hình Firebase cho App

1. Đặt `google-services.json` vào `android/app/`
2. Đảm bảo `applicationId` trong `android/app/build.gradle.kts` khớp Firebase

### Bước 4: Cấu hình kết nối Server

Mở file `lib/services/api_service.dart`:
```dart
static String serverIp = 'YOUR_LOCAL_IP';  // VD: '192.168.1.5'
```

> **Tự động:** Script deploy sẽ tự tìm IP WiFi và cập nhật file này.

### Bước 5: Chạy App (Debug)

```powershell
# Kết nối điện thoại Android (USB debug) hoặc emulator
flutter run
```

### Bước 6: Build APK Release

```powershell
flutter build apk --release
```

APK output: `build/app/outputs/flutter-apk/app-release.apk`

---

## 5. Deploy tự động (1-Click)

### Windows

```powershell
# Chạy tại thư mục gốc dự án
.\deploy.bat
```

### Mac/Linux

```bash
chmod +x deploy.sh   # Chỉ cần lần đầu
./deploy.sh
```

### Script tự động làm gì?
1. **Tìm IP WiFi** hiện tại của máy
2. **Cập nhật IP** vào `api_service.dart`
3. **Tăng version** tự động trong `pubspec.yaml`
4. **Build APK** release
5. **Upload APK** lên server → người dùng app nhận popup cập nhật

---

## 6. Thanh toán Webhook (Ngrok)

Để nhận webhook thanh toán từ SePay (VNPay/Momo/QR), cần tunnel server qua ngrok:

```powershell
# Chạy tại thư mục gốc
.\Start_Webhook.bat
```

- Server local: `http://localhost:8000`
- Ngrok URL: `https://unhealthier-cibarial-lannie.ngrok-free.dev`
- Webhook endpoint: `/api/webhook/sepay`

---

## 7. Cấu trúc thư mục (sau khi dọn dẹp)

```
Violation Detect/
├── Detection Web/                    # 🖥️ Backend + Web UI
│   ├── Web/
│   │   ├── app.py                   # FastAPI server chính (5480 dòng)
│   │   ├── services/
│   │   │   ├── detection_service.py # UnifiedDetector (AI core)
│   │   │   └── fcm_service.py       # Firebase Cloud Messaging
│   │   ├── static/                  # Frontend (JS, CSS, icons)
│   │   │   ├── app.js               # Dashboard JavaScript
│   │   │   ├── style.css            # Dashboard CSS
│   │   │   ├── i18n.js              # Đa ngôn ngữ
│   │   │   └── firebase-config.example.js
│   │   ├── templates/index.html     # Web dashboard HTML
│   │   ├── apk_releases/            # Kho APK cho OTA update
│   │   ├── uploads/                 # Upload complaints
│   │   ├── quota_data.json          # Firestore quota tracking
│   │   └── serviceAccountKey.json   # [GITIGNORE] Firebase key
│   │
│   ├── functions/                   # 🤖 6 Violation Detectors
│   │   ├── __init__.py
│   │   ├── helmet_violation.py      # Không đội mũ bảo hiểm
│   │   ├── redlight_violation.py    # Vượt đèn đỏ
│   │   ├── sidewalk_violation.py    # Đi lên vỉa hè
│   │   ├── sign_violation.py        # Vi phạm biển báo
│   │   ├── wrong_lane_violation.py  # Đi sai làn đường
│   │   └── wrong_way_violation.py   # Đi ngược chiều
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py                # Cấu hình tập trung (40 class YOLO)
│   │   └── custom_tracker.yaml      # Cấu hình ByteTrack
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── draw_utils.py            # Hàm vẽ bounding box, snapshot
│   │
│   ├── assets/
│   │   ├── model/                   # [GITIGNORE] YOLO .pt weights
│   │   ├── video/                   # [GITIGNORE] Video test
│   │   └── image/                   # Icon, favicon
│   │
│   ├── Violations/                  # Snapshot vi phạm (theo loại)
│   ├── output/                      # Video output đã xử lý
│   └── .gitignore
│
├── App/
│   ├── traffic_violation_app/       # 📱 Flutter Mobile App
│   │   ├── lib/
│   │   │   ├── main.dart            # Entry point
│   │   │   ├── screens/             # 11 màn hình UI
│   │   │   ├── services/            # 6 services (API, Auth, FCM...)
│   │   │   ├── models/              # Data models (Violation, User...)
│   │   │   ├── theme/               # App theme (Dark mode)
│   │   │   ├── data/                # Dữ liệu tĩnh
│   │   │   └── widgets/             # Reusable widgets
│   │   ├── android/                 # Android native config
│   │   ├── pubspec.yaml             # Dependencies
│   │   └── pubspec.lock
│   ├── QUICK_START.md
│   └── README.md
│
├── Project info/                    # 📄 Tài liệu dự án
│   ├── Project_Overview.md          # Tổng quan dự án
│   ├── architecture_diagram.md      # Sơ đồ kiến trúc (Mermaid)
│   ├── Fix_Firestore_Permission_Timeout_Plan.md
│   ├── class_index.txt              # 40 YOLO class IDs
│   ├── code_train.txt               # Code train model trên Colab
│   ├── prompt_redlight2.txt         # Prompt gốc vượt đèn đỏ
│   ├── prompt_traffic_nckh.txt      # Prompt gốc full detection
│   ├── prompt_wrong_lane.txt        # Prompt gốc sai làn đường
│   └── firestore-react/             # Mã tham khảo React (unused)
│
├── deploy.bat                       # 🚀 Auto deploy Windows
├── deploy.sh                        # 🚀 Auto deploy Mac/Linux
├── deploy_helper.ps1                # Helper cho deploy.bat
├── Start_Webhook.bat                # Khởi động ngrok webhook
├── ngrok_bin/ngrok.exe              # Ngrok binary
│
├── requirements.txt                 # Python dependencies
├── firebase.json                    # Firebase config (rules)
├── firestore.rules                  # Firestore security rules
├── storage.rules                    # Storage security rules
├── HE_THONG_HOAT_DONG.md           # Tài liệu hệ thống hoạt động
├── README.md
├── HUONG_DAN_CHAY.md               # 📖 FILE NÀY
└── .gitignore
```

---

## 8. Các file đã xóa (cleanup)

### ❌ File/thư mục đã xóa và lý do

| File / Thư mục | Kích thước | Lý do xóa |
|-----------------|-----------|------------|
| `__pycache__/` (tất cả 5 thư mục) | ~215 KB | Cache Python tự sinh, không cần commit |
| `.idea/` (root + App) | ~5 KB | Config IDE JetBrains, tự sinh khi mở project |
| `.vscode/` | 0 KB (rỗng) | Config VSCode rỗng, không cần thiết |
| `ngrok.zip` | **11.2 MB** | File zip gốc đã giải nén thành `ngrok_bin/ngrok.exe` |
| `demo.png` | **957 KB** | Ảnh demo cũ không sử dụng |
| `check_system.py` | 2 KB | Script kiểm tra hệ thống 1 lần, đã dùng xong |
| `COMPLETION_CHECKLIST.md` | 5 KB | Checklist phát triển nội bộ, đã hoàn thành |
| `IMPLEMENTATION_CHANGES.md` | 14 KB | Ghi chú implementation cũ, đã merge code |
| `Detection Web/deploy_helper.ps1` | 3 KB | **Bản trùng** với `deploy_helper.ps1` ở root |
| `Detection Web/Web/web_data.json` | ~73 B | Data debug lỗi (PowerShell output, không phải JSON) |
| `Detection Web/Web/web_data2.json` | ~0 B | File rỗng |
| `App/.../analyze_output.txt` | 15 KB | Output `flutter analyze` cũ, tự chạy lại được |
| `App/.../build_release.log` | **330 KB** | Log build cũ, tự sinh mỗi lần build |
| `App/.../traffic_violation_app.iml` | 14 KB | File IntelliJ tự sinh |
| `Detection Web/output/*.mp4` (2 file) | **18.2 MB** | Video output cũ đã xử lý |
| `Detection Web/Web/apk_releases/v1.0.73.apk` | **72.7 MB** | APK phiên bản cũ (giữ lại v1.0.77) |
| `Detection Web/Web/apk_releases/v1.0.74.apk` | **72.7 MB** | APK phiên bản cũ (giữ lại v1.0.77) |

### 💾 Tổng dung lượng giải phóng: **~175 MB**

### ✅ Các file giữ lại (quan trọng)

| File | Lý do giữ |
|------|-----------|
| `HE_THONG_HOAT_DONG.md` | Tài liệu mô tả hệ thống (sơ đồ khối, lưu đồ) |
| `Project info/` (tất cả) | Tài liệu dự án, kiến trúc, prompt training |
| `deploy.bat` / `deploy.sh` / `deploy_helper.ps1` | Script deploy tự động |
| `Start_Webhook.bat` / `ngrok_bin/` | Webhook thanh toán |
| `firebase.json` / `firestore.rules` / `storage.rules` | Firebase config & rules |
| `requirements.txt` | Python dependencies |
| `.gitignore` (root + Detection Web) | Git rules |

---

## 9. Lưu ý & Troubleshooting

### ⚠️ Không commit các file nhạy cảm
Các file sau **KHÔNG được push lên Git** (đã có trong `.gitignore`):
- `serviceAccountKey.json` — Firebase Admin key
- `google-services.json` — Firebase Android config
- `firebase-config.js` — Firebase Web config
- `Detection Web/assets/model/*.pt` — Model nặng (~23-60MB)
- `Detection Web/assets/video/` — Video test (~130MB)

### 🔧 Lỗi thường gặp

**1. Server không khởi động:**
```
ModuleNotFoundError: No module named 'ultralytics'
```
→ Chạy: `pip install -r requirements.txt`

**2. CUDA không khả dụng:**
```
GPU không được sử dụng, chạy trên CPU
```
→ Cài PyTorch CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**3. Firebase PERMISSION_DENIED:**
→ Kiểm tra `firestore.rules` và deploy rules: `firebase deploy --only firestore:rules`

**4. App không kết nối server:**
→ Kiểm tra IP trong `api_service.dart` khớp IP WiFi máy chạy server
→ Hoặc chạy `deploy.bat` để tự động cập nhật

**5. Flutter build lỗi:**
```powershell
flutter clean
flutter pub get
flutter build apk --release
```

### 📊 Kiểm tra hệ thống nhanh

```powershell
# Kiểm tra Python + thư viện
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import ultralytics; print('YOLO OK')"

# Kiểm tra Flutter
flutter doctor

# Test server API
curl http://localhost:8000/api/app/stats
```

---

> 📝 **Tài liệu này được tạo tự động sau khi phân tích và dọn dẹp dự án.**
> Ngày tạo: 08/05/2026
