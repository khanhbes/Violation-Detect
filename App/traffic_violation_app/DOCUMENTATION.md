# Tài liệu Hệ thống Phạt Nguội Giao Thông

## 🎯 Tổng quan hệ thống

### Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAFFIC VIOLATION SYSTEM                  │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│   WEB ADMIN      │         │   MOBILE APP     │
│   (Detection)    │         │   (User)         │
│                  │         │                  │
│ - YOLOv12 Seg   │         │ - Violations     │
│ - Realtime Det. │         │ - Payments       │
│ - License Plate │◄───────►│ - Notifications  │
│ - Auto Process  │   API   │ - Profile        │
└──────────────────┘         └──────────────────┘
        │                            │
        │                            │
        ▼                            ▼
┌─────────────────────────────────────────────┐
│           BACKEND SERVER (API)              │
│  - Authentication & Authorization           │
│  - Violation Management                     │
│  - Payment Processing                       │
│  - Notification Service (FCM/WebSocket)     │
│  - License Plate → User Matching            │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│              DATABASE                        │
│  - Users & Vehicles                         │
│  - Violations                               │
│  - Payments                                 │
│  - Traffic Laws                             │
└─────────────────────────────────────────────┘
```

## 📊 Data Flow

### 1. Quy trình phát hiện vi phạm

```
Camera → YOLOv12 → Detection → License Plate OCR → 
Database Lookup → User Match → Create Violation → 
Send Notification → User App
```

### 2. Quy trình thanh toán

```
User App → Select Violation → Payment Method → 
Generate QR/Deep Link → Bank Transfer → 
Webhook Callback → Verify Payment → 
Update Status → Confirm Notification
```

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    id VARCHAR(50) PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20) NOT NULL,
    avatar VARCHAR(255),
    id_card VARCHAR(20) UNIQUE NOT NULL,
    address TEXT,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### Vehicles Table
```sql
CREATE TABLE vehicles (
    id VARCHAR(50) PRIMARY KEY,
    license_plate VARCHAR(20) UNIQUE NOT NULL,
    vehicle_type ENUM('Xe máy', 'Ô tô', 'Xe tải', 'Xe khách'),
    brand VARCHAR(50),
    model VARCHAR(50),
    color VARCHAR(30),
    owner_id VARCHAR(50) NOT NULL,
    registration_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_license_plate (license_plate),
    INDEX idx_owner (owner_id)
);
```

### Violations Table
```sql
CREATE TABLE violations (
    id VARCHAR(50) PRIMARY KEY,
    license_plate VARCHAR(20) NOT NULL,
    violation_type VARCHAR(100) NOT NULL,
    violation_code VARCHAR(20) NOT NULL,
    description TEXT,
    timestamp TIMESTAMP NOT NULL,
    location VARCHAR(255) NOT NULL,
    image_url VARCHAR(255),
    fine_amount DECIMAL(10,2) NOT NULL,
    status ENUM('pending', 'paid', 'appealed', 'cancelled') DEFAULT 'pending',
    payment_id VARCHAR(50),
    paid_at TIMESTAMP NULL,
    law_reference VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_license_plate (license_plate),
    INDEX idx_status (status),
    INDEX idx_timestamp (timestamp)
);
```

### Traffic Laws Table
```sql
CREATE TABLE traffic_laws (
    code VARCHAR(20) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    law_reference VARCHAR(100) NOT NULL,
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE fine_levels (
    id VARCHAR(50) PRIMARY KEY,
    law_code VARCHAR(20) NOT NULL,
    vehicle_type VARCHAR(50) NOT NULL,
    min_amount DECIMAL(10,2) NOT NULL,
    max_amount DECIMAL(10,2) NOT NULL,
    additional_penalty TEXT,
    FOREIGN KEY (law_code) REFERENCES traffic_laws(code) ON DELETE CASCADE
);
```

### Payments Table
```sql
CREATE TABLE payments (
    id VARCHAR(50) PRIMARY KEY,
    violation_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    payment_method ENUM('bank_transfer', 'momo', 'vnpay', 'zalopay'),
    transaction_id VARCHAR(100),
    status ENUM('pending', 'success', 'failed') DEFAULT 'pending',
    paid_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (violation_id) REFERENCES violations(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_violation (violation_id),
    INDEX idx_user (user_id)
);
```

### Notifications Table
```sql
CREATE TABLE notifications (
    id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    violation_id VARCHAR(50),
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    type ENUM('violation', 'payment', 'reminder', 'system'),
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_read (user_id, is_read)
);
```

## 🔌 API Endpoints

### Authentication
```
POST   /api/auth/register
POST   /api/auth/login
POST   /api/auth/logout
POST   /api/auth/refresh-token
POST   /api/auth/forgot-password
```

### Users
```
GET    /api/user/profile
PUT    /api/user/profile
PUT    /api/user/password
DELETE /api/user/account
```

### Vehicles
```
GET    /api/vehicles
GET    /api/vehicles/:id
POST   /api/vehicles
PUT    /api/vehicles/:id
DELETE /api/vehicles/:id
```

### Violations
```
GET    /api/violations
GET    /api/violations/:id
GET    /api/violations/stats
POST   /api/violations/appeal/:id
```

### Payments
```
POST   /api/payments/initiate
POST   /api/payments/verify
GET    /api/payments/history
POST   /api/payments/webhook/momo
POST   /api/payments/webhook/vnpay
```

### Traffic Laws
```
GET    /api/traffic-laws
GET    /api/traffic-laws/:code
GET    /api/traffic-laws/search?q=
GET    /api/traffic-laws/category/:category
```

### Notifications
```
GET    /api/notifications
PUT    /api/notifications/:id/read
PUT    /api/notifications/read-all
DELETE /api/notifications/:id
```

### Admin (Web Detection System)
```
POST   /api/admin/violations/create
POST   /api/admin/violations/batch
GET    /api/admin/violations/pending
PUT    /api/admin/violations/:id/verify
```

## 💳 Payment Integration

### VNPay
```javascript
const vnpayPayment = {
  vnp_TmnCode: "YOUR_TMN_CODE",
  vnp_Amount: amount * 100, // VNĐ
  vnp_BankCode: "NCB", // Ngân hàng
  vnp_TxnRef: violationId,
  vnp_OrderInfo: `PHAT ${licensePlate} ${violationCode}`,
  vnp_ReturnUrl: "https://yourapp.com/payment/callback",
};
```

### MoMo
```javascript
const momoPayment = {
  partnerCode: "YOUR_PARTNER_CODE",
  amount: amount,
  orderId: violationId,
  orderInfo: `Vi pham ${violationType}`,
  redirectUrl: "https://yourapp.com/payment/callback",
  ipnUrl: "https://yourapi.com/payment/webhook/momo",
  requestType: "captureWallet",
};
```

### Bank Transfer (QR Code)
```javascript
// VietQR Standard
const qrData = {
  bank: "VCB", // Vietcombank
  accountNo: "1234567890",
  accountName: "CUC CSGT BO CONG AN",
  amount: fineAmount,
  addInfo: `PHAT ${licensePlate} ${violationCode}`,
  template: "compact",
};
```

## 🔔 Push Notifications

### Firebase Cloud Messaging (FCM)

```javascript
// Send notification when new violation detected
const notification = {
  title: "Vi phạm mới",
  body: `Phương tiện ${licensePlate} vừa có vi phạm ${violationType}`,
  data: {
    type: "violation",
    violationId: violation.id,
    action: "open_detail",
  },
  tokens: [userFcmToken],
};
```

## 🔒 Security Best Practices

### 1. Authentication
- JWT tokens với expiry time
- Refresh token rotation
- Rate limiting cho login attempts
- 2FA cho admin accounts

### 2. Data Protection
- Encrypt sensitive data (CCCD, payment info)
- HTTPS only
- Input validation & sanitization
- SQL injection prevention (parameterized queries)

### 3. Payment Security
- PCI DSS compliance
- No storing of card details
- Transaction logging
- Webhook signature verification

## 📱 Mobile App Features

### Implemented ✅
- Modern UI with gradient design
- Violation management
- Payment with QR codes
- Traffic laws database
- Push notifications UI
- Profile management

### To Be Implemented 🔜
- Biometric authentication
- Real FCM integration
- Offline mode
- Receipt generation (PDF)
- Multi-language support
- Dark mode

## 🚀 Deployment

### Mobile App
```bash
# Android
flutter build apk --release
flutter build appbundle --release

# iOS
flutter build ios --release
```

### Backend
```bash
# Docker deployment
docker build -t traffic-api .
docker run -p 3000:3000 traffic-api

# Environment variables
DATABASE_URL=postgresql://...
JWT_SECRET=your_secret
VNPAY_SECRET=...
MOMO_SECRET=...
FCM_SERVER_KEY=...
```

## 📈 Future Enhancements

1. **AI Features**
   - Chatbot hỗ trợ
   - Dự đoán vi phạm
   - Phân tích hành vi lái xe

2. **Analytics**
   - Dashboard thống kê
   - Báo cáo vi phạm theo khu vực
   - Xu hướng vi phạm

3. **Integration**
   - Cổng thanh toán quốc tế
   - Google Maps integration
   - Ví điện tử (ZaloPay, ShopeePay)

---

**Documentation Version**: 1.0  
**Last Updated**: February 2026
