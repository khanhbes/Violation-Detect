# Backend Setup Guide

## Hướng dẫn thiết lập Backend cho Traffic Violation System

### Tech Stack Đề xuất

```
- Node.js + Express.js hoặc NestJS
- PostgreSQL (Database)
- Redis (Caching & Sessions)
- Firebase Cloud Messaging (Push Notifications)
- Socket.io (Real-time notifications)
```

### 1. Cấu trúc thư mục Backend

```
backend/
├── src/
│   ├── controllers/
│   │   ├── auth.controller.js
│   │   ├── user.controller.js
│   │   ├── violation.controller.js
│   │   ├── payment.controller.js
│   │   └── notification.controller.js
│   ├── models/
│   │   ├── User.js
│   │   ├── Vehicle.js
│   │   ├── Violation.js
│   │   ├── Payment.js
│   │   └── TrafficLaw.js
│   ├── routes/
│   │   ├── auth.routes.js
│   │   ├── user.routes.js
│   │   ├── violation.routes.js
│   │   ├── payment.routes.js
│   │   └── admin.routes.js
│   ├── services/
│   │   ├── detection.service.js      # YOLOv12 integration
│   │   ├── ocr.service.js            # License plate OCR
│   │   ├── payment.service.js        # VNPay/MoMo
│   │   └── notification.service.js   # FCM
│   ├── middleware/
│   │   ├── auth.middleware.js
│   │   ├── validation.middleware.js
│   │   └── errorHandler.middleware.js
│   ├── config/
│   │   ├── database.js
│   │   ├── payment.js
│   │   └── firebase.js
│   └── app.js
├── .env.example
├── package.json
└── README.md
```

### 2. Environment Variables (.env)

```env
# Server
NODE_ENV=development
PORT=3000
API_VERSION=v1

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=traffic_violation
DB_USER=postgres
DB_PASSWORD=your_password

# JWT
JWT_SECRET=your_super_secret_key_here
JWT_EXPIRE=7d
JWT_REFRESH_SECRET=your_refresh_token_secret
JWT_REFRESH_EXPIRE=30d

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Firebase Cloud Messaging
FCM_SERVER_KEY=your_fcm_server_key
FCM_PROJECT_ID=your_project_id

# Payment Gateways
# VNPay
VNPAY_TMN_CODE=your_tmn_code
VNPAY_HASH_SECRET=your_hash_secret
VNPAY_URL=https://sandbox.vnpayment.vn/paymentv2/vpcpay.html
VNPAY_RETURN_URL=https://yourapp.com/payment/callback

# MoMo
MOMO_PARTNER_CODE=your_partner_code
MOMO_ACCESS_KEY=your_access_key
MOMO_SECRET_KEY=your_secret_key
MOMO_ENDPOINT=https://test-payment.momo.vn
MOMO_IPN_URL=https://yourapi.com/payment/webhook/momo

# YOLOv12 Detection Service
DETECTION_SERVICE_URL=http://localhost:5000
DETECTION_API_KEY=your_detection_api_key

# OCR Service
OCR_SERVICE_URL=http://localhost:5001
OCR_API_KEY=your_ocr_api_key

# Email (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# File Upload
MAX_FILE_SIZE=10485760  # 10MB
UPLOAD_PATH=./uploads
```

### 3. Package.json

```json
{
  "name": "traffic-violation-backend",
  "version": "1.0.0",
  "description": "Backend API for Traffic Violation Detection System",
  "main": "src/app.js",
  "scripts": {
    "start": "node src/app.js",
    "dev": "nodemon src/app.js",
    "migrate": "node src/database/migrate.js",
    "seed": "node src/database/seed.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "pg": "^8.11.0",
    "pg-hstore": "^2.3.4",
    "sequelize": "^6.32.0",
    "bcryptjs": "^2.4.3",
    "jsonwebtoken": "^9.0.0",
    "cors": "^2.8.5",
    "dotenv": "^16.0.3",
    "helmet": "^7.0.0",
    "express-rate-limit": "^6.8.0",
    "joi": "^17.9.2",
    "axios": "^1.4.0",
    "multer": "^1.4.5-lts.1",
    "socket.io": "^4.6.2",
    "redis": "^4.6.7",
    "firebase-admin": "^11.9.0",
    "crypto": "^1.0.1",
    "moment": "^2.29.4"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
```

### 4. Sample Controller (violation.controller.js)

```javascript
const Violation = require('../models/Violation');
const Vehicle = require('../models/Vehicle');
const User = require('../models/User');
const NotificationService = require('../services/notification.service');

// Get all violations for current user
exports.getUserViolations = async (req, res) => {
  try {
    const userId = req.user.id;
    
    // Get user's vehicles
    const vehicles = await Vehicle.findAll({
      where: { owner_id: userId }
    });
    
    const licensePlates = vehicles.map(v => v.license_plate);
    
    // Get violations for those vehicles
    const violations = await Violation.findAll({
      where: {
        license_plate: licensePlates
      },
      order: [['timestamp', 'DESC']]
    });
    
    res.json({
      success: true,
      data: violations
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// Create new violation (Admin only - from web detection)
exports.createViolation = async (req, res) => {
  try {
    const { licensePlate, violationType, violationCode, location, imageUrl } = req.body;
    
    // Find vehicle and owner
    const vehicle = await Vehicle.findOne({
      where: { license_plate: licensePlate },
      include: [User]
    });
    
    if (!vehicle) {
      return res.status(404).json({
        success: false,
        message: 'Vehicle not found'
      });
    }
    
    // Get fine amount from traffic law
    const trafficLaw = await TrafficLaw.findByCode(violationCode);
    const fineAmount = trafficLaw.getFineAmount(vehicle.vehicle_type);
    
    // Create violation
    const violation = await Violation.create({
      license_plate: licensePlate,
      violation_type: violationType,
      violation_code: violationCode,
      location: location,
      image_url: imageUrl,
      fine_amount: fineAmount,
      timestamp: new Date(),
      status: 'pending'
    });
    
    // Send push notification to vehicle owner
    if (vehicle.User) {
      await NotificationService.sendViolationNotification(
        vehicle.User,
        violation
      );
    }
    
    res.status(201).json({
      success: true,
      data: violation
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};
```

### 5. Payment Service Integration

```javascript
// services/payment.service.js
const crypto = require('crypto');
const axios = require('axios');

class PaymentService {
  // VNPay Payment
  static async createVNPayPayment(violation, returnUrl) {
    const vnpUrl = process.env.VNPAY_URL;
    const tmnCode = process.env.VNPAY_TMN_CODE;
    const secretKey = process.env.VNPAY_HASH_SECRET;
    
    const date = new Date();
    const createDate = moment(date).format('YYYYMMDDHHmmss');
    const orderId = `${violation.id}_${createDate}`;
    
    let vnpParams = {
      vnp_Version: '2.1.0',
      vnp_Command: 'pay',
      vnp_TmnCode: tmnCode,
      vnp_Amount: violation.fine_amount * 100,
      vnp_CreateDate: createDate,
      vnp_CurrCode: 'VND',
      vnp_IpAddr: '127.0.0.1',
      vnp_Locale: 'vn',
      vnp_OrderInfo: `Thanh toan phat nguoi ${violation.license_plate}`,
      vnp_OrderType: 'other',
      vnp_ReturnUrl: returnUrl,
      vnp_TxnRef: orderId,
    };
    
    // Sort and create secure hash
    vnpParams = this.sortObject(vnpParams);
    const signData = querystring.stringify(vnpParams);
    const hmac = crypto.createHmac('sha512', secretKey);
    const signed = hmac.update(Buffer.from(signData, 'utf-8')).digest('hex');
    vnpParams['vnp_SecureHash'] = signed;
    
    const paymentUrl = vnpUrl + '?' + querystring.stringify(vnpParams);
    
    return {
      paymentUrl,
      orderId
    };
  }
  
  // MoMo Payment
  static async createMoMoPayment(violation, returnUrl) {
    const partnerCode = process.env.MOMO_PARTNER_CODE;
    const accessKey = process.env.MOMO_ACCESS_KEY;
    const secretKey = process.env.MOMO_SECRET_KEY;
    const endpoint = process.env.MOMO_ENDPOINT;
    
    const orderId = `${violation.id}_${Date.now()}`;
    const requestId = orderId;
    const amount = violation.fine_amount;
    const orderInfo = `Vi pham ${violation.violation_type}`;
    const ipnUrl = process.env.MOMO_IPN_URL;
    const requestType = 'captureWallet';
    const extraData = '';
    
    // Create signature
    const rawSignature = `accessKey=${accessKey}&amount=${amount}&extraData=${extraData}&ipnUrl=${ipnUrl}&orderId=${orderId}&orderInfo=${orderInfo}&partnerCode=${partnerCode}&redirectUrl=${returnUrl}&requestId=${requestId}&requestType=${requestType}`;
    
    const signature = crypto
      .createHmac('sha256', secretKey)
      .update(rawSignature)
      .digest('hex');
    
    const requestBody = {
      partnerCode,
      accessKey,
      requestId,
      amount,
      orderId,
      orderInfo,
      redirectUrl: returnUrl,
      ipnUrl,
      requestType,
      extraData,
      lang: 'vi',
      signature
    };
    
    const response = await axios.post(`${endpoint}/v2/gateway/api/create`, requestBody);
    
    return {
      paymentUrl: response.data.payUrl,
      orderId
    };
  }
}
```

### 6. Notification Service (FCM)

```javascript
// services/notification.service.js
const admin = require('firebase-admin');

class NotificationService {
  static async sendViolationNotification(user, violation) {
    const message = {
      notification: {
        title: 'Vi phạm mới',
        body: `Phương tiện ${violation.license_plate} vừa có vi phạm ${violation.violation_type}`,
      },
      data: {
        type: 'violation',
        violationId: violation.id,
        action: 'open_detail',
      },
      token: user.fcm_token,
    };
    
    try {
      const response = await admin.messaging().send(message);
      console.log('Successfully sent message:', response);
      return response;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }
  
  static async sendPaymentConfirmation(user, payment) {
    const message = {
      notification: {
        title: 'Thanh toán thành công',
        body: `Giao dịch ${payment.amount.toLocaleString('vi-VN')}đ đã được xác nhận`,
      },
      data: {
        type: 'payment',
        paymentId: payment.id,
      },
      token: user.fcm_token,
    };
    
    await admin.messaging().send(message);
  }
}
```

### 7. Database Migration Script

```javascript
// database/migrate.js
const { Sequelize } = require('sequelize');
const sequelize = require('../config/database');

const User = require('../models/User');
const Vehicle = require('../models/Vehicle');
const Violation = require('../models/Violation');
const TrafficLaw = require('../models/TrafficLaw');
const Payment = require('../models/Payment');

async function migrate() {
  try {
    await sequelize.authenticate();
    console.log('Database connection established');
    
    // Sync all models
    await sequelize.sync({ force: false });
    console.log('All models synced');
    
    process.exit(0);
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  }
}

migrate();
```

### 8. Seed Traffic Laws Data

```javascript
// database/seed.js
const TrafficLaw = require('../models/TrafficLaw');
const FineLevel = require('../models/FineLevel');

async function seedTrafficLaws() {
  const laws = [
    {
      code: 'DD01',
      title: 'Không chấp hành hiệu lệnh đèn tín hiệu giao thông',
      description: 'Người điều khiển phương tiện không chấp hành hiệu lệnh của đèn tín hiệu giao thông...',
      category: 'Đèn đỏ',
      law_reference: 'Điều 6, Nghị định 100/2019/NĐ-CP',
      effective_date: new Date('2020-01-01'),
      fineLevels: [
        { vehicle_type: 'Xe máy', min_amount: 800000, max_amount: 1000000 },
        { vehicle_type: 'Ô tô', min_amount: 4000000, max_amount: 6000000 }
      ]
    },
    // ... more laws
  ];
  
  for (const law of laws) {
    const created = await TrafficLaw.create(law);
    
    for (const level of law.fineLevels) {
      await FineLevel.create({
        ...level,
        law_code: created.code
      });
    }
  }
  
  console.log('Traffic laws seeded successfully');
}
```

### 9. Quick Start Commands

```bash
# Install dependencies
npm install

# Setup database
createdb traffic_violation

# Run migrations
npm run migrate

# Seed data
npm run seed

# Start development server
npm run dev

# Start production server
npm start
```

### 10. Testing the API

```bash
# Test authentication
curl -X POST http://localhost:3000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password123"}'

# Get violations (with auth token)
curl -X GET http://localhost:3000/api/v1/violations \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Create payment
curl -X POST http://localhost:3000/api/v1/payments/initiate \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"violationId":"vio001","method":"vnpay"}'
```

---

**Next Steps:**
1. Setup PostgreSQL database
2. Configure Firebase project for FCM
3. Register for VNPay/MoMo merchant accounts
4. Deploy to cloud (AWS, GCP, or Azure)
5. Setup CI/CD pipeline
