# 🏗️ Sơ Đồ Kiến Trúc Tổng Thể — Traffic Violation Detection System

## 1. Tổng Quan Hệ Thống

Hệ thống gồm **3 tầng chính**: Detection Server (Python/FastAPI), Firebase Cloud, và Flutter Mobile App.

```mermaid
graph TB
    subgraph CLIENT["📱 Client Layer"]
        WEB["🌐 Web Dashboard<br/>(HTML/JS/CSS)"]
        APP["📱 Flutter App<br/>(Android/iOS)"]
    end

    subgraph SERVER["⚙️ Detection Server (FastAPI)"]
        API["🔌 REST API<br/>(app.py)"]
        WS_DET["🔴 WebSocket /ws/detect<br/>(Real-time Detection)"]
        WS_APP["🔵 WebSocket /ws/app<br/>(App Notifications)"]
        DETECTOR["🧠 UnifiedDetector<br/>(detection_service.py)"]
        FCM_SVC["📨 FCM Service<br/>(fcm_service.py)"]
        STORE["💾 store_violation()"]
    end

    subgraph FIREBASE["☁️ Firebase Cloud"]
        AUTH["🔐 Firebase Auth"]
        FS["🗄️ Cloud Firestore"]
        STORAGE["📦 Firebase Storage"]
        FCM["📨 FCM"]
    end

    subgraph AI["🤖 AI Detection Engine"]
        YOLO["YOLOv12 Model<br/>(.pt weights)"]
        CONFIG["⚙️ config.py"]
        UTILS["🎨 draw_utils.py"]
    end

    WEB -- "WebSocket frames" --> WS_DET
    WEB -- "HTTP API calls" --> API
    APP -- "WebSocket real-time" --> WS_APP
    APP -- "REST API (fallback)" --> API
    APP -- "Auth" --> AUTH
    APP -- "Real-time stream" --> FS
    APP -- "Load images" --> STORAGE

    WS_DET --> DETECTOR
    DETECTOR --> YOLO
    DETECTOR --> CONFIG
    DETECTOR --> UTILS
    DETECTOR -- "violations detected" --> STORE
    STORE -- "save violation" --> FS
    STORE -- "upload snapshot" --> STORAGE
    STORE -- "push alert" --> FCM_SVC
    STORE -- "broadcast" --> WS_APP
    FCM_SVC --> FCM
    FCM -- "push notification" --> APP

    style CLIENT fill:#1a1a2e,stroke:#e94560,color:#fff
    style SERVER fill:#16213e,stroke:#0f3460,color:#fff
    style FIREBASE fill:#0f3460,stroke:#533483,color:#fff
    style AI fill:#533483,stroke:#e94560,color:#fff
```

---

## 2. Detection Server — Chi Tiết Backend

```mermaid
graph LR
    subgraph APP_PY["app.py — FastAPI Server (Port 8000)"]
        direction TB
        R1["GET / — Web Dashboard"]
        R2["GET /api/videos — List Videos"]
        R3["GET /api/models — List Models"]
        R4["GET /api/detectors — List Detectors"]
        R5["POST /api/detect/image — Image Detection"]
        R6["POST /api/detect/video — Video Detection"]
        R7["POST /api/lookup — Lookup Violations"]
        R8["GET /api/app/violations — App Violations"]
        R9["POST /api/fcm/register — Register FCM Token"]
        WS1["WS /ws/detect — Real-time Streaming"]
        WS2["WS /ws/app — App Notifications"]
    end

    subgraph SERVICES["Web/services/"]
        DS["detection_service.py<br/>━━━━━━━━━━━━━━━━<br/>UnifiedDetector<br/>• load YOLO model 1 lần<br/>• dispatch tới sub-detectors<br/>• process_frame()"]
        FS["fcm_service.py<br/>━━━━━━━━━━━━━━━━<br/>FCMService (Singleton)<br/>• register/remove tokens<br/>• send/broadcast push<br/>• cleanup stale tokens"]
    end

    WS1 --> DS
    R5 --> DS
    R6 --> DS

    style APP_PY fill:#16213e,stroke:#0f3460,color:#fff
    style SERVICES fill:#0f3460,stroke:#533483,color:#fff
```

---

## 3. AI Detection Engine — 6 Violation Detectors

```mermaid
graph TB
    UD["🧠 UnifiedDetector"]
    
    UD --> H["HelmetDetectorWrapper<br/>━━━━━━━━━━━━━━━━━<br/>helmet_violation.py<br/>• Phát hiện không đội MBH<br/>• Track motorcycle + rider"]
    UD --> R["RedlightDetectorWrapper<br/>━━━━━━━━━━━━━━━━━<br/>redlight_violation.py<br/>• Phát hiện vượt đèn đỏ<br/>• Stopline calibration<br/>• Traffic light detection"]
    UD --> S["SidewalkDetectorWrapper<br/>━━━━━━━━━━━━━━━━━<br/>sidewalk_violation.py<br/>• Phát hiện đi trên vỉa hè<br/>• Zone-based detection"]
    UD --> WW["WrongWayDetectorWrapper<br/>━━━━━━━━━━━━━━━━━<br/>wrong_way_violation.py<br/>• Phát hiện đi ngược chiều<br/>• Direction flow analysis"]
    UD --> WL["WrongLaneDetectorWrapper<br/>━━━━━━━━━━━━━━━━━<br/>wrong_lane_violation.py<br/>• Phát hiện sai làn đường<br/>• Lane segmentation"]
    UD --> SG["SignDetectorWrapper<br/>━━━━━━━━━━━━━━━━━<br/>sign_violation.py<br/>• Phát hiện vi phạm biển báo<br/>• Sign recognition"]

    YOLO["YOLOv12 (.pt)"] -.-> UD
    CFG["config.py<br/>━━━━━━━━━━<br/>• Paths, Class IDs<br/>• Thresholds, Colors<br/>• 40 YOLO classes"] -.-> UD
    DRAW["draw_utils.py<br/>━━━━━━━━━━<br/>• draw_bbox_with_label()<br/>• save_violation_snapshot()"] -.-> UD

    style UD fill:#e94560,stroke:#fff,color:#fff
    style H fill:#533483,stroke:#e94560,color:#fff
    style R fill:#533483,stroke:#e94560,color:#fff
    style S fill:#533483,stroke:#e94560,color:#fff
    style WW fill:#533483,stroke:#e94560,color:#fff
    style WL fill:#533483,stroke:#e94560,color:#fff
    style SG fill:#533483,stroke:#e94560,color:#fff
```

---

## 4. Flutter Mobile App — Chi Tiết Client

```mermaid
graph TB
    subgraph SCREENS["📱 Screens (11)"]
        SP["SplashScreen"]
        LG["LoginScreen"]
        RG["RegisterScreen"]
        HM["HomeScreen"]
        VL["ViolationsScreen"]
        VD["ViolationDetailScreen"]
        PM["PaymentScreen"]
        PF["ProfileScreen"]
        TL["TrafficLawsScreen"]
        NT["NotificationsScreen"]
        VH["VehiclesScreen"]
    end

    subgraph SVC["🔧 Services (6)"]
        AUTH_S["AuthService<br/>(Firebase Auth)"]
        FS_S["FirestoreService<br/>(Firestore CRUD)"]
        API_S["ApiService<br/>(WebSocket + HTTP)"]
        NOTI_S["NotificationService<br/>(Local Notifications)"]
        PUSH_S["PushNotificationService<br/>(FCM Remote)"]
        STOR_S["StorageService<br/>(Firebase Storage)"]
    end

    subgraph MODELS["📋 Models (4)"]
        M_V["Violation"]
        M_U["User"]
        M_VH["Vehicle"]
        M_TL["TrafficLaw"]
    end

    SP --> LG
    LG --> HM
    RG --> HM
    HM --> VL
    HM --> TL
    HM --> PF
    HM --> NT
    HM --> VH
    VL --> VD
    VD --> PM

    LG --> AUTH_S
    RG --> AUTH_S
    VL --> FS_S
    VL --> API_S
    VD --> FS_S
    HM --> API_S
    NT --> PUSH_S
    PF --> FS_S
    VH --> FS_S

    FS_S --> M_V
    FS_S --> M_U
    FS_S --> M_VH
    API_S --> M_V

    style SCREENS fill:#1a1a2e,stroke:#e94560,color:#fff
    style SVC fill:#16213e,stroke:#0f3460,color:#fff
    style MODELS fill:#0f3460,stroke:#533483,color:#fff
```

---

## 5. Luồng Dữ Liệu — Phát Hiện Vi Phạm

```mermaid
sequenceDiagram
    participant Web as 🌐 Web Dashboard
    participant Server as ⚙️ FastAPI Server
    participant YOLO as 🤖 YOLOv12
    participant FS as 🗄️ Firestore
    participant Storage as 📦 Firebase Storage
    participant FCM as 📨 FCM
    participant App as 📱 Flutter App

    Web->>Server: WebSocket: start detection (video + model + detectors)
    loop Mỗi Frame
        Server->>YOLO: process_frame(frame, enabled_detectors)
        YOLO-->>Server: annotated_frame + violations[]
        Server-->>Web: WebSocket: {type: "frame", image: base64, stats}
        
        alt Phát hiện vi phạm
            Server->>Storage: Upload violation snapshot
            Storage-->>Server: image_url
            Server->>FS: Save violation document
            Server->>FCM: broadcast_push_notification()
            FCM-->>App: Push Notification 🔔
            Server-->>App: WebSocket /ws/app: {type: "new_violation"}
        end
    end
    
    App->>FS: Real-time stream violations
    FS-->>App: Violation updates (auto-sync)
```

---

## 6. Cấu Trúc Thư Mục

```
Violation Detect/
├── Detection Web/                    # 🖥️ Server + Web UI
│   ├── Web/
│   │   ├── app.py                    # FastAPI server (1030 lines)
│   │   ├── services/
│   │   │   ├── detection_service.py  # UnifiedDetector (630 lines)
│   │   │   └── fcm_service.py        # FCM push service (360 lines)
│   │   ├── static/
│   │   │   ├── app.js                # Frontend JS
│   │   │   ├── style.css             # Frontend CSS
│   │   │   └── firebase-messaging-sw.js
│   │   ├── templates/index.html      # Web dashboard
│   │   └── serviceAccountKey.json    # Firebase credentials
│   ├── functions/                    # 🤖 6 Violation Detectors
│   │   ├── helmet_violation.py
│   │   ├── redlight_violation.py
│   │   ├── sidewalk_violation.py
│   │   ├── sign_violation.py
│   │   ├── wrong_lane_violation.py
│   │   └── wrong_way_violation.py
│   ├── config/config.py              # ⚙️ Centralized config (232 lines)
│   ├── utils/draw_utils.py           # 🎨 Drawing utilities
│   └── assets/                       # Model weights + test videos
│       ├── model/
│       ├── video/
│       └── image/
│
├── App/traffic_violation_app/        # 📱 Flutter Mobile App
│   └── lib/
│       ├── main.dart                 # App entry point
│       ├── screens/                  # 11 screens (UI)
│       ├── services/                 # 6 services (business logic)
│       ├── models/                   # 4 data models
│       ├── theme/                    # App theme
│       └── data/                     # Static data
│
└── Project info/                     # 📄 Documentation & prompts
```

---

## 7. Công Nghệ Sử Dụng

| Tầng | Công nghệ | Mô tả |
|------|-----------|-------|
| **AI/ML** | YOLOv12 + OpenCV | Object detection & tracking |
| **Backend** | FastAPI + Uvicorn | HTTP + WebSocket server |
| **Database** | Cloud Firestore | Real-time NoSQL database |
| **Storage** | Firebase Storage | Violation snapshot images |
| **Auth** | Firebase Auth | Email/password authentication |
| **Push** | Firebase Cloud Messaging | Push notifications (Android/iOS/Web) |
| **Mobile** | Flutter/Dart | Cross-platform mobile app |
| **Web UI** | HTML/JS/CSS | Detection monitoring dashboard |
| **Config** | Python (centralized) | 40 YOLO classes, paths, thresholds |


```mermaid
flowchart LR
  subgraph Clients["Client Layer"]
    Web["Web UI<br/>`Detection Web/Web/templates/index.html` + `static/app.js`"]
    Mobile["Flutter App<br/>`App/traffic_violation_app/lib/...`"]
  end

  subgraph Backend["Backend Layer (FastAPI)"]
    API["REST API<br/>`/api/*`"]
    WSDetect["WebSocket<br/>`/ws/detect`"]
    WSApp["WebSocket<br/>`/ws/app`"]
    Unified["UnifiedDetector<br/>`Web/services/detection_service.py`"]
    Store["store_violation()<br/>`Web/app.py`"]
    FCM["FCMService<br/>`Web/services/fcm_service.py`"]
  end

  subgraph AI["AI Detection Layer (`Detection Web/functions`)"]
    Helmet["helmet_violation.py"]
    Sidewalk["sidewalk_violation.py"]
    Redlight["redlight_violation.py"]
    WrongWay["wrong_way_violation.py"]
    WrongLane["wrong_lane_violation.py"]
  end

  subgraph Data["Data Layer"]
    Model["YOLO model (.pt)"]
    Videos["Input videos"]
    Snap["Local snapshots<br/>`Detection Web/Violations/*`"]
    Out["Processed outputs<br/>`Detection Web/output/*`"]
    Firestore["Firebase Firestore<br/>violations + device tokens"]
    Storage["Firebase Storage<br/>violation images"]
  end

  subgraph Notify["Notification Layer"]
    Push["FCM Push (Android/iOS/Web)"]
  end

  Web -->|HTTP| API
  Web -->|WS start/stop/update| WSDetect
  Mobile -->|WS realtime alerts| WSApp
  Mobile -->|HTTP fallback| API
  Mobile -->|FCM token register| API
  Mobile -->|read violations stream| Firestore

  WSDetect --> Unified
  Unified --> Model
  Videos --> Unified
  Unified --> Helmet
  Unified --> Sidewalk
  Unified --> Redlight
  Unified --> WrongWay
  Unified --> WrongLane

  Unified -->|violation events| Store
  Store --> Snap
  Store --> Firestore
  Store --> Storage
  Store --> WSApp
  Store --> FCM
  FCM --> Push
  Push --> Mobile
  Push --> Web

  API --> Out

```