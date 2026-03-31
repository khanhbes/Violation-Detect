# Plan xử lý lỗi `PERMISSION_DENIED` + `Timeout` sau login (Firebase + kết nối server)

## 1) Tóm tắt nguyên nhân gốc
- `PERMISSION_DENIED` đồng loạt ở `users/violations/notifications/vehicles` nên app không đọc được Firestore sau login.
- App đang query `whereIn [uid, cccd]`, dễ bị rules chặn toàn bộ khi rules khóa theo `request.auth.uid`.
- App vẫn bám IP cũ (`192.168.1.8`), không đọc được `server/config` nên HTTP/WS fallback bị timeout.
- Có log “success giả” khi tạo profile thất bại.

## 2) Thay đổi triển khai
### A. Firestore rules (ưu tiên chính)
- `users/{uid}`: read/write khi `request.auth.uid == uid`.
- `violations`: read khi `resource.data.userId == request.auth.uid`.
- `notifications`: read/update/delete khi `resource.data.userId == request.auth.uid`.
- `vehicles`: read/write khi `resource.data.ownerId == request.auth.uid`.
- `server/config`: read cho user đã đăng nhập, write chỉ server/admin.
- Deny mặc định path khác.

### B. App query strategy
- Đổi Firestore query sang **UID-only**, không query theo CCCD.
- Giữ fallback API (`/api/app/violations`) khi stream Firestore rỗng/lỗi.

### C. Profile bootstrap + logging
- Không nuốt lỗi khi `createOrUpdateUserProfile` fail.
- Chỉ log “auto-created success” khi ghi Firestore/backend thật sự thành công.

### D. Auto-connect server
- Ưu tiên IP từ `server/config` (LAN).
- Login chỉ coi reconnect thành công khi WS nhận ack connected.
- Retry hữu hạn cho `session/upsert` và call nền.

### E. Backend diagnostics
- Duy trì cập nhật `server/config` lúc startup.
- Chuẩn hóa lỗi để app phân biệt `network_unreachable` / `auth_invalid` / `forbidden`.

## 3) Ảnh hưởng interface
- Không bắt buộc thêm API mới.
- Thay đổi hành vi chính: Firestore query UID-only + bỏ success giả + reconnect chờ ack WS.

## 4) Test plan
1. Rules test: user chỉ đọc/ghi dữ liệu của chính họ, đọc được `server/config`.
2. Login flow: không còn `permission-denied` hàng loạt.
3. Kết nối server: không còn timeout khi backend online.
4. Realtime: tab Vi phạm và chuông thông báo cập nhật đúng user.
5. Negative test: khi Firestore bị chặn, app fallback API, không crash.

## 5) Assumptions
- Dữ liệu nghiệp vụ chuẩn theo Firebase UID.
- Ưu tiên LAN IP từ backend/web.
- Có quyền chỉnh Firestore rules trên Firebase project.

## 6) CODING RULES & BEST PRACTICES (Dành cho các lần sửa code tiếp theo)
*Các quy tắc sau ĐƯỢC RÚT RA TỪ QUÁ TRÌNH FIX LỖI và bám sát kiến trúc hệ thống, BẮT BUỘC tuân thủ mỗi khi viết/sửa code mới:*

1. **Rule 1 - Firestore Query (UID-only & HTTP Fallback):**
   - Mọi truy vấn Firestore (`users`, `violations`, `notifications`, `vehicles`) **PHẢI** dùng `uid` (từ `FirebaseAuth`). KHÔNG query qua field khác (như CCCD) để tránh xung đột rules `request.auth.uid`.
   - Bất kỳ Stream Subscription nào (như `violationsStream`, `notificationsStream`) **LUÔN LUÔN** phải có handler `onError()`. Trong `onError`, bắt buộc fallback gọi HTTP API để tải dữ liệu, đảm bảo app không vỡ giao diện nếu Firestore dính quyền hoặc quota.

2. **Rule 2 - Logging & Xử lý bất đồng bộ (No False Success):**
   - Tuyệt đối **KHÔNG** sử dụng cờ boolean trung gian và bỏ qua ngoại lệ để log "success giả".
   - Các tác vụ ghi dữ liệu (như `createOrUpdateUserProfile`) phải đặt trong `try/catch`. 
   - Lệnh in log `✅ Thành công` KHÔNG ĐƯỢC đặt ở ngoài, mà phải nằm ngay **SAU** khi `await` hoàn tất bên trong khối `try`. Nếu thất bại thì log lỗi và xử lý an toàn tại `catch`.

3. **Rule 3 - WebSocket & Connection Lifecycle:**
   - Khi reconnect (vd: đăng nhập tài khoản khác), **BẮT BUỘC phải reset state** (`_wsConnected = false`, `_isConnected = false`). Tuyệt đối không nhảy cóc (early-return) bằng giá trị state của session cũ.
   - Timeout khi chờ WebSocket ACK (`connected`) phải có fallback sang HTTP Health Check (`testConnection()`), tránh làm app "treo" mạng chỉ do đường truyền WS tạm thời thiếu ổn định.
   - Thêm độ trễ (delay tối thiểu 500-800ms) trước khi build kết nối WebSocket mới để Firebase Auth Token chắc chắn đã đồng bộ.

4. **Rule 4 - Backend Error Handling Standard:**
   - Tại Backend FastAPI, tất cả các endpoint trả data cho Mobile App (`/api/app/...`) **PHẢI bọc trong `try/except Exception as e:`**.
   - Nếu có exception, luôn trả về status 500 kèm cấu trúc JSON chuẩn có chứa field `error_type`. VD: `{"status": "error", "error_type": "server_error", "message": "..."}`.
   - Với lỗi 404/403, cũng phải có `error_type` tương ứng (như `"not_found"`, `"auth_invalid"`). Không trả string text/plain gây parse crash trên Client.

5. **Rule 5 - Firestore Quota Protection (Spark Plan 50k reads/ngày):**
   - **KHÔNG tạo snapshot listener trùng lặp.** Các stream `violationsStream`, `notificationsStream` PHẢI dùng **shared broadcast stream** từ `FirestoreService` singleton. KHÔNG gọi `.snapshots()` trực tiếp từ screen.
   - Mọi query snapshot **PHẢI có `.limit()`** (mặc định 100). Không bao giờ query toàn bộ collection không giới hạn.
   - Khi **signOut**, BẮT BUỘC gọi `FirestoreService().clearSharedStreams()` để đóng listener cũ trước khi user mới login.
   - **Theo dõi quota:** `FirestoreService` có `_countReads()` tự động đếm và log cảnh báo khi reads gần ngưỡng 30k-45k/ngày. Nếu thấy log `⚠️ FIRESTORE QUOTA WARNING` cần kiểm tra ngay source gây reads quá mức.
   - **Không dùng snapshot listener cho dữ liệu ít thay đổi.** Data như `server/config`, `app_config` nên dùng one-shot `.get()` + cache cục bộ thay vì realtime listener.

