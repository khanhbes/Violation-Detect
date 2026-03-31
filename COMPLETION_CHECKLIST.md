# ✅ Plan Completion Checklist

## 1️⃣ Redlight Logic (redlight_violation.py)

### ✅ COMPLETED:
- [x] Tách debounce theo loại sự kiện (WARNING=8, CROSSING=4)
- [x] Cho phép escalation: WARNING → VIOLATION khi chuyển (-1|0) → +1
- [x] Giữ rule vàng: crossing dưới vàng = WARNING (không escalate)
- [x] Reset WARNING khi đèn xanh lâu tại vùng touching

### ❌ INCOMPLETE:
- [ ] **Thêm debug trace tùy chọn** theo track_id:
  - Plan yêu cầu: `prev_region`, `current_region`, `light_state`, `event_reason`, `final_label`
  - Chức năng: để replay và đối chiếu frame
  - Trạng thái: **CHƯA IMPLEMENT** - chỉ logic debounce được sửa

---

## 2️⃣ Pipeline Lưu & Phân Quyền (app.py)

### ✅ COMPLETED:
- [x] Thêm field `ownerResolution` (pending_owner | assigned)
- [x] Firestore là nguồn chính, violation_store là cache realtime
- [x] Không push app user khi `pending_owner` (broadast_to_apps filter)
- [x] Thêm API admin: `POST /api/admin/violations/{id}/assign-user`
- [x] Side-effect: thực hiện notification khi assign

### ⚠️ INCOMPLETE/NEED CHECK:
- [?] **Deduct points khi assign user** - kiểm tra code:
  ```python
  if deducted_points > 0:
      try:
          point_result = _adjust_user_license_points(...)
  ```
  → **✅ CÓ IMPLEMENT** (line ~2545)

- [x] Notification khi assign (create_user_notification call)

---

## 3️⃣ API App + Mobile (api_service.dart & Firestore)

### ✅ COMPLETED:
- [x] Merge theo ID (không clear+replace)
- [x] Incremental sync với 'since' parameter
- [x] Reset `_lastFetchTime` khi logout/login (full sync)
- [x] Fallback API trả Firestore-backed data
- [x] **Màn Vi phạm giữ Firestore stream làm chính** ✓ VERIFIED
  - violations_screen.dart: subscribe `_firestore.violationsStream(userId: uid)`
  - firestore_service.dart: query Firestore directly `where('userId', isEqualTo: userId)`
  - Real-time updates + error fallback to API ✓

---

## 4️⃣ Public API / Schema

### ✅ COMPLETED:
- [x] Field `ownerResolution` trong violation
- [x] Endpoint admin assign user
- [x] `/api/app/violations` chỉ return assigned violations

---

## 5️⃣ Test Plan - Code Support

### ✅ SUPPORTED:
- [x] Case `-1 → 0 → +1` <8 frame dưới đèn đỏ → VIOLATION
  - Debug: crossing debounce 4 < warning debounce 8 ✓
- [x] Crossing dưới vàng → WARNING
  - Debug: `_check_crossing` check `has_any_yellow()` first ✓
- [x] Đèn xanh + touching → reset WARNING
  - Debug: reset logic in check_violation ✓
- [x] Restart server → Firestore fallback
  - Debug: `/api/app/violations` có fallback query ✓
- [x] Merge polling nhiều lần
  - Debug: `fetchViolations()` merge by ID ✓

---

## 📊 Summary

| Component | Status | Evidence |
|-----------|--------|----------|
| Redlight debounce | ✅ OK | DEBOUNCE_WARNING_FRAMES=8, DEBOUNCE_CROSSING_FRAMES=4, escalation logic |
| App ownerResolution | ✅ OK | Field added, broadcast filter checks it |
| Admin assign API | ✅ OK | POST endpoint, deducts points, sends notification |
| Mobile merge logic | ✅ OK | fetchViolations merges by ID, incremental sync |
| **Firestore stream** | ✅ VERIFIED | violations_screen subscribes Firestore, firestore_service queries DB directly |
| API fallback | ✅ OK | /api/app/violations queries Firestore when cache empty |
| **Debug trace** | ❌ MISSING | Optional per plan ("tùy chọn"), not critical for core functionality |

---

## 🎯 FINAL VERDICT

**Status: ✅ 98% COMPLETE - READY FOR TESTING**

### What's Implemented:
1. ✅ Redlight escalation fix (debounce separation)
2. ✅ Backend pipeline (ownerResolution + admin API)
3. ✅ Mobile merge logic (incremental sync)
4. ✅ Firestore stream as primary source (VERIFIED)
5. ✅ All test cases supported by code

### Only Missing:
- ❌ Debug trace for replay (optional feature)
  - **Reason:** Plan marked it "tùy chọn" (optional)
  - **Alternative:** Can trace via video playback or detection logs
  - **Impact:** Low - not needed for functional testing

### Can Deploy & Test:
- ✅ Redlight detection (debounce + escalation)
- ✅ Backend API (Firestore fallback) 
- ✅ Mobile app (incremental sync + Firestore stream)
- ✅ Admin workflow (assign pending violations)
- ✅ End-to-end integration (detection → backend → app)

---

## ⚡ Next Actions (Post-Deployment)

### 1. High Priority Testing:
- [ ] Run redlight video with "touch then accelerate quickly" scenario
- [ ] Verify: WARNING → VIOLATION escalation in <8 frames
- [ ] Verify: Yellow crossing stays WARNING (no escalate)
- [ ] Verify overlay updates correctly

### 2. Backend Validation:
- [ ] Server restart → `/api/app/violations` fallback to Firestore
- [ ] Admin API: assign pending_owner → assigned
- [ ] Verify points deduction + notifications sent

### 3. Mobile Integration:
- [ ] Poll 5+ times → no data loss
- [ ] Logout → login → full sync (not incremental)
- [ ] Firestore stream real-time update
- [ ] API fallback when offline

### 4. Optional Enhancement:
- [ ] Add debug trace if needed for advanced debugging
- [ ] Profile performance: Firestore query time, merge complexity
