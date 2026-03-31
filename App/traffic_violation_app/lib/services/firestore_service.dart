import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:traffic_violation_app/models/notification.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/models/user.dart' as app;
import 'package:traffic_violation_app/models/vehicle.dart';
import 'package:traffic_violation_app/services/api_service.dart';

/// Current data-source mode for the app.
enum DataMode {
  firestore,
  apiFallback,
}

/// Singleton service for Firestore CRUD operations.
///
/// Provides real-time streams and one-shot reads for
/// violations, users, and vehicles.
///
/// Uses **shared broadcast streams** so multiple screens listening
/// to the same data only create ONE Firestore snapshot listener,
/// dramatically reducing daily read count.
class FirestoreService {
  // ── Singleton ──────────────────────────────────────────────────
  static final FirestoreService _instance = FirestoreService._internal();
  factory FirestoreService() => _instance;
  FirestoreService._internal();

  // ── Data mode (firestore vs api-fallback) ─────────────────────
  DataMode _dataMode = DataMode.firestore;
  DataMode get dataMode => _dataMode;
  final StreamController<DataMode> _dataModeStream =
      StreamController<DataMode>.broadcast();
  Stream<DataMode> get dataModeStream => _dataModeStream.stream;

  void _setDataMode(DataMode mode) {
    if (_dataMode == mode) return;
    _dataMode = mode;
    _dataModeStream.add(mode);
    debugPrint('📡 DataMode changed: $mode');
  }

  // ── Permission-denied detection & throttling ──────────────────
  DateTime? _lastPermissionDeniedLog;
  int _permissionDeniedCount = 0;

  /// Check if a Firestore error is a permission-denied error.
  static bool isPermissionDeniedError(Object error) {
    final msg = error.toString().toLowerCase();
    return msg.contains('permission-denied') ||
        msg.contains('permission_denied');
  }

  /// Handle a permission-denied error: throttle logs, switch to API fallback.
  void _onPermissionDenied(String source, Object error) {
    _permissionDeniedCount++;
    final now = DateTime.now();
    final shouldLog = _lastPermissionDeniedLog == null ||
        now.difference(_lastPermissionDeniedLog!) > const Duration(seconds: 10);
    if (shouldLog) {
      debugPrint(
        '🔒 Firestore PERMISSION_DENIED ($source) '
        '[count=$_permissionDeniedCount]: $error',
      );
      _lastPermissionDeniedLog = now;
    }
    _setDataMode(DataMode.apiFallback);
  }

  /// Reset permission-denied state (e.g. after re-login).
  void resetPermissionState() {
    _permissionDeniedCount = 0;
    _lastPermissionDeniedLog = null;
    _setDataMode(DataMode.firestore);
  }

  // ── Shared broadcast stream cache ─────────────────────────────
  // Key = uid, value = broadcast stream.  All screens share one
  // Firestore snapshot listener per uid instead of each creating
  // their own.
  String _cachedViolationsUid = '';
  Stream<List<Violation>>? _sharedViolationsStream;
  StreamSubscription? _sharedViolationsSub;
  List<Violation>? _lastViolationsSnapshot;

  String _cachedNotificationsUid = '';
  Stream<List<AppNotification>>? _sharedNotificationsStream;
  StreamSubscription? _sharedNotificationsSub;
  List<AppNotification>? _lastNotificationsSnapshot;

  // ── Firestore reads quota monitor ─────────────────────────────
  // Approximate counter – resets daily.  Helps detect runaway reads
  // before hitting the Spark plan 50k/day limit.
  static const int _dailyReadWarningThreshold = 30000;
  static const int _dailyReadHardLimit = 45000;
  int _dailyReadCount = 0;
  int _lastResetDay = 0;
  bool _quotaWarningLogged = false;

  /// Current approximate reads today (for monitoring / debug).
  int get approximateDailyReads => _dailyReadCount;

  void _countReads(int count) {
    final today = DateTime.now().day;
    if (today != _lastResetDay) {
      _dailyReadCount = 0;
      _lastResetDay = today;
      _quotaWarningLogged = false;
    }
    _dailyReadCount += count;
    if (_dailyReadCount >= _dailyReadWarningThreshold && !_quotaWarningLogged) {
      debugPrint(
        '⚠️ FIRESTORE QUOTA WARNING: ~$_dailyReadCount reads today '
        '(threshold=$_dailyReadWarningThreshold, hard=$_dailyReadHardLimit)',
      );
      _quotaWarningLogged = true;
    }
  }

  /// Whether we should avoid additional reads to stay within budget.
  bool get isNearQuotaLimit => _dailyReadCount >= _dailyReadHardLimit;

  /// Call on logout to free shared streams.
  void clearSharedStreams() {
    _sharedViolationsSub?.cancel();
    _sharedViolationsSub = null;
    _sharedViolationsStream = null;
    _cachedViolationsUid = '';
    _lastViolationsSnapshot = null;
    _sharedNotificationsSub?.cancel();
    _sharedNotificationsSub = null;
    _sharedNotificationsStream = null;
    _cachedNotificationsUid = '';
    _lastNotificationsSnapshot = null;
    resetPermissionState();
    debugPrint('🧹 FirestoreService shared streams cleared');
  }

  /// Returns [source] prefixed with [seed] if non-null, so late subscribers
  /// to a shared broadcast stream immediately receive the last snapshot.
  Stream<T> _seedStream<T>(Stream<T> source, T? seed) async* {
    if (seed != null) yield seed;
    yield* source;
  }

  // Max documents per snapshot query – prevents reading entire
  // collection when data grows.
  static const int _violationsLimit = 100;
  static const int _notificationsLimit = 100;

  final FirebaseFirestore _db = FirebaseFirestore.instance;

  String _resolveStrictUid(String providedUserId) {
    final primary = providedUserId.trim();
    final authUid = (fb.FirebaseAuth.instance.currentUser?.uid ?? '').trim();

    if (authUid.isNotEmpty) {
      if (primary.isNotEmpty && authUid != primary) {
        debugPrint(
          '⚠️ Firestore UID mismatch: provided=$primary auth=$authUid (using auth uid)',
        );
      }
      return authUid;
    }

    return primary;
  }

  // ═══════════════════════════════════════════════════════════════
  // VIOLATIONS
  // ═══════════════════════════════════════════════════════════════

  CollectionReference<Map<String, dynamic>> get _violationsRef =>
      _db.collection('violations');

  List<Violation> _toViolationListFromDocs(
    Iterable<QueryDocumentSnapshot<Map<String, dynamic>>> docs,
  ) {
    final violations = docs.map((doc) {
      final data = doc.data();
      data['id'] = doc.id;
      return Violation.fromJson(data);
    }).toList();
    violations.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return violations;
  }

  Stream<List<Violation>> _violationsForUserStream(String userId) {
    final strictUid = _resolveStrictUid(userId);
    if (strictUid.isEmpty) {
      return Stream<List<Violation>>.value(<Violation>[]);
    }

    // ── Shared broadcast stream: reuse if uid hasn't changed ───
    if (_cachedViolationsUid == strictUid &&
        _sharedViolationsStream != null) {
      return _seedStream(_sharedViolationsStream!, _lastViolationsSnapshot);
    }

    // Tear down previous stream for old uid
    _sharedViolationsSub?.cancel();
    _lastViolationsSnapshot = null;

    final query = _violationsRef
        .where('userId', isEqualTo: strictUid)
        .limit(_violationsLimit);

    // Create a broadcast StreamController so multiple listeners share
    // one underlying Firestore snapshot listener.
    final controller = StreamController<List<Violation>>.broadcast();
    _sharedViolationsSub = query.snapshots().listen(
      (snapshot) {
        _countReads(snapshot.docs.length);
        final list = _toViolationListFromDocs(snapshot.docs);
        debugPrint(
          '📱 Firestore snapshot (uid=$strictUid): ${list.length} '
          '[reads today≈$_dailyReadCount]',
        );
        _lastViolationsSnapshot = list;
        controller.add(list);
      },
      onError: (Object error) {
        if (isPermissionDeniedError(error)) {
          _onPermissionDenied('violations_stream', error);
        } else {
          debugPrint('❌ Shared violations stream error: $error');
        }
        controller.addError(error);
      },
    );

    _cachedViolationsUid = strictUid;
    _sharedViolationsStream = controller.stream;
    return _seedStream(_sharedViolationsStream!, _lastViolationsSnapshot);
  }

  /// Real-time stream of violations.
  /// Can filter by licensePlate or userId.
  ///
  /// When filtered by [userId], returns a **shared broadcast stream**
  /// so all screens share one Firestore listener.
  Stream<List<Violation>> violationsStream(
      {String? licensePlate, String? userId}) {
    final normalizedUserId = userId?.trim() ?? '';
    if (normalizedUserId.isNotEmpty) {
      return _violationsForUserStream(normalizedUserId);
    }

    Query<Map<String, dynamic>> query = _violationsRef;
    if (licensePlate != null && licensePlate.isNotEmpty) {
      query = query.where('licensePlate', isEqualTo: licensePlate);
    }
    query = query.limit(_violationsLimit);

    return query.snapshots().map((snapshot) {
      _countReads(snapshot.docs.length);
      debugPrint('📱 Firestore snapshot: ${snapshot.docs.length} violations');
      return _toViolationListFromDocs(snapshot.docs);
    });
  }

  /// One-shot fetch of all violations.
  Future<List<Violation>> getViolations(
      {String? licensePlate, String? userId}) async {
    try {
      final normalizedUserId = userId?.trim() ?? '';
      if (normalizedUserId.isNotEmpty) {
        final strictUid = _resolveStrictUid(normalizedUserId);
        if (strictUid.isEmpty) return <Violation>[];

        final userQuery = _violationsRef.where('userId', isEqualTo: strictUid);

        final userSnapshot = await userQuery.get();
        final userViolations = _toViolationListFromDocs(userSnapshot.docs);
        debugPrint(
            '📱 Firestore fetch (uid=$strictUid): ${userViolations.length} violations');
        return userViolations;
      }

      Query<Map<String, dynamic>> query = _violationsRef;
      if (licensePlate != null && licensePlate.isNotEmpty) {
        query = query.where('licensePlate', isEqualTo: licensePlate);
      }

      final snapshot = await query.get();
      debugPrint('📱 Firestore fetch: ${snapshot.docs.length} violations');
      return _toViolationListFromDocs(snapshot.docs);
    } catch (e) {
      debugPrint('❌ Error fetching violations: $e');
      return [];
    }
  }

  /// Get a single violation by ID.
  Future<Violation?> getViolationById(String id) async {
    try {
      final doc = await _violationsRef.doc(id).get();
      if (!doc.exists) return null;
      final data = doc.data()!;
      data['id'] = doc.id;
      return Violation.fromJson(data);
    } catch (e) {
      debugPrint('❌ Error fetching violation $id: $e');
      return null;
    }
  }

  /// Update violation status (e.g., pending → paid).
  Future<void> updateViolationStatus(String id, String status) async {
    await _violationsRef.doc(id).update({'status': status});
  }

  /// Delete one paid violation.
  /// Only paid violations are allowed to be removed from user history.
  Future<void> deletePaidViolation(String violationId) async {
    final violationRef = await _resolveViolationDocRef(violationId);
    if (violationRef == null) {
      throw Exception('Không tìm thấy vi phạm để xóa.');
    }

    final snapshot = await violationRef.get();
    if (!snapshot.exists) {
      throw Exception('Không tìm thấy vi phạm để xóa.');
    }

    final data = snapshot.data() ?? <String, dynamic>{};
    final status = (data['status'] ?? '').toString().toLowerCase().trim();
    if (status != 'paid') {
      throw Exception('Chỉ được xóa vi phạm đã thanh toán.');
    }

    await violationRef.delete();
  }

  Future<DocumentReference<Map<String, dynamic>>?> _resolveViolationDocRef(
    String violationId,
  ) async {
    final normalizedId = violationId.trim();
    if (normalizedId.isEmpty) return null;

    final directRef = _violationsRef.doc(normalizedId);
    final directDoc = await directRef.get();
    if (directDoc.exists) return directRef;

    final byField = await _violationsRef
        .where('id', isEqualTo: normalizedId)
        .limit(1)
        .get();
    if (byField.docs.isNotEmpty) {
      return byField.docs.first.reference;
    }

    final allDocs = await _violationsRef.get();
    for (final doc in allDocs.docs) {
      if (doc.id.toUpperCase() == normalizedId.toUpperCase()) {
        return doc.reference;
      }
    }
    return null;
  }

  // ═══════════════════════════════════════════════════════════════
  // USERS
  // ═══════════════════════════════════════════════════════════════

  CollectionReference<Map<String, dynamic>> get _usersRef =>
      _db.collection('users');

  /// Get user profile by UID.
  Future<app.User?> getUserProfile(String uid) async {
    try {
      final doc = await _usersRef.doc(uid).get();
      if (!doc.exists) return null;
      final data = doc.data()!;
      data['id'] = doc.id;
      return app.User.fromJson(data);
    } catch (e) {
      debugPrint('❌ Error fetching user $uid: $e');
      return null;
    }
  }

  /// Update user profile fields (creates if not exists).
  Future<void> updateUserProfile(String uid, Map<String, dynamic> data) async {
    try {
      await _usersRef.doc(uid).set(data, SetOptions(merge: true));
      debugPrint('✅ User profile updated in Firestore');
    } catch (e) {
      debugPrint('❌ Error updating user profile: $e');
    }
  }

  /// Request profile update (Needs admin approval).
  Future<void> requestProfileUpdate(
      String uid, Map<String, dynamic> requestData) async {
    final sanitized = <String, dynamic>{};
    const reservedKeys = <String>{
      'userId',
      'status',
      'createdAt',
      'updatedAt',
      'reviewedAt',
      'reviewedBy',
      'source',
    };

    requestData.forEach((key, value) {
      final normalizedKey = key.toString().trim();
      if (normalizedKey.isEmpty || reservedKeys.contains(normalizedKey)) return;
      sanitized[normalizedKey] = value;
    });

    if (sanitized.isEmpty) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'invalid-argument',
        message: 'Không có dữ liệu thay đổi hợp lệ để gửi duyệt.',
      );
    }

    try {
      final sentByBackend = await _requestProfileUpdateViaBackend(
        uid: uid,
        requestData: sanitized,
      );
      if (sentByBackend) {
        debugPrint('✅ Profile update request sent via backend API');
        return;
      }
    } on FirebaseException {
      rethrow;
    } catch (e) {
      debugPrint('⚠️ Backend profile update request failed unexpectedly: $e');
    }

    try {
      final payload = <String, dynamic>{
        ...sanitized,
        'userId': uid,
        'status': 'pending',
        'updatedAt': FieldValue.serverTimestamp(),
      };

      payload['createdAt'] = FieldValue.serverTimestamp();

      await _db
          .collection('profile_update_requests')
          .doc(uid)
          .set(payload, SetOptions(merge: true));
      debugPrint(
          '✅ Profile update request sent to Firestore (direct fallback)');
    } catch (e) {
      debugPrint('❌ Error sending profile update request: $e');
      rethrow;
    }
  }

  Future<bool> _requestProfileUpdateViaBackend({
    required String uid,
    required Map<String, dynamic> requestData,
  }) async {
    final user = fb.FirebaseAuth.instance.currentUser;
    if (user == null) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'permission-denied',
        message: 'Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.',
      );
    }

    final idToken = await user.getIdToken(true);
    final uri =
        Uri.parse('${ApiService.baseUrl}/api/app/profile-update-request');

    http.Response response;
    try {
      response = await http
          .post(
            uri,
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
              'Authorization': 'Bearer $idToken',
              'ngrok-skip-browser-warning': 'true',
            },
            body: jsonEncode({
              'user_id': uid,
              'data': requestData,
            }),
          )
          .timeout(const Duration(seconds: 12));
    } on TimeoutException {
      debugPrint(
          '⚠️ profile-update-request API timeout, fallback to Firestore');
      return false;
    } catch (e) {
      debugPrint('⚠️ profile-update-request API unavailable, fallback: $e');
      return false;
    }

    if (response.statusCode == 404) {
      debugPrint(
          'ℹ️ profile-update-request endpoint not found, fallback to Firestore');
      return false;
    }

    Map<String, dynamic> body = const {};
    try {
      final parsed = jsonDecode(response.body);
      if (parsed is Map<String, dynamic>) body = parsed;
    } catch (_) {}

    final status = (body['status'] ?? '').toString().toLowerCase();
    final message = (body['message'] ?? '').toString().trim();

    if (response.statusCode >= 200 &&
        response.statusCode < 300 &&
        status == 'ok') {
      return true;
    }

    if (response.statusCode == 401 || response.statusCode == 403) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'permission-denied',
        message: message.isNotEmpty
            ? message
            : 'Không đủ quyền gửi yêu cầu chỉnh sửa thông tin.',
      );
    }

    if (response.statusCode == 422) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'invalid-argument',
        message: message.isNotEmpty
            ? message
            : 'Dữ liệu yêu cầu chỉnh sửa không hợp lệ.',
      );
    }

    if (response.statusCode >= 500) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'unavailable',
        message: message.isNotEmpty
            ? message
            : 'Server tạm thời không thể xử lý yêu cầu chỉnh sửa.',
      );
    }

    throw FirebaseException(
      plugin: 'cloud_firestore',
      code: 'unknown',
      message: message.isNotEmpty
          ? message
          : 'Không thể gửi yêu cầu chỉnh sửa thông tin.',
    );
  }

  // ═══════════════════════════════════════════════════════════════
  // VEHICLES
  // ═══════════════════════════════════════════════════════════════

  CollectionReference<Map<String, dynamic>> get _vehiclesRef =>
      _db.collection('vehicles');

  /// Real-time stream of vehicles for a user.
  Stream<List<Vehicle>> vehiclesStream(String ownerId) {
    return _vehiclesRef
        .where('ownerId', isEqualTo: ownerId)
        .snapshots()
        .map((snapshot) {
      return snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return Vehicle.fromJson(data);
      }).toList();
    });
  }

  /// One-shot fetch of vehicles for a user.
  Future<List<Vehicle>> getVehicles(String ownerId) async {
    try {
      final snapshot =
          await _vehiclesRef.where('ownerId', isEqualTo: ownerId).get();
      return snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return Vehicle.fromJson(data);
      }).toList();
    } catch (e) {
      debugPrint('❌ Error fetching vehicles: $e');
      return [];
    }
  }

  /// Add a new vehicle.
  Future<String> addVehicle(Vehicle vehicle) async {
    final doc = await _vehiclesRef.add(vehicle.toJson());
    return doc.id;
  }

  /// Update a vehicle.
  Future<void> updateVehicle(String id, Map<String, dynamic> data) async {
    try {
      await _vehiclesRef.doc(id).update(data);
    } catch (e) {
      debugPrint('❌ Error updating vehicle $id: $e');
    }
  }

  /// Delete a vehicle.
  Future<void> deleteVehicle(String id) async {
    try {
      await _vehiclesRef.doc(id).delete();
    } catch (e) {
      debugPrint('❌ Error deleting vehicle $id: $e');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // USER SETTINGS (stored as subcollection or merged into user doc)
  // ═══════════════════════════════════════════════════════════════

  /// Save user settings (theme, language, notifications) to Firestore.
  Future<void> saveUserSettings(
      String uid, Map<String, dynamic> settings) async {
    try {
      await _usersRef.doc(uid).set(
        {'settings': settings},
        SetOptions(merge: true),
      );
      debugPrint('✅ User settings saved to Firestore');
    } catch (e) {
      debugPrint('❌ Error saving user settings: $e');
    }
  }

  /// Load user settings from Firestore.
  Future<Map<String, dynamic>?> getUserSettings(String uid) async {
    try {
      final doc = await _usersRef.doc(uid).get();
      if (!doc.exists) return null;
      return doc.data()?['settings'] as Map<String, dynamic>?;
    } catch (e) {
      debugPrint('❌ Error fetching user settings: $e');
      return null;
    }
  }

  Future<void> syncServerConfig({
    required String ip,
    required int port,
  }) async {
    final normalizedIp = ip.trim();
    if (normalizedIp.isEmpty) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'invalid-argument',
        message: 'Địa chỉ IP máy chủ không hợp lệ.',
      );
    }
    final normalizedPort = port <= 0 ? 8000 : port;
    await _db.collection('server').doc('config').set({
      'ip': normalizedIp,
      'port': normalizedPort,
      'updatedBy': 'mobile_app',
      'updatedAt': FieldValue.serverTimestamp(),
    }, SetOptions(merge: true));
  }

  /// Create or update user profile (called after login if profile doesn't exist).
  Future<bool> createOrUpdateUserProfile(
      String uid, Map<String, dynamic> data) async {
    try {
      await _usersRef.doc(uid).set(data, SetOptions(merge: true));
      debugPrint('✅ User profile created/updated in Firestore');
      return true;
    } catch (e) {
      debugPrint('❌ Error creating user profile: $e');
      rethrow;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // NOTIFICATIONS
  // ═══════════════════════════════════════════════════════════════

  CollectionReference<Map<String, dynamic>> get _notificationsRef =>
      _db.collection('notifications');

  /// Stream of notifications for a specific user, ordered by newest first.
  ///
  /// Returns a **shared broadcast stream** – multiple screens listening
  /// to the same user's notifications share ONE Firestore listener.
  Stream<List<AppNotification>> notificationsStream(String userId) {
    final strictUid = _resolveStrictUid(userId);
    debugPrint(
        '📱 Resolve UID for notifications: requested=$userId uid=$strictUid');
    if (strictUid.isEmpty) {
      return Stream<List<AppNotification>>.value(<AppNotification>[]);
    }

    // ── Shared broadcast stream ─────────────────────────────────
    if (_cachedNotificationsUid == strictUid &&
        _sharedNotificationsStream != null) {
      return _seedStream(_sharedNotificationsStream!, _lastNotificationsSnapshot);
    }

    _sharedNotificationsSub?.cancel();
    _lastNotificationsSnapshot = null;

    final query = _notificationsRef
        .where('userId', isEqualTo: strictUid)
        .limit(_notificationsLimit);

    final controller = StreamController<List<AppNotification>>.broadcast();
    _sharedNotificationsSub = query.snapshots().listen(
      (snapshot) {
        _countReads(snapshot.docs.length);
        final list = snapshot.docs.map((doc) {
          final data = doc.data();
          data['id'] = doc.id;
          return AppNotification.fromJson(data);
        }).toList();
        list.sort((a, b) => b.timestamp.compareTo(a.timestamp));
        debugPrint(
          '📱 Firestore notifications snapshot (uid=$strictUid): '
          '${list.length} [reads today≈$_dailyReadCount]',
        );
        _lastNotificationsSnapshot = list;
        controller.add(list);
      },
      onError: (Object error) {
        if (isPermissionDeniedError(error)) {
          _onPermissionDenied('notifications_stream', error);
        } else {
          debugPrint('❌ Shared notifications stream error: $error');
        }
        controller.addError(error);
      },
    );

    _cachedNotificationsUid = strictUid;
    _sharedNotificationsStream = controller.stream;
    return _seedStream(_sharedNotificationsStream!, _lastNotificationsSnapshot);
  }

  /// One-shot notifications fetch for fallback when realtime stream is delayed.
  Future<List<AppNotification>> getNotificationsOnce(String userId) async {
    try {
      final strictUid = _resolveStrictUid(userId);
      if (strictUid.isEmpty) return <AppNotification>[];

      final snapshot = await _notificationsRef
          .where('userId', isEqualTo: strictUid)
          .limit(_notificationsLimit)
          .get();

      _countReads(snapshot.docs.length);
      final list = snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return AppNotification.fromJson(data);
      }).toList()
        ..sort((a, b) => b.timestamp.compareTo(a.timestamp));
      return list;
    } catch (e) {
      debugPrint('❌ Error fetching notifications once: $e');
      return <AppNotification>[];
    }
  }

  /// Add a new notification.
  Future<void> addNotification(AppNotification notification) async {
    try {
      await _notificationsRef.add(notification.toJson());
    } catch (e) {
      debugPrint('❌ Error adding notification: $e');
    }
  }

  /// Mark a notification as read.
  Future<void> markNotificationRead(String id) async {
    try {
      await _notificationsRef.doc(id).update({'isRead': true});
    } catch (e) {
      debugPrint('❌ Error marking notification as read: $e');
    }
  }

  /// Mark all notifications as read for a user.
  Future<void> markAllNotificationsRead(
    String userId, {
    bool throwOnError = false,
  }) async {
    try {
      final strictUid = _resolveStrictUid(userId);
      if (strictUid.isEmpty) return;

      final snapshot = await _notificationsRef
          .where('userId', isEqualTo: strictUid)
          .where('isRead', isEqualTo: false)
          .get();

      final batch = _db.batch();
      for (final doc in snapshot.docs) {
        batch.update(doc.reference, {'isRead': true});
      }
      await batch.commit();
    } catch (e) {
      debugPrint('❌ Error marking all notifications read: $e');
      if (throwOnError) rethrow;
    }
  }

  /// Delete a notification by id.
  Future<void> deleteNotification(
    String id, {
    bool throwOnError = false,
  }) async {
    try {
      await _notificationsRef.doc(id).delete();
    } catch (e) {
      debugPrint('❌ Error deleting notification $id: $e');
      if (throwOnError) rethrow;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // APP UPDATE (OTA)
  // ═══════════════════════════════════════════════════════════════

  /// Get latest app update info from Firestore.
  /// Document path: app_config/latest_version
  /// Fields: version, buildNumber, downloadUrl, changelog, forceUpdate
  Future<Map<String, dynamic>?> getAppUpdateInfo() async {
    try {
      final doc =
          await _db.collection('app_config').doc('latest_version').get();
      if (!doc.exists) return null;
      return doc.data();
    } catch (e) {
      debugPrint('❌ Error fetching app update info: $e');
      return null;
    }
  }

  /// Set app update info in Firestore (called from admin/server).
  /// This is a convenience method — normally you'd set this from the backend.
  Future<void> setAppUpdateInfo({
    required String version,
    required int buildNumber,
    required String downloadUrl,
    String changelog = '',
    bool forceUpdate = false,
  }) async {
    try {
      await _db.collection('app_config').doc('latest_version').set({
        'version': version,
        'buildNumber': buildNumber,
        'downloadUrl': downloadUrl,
        'changelog': changelog,
        'forceUpdate': forceUpdate,
        'updatedAt': FieldValue.serverTimestamp(),
      });
      debugPrint('✅ App update info saved to Firestore');
    } catch (e) {
      debugPrint('❌ Error saving app update info: $e');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // COMPLAINTS
  // ═══════════════════════════════════════════════════════════════

  CollectionReference<Map<String, dynamic>> get _complaintsRef =>
      _db.collection('complaints');

  /// Submit a new complaint via backend API.
  ///
  /// The backend handles Storage upload + Firestore write atomically
  /// using Admin SDK, eliminating client-side permission/upload issues.
  Future<void> submitComplaint({
    required String userId,
    required String violationId,
    required String reason,
    required String description,
    File? evidenceFile,
  }) async {
    // Validate evidence file locally before sending
    if (evidenceFile != null) {
      if (!evidenceFile.existsSync()) {
        throw Exception('File ảnh bằng chứng không tồn tại.');
      }
      if (evidenceFile.lengthSync() == 0) {
        throw Exception('File ảnh bằng chứng rỗng (0 bytes).');
      }
    }

    final api = ApiService();
    final idToken = await api.currentIdToken();
    if (idToken == null || idToken.isEmpty) {
      throw Exception('Chưa đăng nhập. Vui lòng đăng nhập lại.');
    }

    final uri = Uri.parse('${ApiService.baseUrl}/api/app/complaints/submit');

    final request = http.MultipartRequest('POST', uri)
      ..headers['Authorization'] = 'Bearer $idToken'
      ..headers['ngrok-skip-browser-warning'] = 'true'
      ..fields['violation_id'] = violationId
      ..fields['reason'] = reason
      ..fields['description'] = description;

    if (evidenceFile != null) {
      request.files.add(
        await http.MultipartFile.fromPath('evidence', evidenceFile.path),
      );
    }

    debugPrint('📤 Submitting complaint via backend API...');

    final streamed = await request.send().timeout(
      const Duration(seconds: 30),
    );
    final body = await streamed.stream.bytesToString();

    Map<String, dynamic> json;
    try {
      json = jsonDecode(body) as Map<String, dynamic>;
    } catch (_) {
      throw Exception('Server trả về phản hồi không hợp lệ.');
    }

    final status = (json['status'] ?? '').toString();
    final message = (json['message'] ?? '').toString();

    if (streamed.statusCode == 200) {
      if (status == 'ok' || status == 'already_pending') {
        debugPrint('✅ Complaint submitted: $status — $message');
        return;
      }
    }

    // Map HTTP codes to user-friendly errors
    debugPrint('❌ Complaint submit failed [${streamed.statusCode}]: $message');
    switch (streamed.statusCode) {
      case 401:
        throw Exception('Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại.');
      case 403:
        throw Exception('Vi phạm không thuộc về bạn.');
      case 404:
        throw Exception('Không tìm thấy vi phạm.');
      case 503:
        throw Exception('Server chưa sẵn sàng. Vui lòng thử lại sau.');
      default:
        throw Exception(
          message.isNotEmpty ? message : 'Lỗi gửi khiếu nại (${streamed.statusCode}).',
        );
    }
  }

  /// Stream of complaints for a specific user.
  Stream<List<Map<String, dynamic>>> complaintsStream(String userId) {
    return _complaintsRef
        .where('userId', isEqualTo: userId)
        .snapshots()
        .map((snapshot) {
      final list = snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return data;
      }).toList();
      // Sort by createdAt descending if exists
      list.sort((a, b) {
        final tA = a['createdAt'] as Timestamp?;
        final tB = b['createdAt'] as Timestamp?;
        if (tA == null || tB == null) return 0;
        return tB.compareTo(tA);
      });
      return list;
    });
  }

  /// Delete an approved complaint by ID.
  /// Client validates status before calling; Firestore rules enforce owner + approved.
  Future<void> deleteApprovedComplaint(String complaintId) async {
    try {
      await _complaintsRef.doc(complaintId).delete();
      debugPrint('✅ Approved complaint deleted: $complaintId');
    } catch (e) {
      debugPrint('❌ Error deleting complaint: $e');
      rethrow;
    }
  }
}
