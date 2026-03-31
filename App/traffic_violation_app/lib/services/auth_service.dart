import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/services/push_notification_service.dart';

/// Singleton service for Firebase Authentication.
///
/// Handles email/password sign-in, registration, sign-out,
/// and auth state changes.
class AuthService {
  // ── Singleton ──────────────────────────────────────────────────
  static final AuthService _instance = AuthService._internal();
  factory AuthService() => _instance;
  AuthService._internal();

  final fb.FirebaseAuth _auth = fb.FirebaseAuth.instance;
  static const String technicalLoginDomain = 'vnetraffic.vn';

  static bool isTechnicalLoginEmail(String? value) {
    final email = (value ?? '').trim().toLowerCase();
    if (email.isEmpty || !email.contains('@')) return false;
    final parts = email.split('@');
    if (parts.length != 2) return false;
    final local = parts.first;
    final domain = parts.last;
    return domain == technicalLoginDomain &&
        RegExp(r'^[0-9]{12}$').hasMatch(local);
  }

  static String extractIdCardFromLoginEmail(String? value) {
    if (!isTechnicalLoginEmail(value)) return '';
    return (value ?? '').trim().split('@').first;
  }

  // ── Getters ────────────────────────────────────────────────────
  fb.User? get currentUser => _auth.currentUser;
  bool get isLoggedIn => _auth.currentUser != null;
  Stream<fb.User?> get authStateChanges => _auth.authStateChanges();

  String _buildLoginEmailFromIdCard(String idCard) {
    final normalized = idCard.trim();
    return '$normalized@$technicalLoginDomain';
  }

  // ── Sign In ────────────────────────────────────────────────────
  Future<fb.UserCredential> signIn(String email, String password) async {
    try {
      final credential = await _auth
          .signInWithEmailAndPassword(
            email: email.trim(),
            password: password,
          )
          .timeout(const Duration(seconds: 15));
      debugPrint('✅ Signed in: ${credential.user?.email}');
      return credential;
    } on fb.FirebaseAuthException catch (e) {
      throw _mapAuthError(e.code);
    } on TimeoutException {
      throw 'Kết nối quá chậm. Vui lòng thử lại';
    } catch (e) {
      throw 'Đăng nhập thất bại: ${e.toString()}';
    }
  }

  Future<fb.UserCredential> signInByIdCard(
      String idCard, String password) async {
    return signIn(_buildLoginEmailFromIdCard(idCard), password);
  }

  // ── Register ───────────────────────────────────────────────────
  Future<fb.UserCredential> register({
    required String idCard,
    required String password,
    required String fullName,
    String? phone,
  }) async {
    fb.User? createdUser;
    try {
      final normalizedIdCard = idCard.trim();
      final loginEmail = _buildLoginEmailFromIdCard(normalizedIdCard);
      final credential = await _auth.createUserWithEmailAndPassword(
        email: loginEmail,
        password: password,
      );
      createdUser = credential.user;

      // Update display name
      await credential.user?.updateDisplayName(fullName);

      if (createdUser == null) {
        throw 'Không thể tạo tài khoản. Vui lòng thử lại';
      }
      await _createUserProfileViaBackend(
        user: createdUser,
        fullName: fullName,
        phone: phone ?? '',
        idCard: normalizedIdCard,
        profileEmail: '',
      );

      debugPrint('✅ Registered: ${credential.user?.email}');
      return credential;
    } on fb.FirebaseAuthException catch (e) {
      throw _mapAuthError(e.code);
    } on FirebaseException catch (e) {
      await _rollbackFailedRegistration(createdUser);
      throw _mapFirestoreError(e);
    } on TimeoutException {
      await _rollbackFailedRegistration(createdUser);
      throw 'Kết nối quá chậm. Vui lòng thử lại';
    } catch (e) {
      await _rollbackFailedRegistration(createdUser);
      final msg = e.toString();
      if (msg.contains('permission-denied')) {
        throw _mapFirestoreError(FirebaseException(
          plugin: 'cloud_firestore',
          code: 'permission-denied',
          message: msg,
        ));
      }
      throw 'Đăng ký thất bại: $msg';
    }
  }

  // ── Sign Out ───────────────────────────────────────────────────
  Future<void> signOut() async {
    try {
      await ApiService().clearSession();
    } catch (e) {
      debugPrint('⚠️ Could not clear app session on sign out: $e');
    }
    try {
      await PushNotificationService().clearToken();
    } catch (e) {
      debugPrint('⚠️ Could not clear FCM token on sign out: $e');
    }
    // Release shared Firestore listeners so new user gets fresh streams
    FirestoreService().clearSharedStreams();
    // Stop server config sync (requires auth, will fail after signOut)
    ApiService().stopServerConfigAutoSync();
    await _auth.signOut();
    debugPrint('✅ Signed out');
  }

  // ── Reset Password ─────────────────────────────────────────────
  Future<void> resetPassword(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email.trim());
    } on fb.FirebaseAuthException catch (e) {
      throw _mapAuthError(e.code);
    }
  }

  Future<void> resetPasswordByIdCard(String idCard) async {
    await resetPassword(_buildLoginEmailFromIdCard(idCard));
  }

  // ── Error Mapping ──────────────────────────────────────────────
  String _mapAuthError(String code) {
    switch (code) {
      case 'user-not-found':
        return 'Không tìm thấy tài khoản với số CCCD này';
      case 'wrong-password':
        return 'Mật khẩu không đúng';
      case 'invalid-credential':
        return 'Số CCCD hoặc mật khẩu không đúng';
      case 'email-already-in-use':
        return 'Số CCCD này đã được đăng ký';
      case 'weak-password':
        return 'Mật khẩu quá yếu (tối thiểu 6 ký tự)';
      case 'invalid-email':
        return 'Số CCCD không hợp lệ';
      case 'too-many-requests':
        return 'Quá nhiều lần thử. Vui lòng thử lại sau';
      case 'network-request-failed':
        return 'Lỗi kết nối mạng';
      default:
        return 'Lỗi xác thực: $code';
    }
  }

  Future<void> _createUserProfileViaBackend({
    required fb.User user,
    required String fullName,
    required String phone,
    required String idCard,
    required String profileEmail,
  }) async {
    try {
      await _postRegisterProfileRequest(
        user: user,
        fullName: fullName,
        phone: phone,
        idCard: idCard,
        profileEmail: profileEmail,
      );
      return;
    } on FirebaseException catch (e) {
      // Retry once for transient network/IP mismatch issues.
      if (e.code != 'unavailable') rethrow;
      debugPrint(
        '⚠️ register-profile unavailable, refreshing server config and retrying once: ${e.message}',
      );
    }

    await _refreshServerAddressFromFirestore();
    await ApiService().reconnectWithNewUserAndWait(
      timeout: const Duration(seconds: 6),
    );

    await _postRegisterProfileRequest(
      user: user,
      fullName: fullName,
      phone: phone,
      idCard: idCard,
      profileEmail: profileEmail,
    );
  }

  Future<void> _refreshServerAddressFromFirestore() async {
    try {
      final ds = await FirebaseFirestore.instance
          .collection('server')
          .doc('config')
          .get();
      final data = ds.data();
      if (data == null) return;

      final ip = (data['ip'] ?? '').toString().trim();
      if (ip.isEmpty) return;

      final portRaw = data['port'];
      final port =
          portRaw is num ? portRaw.toInt() : int.tryParse('$portRaw') ?? 8000;
      final reachable = await ApiService().pingServerAddress(ip, port: port);
      if (!reachable) {
        debugPrint(
          '⚠️ AuthService skipped server switch (unreachable): $ip:$port',
        );
        return;
      }
      final changed = ApiService().setServerAddress(ip, port: port);
      debugPrint(
        '🔁 AuthService refreshed server config from Firestore: $ip:$port (changed=$changed)',
      );
    } catch (e) {
      debugPrint('⚠️ AuthService failed to refresh server config: $e');
    }
  }

  Future<void> _postRegisterProfileRequest({
    required fb.User user,
    required String fullName,
    required String phone,
    required String idCard,
    required String profileEmail,
  }) async {
    final idToken = await user.getIdToken(true);
    final endpoint = '${ApiService.baseUrl}/api/app/register-profile';
    final uri = Uri.parse(endpoint);

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
              'full_name': fullName,
              'phone': phone,
              'id_card': idCard,
              'email': profileEmail,
            }),
          )
          .timeout(const Duration(seconds: 12));
    } on TimeoutException {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'unavailable',
        message:
            'Không thể kết nối server tạo hồ sơ ($endpoint). Vui lòng kiểm tra IP server hoặc mạng.',
      );
    } on SocketException catch (e) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'unavailable',
        message: 'Không thể kết nối server tạo hồ sơ ($endpoint): ${e.message}',
      );
    } catch (e) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'unavailable',
        message: 'Không thể gọi API tạo hồ sơ tại $endpoint: $e',
      );
    }

    if (response.statusCode == 404) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'unavailable',
        message:
            'Server chưa hỗ trợ endpoint tạo hồ sơ ($endpoint). Vui lòng cập nhật backend hoặc cấu hình lại IP server.',
      );
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
        (status.isEmpty || status == 'ok')) {
      return;
    }

    if (response.statusCode == 503) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'unavailable',
        message: message.isNotEmpty
            ? message
            : 'Backend Firebase Admin chưa sẵn sàng',
      );
    }

    if (response.statusCode == 401 || response.statusCode == 403) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'permission-denied',
        message: message.isNotEmpty ? message : 'Xác thực token thất bại',
      );
    }

    if (response.statusCode >= 500) {
      throw FirebaseException(
        plugin: 'cloud_firestore',
        code: 'unavailable',
        message: message.isNotEmpty
            ? message
            : 'Không thể tạo hồ sơ người dùng qua backend',
      );
    }

    throw FirebaseException(
      plugin: 'cloud_firestore',
      code: 'permission-denied',
      message: message.isNotEmpty ? message : 'Không thể tạo hồ sơ người dùng',
    );
  }

  Future<void> _rollbackFailedRegistration(fb.User? user) async {
    if (user == null) return;
    try {
      await user.delete();
      debugPrint('↩️ Rolled back auth user after failed profile creation');
    } catch (e) {
      debugPrint('⚠️ Could not delete user during rollback: $e');
    }
    try {
      await _auth.signOut();
    } catch (_) {}
  }

  String _mapFirestoreError(FirebaseException e) {
    final detail = (e.message ?? '').trim();
    switch (e.code) {
      case 'permission-denied':
        if (detail.isNotEmpty) return detail;
        return 'Không đủ quyền tạo hồ sơ người dùng trên hệ thống. Vui lòng liên hệ quản trị viên để cấp quyền Firestore.';
      case 'unavailable':
        if (detail.isNotEmpty) return detail;
        return 'Dịch vụ dữ liệu tạm thời không khả dụng. Vui lòng thử lại sau.';
      default:
        return 'Lỗi dữ liệu (${e.code}): ${detail.isEmpty ? 'Không xác định' : detail}';
    }
  }
}
