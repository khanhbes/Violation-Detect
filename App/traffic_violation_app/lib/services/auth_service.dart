import 'dart:async';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';

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
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  // ── Getters ────────────────────────────────────────────────────
  fb.User? get currentUser => _auth.currentUser;
  bool get isLoggedIn => _auth.currentUser != null;
  Stream<fb.User?> get authStateChanges => _auth.authStateChanges();

  // ── Sign In ────────────────────────────────────────────────────
  Future<fb.UserCredential> signIn(String email, String password) async {
    try {
      final credential = await _auth.signInWithEmailAndPassword(
        email: email.trim(),
        password: password,
      ).timeout(const Duration(seconds: 15));
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

  // ── Register ───────────────────────────────────────────────────
  Future<fb.UserCredential> register({
    required String email,
    required String password,
    required String fullName,
    String? phone,
  }) async {
    try {
      final credential = await _auth.createUserWithEmailAndPassword(
        email: email.trim(),
        password: password,
      );

      // Update display name
      await credential.user?.updateDisplayName(fullName);

      // Create user profile in Firestore
      // Extract CCCD from email (format: cccd@vnetraffic.vn)
      final idCard = email.trim().split('@').first;
      await _firestore.collection('users').doc(credential.user!.uid).set({
        'fullName': fullName,
        'email': email.trim(),
        'phone': phone ?? '',
        'avatar': null,
        'idCard': idCard,
        'address': '',
        'createdAt': FieldValue.serverTimestamp(),
      });

      debugPrint('✅ Registered: ${credential.user?.email}');
      return credential;
    } on fb.FirebaseAuthException catch (e) {
      throw _mapAuthError(e.code);
    }
  }

  // ── Sign Out ───────────────────────────────────────────────────
  Future<void> signOut() async {
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

  // ── Error Mapping ──────────────────────────────────────────────
  String _mapAuthError(String code) {
    switch (code) {
      case 'user-not-found':
        return 'Không tìm thấy tài khoản với email này';
      case 'wrong-password':
        return 'Mật khẩu không đúng';
      case 'invalid-credential':
        return 'Email hoặc mật khẩu không đúng';
      case 'email-already-in-use':
        return 'Email đã được sử dụng';
      case 'weak-password':
        return 'Mật khẩu quá yếu (tối thiểu 6 ký tự)';
      case 'invalid-email':
        return 'Email không hợp lệ';
      case 'too-many-requests':
        return 'Quá nhiều lần thử. Vui lòng thử lại sau';
      case 'network-request-failed':
        return 'Lỗi kết nối mạng';
      default:
        return 'Lỗi xác thực: $code';
    }
  }
}
