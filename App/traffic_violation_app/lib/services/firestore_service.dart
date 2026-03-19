import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/foundation.dart';
import 'package:traffic_violation_app/models/notification.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/models/user.dart' as app;
import 'package:traffic_violation_app/models/vehicle.dart';

/// Singleton service for Firestore CRUD operations.
///
/// Provides real-time streams and one-shot reads for
/// violations, users, and vehicles.
class FirestoreService {
  // ── Singleton ──────────────────────────────────────────────────
  static final FirestoreService _instance = FirestoreService._internal();
  factory FirestoreService() => _instance;
  FirestoreService._internal();

  final FirebaseFirestore _db = FirebaseFirestore.instance;

  // ═══════════════════════════════════════════════════════════════
  // VIOLATIONS
  // ═══════════════════════════════════════════════════════════════

  CollectionReference<Map<String, dynamic>> get _violationsRef =>
      _db.collection('violations');

  /// Real-time stream of violations.
  /// Can filter by licensePlate or userId.
  Stream<List<Violation>> violationsStream(
      {String? licensePlate, String? userId}) {
    Query<Map<String, dynamic>> query = _violationsRef;

    if (userId != null && userId.isNotEmpty) {
      query = query.where('userId', isEqualTo: userId);
    } else if (licensePlate != null && licensePlate.isNotEmpty) {
      query = query.where('licensePlate', isEqualTo: licensePlate);
    }

    return query.snapshots().map((snapshot) {
      debugPrint('📱 Firestore snapshot: ${snapshot.docs.length} violations');
      final violations = snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return Violation.fromJson(data);
      }).toList();

      // Sort client-side by timestamp descending to avoid composite index requirements
      violations.sort((a, b) => b.timestamp.compareTo(a.timestamp));
      return violations;
    });
  }

  /// One-shot fetch of all violations.
  Future<List<Violation>> getViolations(
      {String? licensePlate, String? userId}) async {
    try {
      Query<Map<String, dynamic>> query = _violationsRef;

      if (userId != null && userId.isNotEmpty) {
        query = query.where('userId', isEqualTo: userId);
      } else if (licensePlate != null && licensePlate.isNotEmpty) {
        query = query.where('licensePlate', isEqualTo: licensePlate);
      }

      final snapshot = await query.get();
      debugPrint('📱 Firestore fetch: ${snapshot.docs.length} violations');
      final violations = snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return Violation.fromJson(data);
      }).toList();

      // Sort client-side by timestamp descending
      violations.sort((a, b) => b.timestamp.compareTo(a.timestamp));
      return violations;
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

  Future<void> _setViolationComplaintPending(String violationId) async {
    final violationRef = await _resolveViolationDocRef(violationId);
    if (violationRef == null) {
      debugPrint('⚠️ Violation not found to lock complaint: $violationId');
      return;
    }

    await violationRef.set({
      'status': 'complaint_pending',
      'complaintStatus': 'pending',
      'paymentLocked': true,
      'complaintLocked': true,
      'complaintSubmittedAt': FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
    }, SetOptions(merge: true));
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
    try {
      final payload = <String, dynamic>{
        ...requestData,
        'userId': uid,
        'status': 'pending',
        'updatedAt': FieldValue.serverTimestamp(),
      };

      payload['createdAt'] = FieldValue.serverTimestamp();

      await _db
          .collection('profile_update_requests')
          .doc(uid)
          .set(payload, SetOptions(merge: true));
      debugPrint('✅ Profile update request sent to Firestore');
    } catch (e) {
      debugPrint('❌ Error sending profile update request: $e');
      rethrow;
    }
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

  /// Create or update user profile (called after login if profile doesn't exist).
  Future<void> createOrUpdateUserProfile(
      String uid, Map<String, dynamic> data) async {
    try {
      await _usersRef.doc(uid).set(data, SetOptions(merge: true));
      debugPrint('✅ User profile created/updated in Firestore');
    } catch (e) {
      debugPrint('❌ Error creating user profile: $e');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // NOTIFICATIONS
  // ═══════════════════════════════════════════════════════════════

  CollectionReference<Map<String, dynamic>> get _notificationsRef =>
      _db.collection('notifications');

  /// Stream of notifications for a specific user, ordered by newest first.
  Stream<List<AppNotification>> notificationsStream(String userId) {
    return _notificationsRef
        .where('userId', isEqualTo: userId)
        .snapshots()
        .map((snapshot) {
      final list = snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return AppNotification.fromJson(data);
      }).toList();
      list.sort((a, b) => b.timestamp.compareTo(a.timestamp));
      return list;
    });
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
  Future<void> markAllNotificationsRead(String userId) async {
    try {
      final snapshot = await _notificationsRef
          .where('userId', isEqualTo: userId)
          .where('isRead', isEqualTo: false)
          .get();

      final batch = _db.batch();
      for (final doc in snapshot.docs) {
        batch.update(doc.reference, {'isRead': true});
      }
      await batch.commit();
    } catch (e) {
      debugPrint('❌ Error marking all notifications read: $e');
    }
  }

  /// Delete a notification by id.
  Future<void> deleteNotification(String id) async {
    try {
      await _notificationsRef.doc(id).delete();
    } catch (e) {
      debugPrint('❌ Error deleting notification $id: $e');
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

  /// Submit a new complaint.
  Future<void> submitComplaint({
    required String userId,
    required String violationId,
    required String reason,
    required String description,
    File? evidenceFile,
  }) async {
    try {
      String evidenceUrl = '';
      String evidencePath = '';

      // Upload evidence image if provided
      if (evidenceFile != null) {
        try {
          final storage = FirebaseStorage.instance;
          evidencePath =
              'complaints/$userId/${DateTime.now().millisecondsSinceEpoch}.jpg';
          final ref = storage.ref(evidencePath);
          final uploadTask = await ref.putFile(evidenceFile);
          evidenceUrl = await uploadTask.ref.getDownloadURL();
        } catch (storageErr) {
          debugPrint('❌ Evidence upload failed: $storageErr');
          throw Exception(
            'Không thể tải ảnh bằng chứng. Vui lòng kiểm tra mạng và thử lại.',
          );
        }
      }

      await _complaintsRef.add({
        'userId': userId,
        'violationId': violationId,
        'reason': reason,
        'description': description,
        'status': 'pending',
        'evidenceUrl': evidenceUrl,
        'evidencePath': evidencePath,
        'evidence': {
          'downloadUrl': evidenceUrl,
          'path': evidencePath,
        },
        'adminNote': '',
        'createdAt': FieldValue.serverTimestamp(),
      });

      await _setViolationComplaintPending(violationId);
      debugPrint('✅ Complaint submitted successfully');
    } catch (e) {
      debugPrint('❌ Error submitting complaint: $e');
      throw e;
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
}
