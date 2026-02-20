import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
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

  /// Real-time stream of all violations.
  /// Tries orderBy('createdAt') first, falls back to no ordering.
  Stream<List<Violation>> violationsStream({String? licensePlate}) {
    Query<Map<String, dynamic>> query = _violationsRef;

    // Try ordering by createdAt (server timestamp), fallback to timestamp
    try {
      query = query.orderBy('createdAt', descending: true);
    } catch (e) {
      debugPrint('⚠️ orderBy createdAt failed, using default order: $e');
    }

    if (licensePlate != null) {
      query = query.where('licensePlate', isEqualTo: licensePlate);
    }

    return query.snapshots().map((snapshot) {
      debugPrint('📱 Firestore snapshot: ${snapshot.docs.length} violations');
      return snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return Violation.fromJson(data);
      }).toList();
    }).handleError((error) {
      debugPrint('❌ Firestore stream error: $error');
      debugPrint('📱 Retrying without orderBy...');
      // On error, return a fallback stream without ordering
      return _violationsRefFallbackStream(licensePlate: licensePlate);
    });
  }

  /// Fallback stream without orderBy (in case index doesn't exist)
  Stream<List<Violation>> _violationsRefFallbackStream({String? licensePlate}) {
    Query<Map<String, dynamic>> query = _violationsRef;

    if (licensePlate != null) {
      query = query.where('licensePlate', isEqualTo: licensePlate);
    }

    return query.snapshots().map((snapshot) {
      debugPrint('📱 Firestore fallback: ${snapshot.docs.length} violations');
      final violations = snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return Violation.fromJson(data);
      }).toList();

      // Sort client-side by timestamp descending
      violations.sort((a, b) => b.timestamp.compareTo(a.timestamp));
      return violations;
    });
  }

  /// One-shot fetch of all violations.
  Future<List<Violation>> getViolations({String? licensePlate}) async {
    try {
      // Try with orderBy first
      Query<Map<String, dynamic>> query = _violationsRef;

      try {
        query = query.orderBy('createdAt', descending: true);
      } catch (_) {
        // If orderBy fails at build time, skip it
      }

      if (licensePlate != null) {
        query = query.where('licensePlate', isEqualTo: licensePlate);
      }

      final snapshot = await query.get();
      debugPrint('📱 Firestore fetch: ${snapshot.docs.length} violations');
      return snapshot.docs.map((doc) {
        final data = doc.data();
        data['id'] = doc.id;
        return Violation.fromJson(data);
      }).toList();
    } catch (e) {
      debugPrint('❌ Error fetching violations (trying fallback): $e');
      // Fallback: fetch without ordering
      try {
        Query<Map<String, dynamic>> query = _violationsRef;
        if (licensePlate != null) {
          query = query.where('licensePlate', isEqualTo: licensePlate);
        }
        final snapshot = await query.get();
        final violations = snapshot.docs.map((doc) {
          final data = doc.data();
          data['id'] = doc.id;
          return Violation.fromJson(data);
        }).toList();
        violations.sort((a, b) => b.timestamp.compareTo(a.timestamp));
        return violations;
      } catch (e2) {
        debugPrint('❌ Fallback also failed: $e2');
        return [];
      }
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

  /// Update user profile fields.
  Future<void> updateUserProfile(String uid, Map<String, dynamic> data) async {
    await _usersRef.doc(uid).update(data);
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
      final snapshot = await _vehiclesRef
          .where('ownerId', isEqualTo: ownerId)
          .get();
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
}

