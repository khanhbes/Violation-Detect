import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/foundation.dart';

/// Singleton service for Firebase Storage.
///
/// Handles retrieval of violation evidence image URLs.
class StorageService {
  // ── Singleton ──────────────────────────────────────────────────
  static final StorageService _instance = StorageService._internal();
  factory StorageService() => _instance;
  StorageService._internal();

  final FirebaseStorage _storage = FirebaseStorage.instance;

  /// Get download URL for a violation snapshot image.
  ///
  /// [imagePath] is the path in Storage, e.g. "violations/helmet/snapshot_001.jpg"
  Future<String?> getImageUrl(String imagePath) async {
    try {
      final ref = _storage.ref().child(imagePath);
      return await ref.getDownloadURL();
    } catch (e) {
      debugPrint('❌ Error getting image URL for $imagePath: $e');
      return null;
    }
  }

  /// Get download URL directly from a full gs:// or https:// URL.
  ///
  /// If it's already a valid https URL, return as-is.
  /// If it's a gs:// URL or a Storage path, resolve it.
  Future<String> resolveImageUrl(String urlOrPath) async {
    // Already an HTTPS URL (from Firestore record)
    if (urlOrPath.startsWith('http://') || urlOrPath.startsWith('https://')) {
      return urlOrPath;
    }

    // gs:// URL
    if (urlOrPath.startsWith('gs://')) {
      try {
        final ref = _storage.refFromURL(urlOrPath);
        return await ref.getDownloadURL();
      } catch (e) {
        debugPrint('❌ Error resolving gs:// URL: $e');
        return urlOrPath;
      }
    }

    // Assume it's a Storage path
    final url = await getImageUrl(urlOrPath);
    return url ?? urlOrPath;
  }
}
