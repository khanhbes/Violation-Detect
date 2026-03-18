import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:traffic_violation_app/services/firestore_service.dart';


/// Global app settings singleton — manages theme, locale, notifications, user profile.
/// Now persists all data to Firebase Firestore.
class AppSettings extends ChangeNotifier {
  // ── Singleton ──────────────────────────────────────────────────
  static final AppSettings _instance = AppSettings._internal();
  factory AppSettings() => _instance;
  AppSettings._internal();

  final FirestoreService _firestore = FirestoreService();

  // ── Auth UID (set after login) ─────────────────────────────────
  String? _uid;
  String? get uid => _uid;

  // ── Theme ──────────────────────────────────────────────────────
  ThemeMode _themeMode = ThemeMode.light;
  ThemeMode get themeMode => _themeMode;
  bool get isDarkMode => _themeMode == ThemeMode.dark;

  void setThemeMode(ThemeMode mode) {
    _themeMode = mode;
    notifyListeners();
    _saveSettingsToFirestore();
  }

  void toggleDarkMode() {
    _themeMode = _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    notifyListeners();
    _saveSettingsToFirestore();
  }

  // ── Locale ─────────────────────────────────────────────────────
  Locale _locale = const Locale('vi', 'VN');
  Locale get locale => _locale;
  bool get isVietnamese => _locale.languageCode == 'vi';

  void setLocale(Locale locale) {
    _locale = locale;
    notifyListeners();
    _saveSettingsToFirestore();
  }

  void toggleLanguage() {
    _locale = isVietnamese
        ? const Locale('en', 'US')
        : const Locale('vi', 'VN');
    notifyListeners();
    _saveSettingsToFirestore();
  }

  // ── Notification Badge Count ───────────────────────────────────
  int _unreadNotifications = 0; // Will be set from Firestore realtime notifications
  int get unreadNotifications => _unreadNotifications;

  void addNotification() {
    _unreadNotifications++;
    notifyListeners();
  }

  void clearNotifications() {
    _unreadNotifications = 0;
    notifyListeners();
  }

  void setNotificationCount(int count) {
    _unreadNotifications = count;
    notifyListeners();
  }

  void setUserPoints(int points) {
    if (_userPoints != points) {
      _userPoints = points;
      notifyListeners();
    }
  }

  // ── Notification Toggle (on/off) ───────────────────────────────
  bool _notificationsEnabled = true;
  bool get notificationsEnabled => _notificationsEnabled;

  void setNotificationsEnabled(bool enabled) {
    _notificationsEnabled = enabled;
    notifyListeners();
    _saveSettingsToFirestore();
  }

  void toggleNotifications() {
    _notificationsEnabled = !_notificationsEnabled;
    notifyListeners();
    _saveSettingsToFirestore();
  }

  // ── User Profile Editable Fields ───────────────────────────────
  String _userName = '';
  String _userEmail = '';
  String _userPhone = '';
  String _userAddress = '';
  String _userAvatar = '';
  String _userIdCard = '';
  String _userIdCardIssueDate = '';
  String _userOccupation = '';
  String _userDateOfBirth = '';
  int _userPoints = 12;

  String get userName => _userName;
  String get userEmail => _userEmail;
  String get userPhone => _userPhone;
  String get userAddress => _userAddress;
  String get userAvatar => _userAvatar;
  String get userIdCardIssueDate => _userIdCardIssueDate;
  String get userOccupation => _userOccupation;
  String get userDateOfBirth => _userDateOfBirth;
  int get userPoints => _userPoints;
  String get userIdCard {
    if (_userIdCard.isNotEmpty) return _userIdCard;
    if (_userEmail.isNotEmpty && _userEmail.contains('@')) {
      final prefix = _userEmail.split('@').first;
      if (RegExp(r'^[0-9]+$').hasMatch(prefix)) return prefix;
    }
    // Fallback: lay truc tiep tu email dang nhap cua Firebase (la CCCD)
    final authUser = fb.FirebaseAuth.instance.currentUser;
    if (authUser != null && authUser.email != null) {
      final prefix = authUser.email!.split('@').first;
      if (RegExp(r'^[0-9]+$').hasMatch(prefix)) return prefix;
    }
    return '';
  }

  bool _profileInitialized = false;
  bool get profileInitialized => _profileInitialized;

  /// Initialize profile from MockData (legacy fallback, only if not loaded from Firestore)
  void initProfile({
    required String name,
    required String email,
    required String phone,
    required String address,
    required String avatar,
    required String idCard,
    String idCardIssueDate = '',
    String occupation = '',
    String dateOfBirth = '',
  }) {
    if (!_profileInitialized) {
      _userName = name;
      _userEmail = email;
      _userPhone = phone;
      _userAddress = address;
      _userAvatar = avatar;
      _userIdCard = idCard;
      _userIdCardIssueDate = idCardIssueDate;
      _userOccupation = occupation;
      _userDateOfBirth = dateOfBirth;
      _profileInitialized = true;
    }
  }

  void updateProfile({
    String? name,
    String? email,
    String? phone,
    String? address,
    String? avatar,
    String? idCard,
    String? idCardIssueDate,
    String? occupation,
    String? dateOfBirth,
  }) {
    if (name != null) _userName = name;
    if (email != null) _userEmail = email;
    if (phone != null) _userPhone = phone;
    if (address != null) _userAddress = address;
    if (avatar != null) _userAvatar = avatar;
    if (idCard != null) _userIdCard = idCard;
    if (idCardIssueDate != null) _userIdCardIssueDate = idCardIssueDate;
    if (occupation != null) _userOccupation = occupation;
    if (dateOfBirth != null) _userDateOfBirth = dateOfBirth;
    notifyListeners();
    _saveProfileToFirestore();
  }

  void updateAvatar(String url) {
    _userAvatar = url;
    notifyListeners();
    _saveProfileToFirestore();
  }

  // ═══════════════════════════════════════════════════════════════
  //  FIRESTORE PERSISTENCE
  // ═══════════════════════════════════════════════════════════════

  /// Call this after successful login/register to load all user data from Firestore.
  Future<void> loadFromFirestore(String uid) async {
    _uid = uid;

    try {
      // 1. Load user profile
      final user = await _firestore.getUserProfile(uid);
      if (user != null) {
        _userName = user.fullName;
        _userEmail = user.email;
        _userPhone = user.phone;
        _userAddress = user.address;
        _userAvatar = user.avatar ?? '';
        _userIdCard = user.idCard;
        _userIdCardIssueDate = user.idCardIssueDate ?? '';
        _userOccupation = user.occupation ?? '';
        _userDateOfBirth = user.dateOfBirth ?? '';
        _userPoints = user.points;
        _profileInitialized = true;
        debugPrint('✅ Profile loaded from Firestore: $_userName');
      } else {
        // Profile not in Firestore — try to get from Firebase Auth
        debugPrint('⚠️ No profile found in Firestore for $uid, creating from Auth...');
        try {
          final auth = _getFirebaseAuthUser();
          if (auth != null) {
            _userName = auth['displayName'] ?? '';
            _userEmail = auth['email'] ?? '';
            _userIdCard = _userEmail.split('@').first;
            _profileInitialized = true;

            // Auto-create profile in Firestore so next time it loads properly
            await _firestore.createOrUpdateUserProfile(uid, {
              'fullName': _userName,
              'email': _userEmail,
              'phone': _userPhone,
              'address': _userAddress,
              'avatar': _userAvatar,
              'idCard': _userIdCard,
              'idCardIssueDate': _userIdCardIssueDate,
              'occupation': _userOccupation,
              'dateOfBirth': _userDateOfBirth,
            });
            debugPrint('✅ Profile auto-created in Firestore from Auth data');
          }
        } catch (authErr) {
          debugPrint('❌ Fallback auth data also failed: $authErr');
        }
      }

      // 2. Load user settings
      final settings = await _firestore.getUserSettings(uid);
      if (settings != null) {
        _themeMode = settings['isDarkMode'] == true
            ? ThemeMode.dark
            : ThemeMode.light;
        final lang = settings['language'] as String?;
        _locale = (lang == 'en')
            ? const Locale('en', 'US')
            : const Locale('vi', 'VN');
        _notificationsEnabled = settings['notificationsEnabled'] ?? true;
        debugPrint('✅ Settings loaded from Firestore');
      }
    } catch (e) {
      debugPrint('❌ Error loading from Firestore: $e');
    }

    notifyListeners();
  }

  /// Helper to get Firebase Auth current user info
  Map<String, String?>? _getFirebaseAuthUser() {
    try {
      final user = fb.FirebaseAuth.instance.currentUser;
      if (user == null) return null;
      return {
        'displayName': user.displayName,
        'email': user.email,
      };
    } catch (_) {
      return null;
    }
  }

  /// Save user profile fields to Firestore.
  Future<void> _saveProfileToFirestore() async {
    if (_uid == null) return;

    try {
      await _firestore.updateUserProfile(_uid!, {
        'fullName': _userName,
        'email': _userEmail,
        'phone': _userPhone,
        'address': _userAddress,
        'avatar': _userAvatar,
        'idCard': _userIdCard,
        'idCardIssueDate': _userIdCardIssueDate,
        'occupation': _userOccupation,
        'dateOfBirth': _userDateOfBirth,
      });
      debugPrint('✅ Profile saved to Firestore');
    } catch (e) {
      debugPrint('❌ Error saving profile: $e');
    }
  }

  /// Save settings (theme, language, notifications) to Firestore.
  Future<void> _saveSettingsToFirestore() async {
    if (_uid == null) return;

    try {
      await _firestore.saveUserSettings(_uid!, {
        'isDarkMode': _themeMode == ThemeMode.dark,
        'language': _locale.languageCode,
        'notificationsEnabled': _notificationsEnabled,
      });
    } catch (e) {
      debugPrint('❌ Error saving settings: $e');
    }
  }

  /// Reset all data on logout.
  void resetOnLogout() {
    _uid = null;
    _profileInitialized = false;
    _userName = '';
    _userEmail = '';
    _userPhone = '';
    _userAddress = '';
    _userAvatar = '';
    _userIdCard = '';
    _themeMode = ThemeMode.light;
    _locale = const Locale('vi', 'VN');
    _notificationsEnabled = true;
    _unreadNotifications = 0;
    notifyListeners();
  }

  // ── Localization Strings ───────────────────────────────────────
  // Quick i18n helper — returns Vietnamese or English strings
  String tr(String viText, String enText) {
    return isVietnamese ? viText : enText;
  }
}
