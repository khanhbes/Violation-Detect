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
    _themeMode =
        _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
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
    _locale =
        isVietnamese ? const Locale('en', 'US') : const Locale('vi', 'VN');
    notifyListeners();
    _saveSettingsToFirestore();
  }

  // ── Notification Badge Count ───────────────────────────────────
  int _unreadNotifications =
      0; // Will be set from Firestore realtime notifications
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
    final safe = points.clamp(0, 12).toInt();
    if (_userPoints != safe ||
        _motoLicensePoints != safe ||
        _carLicensePoints != safe) {
      _userPoints = safe;
      _motoLicensePoints = safe;
      _carLicensePoints = safe;
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
  String _userIdCardExpiryDate = '';
  String _userGender = '';
  String _userNationality = '';
  String _userPlaceOfOrigin = '';
  String _userLicenseIssuedBy = '';
  String _userOccupation = '';
  String _userDateOfBirth = '';
  List<Map<String, String>> _driverLicenses = [];
  int _motoLicensePoints = 12;
  int _carLicensePoints = 12;
  int _userPoints = 12;

  String get userName => _userName;
  String get userEmail => _userEmail;
  String get userPhone => _userPhone;
  String get userAddress => _userAddress;
  String get userAvatar => _userAvatar;
  String get userIdCardIssueDate => _userIdCardIssueDate;
  String get userIdCardExpiryDate => _userIdCardExpiryDate;
  String get userGender => _userGender;
  String get userNationality => _userNationality;
  String get userPlaceOfOrigin => _userPlaceOfOrigin;
  String get userLicenseIssuedBy => _userLicenseIssuedBy;
  String get userOccupation => _userOccupation;
  String get userDateOfBirth => _userDateOfBirth;
  List<Map<String, String>> get driverLicenses =>
      List<Map<String, String>>.unmodifiable(_driverLicenses);
  int get motoLicensePoints => _motoLicensePoints;
  int get carLicensePoints => _carLicensePoints;
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

  List<Map<String, String>> _normalizeDriverLicenses(dynamic raw) {
    if (raw is! List) return <Map<String, String>>[];
    return raw.whereType<Map>().map((item) {
      String read(String key) => item[key]?.toString().trim() ?? '';
      return <String, String>{
        'class': read('class'),
        'vehicleType': read('vehicleType'),
        'issueDate': read('issueDate'),
        'expiryDate': read('expiryDate'),
        'licenseNumber': read('licenseNumber'),
        'issuedBy': read('issuedBy'),
      };
    }).toList();
  }

  bool _isMotoLicense(Map<String, String> l) {
    final vehicleType = (l['vehicleType'] ?? '').toLowerCase();
    final cls = (l['class'] ?? '').toUpperCase();
    return vehicleType.contains('xe máy') ||
        vehicleType.contains('motor') ||
        cls.startsWith('A');
  }

  bool _isCarLicense(Map<String, String> l) {
    final vehicleType = (l['vehicleType'] ?? '').toLowerCase();
    final cls = (l['class'] ?? '').toUpperCase();
    return vehicleType.contains('ô tô') ||
        vehicleType.contains('o to') ||
        vehicleType.contains('car') ||
        cls.startsWith('B') ||
        cls.startsWith('C') ||
        cls.startsWith('D') ||
        cls.startsWith('E') ||
        cls.startsWith('F');
  }

  bool get _hasMotoLicense {
    if (_driverLicenses.any(_isMotoLicense)) return true;
    return false;
  }

  bool get _hasCarLicense {
    if (_driverLicenses.any(_isCarLicense)) return true;
    return false;
  }

  int _readLicensePoints(
    Map<String, dynamic> data, {
    required String primaryKey,
    required String legacyKey,
    required int fallback,
  }) {
    final rawPrimary = data[primaryKey];
    if (rawPrimary is num) return rawPrimary.toInt().clamp(0, 12).toInt();
    final rawLegacy = data[legacyKey];
    if (rawLegacy is num) return rawLegacy.toInt().clamp(0, 12).toInt();
    return fallback.clamp(0, 12).toInt();
  }

  int _resolveAggregatePoints() {
    final hasMoto = _hasMotoLicense;
    final hasCar = _hasCarLicense;
    if (hasMoto && hasCar) {
      return (_motoLicensePoints < _carLicensePoints
              ? _motoLicensePoints
              : _carLicensePoints)
          .clamp(0, 12)
          .toInt();
    }
    if (hasMoto) return _motoLicensePoints.clamp(0, 12).toInt();
    if (hasCar) return _carLicensePoints.clamp(0, 12).toInt();
    return _userPoints.clamp(0, 12).toInt();
  }

  Map<String, dynamic> _legacyLicenseFields(
      List<Map<String, String>> licenses) {
    Map<String, String>? car;
    Map<String, String>? moto;

    for (final l in licenses) {
      final vehicleType = (l['vehicleType'] ?? '').toLowerCase();
      final cls = (l['class'] ?? '').toUpperCase();
      if (car == null &&
          (vehicleType.contains('ô tô') ||
              vehicleType.contains('car') ||
              cls.startsWith('B') ||
              cls.startsWith('C') ||
              cls.startsWith('D') ||
              cls.startsWith('E') ||
              cls == 'FB2')) {
        car = l;
      }
      if (moto == null &&
          (vehicleType.contains('xe máy') ||
              vehicleType.contains('motor') ||
              cls.startsWith('A'))) {
        moto = l;
      }
    }

    final first = licenses.isNotEmpty ? licenses.first : null;
    return {
      'licenseNumber': first?['licenseNumber'] ?? '',
      'licenseIssueDate': first?['issueDate'] ?? '',
      'licenseExpiryDate': first?['expiryDate'] ?? '',
      'licenseIssuedBy': first?['issuedBy'] ?? _userLicenseIssuedBy,
      'carLicenseClass': car?['class'] ?? '',
      'motoLicenseClass': moto?['class'] ?? '',
    };
  }

  List<Map<String, String>> _licensesFromLegacy({
    String? licenseNumber,
    String? carLicenseClass,
    String? motoLicenseClass,
    String? licenseIssueDate,
    String? licenseExpiryDate,
    String? licenseIssuedBy,
  }) {
    final result = <Map<String, String>>[];
    final normalizedNumber = (licenseNumber ?? '').trim();
    final normalizedIssue = (licenseIssueDate ?? '').trim();
    final normalizedExpiry = (licenseExpiryDate ?? '').trim();
    final normalizedIssuedBy = (licenseIssuedBy ?? '').trim();

    if ((carLicenseClass ?? '').trim().isNotEmpty ||
        normalizedNumber.isNotEmpty) {
      result.add({
        'class': (carLicenseClass ?? '').trim(),
        'vehicleType': 'Ô tô',
        'issueDate': normalizedIssue,
        'expiryDate': normalizedExpiry,
        'licenseNumber': normalizedNumber,
        'issuedBy': normalizedIssuedBy,
      });
    }

    if ((motoLicenseClass ?? '').trim().isNotEmpty) {
      result.add({
        'class': (motoLicenseClass ?? '').trim(),
        'vehicleType': 'Xe máy',
        'issueDate': normalizedIssue,
        'expiryDate': normalizedExpiry,
        'licenseNumber': normalizedNumber,
        'issuedBy': normalizedIssuedBy,
      });
    }

    return result;
  }

  /// Initialize profile from MockData (legacy fallback, only if not loaded from Firestore)
  void initProfile({
    required String name,
    required String email,
    required String phone,
    required String address,
    required String avatar,
    required String idCard,
    String idCardIssueDate = '',
    String idCardExpiryDate = '',
    String gender = '',
    String nationality = '',
    String placeOfOrigin = '',
    String licenseIssuedBy = '',
    String occupation = '',
    String dateOfBirth = '',
    List<Map<String, String>> driverLicenses = const [],
  }) {
    if (!_profileInitialized) {
      _userName = name;
      _userEmail = email;
      _userPhone = phone;
      _userAddress = address;
      _userAvatar = avatar;
      _userIdCard = idCard;
      _userIdCardIssueDate = idCardIssueDate;
      _userIdCardExpiryDate = idCardExpiryDate;
      _userGender = gender;
      _userNationality = nationality;
      _userPlaceOfOrigin = placeOfOrigin;
      _userLicenseIssuedBy = licenseIssuedBy;
      _userOccupation = occupation;
      _userDateOfBirth = dateOfBirth;
      _driverLicenses = List<Map<String, String>>.from(driverLicenses);
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
    String? idCardExpiryDate,
    String? gender,
    String? nationality,
    String? placeOfOrigin,
    String? licenseIssuedBy,
    String? occupation,
    String? dateOfBirth,
    List<Map<String, String>>? driverLicenses,
  }) {
    if (name != null) _userName = name;
    if (email != null) _userEmail = email;
    if (phone != null) _userPhone = phone;
    if (address != null) _userAddress = address;
    if (avatar != null) _userAvatar = avatar;
    if (idCard != null) _userIdCard = idCard;
    if (idCardIssueDate != null) _userIdCardIssueDate = idCardIssueDate;
    if (idCardExpiryDate != null) _userIdCardExpiryDate = idCardExpiryDate;
    if (gender != null) _userGender = gender;
    if (nationality != null) _userNationality = nationality;
    if (placeOfOrigin != null) _userPlaceOfOrigin = placeOfOrigin;
    if (licenseIssuedBy != null) _userLicenseIssuedBy = licenseIssuedBy;
    if (occupation != null) _userOccupation = occupation;
    if (dateOfBirth != null) _userDateOfBirth = dateOfBirth;
    if (driverLicenses != null) {
      _driverLicenses = List<Map<String, String>>.from(driverLicenses);
    }
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
        _userIdCardExpiryDate = user.idCardExpiryDate ?? '';
        _userGender = user.gender ?? '';
        _userNationality = user.nationality ?? '';
        _userPlaceOfOrigin = user.placeOfOrigin ?? '';
        _userLicenseIssuedBy = user.licenseIssuedBy ?? '';
        _userOccupation = user.occupation ?? '';
        _userDateOfBirth = user.dateOfBirth ?? '';
        _driverLicenses = List<Map<String, String>>.from(user.driverLicenses);
        if (_driverLicenses.isEmpty) {
          _driverLicenses = _licensesFromLegacy(
            licenseNumber: user.licenseNumber,
            carLicenseClass: user.carLicenseClass,
            motoLicenseClass: user.motoLicenseClass,
            licenseIssueDate: user.licenseIssueDate,
            licenseExpiryDate: user.licenseExpiryDate,
            licenseIssuedBy: user.licenseIssuedBy,
          );
        }
        _motoLicensePoints = user.motoPoints.clamp(0, 12).toInt();
        _carLicensePoints = user.carPoints.clamp(0, 12).toInt();
        _userPoints = _resolveAggregatePoints();
        _profileInitialized = true;
        debugPrint('✅ Profile loaded from Firestore: $_userName');
      } else {
        // Profile not in Firestore — try to get from Firebase Auth
        debugPrint(
            '⚠️ No profile found in Firestore for $uid, creating from Auth...');
        try {
          final auth = _getFirebaseAuthUser();
          if (auth != null) {
            _userName = auth['displayName'] ?? '';
            _userEmail = auth['email'] ?? '';
            _userIdCard = _userEmail.split('@').first;
            _profileInitialized = true;
            final bootstrapLicenses = _driverLicenses.isNotEmpty
                ? _driverLicenses
                : _licensesFromLegacy(
                    licenseNumber: '079201001234',
                    carLicenseClass: 'B2',
                    motoLicenseClass: 'A1',
                    licenseIssueDate: '15/03/2020',
                    licenseExpiryDate: '15/03/2030',
                    licenseIssuedBy: _userLicenseIssuedBy,
                  );
            _driverLicenses = bootstrapLicenses;

            // Auto-create profile in Firestore so next time it loads properly
            await _firestore.createOrUpdateUserProfile(uid, {
              'fullName': _userName,
              'email': _userEmail,
              'phone': _userPhone,
              'address': _userAddress,
              'avatar': _userAvatar,
              'idCard': _userIdCard,
              'idCardIssueDate': _userIdCardIssueDate,
              'idCardExpiryDate': _userIdCardExpiryDate,
              'gender': _userGender,
              'nationality': _userNationality,
              'placeOfOrigin': _userPlaceOfOrigin,
              'licenseIssuedBy': _userLicenseIssuedBy,
              'occupation': _userOccupation,
              'dateOfBirth': _userDateOfBirth,
              'driverLicenses': bootstrapLicenses,
              ..._legacyLicenseFields(bootstrapLicenses),
              'motoPoints': _motoLicensePoints,
              'carPoints': _carLicensePoints,
              'points': _userPoints,
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
        _themeMode =
            settings['isDarkMode'] == true ? ThemeMode.dark : ThemeMode.light;
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
        'idCardExpiryDate': _userIdCardExpiryDate,
        'gender': _userGender,
        'nationality': _userNationality,
        'placeOfOrigin': _userPlaceOfOrigin,
        'licenseIssuedBy': _userLicenseIssuedBy,
        'occupation': _userOccupation,
        'dateOfBirth': _userDateOfBirth,
        'driverLicenses': _driverLicenses,
        ..._legacyLicenseFields(_driverLicenses),
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
      final existingSettings = await _firestore.getUserSettings(_uid!) ?? {};
      await _firestore.saveUserSettings(_uid!, {
        ...existingSettings,
        'isDarkMode': _themeMode == ThemeMode.dark,
        'language': _locale.languageCode,
        'notificationsEnabled': _notificationsEnabled,
      });
    } catch (e) {
      debugPrint('❌ Error saving settings: $e');
    }
  }

  /// Apply realtime profile updates coming from Firestore snapshots
  /// without writing back to Firestore.
  void applyRemoteProfileData(Map<String, dynamic> data) {
    if (data.isEmpty) return;

    _userName = data['fullName']?.toString() ?? _userName;
    _userEmail = data['email']?.toString() ?? _userEmail;
    _userPhone = data['phone']?.toString() ?? _userPhone;
    _userAddress = data['address']?.toString() ?? _userAddress;
    _userAvatar = data['avatar']?.toString() ?? _userAvatar;
    _userIdCard = data['idCard']?.toString() ?? _userIdCard;
    _userIdCardIssueDate =
        data['idCardIssueDate']?.toString() ?? _userIdCardIssueDate;
    _userIdCardExpiryDate =
        data['idCardExpiryDate']?.toString() ?? _userIdCardExpiryDate;
    _userGender = data['gender']?.toString() ?? _userGender;
    _userNationality = data['nationality']?.toString() ?? _userNationality;
    _userPlaceOfOrigin =
        data['placeOfOrigin']?.toString() ?? _userPlaceOfOrigin;
    _userLicenseIssuedBy =
        data['licenseIssuedBy']?.toString() ?? _userLicenseIssuedBy;
    _userOccupation = data['occupation']?.toString() ?? _userOccupation;
    _userDateOfBirth = data['dateOfBirth']?.toString() ?? _userDateOfBirth;
    final incomingLicenses = _normalizeDriverLicenses(data['driverLicenses']);
    if (incomingLicenses.isNotEmpty) {
      _driverLicenses = incomingLicenses;
    } else {
      final legacyLicenses = _licensesFromLegacy(
        licenseNumber: data['licenseNumber']?.toString(),
        carLicenseClass: data['carLicenseClass']?.toString(),
        motoLicenseClass: data['motoLicenseClass']?.toString(),
        licenseIssueDate: data['licenseIssueDate']?.toString(),
        licenseExpiryDate: data['licenseExpiryDate']?.toString(),
        licenseIssuedBy: data['licenseIssuedBy']?.toString(),
      );
      if (legacyLicenses.isNotEmpty) {
        _driverLicenses = legacyLicenses;
      }
    }
    final legacyPoints = (data['points'] as num?)?.toInt() ?? _userPoints;
    _motoLicensePoints = _readLicensePoints(
      data,
      primaryKey: 'motoPoints',
      legacyKey: 'motoLicensePoints',
      fallback: legacyPoints,
    );
    _carLicensePoints = _readLicensePoints(
      data,
      primaryKey: 'carPoints',
      legacyKey: 'carLicensePoints',
      fallback: legacyPoints,
    );
    _userPoints = _resolveAggregatePoints();
    _profileInitialized = true;

    notifyListeners();
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
    _userIdCardIssueDate = '';
    _userIdCardExpiryDate = '';
    _userGender = '';
    _userNationality = '';
    _userPlaceOfOrigin = '';
    _userLicenseIssuedBy = '';
    _userOccupation = '';
    _userDateOfBirth = '';
    _driverLicenses = [];
    _motoLicensePoints = 12;
    _carLicensePoints = 12;
    _userPoints = 12;
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
