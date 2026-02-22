import 'package:flutter/material.dart';

/// Global app settings singleton — manages theme, locale, notifications, user profile.
class AppSettings extends ChangeNotifier {
  // ── Singleton ──────────────────────────────────────────────────
  static final AppSettings _instance = AppSettings._internal();
  factory AppSettings() => _instance;
  AppSettings._internal();

  // ── Theme ──────────────────────────────────────────────────────
  ThemeMode _themeMode = ThemeMode.light;
  ThemeMode get themeMode => _themeMode;
  bool get isDarkMode => _themeMode == ThemeMode.dark;

  void setThemeMode(ThemeMode mode) {
    _themeMode = mode;
    notifyListeners();
  }

  void toggleDarkMode() {
    _themeMode = _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    notifyListeners();
  }

  // ── Locale ─────────────────────────────────────────────────────
  Locale _locale = const Locale('vi', 'VN');
  Locale get locale => _locale;
  bool get isVietnamese => _locale.languageCode == 'vi';

  void setLocale(Locale locale) {
    _locale = locale;
    notifyListeners();
  }

  void toggleLanguage() {
    _locale = isVietnamese
        ? const Locale('en', 'US')
        : const Locale('vi', 'VN');
    notifyListeners();
  }

  // ── Notification Badge Count ───────────────────────────────────
  int _unreadNotifications = 3; // Default some unread on start
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

  // ── User Profile Editable Fields ───────────────────────────────
  String _userName = '';
  String _userEmail = '';
  String _userPhone = '';
  String _userAddress = '';
  String _userAvatar = '';
  String _userIdCard = '';

  String get userName => _userName;
  String get userEmail => _userEmail;
  String get userPhone => _userPhone;
  String get userAddress => _userAddress;
  String get userAvatar => _userAvatar;
  String get userIdCard => _userIdCard;

  bool _profileInitialized = false;
  bool get profileInitialized => _profileInitialized;

  void initProfile({
    required String name,
    required String email,
    required String phone,
    required String address,
    required String avatar,
    required String idCard,
  }) {
    if (!_profileInitialized) {
      _userName = name;
      _userEmail = email;
      _userPhone = phone;
      _userAddress = address;
      _userAvatar = avatar;
      _userIdCard = idCard;
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
  }) {
    if (name != null) _userName = name;
    if (email != null) _userEmail = email;
    if (phone != null) _userPhone = phone;
    if (address != null) _userAddress = address;
    if (avatar != null) _userAvatar = avatar;
    if (idCard != null) _userIdCard = idCard;
    notifyListeners();
  }

  void updateAvatar(String url) {
    _userAvatar = url;
    notifyListeners();
  }

  // ── Localization Strings ───────────────────────────────────────
  // Quick i18n helper — returns Vietnamese or English strings
  String tr(String viText, String enText) {
    return isVietnamese ? viText : enText;
  }
}
