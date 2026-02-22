import 'dart:io';
import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/data/mock_data.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:image_picker/image_picker.dart';

class ProfileScreen extends StatefulWidget {
  final bool embedded;

  const ProfileScreen({super.key, this.embedded = false});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final AppSettings _settings = AppSettings();

  @override
  void initState() {
    super.initState();
    final user = MockData.currentUser;
    _settings.initProfile(
      name: user.fullName,
      email: user.email,
      phone: user.phone,
      address: user.address,
      avatar: user.avatar ?? '',
      idCard: user.idCard,
    );
    _settings.addListener(_onSettingsChanged);
  }

  @override
  void dispose() {
    _settings.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final s = _settings;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF121212) : AppTheme.surfaceColor;
    final cardBg = isDark ? const Color(0xFF1E1E1E) : Colors.white;
    final textPrimary = isDark ? const Color(0xFFE0E0E0) : AppTheme.textPrimary;
    final textSecondary = isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary;
    final divider = isDark ? const Color(0xFF333333) : AppTheme.dividerColor;

    return Scaffold(
      backgroundColor: bgColor,
      body: SingleChildScrollView(
        child: Column(
          children: [
            // ── Profile Header (Red gradient) ──────────────
            Container(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [Color(0xFFD32F2F), Color(0xFFB71C1C), Color(0xFF880E4F)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.only(
                  bottomLeft: Radius.circular(32),
                  bottomRight: Radius.circular(32),
                ),
              ),
              child: SafeArea(
                bottom: false,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
                  child: Column(
                    children: [
                      // Title row — no gear icon
                      Align(
                        alignment: Alignment.centerLeft,
                        child: Text(
                          s.tr('Tài khoản', 'Account'),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 24,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ),
                      const SizedBox(height: 28),

                      // Avatar with gradient ring
                      GestureDetector(
                        onTap: _pickAvatar,
                        child: Stack(
                          alignment: Alignment.center,
                          children: [
                            // Outer glow ring
                            Container(
                              width: 104,
                              height: 104,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                gradient: const LinearGradient(
                                  colors: [Colors.amber, Colors.orangeAccent, Colors.white],
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                ),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.amber.withOpacity(0.4),
                                    blurRadius: 20,
                                    spreadRadius: 2,
                                  ),
                                ],
                              ),
                            ),
                            // Inner avatar
                            Container(
                              width: 96,
                              height: 96,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: const Color(0xFFB71C1C),
                                border: Border.all(color: Colors.white.withOpacity(0.1), width: 2),
                                image: _avatarImage(),
                              ),
                              child: _avatarFallback(),
                            ),
                            // Camera badge
                            Positioned(
                              bottom: 2,
                              right: 2,
                              child: Container(
                                width: 32,
                                height: 32,
                                decoration: BoxDecoration(
                                  gradient: const LinearGradient(
                                    colors: [Color(0xFFF57C00), Color(0xFFE65100)],
                                  ),
                                  shape: BoxShape.circle,
                                  border: Border.all(color: Colors.white, width: 2.5),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.2),
                                      blurRadius: 6,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: const Icon(Icons.camera_alt_rounded, color: Colors.white, size: 15),
                              ),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 16),

                      // User name
                      Text(
                        s.userName,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.3,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        s.userEmail,
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.75),
                          fontSize: 14,
                          fontWeight: FontWeight.w400,
                        ),
                      ),
                      const SizedBox(height: 16),

                      // Verification chip
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [Colors.green.shade600, Colors.green.shade800],
                          ),
                          borderRadius: BorderRadius.circular(24),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.green.withOpacity(0.3),
                              blurRadius: 10,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(Icons.verified_rounded, color: Colors.white, size: 16),
                            const SizedBox(width: 6),
                            Text(
                              s.tr('Đã xác minh tài khoản', 'Account verified'),
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                                letterSpacing: 0.3,
                              ),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 20),

                      // Stats row — glassmorphism card
                      Container(
                        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(18),
                          border: Border.all(color: Colors.white.withOpacity(0.15)),
                        ),
                        child: Row(
                          children: [
                            _buildStatItem(s.userIdCard.length > 8 ? '${s.userIdCard.substring(0, 4)}...${s.userIdCard.substring(s.userIdCard.length - 4)}' : s.userIdCard, 'CCCD'),
                            _buildStatDivider(),
                            _buildStatItem('12', s.tr('Điểm', 'Points')),
                            _buildStatDivider(),
                            _buildStatItem('2', s.tr('Phương tiện', 'Vehicles')),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // ── Content ───────────────────────────────────────
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Personal Info
                  Text(
                    s.tr('Thông tin cá nhân', 'Personal Info'),
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: textPrimary,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Container(
                    decoration: BoxDecoration(
                      color: cardBg,
                      borderRadius: BorderRadius.circular(AppTheme.radiusL),
                      boxShadow: isDark ? [] : AppTheme.cardShadow,
                    ),
                    child: Column(
                      children: [
                        _buildInfoTile(Icons.person_rounded, AppTheme.primaryColor,
                            s.tr('Họ và tên', 'Full Name'), s.userName, textPrimary, textSecondary, divider,
                            onTap: () => _editField(s.tr('Họ và tên', 'Full Name'), s.userName, (v) {
                                  s.updateProfile(name: v);
                                })),
                        Container(margin: const EdgeInsets.symmetric(horizontal: 16), height: 1, color: divider),
                        _buildInfoTile(Icons.phone_rounded, AppTheme.successColor,
                            s.tr('Số điện thoại', 'Phone'), s.userPhone, textPrimary, textSecondary, divider,
                            onTap: () => _editField(s.tr('Số điện thoại', 'Phone'), s.userPhone, (v) {
                                  s.updateProfile(phone: v);
                                })),
                        Container(margin: const EdgeInsets.symmetric(horizontal: 16), height: 1, color: divider),
                        _buildInfoTile(Icons.badge_rounded, AppTheme.infoColor,
                            'CCCD/CMND', s.userIdCard, textPrimary, textSecondary, divider),
                        Container(margin: const EdgeInsets.symmetric(horizontal: 16), height: 1, color: divider),
                        _buildInfoTile(Icons.email_rounded, AppTheme.warningColor,
                            'Email', s.userEmail, textPrimary, textSecondary, divider,
                            onTap: () => _editField('Email', s.userEmail, (v) {
                                  s.updateProfile(email: v);
                                })),
                        Container(margin: const EdgeInsets.symmetric(horizontal: 16), height: 1, color: divider),
                        _buildInfoTile(Icons.location_on_rounded, AppTheme.secondaryColor,
                            s.tr('Địa chỉ', 'Address'), s.userAddress, textPrimary, textSecondary, divider,
                            onTap: () => _editField(s.tr('Địa chỉ', 'Address'), s.userAddress, (v) {
                                  s.updateProfile(address: v);
                                })),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),

                  // Vehicles
                  Text(
                    s.tr('Phương tiện của tôi', 'My Vehicles'),
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: textPrimary,
                    ),
                  ),
                  const SizedBox(height: 12),
                  ...MockData.vehicles.map((vehicle) => _buildVehicleCard(context, vehicle)),

                  const SizedBox(height: 24),

                  // Menu
                  Text(
                    s.tr('Tiện ích', 'Utilities'),
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: textPrimary,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildMenuList(context, cardBg, textPrimary, textSecondary, divider, isDark),

                  const SizedBox(height: 24),

                  // Logout
                  SizedBox(
                    width: double.infinity,
                    height: 50,
                    child: OutlinedButton.icon(
                      onPressed: () => _showLogoutDialog(context),
                      icon: const Icon(Icons.logout_rounded, color: AppTheme.dangerColor, size: 20),
                      label: Text(
                        s.tr('Đăng xuất', 'Sign out'),
                        style: const TextStyle(
                          color: AppTheme.dangerColor,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      style: OutlinedButton.styleFrom(
                        side: const BorderSide(color: AppTheme.dangerColor),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(AppTheme.radiusM),
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 30),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  STAT HELPERS (Profile header)
  // ═══════════════════════════════════════════════════════════════
  Widget _buildStatItem(String value, String label) {
    return Expanded(
      child: Column(
        children: [
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              color: Colors.white.withOpacity(0.6),
              fontSize: 11,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatDivider() {
    return Container(
      width: 1,
      height: 30,
      color: Colors.white.withOpacity(0.2),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  AVATAR HELPERS
  // ═══════════════════════════════════════════════════════════════
  DecorationImage? _avatarImage() {
    final url = _settings.userAvatar;
    if (url.isEmpty) return null;
    if (url.startsWith('http')) {
      return DecorationImage(image: NetworkImage(url), fit: BoxFit.cover);
    }
    // Local file
    return DecorationImage(image: FileImage(File(url)), fit: BoxFit.cover);
  }

  Widget? _avatarFallback() {
    if (_settings.userAvatar.isNotEmpty) return null;
    final name = _settings.userName;
    return Center(
      child: Text(
        name.isNotEmpty ? name.substring(0, 1).toUpperCase() : '?',
        style: const TextStyle(color: Colors.white, fontSize: 32, fontWeight: FontWeight.bold),
      ),
    );
  }

  Future<void> _pickAvatar() async {
    final s = _settings;
    showModalBottomSheet(
      context: context,
      backgroundColor: Theme.of(context).brightness == Brightness.dark
          ? const Color(0xFF1E1E1E) : Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40, height: 4,
              margin: const EdgeInsets.only(top: 12, bottom: 16),
              decoration: BoxDecoration(
                color: AppTheme.dividerColor,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Text(
              s.tr('Đổi ảnh đại diện', 'Change Avatar'),
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 16),
            ListTile(
              leading: Container(
                width: 42, height: 42,
                decoration: BoxDecoration(
                  color: AppTheme.primaryColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.photo_library_rounded, color: AppTheme.primaryColor),
              ),
              title: Text(s.tr('Chọn từ thư viện', 'Choose from gallery')),
              onTap: () async {
                Navigator.pop(ctx);
                final picker = ImagePicker();
                final file = await picker.pickImage(source: ImageSource.gallery, maxWidth: 512);
                if (file != null) {
                  _settings.updateAvatar(file.path);
                }
              },
            ),
            ListTile(
              leading: Container(
                width: 42, height: 42,
                decoration: BoxDecoration(
                  color: AppTheme.infoColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.camera_alt_rounded, color: AppTheme.infoColor),
              ),
              title: Text(s.tr('Chụp ảnh mới', 'Take a photo')),
              onTap: () async {
                Navigator.pop(ctx);
                final picker = ImagePicker();
                final file = await picker.pickImage(source: ImageSource.camera, maxWidth: 512);
                if (file != null) {
                  _settings.updateAvatar(file.path);
                }
              },
            ),
            if (_settings.userAvatar.isNotEmpty)
              ListTile(
                leading: Container(
                  width: 42, height: 42,
                  decoration: BoxDecoration(
                    color: AppTheme.dangerColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.delete_outline_rounded, color: AppTheme.dangerColor),
                ),
                title: Text(s.tr('Xóa ảnh đại diện', 'Remove avatar')),
                onTap: () {
                  Navigator.pop(ctx);
                  _settings.updateAvatar('');
                },
              ),
            const SizedBox(height: 12),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  EDIT FIELD DIALOG
  // ═══════════════════════════════════════════════════════════════
  void _editField(String label, String currentValue, ValueChanged<String> onSave) {
    final controller = TextEditingController(text: currentValue);
    final isDark = Theme.of(context).brightness == Brightness.dark;

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: AppTheme.primaryColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.edit_rounded, color: AppTheme.primaryColor, size: 20),
            ),
            const SizedBox(width: 12),
            Text(
              _settings.tr('Chỉnh sửa', 'Edit'),
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
            ),
          ],
        ),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: InputDecoration(
            labelText: label,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(color: AppTheme.primaryColor, width: 2),
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(_settings.tr('Hủy', 'Cancel'), style: const TextStyle(color: AppTheme.textSecondary)),
          ),
          ElevatedButton(
            onPressed: () {
              final v = controller.text.trim();
              if (v.isNotEmpty) {
                onSave(v);
                Navigator.pop(ctx);
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryColor,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: Text(_settings.tr('Lưu', 'Save')),
          ),
        ],
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  INFO TILE
  // ═══════════════════════════════════════════════════════════════
  Widget _buildInfoTile(IconData icon, Color color, String label, String value,
      Color textPrimary, Color textSecondary, Color divider,
      {VoidCallback? onTap}) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label,
                    style: TextStyle(fontSize: 12, color: textSecondary),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    value,
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w500,
                      color: textPrimary,
                    ),
                  ),
                ],
              ),
            ),
            if (onTap != null)
              Icon(Icons.edit_outlined, color: textSecondary, size: 18),
          ],
        ),
      ),
    );
  }

  Widget _buildVehicleCard(BuildContext context, vehicle) {
    final isMotorcycle = vehicle.vehicleType.contains('máy');

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isMotorcycle
              ? [const Color(0xFFD32F2F), const Color(0xFFB71C1C)]
              : [const Color(0xFF1565C0), const Color(0xFF0D47A1)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        boxShadow: [
          BoxShadow(
            color: (isMotorcycle ? AppTheme.primaryColor : AppTheme.infoColor).withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.15),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                isMotorcycle ? Icons.two_wheeler_rounded : Icons.directions_car_rounded,
                color: Colors.white,
                size: 26,
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      vehicle.licensePlate,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 2,
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    '${vehicle.brand} ${vehicle.model}',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.85),
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right_rounded, color: Colors.white.withOpacity(0.6)),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  MENU LIST — Dark mode, Language, IP, etc.
  // ═══════════════════════════════════════════════════════════════
  Widget _buildMenuList(BuildContext context, Color cardBg, Color textPrimary,
      Color textSecondary, Color divider, bool isDark) {
    final s = _settings;

    return Container(
      decoration: BoxDecoration(
        color: cardBg,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        boxShadow: isDark ? [] : AppTheme.cardShadow,
      ),
      child: Column(
        children: [
          _buildMenuItem(Icons.notifications_outlined, AppTheme.warningColor,
              s.tr('Thông báo', 'Notifications'), textPrimary, textSecondary, () {
            Navigator.pushNamed(context, '/notifications');
          }),
          _menuDivider(divider),
          _buildMenuItem(Icons.router_rounded, AppTheme.primaryColor,
              s.tr('Chỉnh IP máy chủ', 'Server IP Settings'), textPrimary, textSecondary, () {
            _showIpSettingsDialog(context);
          },
              trailing: Text(
                '${ApiService.serverIp}:${ApiService.serverPort}',
                style: TextStyle(
                  fontSize: 12,
                  color: textSecondary,
                  fontWeight: FontWeight.w500,
                ),
              )),
          _menuDivider(divider),
          // ── Dark Mode Toggle ──
          _buildMenuItem(Icons.dark_mode_outlined, Colors.deepPurple,
              s.tr('Giao diện tối', 'Dark Mode'), textPrimary, textSecondary, () {
            s.toggleDarkMode();
          },
              trailing: Switch(
                value: s.isDarkMode,
                onChanged: (v) => s.setThemeMode(v ? ThemeMode.dark : ThemeMode.light),
                activeColor: AppTheme.primaryColor,
              )),
          _menuDivider(divider),
          // ── Language Toggle ──
          _buildMenuItem(Icons.language_rounded, AppTheme.infoColor,
              s.tr('Ngôn ngữ', 'Language'), textPrimary, textSecondary, () {
            _showLanguageDialog(context);
          },
              trailing: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: AppTheme.infoColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  s.isVietnamese ? 'Tiếng Việt' : 'English',
                  style: TextStyle(
                    fontSize: 12,
                    color: AppTheme.infoColor,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              )),
          _menuDivider(divider),
          _buildMenuItem(Icons.help_outline_rounded, AppTheme.successColor,
              s.tr('Trợ giúp & Hỗ trợ', 'Help & Support'), textPrimary, textSecondary, () {}),
          _menuDivider(divider),
          _buildMenuItem(Icons.shield_outlined, AppTheme.secondaryColor,
              s.tr('Chính sách bảo mật', 'Privacy Policy'), textPrimary, textSecondary, () {}),
          _menuDivider(divider),
          _buildMenuItem(Icons.info_outline_rounded, textSecondary,
              s.tr('Về ứng dụng', 'About'), textPrimary, textSecondary, () {}),
        ],
      ),
    );
  }

  Widget _buildMenuItem(IconData icon, Color color, String title,
      Color textPrimary, Color textSecondary, VoidCallback onTap,
      {Widget? trailing}) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                title,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                  color: textPrimary,
                ),
              ),
            ),
            trailing ?? Icon(Icons.chevron_right_rounded, color: textSecondary, size: 22),
          ],
        ),
      ),
    );
  }

  Widget _menuDivider(Color color) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      height: 1,
      color: color,
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  LANGUAGE DIALOG
  // ═══════════════════════════════════════════════════════════════
  void _showLanguageDialog(BuildContext context) {
    final s = _settings;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: AppTheme.infoColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.language_rounded, color: AppTheme.infoColor, size: 22),
            ),
            const SizedBox(width: 12),
            Text(
              s.tr('Chọn ngôn ngữ', 'Select Language'),
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildLanguageOption(
              ctx,
              '🇻🇳',
              'Tiếng Việt',
              s.tr('Ngôn ngữ mặc định', 'Default language'),
              s.isVietnamese,
              () {
                s.setLocale(const Locale('vi', 'VN'));
                Navigator.pop(ctx);
              },
            ),
            const SizedBox(height: 8),
            _buildLanguageOption(
              ctx,
              '🇺🇸',
              'English',
              s.tr('Ngôn ngữ phụ', 'Secondary language'),
              !s.isVietnamese,
              () {
                s.setLocale(const Locale('en', 'US'));
                Navigator.pop(ctx);
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLanguageOption(BuildContext ctx, String flag, String title,
      String subtitle, bool isSelected, VoidCallback onTap) {
    final isDark = Theme.of(ctx).brightness == Brightness.dark;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: isSelected
              ? AppTheme.primaryColor.withOpacity(isDark ? 0.2 : 0.06)
              : (isDark ? const Color(0xFF2A2A2A) : AppTheme.surfaceColor),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: isSelected ? AppTheme.primaryColor : Colors.transparent,
            width: 1.5,
          ),
        ),
        child: Row(
          children: [
            Text(flag, style: const TextStyle(fontSize: 28)),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 15,
                        color: isDark ? Colors.white : AppTheme.textPrimary,
                      )),
                  Text(subtitle,
                      style: TextStyle(
                        fontSize: 12,
                        color: isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary,
                      )),
                ],
              ),
            ),
            if (isSelected)
              const Icon(Icons.check_circle_rounded, color: AppTheme.primaryColor, size: 22),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  IP SETTINGS DIALOG
  // ═══════════════════════════════════════════════════════════════
  void _showIpSettingsDialog(BuildContext context) {
    final ipController = TextEditingController(text: ApiService.serverIp);
    final portController = TextEditingController(text: ApiService.serverPort.toString());
    final isDark = Theme.of(context).brightness == Brightness.dark;

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: AppTheme.primaryColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.router_rounded, color: AppTheme.primaryColor, size: 22),
            ),
            const SizedBox(width: 12),
            Text(
              _settings.tr('Cài đặt IP máy chủ', 'Server IP Settings'),
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              _settings.tr(
                'Nhập địa chỉ IP và cổng của máy chủ phát hiện vi phạm.',
                'Enter server IP address and port for violation detection.',
              ),
              style: TextStyle(
                fontSize: 13,
                color: isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary,
              ),
            ),
            const SizedBox(height: 18),
            TextField(
              controller: ipController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: _settings.tr('Địa chỉ IP', 'IP Address'),
                hintText: '192.168.1.93',
                prefixIcon: const Icon(Icons.computer_rounded, size: 20),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: AppTheme.primaryColor, width: 2),
                ),
              ),
            ),
            const SizedBox(height: 14),
            TextField(
              controller: portController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: _settings.tr('Cổng (Port)', 'Port'),
                hintText: '8000',
                prefixIcon: const Icon(Icons.numbers_rounded, size: 20),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: AppTheme.primaryColor, width: 2),
                ),
              ),
            ),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: isDark ? const Color(0xFF2A2A2A) : AppTheme.surfaceColor,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Icon(
                    ApiService().isConnected ? Icons.check_circle : Icons.error_outline,
                    color: ApiService().isConnected ? AppTheme.successColor : AppTheme.dangerColor,
                    size: 16,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    ApiService().isConnected
                        ? _settings.tr('Đang kết nối', 'Connected')
                        : _settings.tr('Chưa kết nối', 'Not connected'),
                    style: TextStyle(
                      fontSize: 12,
                      color: ApiService().isConnected ? AppTheme.successColor : AppTheme.dangerColor,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(_settings.tr('Hủy', 'Cancel'),
                style: const TextStyle(color: AppTheme.textSecondary)),
          ),
          ElevatedButton.icon(
            onPressed: () {
              final ip = ipController.text.trim();
              final port = int.tryParse(portController.text.trim()) ?? 8000;
              if (ip.isNotEmpty) {
                ApiService().setServerAddress(ip, port: port);
                Navigator.pop(ctx);
                setState(() {});
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Row(
                      children: [
                        const Icon(Icons.check_circle, color: Colors.white, size: 18),
                        const SizedBox(width: 8),
                        Text('${_settings.tr("Đã cập nhật IP", "IP updated")}: $ip:$port'),
                      ],
                    ),
                    backgroundColor: AppTheme.successColor,
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                );
              }
            },
            icon: const Icon(Icons.save_rounded, size: 18),
            label: Text(_settings.tr('Lưu & Kết nối', 'Save & Connect')),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryColor,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ],
      ),
    );
  }

  void _showLogoutDialog(BuildContext context) {
    final s = _settings;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppTheme.radiusXL)),
        title: Text(
          s.tr('Đăng xuất', 'Sign out'),
          style: const TextStyle(fontWeight: FontWeight.w700),
        ),
        content: Text(s.tr(
          'Bạn có chắc chắn muốn đăng xuất khỏi ứng dụng?',
          'Are you sure you want to sign out?',
        )),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text(s.tr('Hủy', 'Cancel'),
                style: const TextStyle(color: AppTheme.textSecondary)),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pushNamedAndRemoveUntil(context, '/login', (route) => false);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.dangerColor,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppTheme.radiusM)),
            ),
            child: Text(s.tr('Đăng xuất', 'Sign out')),
          ),
        ],
      ),
    );
  }
}
