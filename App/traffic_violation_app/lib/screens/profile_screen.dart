import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/models/vehicle.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/services/auth_service.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/update_service.dart';
import 'package:image_picker/image_picker.dart';
import 'package:traffic_violation_app/screens/vehicles_screen.dart'
    as traffic_vehicles;

class ProfileScreen extends StatefulWidget {
  final bool embedded;

  const ProfileScreen({super.key, this.embedded = false});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final AppSettings _settings = AppSettings();
  List<Vehicle> _vehicles = [];
  StreamSubscription? _vehicleSub;

  @override
  void initState() {
    super.initState();
    _settings.addListener(_onSettingsChanged);
    _loadVehicles();
  }

  void _loadVehicles() {
    final uid = _settings.uid;
    if (uid != null) {
      _vehicleSub = FirestoreService().vehiclesStream(uid).listen((vehicles) {
        if (mounted) setState(() => _vehicles = vehicles);
      });
    }
  }

  @override
  void dispose() {
    _vehicleSub?.cancel();
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
    final textSecondary =
        isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary;
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
                  colors: [
                    Color(0xFFD32F2F),
                    Color(0xFFB71C1C),
                    Color(0xFF880E4F)
                  ],
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
                                  colors: [
                                    Colors.amber,
                                    Colors.orangeAccent,
                                    Colors.white
                                  ],
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
                                border: Border.all(
                                    color: Colors.white.withOpacity(0.1),
                                    width: 2),
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
                                    colors: [
                                      Color(0xFFF57C00),
                                      Color(0xFFE65100)
                                    ],
                                  ),
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                      color: Colors.white, width: 2.5),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.2),
                                      blurRadius: 6,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: const Icon(Icons.camera_alt_rounded,
                                    color: Colors.white, size: 15),
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
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              Colors.green.shade600,
                              Colors.green.shade800
                            ],
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
                            const Icon(Icons.verified_rounded,
                                color: Colors.white, size: 16),
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
                        padding: const EdgeInsets.symmetric(
                            vertical: 14, horizontal: 8),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(18),
                          border:
                              Border.all(color: Colors.white.withOpacity(0.15)),
                        ),
                        child: Row(
                          children: [
                            _buildStatItem(_settings.userPoints.toString(), s.tr('Điểm', 'Points'),
                                valueColor: Colors.amber),
                            _buildStatDivider(),
                            _buildStatItem(
                              _vehicles.length.toString(),
                              s.tr('Phương tiện', 'Vehicles'),
                              onTap: () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                      builder: (_) => const traffic_vehicles
                                          .VehiclesScreen()),
                                );
                              },
                            ),
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
                        _buildInfoTile(
                            Icons.person_rounded,
                            AppTheme.primaryColor,
                            s.tr('Họ và tên', 'Full Name'),
                            s.userName,
                            textPrimary,
                            textSecondary,
                            divider,
                            onTap: () => _editField(
                                s.tr('Họ và tên', 'Full Name'),
                                s.userName,
                                (v) => _submitProfileUpdate('fullName', v))),
                        Container(
                            margin: const EdgeInsets.symmetric(horizontal: 16),
                            height: 1,
                            color: divider),
                        _buildInfoTile(
                            Icons.phone_rounded,
                            AppTheme.successColor,
                            s.tr('Số điện thoại', 'Phone'),
                            s.userPhone,
                            textPrimary,
                            textSecondary,
                            divider,
                            onTap: () => _editField(
                                s.tr('Số điện thoại', 'Phone'),
                                s.userPhone,
                                (v) => _submitProfileUpdate('phone', v))),
                        Container(
                            margin: const EdgeInsets.symmetric(horizontal: 16),
                            height: 1,
                            color: divider),
                        _buildInfoTile(
                            Icons.badge_rounded,
                            AppTheme.infoColor,
                            'CCCD/CMND',
                            s.userIdCard,
                            textPrimary,
                            textSecondary,
                            divider,
                            onTap: () => _editField('CCCD/CMND', s.userIdCard,
                                (v) => _submitProfileUpdate('idCard', v))),
                        Container(
                            margin: const EdgeInsets.symmetric(horizontal: 16),
                            height: 1,
                            color: divider),
                        _buildInfoTile(
                            Icons.calendar_month_rounded,
                            AppTheme.infoColor,
                            s.tr('Ngày cấp CCCD', 'ID Card Issue Date'),
                            s.userIdCardIssueDate.isEmpty ? s.tr('Chưa cập nhật', 'Not updated') : s.userIdCardIssueDate,
                            textPrimary,
                            textSecondary,
                            divider,
                            onTap: () => _editField(s.tr('Ngày cấp CCCD', 'ID Card Issue Date'), s.userIdCardIssueDate,
                                (v) => _submitProfileUpdate('idCardIssueDate', v))),
                        Container(
                            margin: const EdgeInsets.symmetric(horizontal: 16),
                            height: 1,
                            color: divider),
                        _buildInfoTile(
                            Icons.cake_rounded,
                            AppTheme.warningColor,
                            s.tr('Ngày sinh', 'Date of Birth'),
                            s.userDateOfBirth.isEmpty ? s.tr('Chưa cập nhật', 'Not updated') : s.userDateOfBirth,
                            textPrimary,
                            textSecondary,
                            divider,
                            onTap: () => _editField(s.tr('Ngày sinh', 'Date of Birth'), s.userDateOfBirth,
                                (v) => _submitProfileUpdate('dateOfBirth', v))),
                        Container(
                            margin: const EdgeInsets.symmetric(horizontal: 16),
                            height: 1,
                            color: divider),
                        _buildInfoTile(
                            Icons.work_rounded,
                            AppTheme.secondaryColor,
                            s.tr('Nghề nghiệp', 'Occupation'),
                            s.userOccupation.isEmpty ? s.tr('Chưa cập nhật', 'Not updated') : s.userOccupation,
                            textPrimary,
                            textSecondary,
                            divider,
                            onTap: () => _editField(s.tr('Nghề nghiệp', 'Occupation'), s.userOccupation,
                                (v) => _submitProfileUpdate('occupation', v))),
                        Container(
                            margin: const EdgeInsets.symmetric(horizontal: 16),
                            height: 1,
                            color: divider),
                        _buildInfoTile(
                            Icons.email_rounded,
                            AppTheme.warningColor,
                            'Email',
                            s.userEmail,
                            textPrimary,
                            textSecondary,
                            divider,
                            onTap: () => _editField('Email', s.userEmail,
                                (v) => _submitProfileUpdate('email', v))),
                        Container(
                            margin: const EdgeInsets.symmetric(horizontal: 16),
                            height: 1,
                            color: divider),
                        _buildInfoTile(
                            Icons.location_on_rounded,
                            AppTheme.secondaryColor,
                            s.tr('Địa chỉ', 'Address'),
                            s.userAddress,
                            textPrimary,
                            textSecondary,
                            divider,
                            onTap: () => _editField(
                                s.tr('Địa chỉ', 'Address'),
                                s.userAddress,
                                (v) => _submitProfileUpdate('address', v))),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),

                  // Vehicles
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        s.tr('Phương tiện của tôi', 'My Vehicles'),
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                          color: textPrimary,
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (_) =>
                                    const traffic_vehicles.VehiclesScreen()),
                          );
                        },
                        child: Text(s.tr('Xem tất cả', 'View all')),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (_vehicles.isEmpty)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Text(
                        s.tr('Chưa có phương tiện. Nhấn "Xem tất cả" để thêm.',
                            'No vehicles yet. Tap "View all" to add.'),
                        style: TextStyle(
                            color: textSecondary, fontStyle: FontStyle.italic),
                      ),
                    ),
                  ..._vehicles
                      .map((vehicle) => _buildVehicleCard(context, vehicle)),

                  const SizedBox(height: 24),

                  // Menu
                  Text(
                    s.tr('Cài đặt', 'Settings'),
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: textPrimary,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildMenuList(context, cardBg, textPrimary, textSecondary,
                      divider, isDark),

                  const SizedBox(height: 24),

                  // Logout
                  SizedBox(
                    width: double.infinity,
                    height: 50,
                    child: OutlinedButton.icon(
                      onPressed: () => _showLogoutDialog(context),
                      icon: const Icon(Icons.logout_rounded,
                          color: AppTheme.dangerColor, size: 20),
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
                  if (widget.embedded) const SizedBox(height: 80),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  // ═══════════════════════════════════════════════════════════════
  //  STAT HELPERS (Profile header)
  // ═══════════════════════════════════════════════════════════════
  Widget _buildStatItem(String value, String label,
      {Color? valueColor, VoidCallback? onTap}) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        behavior: HitTestBehavior.opaque,
        child: Column(
          children: [
            Text(
              value,
              style: TextStyle(
                color: valueColor ?? Colors.white,
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
    final file = File(url);
    if (!file.existsSync()) return null;
    return DecorationImage(image: FileImage(file), fit: BoxFit.cover);
  }

  Widget? _avatarFallback() {
    final url = _settings.userAvatar;
    bool hasValidAvatar = false;
    if (url.isNotEmpty) {
      if (url.startsWith('http')) {
        hasValidAvatar = true;
      } else {
        hasValidAvatar = File(url).existsSync();
      }
    }

    if (hasValidAvatar) return null;

    final name = _settings.userName;
    return Center(
      child: Text(
        name.isNotEmpty ? name.substring(0, 1).toUpperCase() : '?',
        style: const TextStyle(
            color: Colors.white, fontSize: 32, fontWeight: FontWeight.bold),
      ),
    );
  }

  Future<void> _pickAvatar() async {
    final s = _settings;
    showModalBottomSheet(
      context: context,
      backgroundColor: Theme.of(context).brightness == Brightness.dark
          ? const Color(0xFF1E1E1E)
          : Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40,
              height: 4,
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
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: AppTheme.primaryColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.photo_library_rounded,
                    color: AppTheme.primaryColor),
              ),
              title: Text(s.tr('Chọn từ thư viện', 'Choose from gallery')),
              onTap: () async {
                Navigator.pop(ctx);
                final picker = ImagePicker();
                final file = await picker.pickImage(
                    source: ImageSource.gallery, maxWidth: 512);
                if (file != null) {
                  _settings.updateAvatar(file.path);
                }
              },
            ),
            ListTile(
              leading: Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: AppTheme.infoColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.camera_alt_rounded,
                    color: AppTheme.infoColor),
              ),
              title: Text(s.tr('Chụp ảnh mới', 'Take a photo')),
              onTap: () async {
                Navigator.pop(ctx);
                final picker = ImagePicker();
                final file = await picker.pickImage(
                    source: ImageSource.camera, maxWidth: 512);
                if (file != null) {
                  _settings.updateAvatar(file.path);
                }
              },
            ),
            if (_settings.userAvatar.isNotEmpty)
              ListTile(
                leading: Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: AppTheme.dangerColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.delete_outline_rounded,
                      color: AppTheme.dangerColor),
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

  Future<void> _submitProfileUpdate(String key, String newValue) async {
    final uid = _settings.uid;
    if (uid == null) return;

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => const Center(child: CircularProgressIndicator()),
    );

    try {
      await FirestoreService().requestProfileUpdate(uid, {
        key: newValue,
      });
      if (mounted) {
        Navigator.pop(context); // close loading
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(_settings.tr('Đã gửi yêu cầu cập nhật. Đang chờ duyệt.',
              'Update request sent. Waiting for approval.')),
          backgroundColor: AppTheme.successColor,
        ));
      }
    } catch (e) {
      if (mounted) {
        Navigator.pop(context); // close loading
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(
              _settings.tr('Lỗi khi gửi yêu cầu', 'Error sending request')),
          backgroundColor: AppTheme.dangerColor,
        ));
      }
    }
  }

  void _editField(
      String label, String currentValue, ValueChanged<String> onSave) {
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
              child: const Icon(Icons.edit_rounded,
                  color: AppTheme.primaryColor, size: 20),
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
              borderSide:
                  const BorderSide(color: AppTheme.primaryColor, width: 2),
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(_settings.tr('Hủy', 'Cancel'),
                style: const TextStyle(color: AppTheme.textSecondary)),
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
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
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
            color: (isMotorcycle ? AppTheme.primaryColor : AppTheme.infoColor)
                .withOpacity(0.3),
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
                isMotorcycle
                    ? Icons.two_wheeler_rounded
                    : Icons.directions_car_rounded,
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
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
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
            Icon(Icons.chevron_right_rounded,
                color: Colors.white.withOpacity(0.6)),
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
          _buildMenuItem(
              Icons.notifications_outlined,
              AppTheme.warningColor,
              s.tr('Thông báo', 'Notifications'),
              textPrimary,
              textSecondary, () {
            s.toggleNotifications();
          },
              trailing: Switch(
                value: s.notificationsEnabled,
                onChanged: (v) => s.setNotificationsEnabled(v),
                activeColor: AppTheme.primaryColor,
              )),
          _menuDivider(divider),
          _buildMenuItem(
              Icons.router_rounded,
              AppTheme.primaryColor,
              s.tr('Chỉnh IP máy chủ', 'Server IP Settings'),
              textPrimary,
              textSecondary, () {
            _showIpSettingsDialog(context);
          },
              trailing: Flexible(
                child: Text(
                  ApiService.serverIp,
                  style: TextStyle(
                    fontSize: 12,
                    color: textSecondary,
                    fontWeight: FontWeight.w500,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  textAlign: TextAlign.right,
                ),
              )),
          _menuDivider(divider),
          // ── Dark Mode Toggle ──
          _buildMenuItem(
              Icons.dark_mode_outlined,
              Colors.deepPurple,
              s.tr('Giao diện tối', 'Dark Mode'),
              textPrimary,
              textSecondary, () {
            s.toggleDarkMode();
          },
              trailing: Switch(
                value: s.isDarkMode,
                onChanged: (v) =>
                    s.setThemeMode(v ? ThemeMode.dark : ThemeMode.light),
                activeColor: AppTheme.primaryColor,
              )),
          _menuDivider(divider),
          // ── Language Toggle ──
          _buildMenuItem(Icons.language_rounded, AppTheme.infoColor,
              s.tr('Ngôn ngữ', 'Language'), textPrimary, textSecondary, () {
            _showLanguageDialog(context);
          },
              trailing: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
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
          _buildMenuItem(
              Icons.help_outline_rounded,
              AppTheme.successColor,
              s.tr('Trợ giúp & Hỗ trợ', 'Help & Support'),
              textPrimary,
              textSecondary, () {
            Navigator.pushNamed(context, '/support');
          }),
          _menuDivider(divider),
          _buildMenuItem(
              Icons.shield_outlined,
              AppTheme.secondaryColor,
              s.tr('Chính sách bảo mật', 'Privacy Policy'),
              textPrimary,
              textSecondary,
              () {}),
          _menuDivider(divider),
          _buildMenuItem(Icons.info_outline_rounded, textSecondary,
              s.tr('Về ứng dụng', 'About'), textPrimary, textSecondary, () {
            _showAboutDialog(context);
          }),
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
            trailing ??
                Icon(Icons.chevron_right_rounded,
                    color: textSecondary, size: 22),
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
  //  ABOUT DIALOG
  // ═══════════════════════════════════════════════════════════════
  void _showAboutDialog(BuildContext context) {
    final s = _settings;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final updateService = UpdateService();

    // UI Helpers inside dialog
    Widget _buildSectionHeader(IconData icon, String title) {
      return Padding(
        padding: const EdgeInsets.only(top: 20, bottom: 10),
        child: Row(
          children: [
            Icon(icon, size: 18, color: AppTheme.primaryColor),
            const SizedBox(width: 8),
            Text(
              title,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w800,
                color: isDark ? Colors.white : Colors.black87,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      );
    }

    Widget _buildBullet(String text) {
      return Padding(
        padding: const EdgeInsets.only(bottom: 6),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.only(top: 5, right: 8),
              child: Icon(Icons.circle, size: 6, color: AppTheme.primaryColor),
            ),
            Expanded(
              child: Text(
                text,
                style: TextStyle(
                  fontSize: 13,
                  color: isDark ? Colors.white70 : Colors.black87,
                  height: 1.4,
                ),
              ),
            ),
          ],
        ),
      );
    }

    Widget _buildTechChip(String label) {
      return Container(
        margin: const EdgeInsets.only(right: 6, bottom: 6),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: AppTheme.primaryColor.withOpacity(0.1),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: AppTheme.primaryColor.withOpacity(0.3)),
        ),
        child: Text(
          label,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: AppTheme.primaryColor,
          ),
        ),
      );
    }

    updateService.init().then((_) {
      if (!context.mounted) return;

      showDialog(
        context: context,
        builder: (ctx) => Dialog(
          backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
          insetPadding:
              const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header Image
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(24),
                decoration: const BoxDecoration(
                  color: Color(0xFFF8F9FA),
                  borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
                ),
                child: Column(
                  children: [
                    Container(
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                              color: Colors.black.withOpacity(0.1),
                              blurRadius: 15,
                              offset: const Offset(0, 5)),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(20),
                        child: Image.asset('assets/images/app_icon.png',
                            fit: BoxFit.cover),
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Text('VNeTraffic',
                        style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.w900,
                            color: Colors.black)),
                    const SizedBox(height: 4),
                    Text(
                      '${s.tr("Phiên bản", "Version")} ${updateService.currentVersion} (Build ${updateService.currentBuildNumber})',
                      style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: Colors.grey),
                    ),
                  ],
                ),
              ),

              // Scrollable Content
              Flexible(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildSectionHeader(Icons.info_outline_rounded,
                          s.tr('Về hệ thống', 'About the System')),
                      Text(
                        s.tr(
                            'VNeTraffic là hệ thống thông minh, tự động phát hiện vi phạm giao thông bằng AI Camera và hỗ trợ công dân tra cứu, nộp phạt, quản lý giấy tờ điện tử.',
                            'VNeTraffic is a smart system that automatically detects traffic violations using AI Camera and helps citizens look up, pay fines, and manage digital documents.'),
                        style: TextStyle(
                            fontSize: 13,
                            color: isDark ? Colors.white70 : Colors.black87,
                            height: 1.5),
                      ),
                      _buildSectionHeader(Icons.star_rounded,
                          s.tr('Điểm nổi bật', 'Key Features')),
                      _buildBullet(s.tr(
                          'Nhận diện vi phạm giao thông bằng AI (YOLO & OpenCV).',
                          'Traffic violation detection via AI (YOLO & OpenCV).')),
                      _buildBullet(s.tr(
                          'Thông báo đẩy (Push Notifications) thời gian thực theo biển số xe vi phạm.',
                          'Real-time push notifications based on license plate violations.')),
                      _buildBullet(s.tr(
                          'Quản lý "Ví giấy tờ" và điểm bằng lái của công dân.',
                          'Digital Document Wallet and driver license points management.')),
                      _buildBullet(s.tr(
                          'Nộp phạt trực tuyến nhanh chóng tiện lợi.',
                          'Fast and convenient online fine payment.')),
                      _buildBullet(s.tr(
                          'Cập nhật ứng dụng tự động (OTA Update) liền mạch.',
                          'Seamless Over-The-Air (OTA) application updates.')),
                      _buildSectionHeader(Icons.code_rounded,
                          s.tr('Công nghệ sử dụng', 'Tech Stack')),
                      Wrap(
                        children: [
                          _buildTechChip('Flutter'),
                          _buildTechChip('Dart'),
                          _buildTechChip('Python'),
                          _buildTechChip('FastAPI'),
                          _buildTechChip('YOLO'),
                          _buildTechChip('OpenCV'),
                          _buildTechChip('Firebase (Auth/Firestore/FCM)'),
                          _buildTechChip('PowerShell / Bash'),
                        ],
                      ),
                      _buildSectionHeader(Icons.developer_mode_rounded,
                          s.tr('Đội ngũ phát triển', 'Development Team')),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: isDark
                              ? Colors.white.withOpacity(0.05)
                              : Colors.grey.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          children: [
                            const CircleAvatar(
                              radius: 20,
                              backgroundColor: AppTheme.primaryColor,
                              child: Icon(Icons.person, color: Colors.white),
                            ),
                            const SizedBox(width: 12),
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text('Khánh Bes',
                                    style: TextStyle(
                                        fontSize: 15,
                                        fontWeight: FontWeight.bold)),
                                Text(
                                    s.tr('Nhà sáng lập & Kỹ sư phát triển',
                                        'Founder & Lead Developer'),
                                    style: const TextStyle(
                                        fontSize: 11, color: Colors.grey)),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // Footer Button
              Padding(
                padding: const EdgeInsets.all(24).copyWith(top: 0),
                child: ElevatedButton(
                  onPressed: () => Navigator.pop(ctx),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryColor,
                    foregroundColor: Colors.white,
                    minimumSize: const Size(double.infinity, 50),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16)),
                    elevation: 0,
                  ),
                  child: Text(s.tr('Đóng', 'Close'),
                      style: const TextStyle(
                          fontWeight: FontWeight.w700, fontSize: 16)),
                ),
              ),
            ],
          ),
        ),
      );
    });
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
              child: const Icon(Icons.language_rounded,
                  color: AppTheme.infoColor, size: 22),
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
                        color: isDark
                            ? const Color(0xFF9E9E9E)
                            : AppTheme.textSecondary,
                      )),
                ],
              ),
            ),
            if (isSelected)
              const Icon(Icons.check_circle_rounded,
                  color: AppTheme.primaryColor, size: 22),
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
    final portController =
        TextEditingController(text: ApiService.serverPort.toString());
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
              child: const Icon(Icons.router_rounded,
                  color: AppTheme.primaryColor, size: 22),
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
                color:
                    isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary,
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
                border:
                    OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide:
                      const BorderSide(color: AppTheme.primaryColor, width: 2),
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
                border:
                    OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide:
                      const BorderSide(color: AppTheme.primaryColor, width: 2),
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
                    ApiService().isConnected
                        ? Icons.check_circle
                        : Icons.error_outline,
                    color: ApiService().isConnected
                        ? AppTheme.successColor
                        : AppTheme.dangerColor,
                    size: 16,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    ApiService().isConnected
                        ? _settings.tr('Đang kết nối', 'Connected')
                        : _settings.tr('Chưa kết nối', 'Not connected'),
                    style: TextStyle(
                      fontSize: 12,
                      color: ApiService().isConnected
                          ? AppTheme.successColor
                          : AppTheme.dangerColor,
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
                        const Icon(Icons.check_circle,
                            color: Colors.white, size: 18),
                        const SizedBox(width: 8),
                        Text(
                            '${_settings.tr("Đã cập nhật IP", "IP updated")}: $ip:$port'),
                      ],
                    ),
                    backgroundColor: AppTheme.successColor,
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                  ),
                );
              }
            },
            icon: const Icon(Icons.save_rounded, size: 18),
            label: Text(_settings.tr('Lưu & Kết nối', 'Save & Connect')),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryColor,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
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
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppTheme.radiusXL)),
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
            onPressed: () async {
              Navigator.pop(context);
              // Sign out from Firebase and reset local state
              await AuthService().signOut();
              _settings.resetOnLogout();
              if (mounted) {
                Navigator.pushNamedAndRemoveUntil(
                    context, '/login', (route) => false);
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.dangerColor,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(AppTheme.radiusM)),
            ),
            child: Text(s.tr('Đăng xuất', 'Sign out')),
          ),
        ],
      ),
    );
  }
}
