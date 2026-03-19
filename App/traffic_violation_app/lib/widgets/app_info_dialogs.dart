import 'package:flutter/material.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/update_service.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';

class AppInfoDialogs {
  static Future<void> showAboutDialog(
      BuildContext context, AppSettings settings) async {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final updateService = UpdateService();
    await updateService.init();
    if (!context.mounted) return;

    Widget buildSectionHeader(IconData icon, String title) {
      return Padding(
        padding: const EdgeInsets.only(top: 18, bottom: 10),
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
                letterSpacing: 0.4,
              ),
            ),
          ],
        ),
      );
    }

    Widget buildBullet(String text) {
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
                  height: 1.45,
                ),
              ),
            ),
          ],
        ),
      );
    }

    showDialog(
      context: context,
      builder: (ctx) => Dialog(
        backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
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
                          offset: const Offset(0, 5),
                        ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: Image.asset('assets/images/app_icon.png',
                          fit: BoxFit.cover),
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'VNeTraffic',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w900,
                      color: Colors.black,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${settings.tr("Phiên bản", "Version")} ${updateService.currentVersion} '
                    '(Build ${updateService.currentBuildNumber})',
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: Colors.grey,
                    ),
                  ),
                ],
              ),
            ),
            Flexible(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    buildSectionHeader(Icons.info_outline_rounded,
                        settings.tr('Về hệ thống', 'About the System')),
                    Text(
                      settings.tr(
                        'VNeTraffic là hệ thống thông minh, tự động phát hiện vi phạm giao thông bằng AI Camera '
                            'và hỗ trợ công dân tra cứu, nộp phạt, quản lý giấy tờ điện tử.',
                        'VNeTraffic is a smart system that automatically detects traffic violations using AI Camera '
                            'and helps citizens look up, pay fines, and manage digital documents.',
                      ),
                      style: TextStyle(
                        fontSize: 13,
                        color: isDark ? Colors.white70 : Colors.black87,
                        height: 1.5,
                      ),
                    ),
                    buildSectionHeader(Icons.star_rounded,
                        settings.tr('Điểm nổi bật', 'Key Features')),
                    buildBullet(settings.tr(
                      'Nhận diện vi phạm giao thông bằng AI (YOLO & OpenCV).',
                      'Traffic violation detection via AI (YOLO & OpenCV).',
                    )),
                    buildBullet(settings.tr(
                      'Thông báo đẩy thời gian thực: vi phạm, hạn đóng phạt, đã đóng phạt, cập nhật ứng dụng.',
                      'Real-time notifications: violation, due date, paid fines, and app updates.',
                    )),
                    buildBullet(settings.tr(
                      'Quản lý Ví giấy tờ và điểm giấy phép lái xe đồng bộ thời gian thực.',
                      'Real-time synced Document Wallet and driver license points management.',
                    )),
                    buildBullet(settings.tr(
                      'Nộp phạt trực tuyến nhanh chóng, tiện lợi.',
                      'Fast and convenient online fine payment.',
                    )),
                    buildSectionHeader(Icons.code_rounded,
                        settings.tr('Công nghệ sử dụng', 'Tech Stack')),
                    Text(
                      'Flutter, Dart, Python, FastAPI, YOLO, OpenCV, Firebase',
                      style: TextStyle(
                        fontSize: 13,
                        color: isDark ? Colors.white70 : Colors.black87,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(24).copyWith(top: 0),
              child: ElevatedButton(
                onPressed: () => Navigator.pop(ctx),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.primaryColor,
                  foregroundColor: Colors.white,
                  minimumSize: const Size(double.infinity, 50),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  elevation: 0,
                ),
                child: Text(
                  settings.tr('Đóng', 'Close'),
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  static Future<void> showPrivacyPolicyDialog(
      BuildContext context, AppSettings settings) async {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final textPrimary = isDark ? const Color(0xFFE0E0E0) : AppTheme.textPrimary;
    final textSecondary =
        isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary;

    Widget section(String title, String body) {
      return Padding(
        padding: const EdgeInsets.only(bottom: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w800,
                color: textPrimary,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              body,
              style: TextStyle(
                fontSize: 13,
                color: textSecondary,
                height: 1.55,
              ),
            ),
          ],
        ),
      );
    }

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.82,
        ),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1E1E1E) : Colors.white,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: Column(
          children: [
            Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.only(top: 12),
              decoration: BoxDecoration(
                color: AppTheme.dividerColor,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 18, 20, 12),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: AppTheme.secondaryColor.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(Icons.shield_outlined,
                        color: AppTheme.secondaryColor),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      settings.tr('Chính sách bảo mật', 'Privacy Policy'),
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        color: textPrimary,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Divider(
              height: 1,
              color: isDark ? const Color(0xFF333333) : AppTheme.dividerColor,
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    section(
                      settings.tr('1. Dữ liệu thu thập', '1. Data Collected'),
                      settings.tr(
                        'Ứng dụng thu thập thông tin tài khoản, thông tin phương tiện, lịch sử vi phạm, '
                            'và thông tin thiết bị phục vụ định danh và gửi thông báo.',
                        'The app collects account data, vehicle data, violation history, and device info '
                            'for identity verification and notifications.',
                      ),
                    ),
                    section(
                      settings.tr(
                          '2. Mục đích sử dụng', '2. Purpose of Processing'),
                      settings.tr(
                        'Dữ liệu được dùng để tra cứu vi phạm, xử lý nộp phạt, đồng bộ ví giấy tờ và cải thiện '
                            'chất lượng dịch vụ.',
                        'Data is used for violation lookup, fine payment processing, document wallet sync, and '
                            'service quality improvements.',
                      ),
                    ),
                    section(
                      settings.tr(
                          '3. Chia sẻ dữ liệu', '3. Data Sharing Policy'),
                      settings.tr(
                        'Ứng dụng không bán dữ liệu cá nhân. Dữ liệu chỉ chia sẻ với cơ quan có thẩm quyền và '
                            'hệ thống kỹ thuật liên quan để xử lý nghiệp vụ.',
                        'The app does not sell personal data. Data is shared only with authorized agencies and '
                            'related technical systems for service operations.',
                      ),
                    ),
                    section(
                      settings.tr('4. Quyền của người dùng', '4. User Rights'),
                      settings.tr(
                        'Người dùng có quyền yêu cầu cập nhật, chỉnh sửa hoặc xóa dữ liệu theo quy định hiện hành.',
                        'Users have the right to request updates, corrections, or deletion of data as permitted by law.',
                      ),
                    ),
                    section(
                      settings.tr('5. Liên hệ', '5. Contact'),
                      settings.tr(
                        'Mọi thắc mắc về bảo mật vui lòng liên hệ bộ phận hỗ trợ trong mục Trợ giúp & Hỗ trợ.',
                        'For privacy concerns, please contact support via Help & Support in the app.',
                      ),
                    ),
                    Text(
                      settings.tr(
                        'Cập nhật lần cuối: 18/03/2026',
                        'Last updated: 18/03/2026',
                      ),
                      style: TextStyle(
                        fontSize: 12,
                        color: textSecondary.withOpacity(0.85),
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 14),
                child: SizedBox(
                  width: double.infinity,
                  height: 48,
                  child: ElevatedButton(
                    onPressed: () => Navigator.pop(ctx),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.primaryColor,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                      elevation: 0,
                    ),
                    child: Text(
                      settings.tr('Đã hiểu', 'Got it'),
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
