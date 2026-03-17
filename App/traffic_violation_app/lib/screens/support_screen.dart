import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:url_launcher/url_launcher.dart';

class SupportScreen extends StatefulWidget {
  const SupportScreen({super.key});

  @override
  State<SupportScreen> createState() => _SupportScreenState();
}

class _SupportScreenState extends State<SupportScreen> {
  final AppSettings _s = AppSettings();

  final List<Map<String, String>> _faqs = [
    {
      'q_vi': 'Làm thế nào để tra cứu vi phạm của tôi?',
      'a_vi': 'Bạn có thể vào mục "Vi phạm" (Violations) trên thanh điều hướng bên dưới để xem danh sách tất cả các vi phạm đã được ghi nhận bằng camera thông minh của hệ thống liên kết với số điện thoại/CCCD của bạn.',
      'q_en': 'How can I lookup my violations?',
      'a_en': 'You can go to the "Violations" tab on the bottom navigation bar to see a list of all violations associated with your account/ID.',
    },
    {
      'q_vi': 'Thanh toán trực tuyến có an toàn không?',
      'a_vi': 'Rất an toàn. Hệ thống áp dụng mã QR chuẩn VietQR tự động kết nối qua Open Banking và mã hóa RSA. Biến động được ghi nhận tự động qua cổng SePay mà không ghi nhớ thông tin ngân hàng của bạn.',
      'q_en': 'Is online payment safe?',
      'a_en': 'Very safe. The system uses VietQR standards automatically connecting via Open Banking and RSA encryption. We do not store any of your bank details.',
    },
    {
      'q_vi': 'Tôi phải làm gì nếu thông tin xe không đúng?',
      'a_vi': 'Vui lòng truy cập mục "Phương tiện", chọn "Sửa thông tin" hoặc bấm "Báo cáo lỗi" để gửi phiếu yêu cầu đính chính đến CSGT địa phương.',
      'q_en': 'What if my vehicle information is incorrect?',
      'a_en': 'Please go to the "Vehicles" section to edit or send an error report to the local traffic authorities.',
    },
    {
      'q_vi': 'Làm sao để tôi khiếu nại một lỗi sai?',
      'a_vi': 'Để khiếu nại, bạn ấn vào chi tiết vi phạm đó, cuộn xuống dưới cùng và chọn nút chức năng "Khiếu nại/Góp ý". Hệ thống sẽ tạo một mẫu đơn cho bạn.',
      'q_en': 'How do I appeal a violation?',
      'a_en': 'To appeal, click on the specific violation, scroll to the bottom and select the "Appeal/Report" button. The system will create form for you.',
    },
  ];

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    if (!await launchUrl(uri)) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(_s.tr('Không thể mở liên kết này.', 'Could not launch URL'))),
        );
      }
    }
  }

  Future<void> _launchPhone(String phone) async {
    final uri = Uri.parse('tel:$phone');
    if (!await launchUrl(uri)) {
      // ignore
    }
  }

  Future<void> _launchEmail(String email) async {
    final uri = Uri.parse('mailto:$email?subject=Hỗ trợ ứng dụng VNETraffic');
    if (!await launchUrl(uri)) {
      // ignore
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: AppTheme.surfaceColor,
      appBar: AppBar(
        title: Text(
          _s.tr('Trợ giúp & Hỗ trợ', 'Help & Support'),
          style: const TextStyle(fontWeight: FontWeight.w700),
        ),
        backgroundColor: AppTheme.primaryColor,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // ── HERO HEADER ──
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(24, 16, 24, 40),
              decoration: const BoxDecoration(
                color: AppTheme.primaryColor,
                borderRadius: BorderRadius.only(
                  bottomLeft: Radius.circular(32),
                  bottomRight: Radius.circular(32),
                ),
              ),
              child: Column(
                children: [
                  const Icon(Icons.support_agent_rounded, size: 72, color: Colors.white),
                  const SizedBox(height: 16),
                  Text(
                    _s.tr('Xin chào, chúng tôi có thể giúp gì?', 'Hi, how can we help you?'),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.w800,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _s.tr('Tìm kiếm câu trả lời nhanh qua Danh mục hoặc Liên hệ tổng đài 24/7.', 'Find quick answers or contact our 24/7 hotline.'),
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.8),
                      fontSize: 14,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── CONTACT CARDS ──
                  Text(
                    _s.tr('Liên hệ trực tiếp', 'Contact Us'),
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: _buildContactCard(
                          icon: Icons.headset_mic_rounded,
                          title: 'Hotline',
                          subtitle: '1900 8888',
                          color: AppTheme.primaryColor,
                          onTap: () => _launchPhone('19008888'),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: _buildContactCard(
                          icon: Icons.email_rounded,
                          title: 'Email',
                          subtitle: 'vne@police.vn',
                          color: AppTheme.infoColor,
                          onTap: () => _launchEmail('vne@police.vn'),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 32),

                  // ── FAQ SECTION ──
                  Text(
                    _s.tr('Câu hỏi thường gặp (FAQ)', 'Frequently Asked Questions'),
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 12),
                  
                  Container(
                    decoration: BoxDecoration(
                      color: isDark ? const Color(0xFF1E1E1E) : Colors.white,
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: isDark ? [] : AppTheme.cardShadow,
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(16),
                      child: ListView.separated(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: _faqs.length,
                        separatorBuilder: (_, __) => Divider(
                          color: isDark ? Colors.white12 : Colors.black12,
                          height: 1,
                        ),
                        itemBuilder: (context, index) {
                          final faq = _faqs[index];
                          return Theme(
                            data: Theme.of(context).copyWith(
                              dividerColor: Colors.transparent, // remove expansion borders
                            ),
                            child: ExpansionTile(
                              iconColor: AppTheme.primaryColor,
                              textColor: AppTheme.primaryColor,
                              collapsedIconColor: isDark ? Colors.white70 : Colors.black54,
                              title: Text(
                                _s.isVietnamese ? faq['q_vi']! : faq['q_en']!,
                                style: const TextStyle(
                                  fontWeight: FontWeight.w600,
                                  fontSize: 15,
                                ),
                              ),
                              childrenPadding: const EdgeInsets.only(left: 16, right: 16, bottom: 16),
                              children: [
                                Text(
                                  _s.isVietnamese ? faq['a_vi']! : faq['a_en']!,
                                  style: TextStyle(
                                    color: isDark ? Colors.white70 : AppTheme.textSecondary,
                                    height: 1.5,
                                  ),
                                ),
                              ],
                            ),
                          );
                        },
                      ),
                    ),
                  ),

                  const SizedBox(height: 40),

                  // ── SUBMIT TICKET ──
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFF263238), Color(0xFF37474F)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.2),
                          blurRadius: 10,
                          offset: const Offset(0, 4),
                        )
                      ],
                    ),
                    child: Column(
                      children: [
                        const Icon(Icons.rate_review_rounded, color: Colors.white, size: 40),
                        const SizedBox(height: 12),
                        Text(
                          _s.tr('Bạn vẫn cần hỗ trợ thêm?', 'Still need more help?'),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _s.tr('Gửi yêu cầu hỗ trợ (Ticket) đến chuyên viên giải quyết khiếu nại của chúng tôi.', 'Submit a ticket to our complaint resolving specialist.'),
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.8),
                            fontSize: 12,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            onPressed: () {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text(_s.tr('Tính năng đang được phát triển.', 'Feature is in development.'))),
                              );
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.white,
                              foregroundColor: const Color(0xFF263238),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                            ),
                            child: Text(
                              _s.tr('Tạo Ticket Hỗ Trợ', 'Create Support Ticket'),
                              style: const TextStyle(fontWeight: FontWeight.w800),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 40),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildContactCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
    required VoidCallback onTap,
  }) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1E1E1E) : Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: isDark ? [] : AppTheme.cardShadow,
          border: Border.all(color: color.withOpacity(0.2)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: color, size: 28),
            ),
            const SizedBox(height: 16),
            Text(
              title,
              style: TextStyle(
                color: isDark ? Colors.white70 : AppTheme.textSecondary,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              subtitle,
              style: TextStyle(
                color: isDark ? Colors.white : AppTheme.textPrimary,
                fontSize: 16,
                fontWeight: FontWeight.w800,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
