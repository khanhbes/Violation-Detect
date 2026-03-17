import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:intl/intl.dart';
import 'package:qr_flutter/qr_flutter.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class PaymentScreen extends StatefulWidget {
  const PaymentScreen({super.key});

  static final Map<String, DateTime> paymentStarts = {};

  static bool isProcessing(String violationId) {
    if (!paymentStarts.containsKey(violationId)) return false;
    final start = paymentStarts[violationId]!;
    final elapsed = DateTime.now().difference(start).inSeconds;
    if (elapsed >= 300) {
      paymentStarts.remove(violationId);
      return false;
    }
    return true;
  }

  @override
  State<PaymentScreen> createState() => _PaymentScreenState();
}

class _PaymentScreenState extends State<PaymentScreen>
    with SingleTickerProviderStateMixin {
  bool _isProcessing = false;
  bool _paymentDone = false;
  final AppSettings _s = AppSettings();

  // ── Timer State ──
  Timer? _timer;
  int _remainingSeconds = 300;

  late AnimationController _animController;
  late Animation<double> _fadeAnim;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments;
    if (args is Violation && !_paymentDone) {
      _initTimer(args.id);
    }
  }

  void _initTimer(String violationId) {
    if (_timer != null) return; // already initialized

    if (!PaymentScreen.paymentStarts.containsKey(violationId)) {
      PaymentScreen.paymentStarts[violationId] = DateTime.now();
    }
    
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      
      final start = PaymentScreen.paymentStarts[violationId]!;
      final elapsed = DateTime.now().difference(start).inSeconds;
      final remaining = 300 - elapsed;
      
      if (remaining <= 0) {
        timer.cancel();
        PaymentScreen.paymentStarts.remove(violationId);
        if (mounted && Navigator.canPop(context)) {
          Navigator.pop(context);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(_s.tr('Phiên thanh toán đã hết hạn (5 phút)', 'Payment session expired')),
              backgroundColor: AppTheme.dangerColor,
            ),
          );
        }
      } else {
        setState(() {
          _remainingSeconds = remaining;
        });
      }
    });

    setState(() {
      final start = PaymentScreen.paymentStarts[violationId]!;
      _remainingSeconds = 300 - DateTime.now().difference(start).inSeconds;
    });
  }

  String get _formattedTime {
    final m = (_remainingSeconds / 60).floor();
    final s = _remainingSeconds % 60;
    return '${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 700));
    _fadeAnim = CurvedAnimation(parent: _animController, curve: Curves.easeOut);
    _animController.forward();
  }

  @override
  void dispose() {
    _animController.dispose();
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final args = ModalRoute.of(context)?.settings.arguments;
    if (args == null || args is! Violation) {
      return Scaffold(
        appBar: AppBar(
          title: Text(_s.tr('Nộp phạt', 'Payment')),
          backgroundColor: AppTheme.primaryColor,
          foregroundColor: Colors.white,
        ),
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  color: AppTheme.textHint.withOpacity(0.15),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.error_outline,
                    size: 36, color: AppTheme.textSecondary),
              ),
              const SizedBox(height: 16),
              Text(
                _s.tr('Không tìm thấy thông tin vi phạm',
                    'Violation info not found'),
                style: const TextStyle(color: AppTheme.textSecondary),
              ),
            ],
          ),
        ),
      );
    }

    final violation = args;
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    if (_paymentDone) return _buildSuccessPage(violation, formatter);

    return Scaffold(
      backgroundColor: AppTheme.surfaceColor,
      appBar: AppBar(
        title: Text(_s.tr('Thanh toán Vi phạm', 'Violation Payment')),
        backgroundColor: AppTheme.primaryColor,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: FadeTransition(
        opacity: _fadeAnim,
        child: SingleChildScrollView(
          child: Column(
            children: [
              // ── Fine Summary ─────────────────────────
              Container(
                width: double.infinity,
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
                decoration: const BoxDecoration(
                  color: AppTheme.primaryColor,
                  borderRadius: BorderRadius.only(
                    bottomLeft: Radius.circular(28),
                    bottomRight: Radius.circular(28),
                  ),
                ),
                child: Column(
                  children: [
                    Text(
                      _s.tr('Số tiền phạt', 'Fine amount'),
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.8),
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      formatter.format(violation.fineAmount),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 32,
                        fontWeight: FontWeight.w800,
                        letterSpacing: -0.5,
                      ),
                    ),
                  ],
                ),
              ),

              Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    // ── Professional QR Code Section ─
                    _buildProfessionalQR(violation, formatter),

                    const SizedBox(height: 32),

                    // ── Pay Button (Auto Verification) ─
                    // ── Auto Verification Listener ─
                    StreamBuilder<DocumentSnapshot>(
                      stream: FirebaseFirestore.instance.collection('violations').doc(violation.id).snapshots(),
                      builder: (context, snapshot) {
                        if (snapshot.hasData && snapshot.data != null && snapshot.data!.exists) {
                          final data = snapshot.data!.data() as Map<String, dynamic>?;
                          if (data != null && data['status'] == 'paid' && !_paymentDone) {
                            WidgetsBinding.instance.addPostFrameCallback((_) {
                              if (mounted) {
                                setState(() {
                                  _paymentDone = true;
                                });
                              }
                            });
                          }
                        }

                        return Container(
                          width: double.infinity,
                          padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
                          decoration: BoxDecoration(
                            color: AppTheme.infoColor.withOpacity(0.08),
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(color: AppTheme.infoColor.withOpacity(0.3)),
                          ),
                          child: Column(
                            children: [
                              const SizedBox(
                                width: 28,
                                height: 28,
                                child: CircularProgressIndicator(strokeWidth: 2.5, color: AppTheme.infoColor),
                              ),
                              const SizedBox(height: 16),
                              Text(
                                _s.tr('Hệ thống đang kiểm tra giao dịch...', 'Waiting for money transfer...'),
                                style: const TextStyle(fontWeight: FontWeight.w700, color: AppTheme.infoColor, fontSize: 16),
                              ),
                              const SizedBox(height: 6),
                                Text(
                                  _s.tr('Trang này sẽ tự động chuyển hướng khi hệ thống ghi nhận bạn đã thanh toán thành công (thường mất 10-30 giây).', 'This page will auto redirect upon successful transfer.'),
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: AppTheme.textSecondary.withOpacity(0.8),
                                  ),
                                ),
                                const SizedBox(height: 12),
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                                  decoration: BoxDecoration(
                                    color: AppTheme.dangerColor.withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Text(
                                    '⏳ Hủy giao dịch trong: $_formattedTime',
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color: AppTheme.dangerColor,
                                    ),
                                  ),
                                ),
                              ],
                          ),
                        );
                      },
                    ),
                    const SizedBox(height: 16),

                    const SizedBox(height: 32),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _removeDiacritics(String str) {
    const withDia = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ';
    const withoutDia = 'aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyydAAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD';
    for (int i = 0; i < withDia.length; i++) {
      str = str.replaceAll(withDia[i], withoutDia[i]);
    }
    return str;
  }

  Widget _buildProfessionalQR(Violation v, NumberFormat fmt) {
    // Determine a safe string for the name
    final String rawName = _s.userName.trim().isNotEmpty ? _s.userName : 'VNETRAFFIC_USER';
    final String cleanName = _removeDiacritics(rawName);
    final safeNameRegex = cleanName.replaceAll(RegExp(r'[^a-zA-Z0-9]'), '');
    final String contentMsg = 'NP${v.id}$safeNameRegex'.toUpperCase();

    // VietQR dynamic endpoint that automatically sets bank details + amount + message
    final String encodedMsg = Uri.encodeComponent(contentMsg);
    final String accountNameEncoded = Uri.encodeComponent('PHAN NAM KHANH');
    // Using VietQR compact API
    final String qrImageUrl =
        'https://img.vietqr.io/image/MB-0852232174-compact2.png?amount=${v.fineAmount}&addInfo=$encodedMsg&accountName=$accountNameEncoded';

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        children: [
          // White professional top for QR
          Container(
            padding: const EdgeInsets.all(24),
            decoration: const BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
            ),
            child: Column(
              children: [
                Text(
                  _s.tr('Quét mã qua Ứng dụng Ngân hàng',
                      'Scan via Banking App'),
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _s.tr('Số tiền & Nội dung sẽ được nhập tự động',
                      'Amount & Content will be auto-filled'),
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppTheme.successColor,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 24),
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.network(
                    qrImageUrl,
                    width: 260,
                    height: 260,
                    fit: BoxFit.contain,
                    loadingBuilder: (ctx, child, progress) {
                      if (progress == null) return child;
                      return const SizedBox(
                        width: 260,
                        height: 260,
                        child: Center(
                          child: CircularProgressIndicator(
                              color: AppTheme.primaryColor),
                        ),
                      );
                    },
                    errorBuilder: (ctx, err, stack) {
                      // Fallback QR if network fails to fetch from vietqr.io
                      return QrImageView(
                        data:
                            'VIETQR|MB|0852232174|${v.fineAmount}|$contentMsg',
                        version: QrVersions.auto,
                        size: 260,
                      );
                    },
                  ),
                ),
              ],
            ),
          ),

          // Separator line
          Container(height: 1, color: AppTheme.dividerColor),

          // Info below QR code
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: const BoxDecoration(
              color: Color(0xFFFAFAFA),
              borderRadius: BorderRadius.vertical(bottom: Radius.circular(24)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildDataRow(
                    _s.tr('Tên tài khoản', 'Account Name'), 'PHAN NAM KHANH',
                    isBold: true),
                _buildDataRow(
                    _s.tr('Số tài khoản', 'Account Number'), '0852232174',
                    copy: true),
                _buildDataRow(_s.tr('Ngân hàng', 'Bank'), 'MB Bank'),
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 8),
                  child: Divider(height: 1, color: AppTheme.dividerColor),
                ),
                _buildDataRow(_s.tr('Số tiền nộp', 'Transfer Amount'),
                    fmt.format(v.fineAmount),
                    isBold: true, valueColor: AppTheme.dangerColor),
                _buildDataRow(
                    _s.tr('Nội dung CK', 'Transfer Content'), contentMsg,
                    copy: true),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDataRow(String label, String value,
      {bool copy = false, bool isBold = false, Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 95,
            child: Text(
              label,
              style:
                  const TextStyle(fontSize: 13, color: AppTheme.textSecondary),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                fontSize: 14,
                fontWeight: isBold ? FontWeight.w700 : FontWeight.w500,
                color: valueColor ?? AppTheme.textPrimary,
              ),
            ),
          ),
          if (copy)
            GestureDetector(
              onTap: () {
                Clipboard.setData(ClipboardData(text: value));
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Row(
                      children: [
                        const Icon(Icons.check_circle,
                            color: Colors.white, size: 18),
                        const SizedBox(width: 8),
                        Text('${_s.tr("Đã sao chép", "Copied")} $label'),
                      ],
                    ),
                    backgroundColor: AppTheme.successColor,
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                  ),
                );
              },
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: AppTheme.primaryColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: const Text('COPY',
                    style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.primaryColor)),
              ),
            ),
        ],
      ),
    );
  }



  Widget _buildSuccessPage(Violation v, NumberFormat fmt) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0, end: 1),
                  duration: const Duration(milliseconds: 600),
                  curve: Curves.elasticOut,
                  builder: (context, anim, child) {
                    return Transform.scale(scale: anim, child: child);
                  },
                  child: Container(
                    width: 100,
                    height: 100,
                    decoration: const BoxDecoration(
                      color: Color(0xFFE8F5E9),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.check_circle_rounded,
                      size: 60,
                      color: AppTheme.successColor,
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                Text(
                  _s.tr('Thanh toán thành công!', 'Payment successful!'),
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w800,
                    color: AppTheme.successColor,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  fmt.format(v.fineAmount),
                  style: const TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  v.violationType,
                  style: const TextStyle(
                      fontSize: 14, color: AppTheme.textSecondary),
                ),
                const SizedBox(height: 32),
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pushNamedAndRemoveUntil(
                          context, '/home', (route) => false);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.primaryColor,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                    ),
                    child: Text(_s.tr('Về trang chủ', 'Go to home'),
                        style: const TextStyle(fontWeight: FontWeight.w700)),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
