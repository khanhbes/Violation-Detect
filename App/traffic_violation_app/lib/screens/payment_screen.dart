import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:intl/intl.dart';
import 'package:qr_flutter/qr_flutter.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

class PaymentScreen extends StatefulWidget {
  const PaymentScreen({super.key});

  @override
  State<PaymentScreen> createState() => _PaymentScreenState();
}

class _PaymentScreenState extends State<PaymentScreen>
    with SingleTickerProviderStateMixin {
  int _selectedMethod = 0;
  bool _isProcessing = false;
  bool _paymentDone = false;
  final AppSettings _s = AppSettings();

  late AnimationController _animController;
  late Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    _fadeAnim = CurvedAnimation(parent: _animController, curve: Curves.easeOut);
    _animController.forward();
  }

  @override
  void dispose() {
    _animController.dispose();
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
                child: const Icon(Icons.error_outline, size: 36, color: AppTheme.textSecondary),
              ),
              const SizedBox(height: 16),
              Text(_s.tr('Không tìm thấy thông tin vi phạm', 'Violation info not found'), style: const TextStyle(color: AppTheme.textSecondary)),
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
        title: Text(_s.tr('Nộp phạt', 'Payment')),
        backgroundColor: AppTheme.primaryColor,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: FadeTransition(
        opacity: _fadeAnim,
        child: SingleChildScrollView(
          child: Column(
            children: [
              // ── Fine Summary Card ─────────────────────────
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
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(AppTheme.radiusXL),
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
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          violation.violationType,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // ── Violation Info ─────────────────────────
                    Text(
                      _s.tr('Thông tin vi phạm', 'Violation info'),
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(AppTheme.radiusL),
                        boxShadow: AppTheme.cardShadow,
                      ),
                      child: Column(
                        children: [
                          _buildInfoRow(Icons.qr_code_rounded, _s.tr('Mã vi phạm', 'Violation code'), violation.violationCode.isNotEmpty ? violation.violationCode : violation.id),
                          Container(margin: const EdgeInsets.symmetric(horizontal: 16), height: 1, color: AppTheme.dividerColor),
                          _buildInfoRow(Icons.directions_car_rounded, _s.tr('Biển số', 'License plate'), violation.licensePlate),
                          Container(margin: const EdgeInsets.symmetric(horizontal: 16), height: 1, color: AppTheme.dividerColor),
                          _buildInfoRow(Icons.location_on_rounded, _s.tr('Địa điểm', 'Location'), violation.location.isNotEmpty ? violation.location : _s.tr('Camera giám sát', 'Surveillance camera')),
                        ],
                      ),
                    ),

                    const SizedBox(height: 24),

                    // ── Payment Method ────────────────────────
                    Text(
                      _s.tr('Phương thức thanh toán', 'Payment method'),
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 10),
                    _buildPaymentMethod(0, Icons.qr_code_2_rounded, 'QR Code', _s.tr('Quét mã QR để thanh toán', 'Scan QR to pay'), AppTheme.primaryColor),
                    const SizedBox(height: 8),
                    _buildPaymentMethod(1, Icons.account_balance_rounded, _s.tr('Chuyển khoản', 'Bank transfer'), _s.tr('Chuyển khoản ngân hàng', 'Bank transfer payment'), AppTheme.infoColor),
                    const SizedBox(height: 8),
                    _buildPaymentMethod(2, Icons.credit_card_rounded, _s.tr('Thẻ ngân hàng', 'Bank card'), 'Visa, Mastercard, JCB', AppTheme.secondaryColor),

                    const SizedBox(height: 24),

                    // ── QR Code Section ───────────────────────
                    if (_selectedMethod == 0) _buildQRSection(violation, formatter),

                    // ── Bank Transfer Section ─────────────────
                    if (_selectedMethod == 1) _buildBankSection(violation, formatter),

                    const SizedBox(height: 24),

                    // ── Pay Button ────────────────────────────
                    SizedBox(
                      width: double.infinity,
                      height: 54,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: AppTheme.primaryGradient,
                          borderRadius: BorderRadius.circular(AppTheme.radiusM),
                          boxShadow: AppTheme.redShadow,
                        ),
                        child: ElevatedButton.icon(
                          onPressed: _isProcessing ? null : () => _processPayment(),
                          icon: _isProcessing
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                                )
                              : const Icon(Icons.payment_rounded),
                          label: Text(
                            _isProcessing ? _s.tr('Đang xử lý...', 'Processing...') : _s.tr('Xác nhận thanh toán', 'Confirm payment'),
                            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.transparent,
                            foregroundColor: Colors.white,
                            shadowColor: Colors.transparent,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(AppTheme.radiusM),
                            ),
                          ),
                        ),
                      ),
                    ),

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

  Widget _buildInfoRow(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: [
          Icon(icon, size: 18, color: AppTheme.textSecondary),
          const SizedBox(width: 10),
          Text(label, style: const TextStyle(fontSize: 13, color: AppTheme.textSecondary)),
          const Spacer(),
          Flexible(
            child: Text(
              value,
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: AppTheme.textPrimary),
              textAlign: TextAlign.right,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPaymentMethod(int index, IconData icon, String title, String subtitle, Color color) {
    final isSelected = _selectedMethod == index;

    return GestureDetector(
      onTap: () => setState(() => _selectedMethod = index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          border: Border.all(
            color: isSelected ? color : AppTheme.dividerColor,
            width: isSelected ? 2 : 1,
          ),
          boxShadow: isSelected ? [BoxShadow(color: color.withOpacity(0.12), blurRadius: 10, offset: const Offset(0, 4))] : null,
        ),
        child: Row(
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: color.withOpacity(isSelected ? 0.12 : 0.06),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: color, size: 22),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                      color: isSelected ? color : AppTheme.textPrimary,
                    ),
                  ),
                  Text(
                    subtitle,
                    style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary),
                  ),
                ],
              ),
            ),
            Icon(
              isSelected ? Icons.check_circle_rounded : Icons.radio_button_off_rounded,
              color: isSelected ? color : AppTheme.textHint,
              size: 22,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildQRSection(Violation v, NumberFormat fmt) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(AppTheme.radiusXL),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Column(
        children: [
          Text(
            _s.tr('Quét mã QR để thanh toán', 'Scan QR code to pay'),
            style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(AppTheme.radiusL),
              border: Border.all(color: AppTheme.dividerColor),
            ),
            child: QrImageView(
              data: 'VIOLATION|${v.id}|${v.fineAmount}|${v.licensePlate}',
              version: QrVersions.auto,
              size: 180,
              eyeStyle: const QrEyeStyle(
                eyeShape: QrEyeShape.square,
                color: AppTheme.primaryDark,
              ),
              dataModuleStyle: const QrDataModuleStyle(
                dataModuleShape: QrDataModuleShape.square,
                color: AppTheme.textPrimary,
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            fmt.format(v.fineAmount),
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w800,
              color: AppTheme.primaryColor,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            '${_s.tr('Mã', 'Code')}: ${v.id}',
            style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary),
          ),
        ],
      ),
    );
  }

  Widget _buildBankSection(Violation v, NumberFormat fmt) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(AppTheme.radiusXL),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _s.tr('Thông tin chuyển khoản', 'Bank transfer info'),
            style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: AppTheme.textPrimary),
          ),
          const SizedBox(height: 14),
          _buildBankRow(_s.tr('Ngân hàng', 'Bank'), 'Vietcombank'),
          _buildBankRow(_s.tr('Số TK', 'Account'), '1234 5678 9012'),
          _buildBankRow(_s.tr('Chủ TK', 'Owner'), _s.tr('KHO BẠC NHÀ NƯỚC', 'STATE TREASURY')),
          _buildBankRow(_s.tr('Nội dung', 'Content'), 'NP ${v.id}'),
          _buildBankRow(_s.tr('Số tiền', 'Amount'), fmt.format(v.fineAmount)),
          const SizedBox(height: 10),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () {
                Clipboard.setData(ClipboardData(text: 'NP ${v.id}'));
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(_s.tr('Đã sao chép nội dung chuyển khoản', 'Transfer content copied')),
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                );
              },
              icon: const Icon(Icons.copy_rounded, size: 16),
              label: Text(_s.tr('Sao chép nội dung CK', 'Copy transfer content')),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBankRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 13, color: AppTheme.textSecondary)),
          Text(value, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
        ],
      ),
    );
  }

  void _processPayment() async {
    setState(() => _isProcessing = true);
    await Future.delayed(const Duration(seconds: 2));
    if (mounted) {
      setState(() {
        _isProcessing = false;
        _paymentDone = true;
      });
    }
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
                  style: const TextStyle(fontSize: 14, color: AppTheme.textSecondary),
                ),
                const SizedBox(height: 32),
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
                    },
                    child: Text(_s.tr('Về trang chủ', 'Go to home')),
                  ),
                ),
                const SizedBox(height: 12),
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: Text(
                    _s.tr('Xem lại vi phạm', 'View violation'),
                    style: const TextStyle(color: AppTheme.primaryColor, fontWeight: FontWeight.w600),
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
