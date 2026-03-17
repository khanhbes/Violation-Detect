import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/screens/payment_screen.dart';

class ViolationDetailScreen extends StatefulWidget {
  const ViolationDetailScreen({super.key});

  @override
  State<ViolationDetailScreen> createState() => _ViolationDetailScreenState();
}

class _ViolationDetailScreenState extends State<ViolationDetailScreen>
    with SingleTickerProviderStateMixin {
  final AppSettings _settings = AppSettings();
  late AnimationController _animController;
  late Animation<double> _fadeAnim;
  late Animation<Offset> _slideAnim;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    _fadeAnim = CurvedAnimation(parent: _animController, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(
      begin: const Offset(0, 0.12),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _animController, curve: Curves.easeOutCubic));

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
          title: Text(_settings.tr('Chi tiết vi phạm', 'Violation Detail')),
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
              Text(
                _settings.tr('Không tìm thấy thông tin vi phạm', 'Violation info not found'),
                style: const TextStyle(fontSize: 16, color: AppTheme.textSecondary),
              ),
            ],
          ),
        ),
      );
    }

    final violation = args;
    final df = DateFormat('HH:mm — dd/MM/yyyy');
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    return Scaffold(
      backgroundColor: AppTheme.surfaceColor,
      body: CustomScrollView(
        slivers: [
          // ── Image AppBar ──────────────────────────────────
          SliverAppBar(
            expandedHeight: 260,
            pinned: true,
            backgroundColor: AppTheme.primaryColor,
            foregroundColor: Colors.white,
            flexibleSpace: FlexibleSpaceBar(
              background: Hero(
                tag: 'violation_image_${violation.id}',
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.network(
                      violation.imageUrl,
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => Container(
                        color: Colors.grey[300],
                        child: const Icon(Icons.image_not_supported, size: 64, color: Colors.grey),
                      ),
                    ),
                    // Gradient overlay
                    Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.transparent,
                            Colors.black.withOpacity(0.7),
                          ],
                        ),
                      ),
                    ),
                    // Bottom info
                    Positioned(
                      bottom: 16,
                      left: 16,
                      right: 16,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _buildStatusBadge(violation),
                          const SizedBox(height: 8),
                          Text(
                            violation.violationType,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 22,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Row(
                            children: [
                              const Icon(Icons.access_time_rounded, color: Colors.white70, size: 14),
                              const SizedBox(width: 4),
                              Text(
                                df.format(violation.timestamp),
                                style: const TextStyle(color: Colors.white70, fontSize: 13),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // ── Content ──────────────────────────────────────
          SliverToBoxAdapter(
            child: FadeTransition(
              opacity: _fadeAnim,
              child: SlideTransition(
                position: _slideAnim,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // ── Fine Amount Card ────────────────────────
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Color(0xFFD32F2F), Color(0xFFB71C1C)],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(AppTheme.radiusXL),
                          boxShadow: AppTheme.redShadow,
                        ),
                        child: Row(
                          children: [
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Row(
                                    children: [
                                      Container(
                                        padding: const EdgeInsets.all(6),
                                        decoration: BoxDecoration(
                                          color: Colors.white.withOpacity(0.15),
                                          borderRadius: BorderRadius.circular(8),
                                        ),
                                        child: const Icon(Icons.monetization_on_rounded, color: Colors.white, size: 18),
                                      ),
                                      const SizedBox(width: 8),
                                      Text(
                                        _settings.tr('Mức tiền phạt', 'Fine amount'),
                                        style: const TextStyle(color: Colors.white70, fontSize: 13),
                                      ),
                                    ],
                                  ),
                                  const SizedBox(height: 10),
                                  Text(
                                    formatter.format(violation.fineAmount),
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 30,
                                      fontWeight: FontWeight.w800,
                                      letterSpacing: -0.5,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            if (violation.isPending)
                              Container(
                                padding: const EdgeInsets.all(10),
                                decoration: BoxDecoration(
                                  color: Colors.white.withOpacity(0.2),
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: const Icon(Icons.payment_rounded, color: Colors.white, size: 28),
                              ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 24),

                      // ── Info Cards ──────────────────────────────
                      Text(
                        _settings.tr('Thông tin vi phạm', 'Violation info'),
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.textPrimary,
                        ),
                      ),
                      const SizedBox(height: 14),
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(AppTheme.radiusL),
                          boxShadow: AppTheme.cardShadow,
                        ),
                        child: Column(
                          children: [
                            _buildInfoRow(
                              Icons.access_time_rounded,
                              AppTheme.infoColor,
                              _settings.tr('Thời gian', 'Time'),
                              df.format(violation.timestamp),
                            ),
                            _divider(),
                            _buildInfoRow(
                              Icons.location_on_rounded,
                              AppTheme.successColor,
                              _settings.tr('Địa điểm', 'Location'),
                              violation.location.isNotEmpty
                                  ? violation.location
                                  : _settings.tr('Camera giám sát giao thông', 'Traffic surveillance camera'),
                            ),
                            _divider(),
                            _buildInfoRow(
                              Icons.directions_car_rounded,
                              AppTheme.secondaryColor,
                              _settings.tr('Biển số xe', 'License plate'),
                              violation.licensePlate,
                            ),
                            _divider(),
                            _buildInfoRow(
                              Icons.warning_amber_rounded,
                              AppTheme.primaryColor,
                              _settings.tr('Loại vi phạm', 'Violation type'),
                              violation.violationType,
                            ),
                            if (violation.violationCode.isNotEmpty) ...[
                              _divider(),
                              _buildInfoRow(
                                Icons.qr_code_rounded,
                                Colors.purple,
                                _settings.tr('Mã vi phạm', 'Violation code'),
                                violation.violationCode,
                              ),
                            ],
                          ],
                        ),
                      ),

                      const SizedBox(height: 20),

                      // ── Description ─────────────────────────────
                      if (violation.description.isNotEmpty) ...[
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: AppTheme.infoColor.withOpacity(0.05),
                            borderRadius: BorderRadius.circular(AppTheme.radiusL),
                            border: Border.all(color: AppTheme.infoColor.withOpacity(0.15)),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Container(
                                    padding: const EdgeInsets.all(6),
                                    decoration: BoxDecoration(
                                      color: AppTheme.infoColor.withOpacity(0.1),
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: const Icon(Icons.description_rounded, size: 16, color: AppTheme.infoColor),
                                  ),
                                  const SizedBox(width: 8),
                                  Text(
                                    _settings.tr('Mô tả vi phạm', 'Violation description'),
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w600,
                                      color: AppTheme.infoColor,
                                      fontSize: 14,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 10),
                              Text(
                                violation.description,
                                style: const TextStyle(
                                  fontSize: 14,
                                  color: AppTheme.textPrimary,
                                  height: 1.5,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                      ],

                      // ── Law Reference ───────────────────────────
                      if (violation.lawReference.isNotEmpty)
                        _buildLawSection(violation),

                      const SizedBox(height: 24),

                      // ── Action Buttons ──────────────────────────
                      if (violation.isPending) ...[
                        // Pay button
                        SizedBox(
                          width: double.infinity,
                          height: 52,
                          child: Container(
                            decoration: BoxDecoration(
                              gradient: AppTheme.primaryGradient,
                              borderRadius: BorderRadius.circular(AppTheme.radiusM),
                              boxShadow: AppTheme.redShadow,
                            ),
                            child: ElevatedButton.icon(
                                onPressed: () {
                                  Navigator.pushNamed(context, '/payment', arguments: violation).then((_) {
                                    if (mounted) setState(() {});
                                  });
                                },
                                icon: const Icon(Icons.payment_rounded),
                                label: Text(
                                  PaymentScreen.isProcessing(violation.id) 
                                      ? _settings.tr('Tiếp tục nộp phạt', 'Continue payment')
                                      : _settings.tr('Nộp phạt ngay', 'Pay fine now'),
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
                        const SizedBox(height: 12),
                        // Complaint button
                        SizedBox(
                          width: double.infinity,
                          height: 52,
                          child: OutlinedButton.icon(
                            onPressed: () {
                              Navigator.pushNamed(context, '/complaint');
                            },
                            icon: const Icon(Icons.rate_review_rounded),
                            label: Text(
                              _settings.tr('Khiếu nại vi phạm', 'File complaint'),
                              style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                            ),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppTheme.primaryColor,
                              side: const BorderSide(color: AppTheme.primaryColor, width: 1.5),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(AppTheme.radiusM),
                              ),
                            ),
                          ),
                        ),
                      ],

                      if (violation.isPaid)
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: AppTheme.successColor.withOpacity(0.08),
                            borderRadius: BorderRadius.circular(AppTheme.radiusL),
                            border: Border.all(color: AppTheme.successColor.withOpacity(0.25)),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Container(
                                padding: const EdgeInsets.all(6),
                                decoration: BoxDecoration(
                                  color: AppTheme.successColor.withOpacity(0.15),
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(Icons.check_rounded, color: AppTheme.successColor, size: 20),
                              ),
                              const SizedBox(width: 10),
                              Text(
                                _settings.tr('Đã nộp phạt', 'Fine paid'),
                                style: const TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                  color: AppTheme.successColor,
                                ),
                              ),
                            ],
                          ),
                        ),

                      const SizedBox(height: 32),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusBadge(Violation v) {
    final bool isProcessing = PaymentScreen.isProcessing(v.id);
    final Color bgColor = v.isPaid ? AppTheme.successColor : (isProcessing ? AppTheme.infoColor : AppTheme.dangerColor);
    final IconData iconData = v.isPaid ? Icons.check_circle_rounded : (isProcessing ? Icons.hourglass_top_rounded : Icons.warning_rounded);
    final String actText = v.isPaid 
        ? _settings.tr('Đã nộp phạt', 'Paid') 
        : (isProcessing ? _settings.tr('Đang nộp', 'Processing') : _settings.tr('Chưa nộp phạt', 'Unpaid'));

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            iconData,
            color: Colors.white,
            size: 14,
          ),
          const SizedBox(width: 4),
          Text(
            actText,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoRow(IconData icon, Color color, String title, String value) {
    return Padding(
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
            child: Icon(icon, size: 18, color: color),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppTheme.textSecondary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w500,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _divider() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      height: 1,
      color: AppTheme.dividerColor,
    );
  }

  Widget _buildLawSection(Violation v) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.warningColor.withOpacity(0.06),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        border: Border.all(color: AppTheme.warningColor.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                  color: AppTheme.warningColor.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(Icons.gavel_rounded, size: 16, color: AppTheme.warningColor),
              ),
              const SizedBox(width: 8),
              Text(
                _settings.tr('Căn cứ pháp luật', 'Legal basis'),
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: AppTheme.warningColor,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            v.lawReference,
            style: const TextStyle(
              fontSize: 14,
              color: AppTheme.textPrimary,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }
}
