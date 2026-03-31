import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/auth_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _formKey = GlobalKey<FormState>();
  final _cccdController = TextEditingController();
  final _auth = AuthService();
  final _s = AppSettings();
  bool _isLoading = false;

  @override
  void dispose() {
    _cccdController.dispose();
    super.dispose();
  }

  Future<void> _handleReset() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);
    try {
      await _auth.resetPasswordByIdCard(_cccdController.text.trim());
      if (mounted) {
        showDialog(
          context: context,
          builder: (ctx) => AlertDialog(
            icon: const Icon(Icons.mark_email_read_rounded,
                color: AppTheme.successColor, size: 48),
            title: Text(_s.tr(
                'Đã gửi email đặt lại mật khẩu',
                'Password Reset Email Sent')),
            content: Text(_s.tr(
              'Vui lòng kiểm tra hộp thư email liên kết với CCCD của bạn để đặt lại mật khẩu.',
              'Please check the email linked to your ID card to reset your password.',
            )),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(ctx);
                  Navigator.pop(context);
                },
                child: Text(_s.tr('Quay lại đăng nhập', 'Back to login')),
              ),
            ],
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        final msg = e.toString().replaceFirst('Exception:', '').trim();
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Row(
            children: [
              const Icon(Icons.error_outline, color: Colors.white, size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  msg.isNotEmpty
                      ? msg
                      : _s.tr('Không thể gửi email đặt lại mật khẩu',
                          'Could not send password reset email'),
                ),
              ),
            ],
          ),
          backgroundColor: AppTheme.dangerColor,
          behavior: SnackBarBehavior.floating,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ));
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF121212) : AppTheme.surfaceColor,
      appBar: AppBar(
        title: Text(_s.tr('Quên mật khẩu', 'Forgot Password')),
        backgroundColor: AppTheme.primaryColor,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Container(
            constraints: const BoxConstraints(maxWidth: 400),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: isDark ? const Color(0xFF1E1E1E) : Colors.white,
              borderRadius: BorderRadius.circular(AppTheme.radiusXL),
              boxShadow: isDark ? [] : AppTheme.cardShadow,
            ),
            child: Form(
              key: _formKey,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 72,
                    height: 72,
                    decoration: BoxDecoration(
                      color: AppTheme.primaryColor.withValues(alpha: 0.1),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.lock_reset_rounded,
                        color: AppTheme.primaryColor, size: 36),
                  ),
                  const SizedBox(height: 20),
                  Text(
                    _s.tr(
                        'Nhập số CCCD để đặt lại mật khẩu',
                        'Enter your ID card number to reset password'),
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: isDark ? Colors.white70 : AppTheme.textSecondary,
                      fontSize: 14,
                      height: 1.4,
                    ),
                  ),
                  const SizedBox(height: 24),
                  TextFormField(
                    controller: _cccdController,
                    keyboardType: TextInputType.number,
                    inputFormatters: [
                      FilteringTextInputFormatter.digitsOnly,
                      LengthLimitingTextInputFormatter(12),
                    ],
                    decoration: InputDecoration(
                      labelText: _s.tr('Số CCCD', 'ID Card Number'),
                      hintText: _s.tr('Nhập 12 số CCCD', 'Enter 12-digit ID'),
                      prefixIcon: const Icon(Icons.badge_outlined),
                      filled: true,
                      fillColor: isDark
                          ? Colors.white.withValues(alpha: 0.05)
                          : AppTheme.surfaceColor,
                    ),
                    validator: (value) {
                      final v = (value ?? '').trim();
                      if (v.isEmpty) {
                        return _s.tr(
                            'Vui lòng nhập số CCCD', 'Please enter ID number');
                      }
                      if (v.length != 12 ||
                          !RegExp(r'^\d{12}$').hasMatch(v)) {
                        return _s.tr('CCCD phải gồm đúng 12 chữ số',
                            'ID must be exactly 12 digits');
                      }
                      return null;
                    },
                  ),
                  const SizedBox(height: 24),
                  SizedBox(
                    width: double.infinity,
                    height: 50,
                    child: ElevatedButton.icon(
                      onPressed: _isLoading ? null : _handleReset,
                      icon: _isLoading
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(
                                  color: Colors.white, strokeWidth: 2))
                          : const Icon(Icons.send_rounded, size: 18),
                      label: Text(_isLoading
                          ? _s.tr('Đang gửi...', 'Sending...')
                          : _s.tr('Gửi email đặt lại', 'Send Reset Email')),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.primaryColor,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                            borderRadius:
                                BorderRadius.circular(AppTheme.radiusM)),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
