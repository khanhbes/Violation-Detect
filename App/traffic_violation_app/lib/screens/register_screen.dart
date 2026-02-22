import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/auth_service.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _formKey = GlobalKey<FormState>();
  final _fullNameController = TextEditingController();
  final _cccdController = TextEditingController();
  final _phoneController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _auth = AuthService();
  bool _isPasswordVisible = false;
  bool _isConfirmVisible = false;
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void dispose() {
    _fullNameController.dispose();
    _cccdController.dispose();
    _phoneController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  Future<void> _handleRegister() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
      });

      try {
        final cccd = _cccdController.text.trim();
        final email = '$cccd@vnetraffic.vn';

        await _auth.register(
          email: email,
          password: _passwordController.text,
          fullName: _fullNameController.text.trim(),
          phone: _phoneController.text.trim(),
        );

        if (mounted) {
          setState(() => _isLoading = false);
          // Go straight to home after registration
          Navigator.pushReplacementNamed(context, '/home');
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _isLoading = false;
            _errorMessage = e.toString();
          });
        }
      }
    }
  }

  // Password strength check
  double _getPasswordStrength(String password) {
    if (password.isEmpty) return 0;
    double strength = 0;
    if (password.length >= 8) strength += 0.2;
    if (password.length >= 12) strength += 0.1;
    if (RegExp(r'[A-Z]').hasMatch(password)) strength += 0.2;
    if (RegExp(r'[a-z]').hasMatch(password)) strength += 0.15;
    if (RegExp(r'[0-9]').hasMatch(password)) strength += 0.15;
    if (RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(password)) strength += 0.2;
    return strength.clamp(0.0, 1.0);
  }

  String _getStrengthLabel(double strength) {
    if (strength < 0.3) return 'Yếu';
    if (strength < 0.6) return 'Trung bình';
    if (strength < 0.9) return 'Mạnh';
    return 'Rất mạnh';
  }

  Color _getStrengthColor(double strength) {
    if (strength < 0.3) return AppTheme.dangerColor;
    if (strength < 0.6) return AppTheme.warningColor;
    if (strength < 0.9) return AppTheme.secondaryColor;
    return AppTheme.successColor;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFD32F2F), Color(0xFFB71C1C)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Back button & title
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                child: Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
                    ),
                    const SizedBox(width: 4),
                    const Text(
                      'Đăng ký tài khoản',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
              // Form
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
                    boxShadow: [
                      BoxShadow(
                        color: Color(0x33000000),
                        blurRadius: 20,
                        offset: Offset(0, -4),
                      ),
                    ],
                  ),
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(24, 28, 24, 16),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          // Error message
                          if (_errorMessage != null) ...[
                            Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: AppTheme.dangerColor.withOpacity(0.06),
                                borderRadius: BorderRadius.circular(12),
                                border: Border.all(color: AppTheme.dangerColor.withOpacity(0.2)),
                              ),
                              child: Row(
                                children: [
                                  const Icon(Icons.error_outline, color: AppTheme.dangerColor, size: 20),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      _errorMessage!,
                                      style: const TextStyle(color: AppTheme.dangerColor, fontSize: 13),
                                    ),
                                  ),
                                  GestureDetector(
                                    onTap: () => setState(() => _errorMessage = null),
                                    child: const Icon(Icons.close, color: AppTheme.dangerColor, size: 18),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 16),
                          ],

                          // Full Name
                          _buildFieldLabel('Họ và tên', true),
                          const SizedBox(height: 8),
                          TextFormField(
                            controller: _fullNameController,
                            textCapitalization: TextCapitalization.words,
                            decoration: _buildInputDecoration(
                              hint: 'Nguyễn Văn A',
                              icon: Icons.person_outline,
                            ),
                            validator: (value) {
                              if (value == null || value.trim().isEmpty) {
                                return 'Vui lòng nhập họ tên';
                              }
                              return null;
                            },
                          ),
                          const SizedBox(height: 18),

                          // CCCD
                          _buildFieldLabel('Số căn cước công dân', true),
                          const SizedBox(height: 8),
                          TextFormField(
                            controller: _cccdController,
                            keyboardType: TextInputType.number,
                            maxLength: 12,
                            style: const TextStyle(letterSpacing: 1),
                            decoration: _buildInputDecoration(
                              hint: 'Nhập 12 số CCCD',
                              icon: Icons.credit_card_rounded,
                            ).copyWith(counterText: ''),
                            validator: (value) {
                              if (value == null || value.isEmpty) return 'Vui lòng nhập số CCCD';
                              if (value.length != 12) return 'Số CCCD phải gồm 12 chữ số';
                              if (!RegExp(r'^[0-9]{12}$').hasMatch(value)) return 'CCCD chỉ bao gồm chữ số';
                              return null;
                            },
                          ),
                          const SizedBox(height: 18),

                          // Phone
                          _buildFieldLabel('Số điện thoại', false),
                          const SizedBox(height: 8),
                          TextFormField(
                            controller: _phoneController,
                            keyboardType: TextInputType.phone,
                            decoration: _buildInputDecoration(
                              hint: '0901234567',
                              icon: Icons.phone_outlined,
                            ),
                          ),
                          const SizedBox(height: 18),

                          // Password
                          _buildFieldLabel('Mật khẩu', true),
                          const SizedBox(height: 8),
                          TextFormField(
                            controller: _passwordController,
                            obscureText: !_isPasswordVisible,
                            onChanged: (v) => setState(() {}),
                            decoration: _buildInputDecoration(
                              hint: 'Tối thiểu 8 ký tự',
                              icon: Icons.lock_outline,
                            ).copyWith(
                              suffixIcon: IconButton(
                                icon: Icon(
                                  _isPasswordVisible
                                      ? Icons.visibility_outlined
                                      : Icons.visibility_off_outlined,
                                  color: AppTheme.textSecondary,
                                  size: 20,
                                ),
                                onPressed: () {
                                  setState(() => _isPasswordVisible = !_isPasswordVisible);
                                },
                              ),
                            ),
                            validator: (value) {
                              if (value == null || value.isEmpty) return 'Vui lòng nhập mật khẩu';
                              if (value.length < 8) return 'Mật khẩu phải có ít nhất 8 ký tự';
                              if (!RegExp(r'[A-Z]').hasMatch(value)) return 'Phải có ít nhất 1 chữ hoa';
                              if (!RegExp(r'[a-z]').hasMatch(value)) return 'Phải có ít nhất 1 chữ thường';
                              if (!RegExp(r'[0-9]').hasMatch(value)) return 'Phải có ít nhất 1 chữ số';
                              if (!RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(value)) return 'Phải có ít nhất 1 ký tự đặc biệt';
                              return null;
                            },
                          ),
                          const SizedBox(height: 8),
                          // Password Strength Indicator
                          if (_passwordController.text.isNotEmpty) ...[
                            _buildPasswordStrengthBar(),
                            const SizedBox(height: 6),
                            _buildPasswordRequirements(),
                            const SizedBox(height: 12),
                          ] else
                            const SizedBox(height: 18),

                          // Confirm Password
                          _buildFieldLabel('Xác nhận mật khẩu', true),
                          const SizedBox(height: 8),
                          TextFormField(
                            controller: _confirmPasswordController,
                            obscureText: !_isConfirmVisible,
                            decoration: _buildInputDecoration(
                              hint: 'Nhập lại mật khẩu',
                              icon: Icons.lock_outline,
                            ).copyWith(
                              suffixIcon: IconButton(
                                icon: Icon(
                                  _isConfirmVisible
                                      ? Icons.visibility_outlined
                                      : Icons.visibility_off_outlined,
                                  color: AppTheme.textSecondary,
                                  size: 20,
                                ),
                                onPressed: () {
                                  setState(() => _isConfirmVisible = !_isConfirmVisible);
                                },
                              ),
                            ),
                            validator: (value) {
                              if (value != _passwordController.text) {
                                return 'Mật khẩu không khớp';
                              }
                              return null;
                            },
                          ),
                          const SizedBox(height: 28),

                          // Register Button
                          Container(
                            height: 56,
                            decoration: BoxDecoration(
                              gradient: AppTheme.primaryGradient,
                              borderRadius: BorderRadius.circular(AppTheme.radiusM),
                              boxShadow: AppTheme.redShadow,
                            ),
                            child: ElevatedButton(
                              onPressed: _isLoading ? null : _handleRegister,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.transparent,
                                shadowColor: Colors.transparent,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(AppTheme.radiusM),
                                ),
                              ),
                              child: _isLoading
                                  ? const SizedBox(
                                      height: 24,
                                      width: 24,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2.5,
                                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                      ),
                                    )
                                  : const Text(
                                      'Đăng ký',
                                      style: TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.w700,
                                        color: Colors.white,
                                      ),
                                    ),
                            ),
                          ),
                          const SizedBox(height: 20),

                          // Already have account
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                'Đã có tài khoản? ',
                                style: TextStyle(color: AppTheme.textSecondary, fontSize: 13),
                              ),
                              GestureDetector(
                                onTap: () => Navigator.pop(context),
                                child: const Text(
                                  'Đăng nhập',
                                  style: TextStyle(
                                    fontWeight: FontWeight.w700,
                                    color: AppTheme.primaryColor,
                                    fontSize: 14,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Helper: Password Strength Bar ───────────────────────────────
  Widget _buildPasswordStrengthBar() {
    final strength = _getPasswordStrength(_passwordController.text);
    final color = _getStrengthColor(strength);
    final label = _getStrengthLabel(strength);

    return Row(
      children: [
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: strength,
              backgroundColor: AppTheme.dividerColor,
              valueColor: AlwaysStoppedAnimation(color),
              minHeight: 6,
            ),
          ),
        ),
        const SizedBox(width: 10),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
      ],
    );
  }

  // ── Helper: Password Requirements Checklist ─────────────────────
  Widget _buildPasswordRequirements() {
    final pwd = _passwordController.text;
    return Wrap(
      spacing: 8,
      runSpacing: 4,
      children: [
        _buildReqChip('≥ 8 ký tự', pwd.length >= 8),
        _buildReqChip('Chữ hoa', RegExp(r'[A-Z]').hasMatch(pwd)),
        _buildReqChip('Chữ thường', RegExp(r'[a-z]').hasMatch(pwd)),
        _buildReqChip('Chữ số', RegExp(r'[0-9]').hasMatch(pwd)),
        _buildReqChip('Ký tự ĐB', RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(pwd)),
      ],
    );
  }

  Widget _buildReqChip(String label, bool met) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: met
            ? AppTheme.successColor.withOpacity(0.08)
            : AppTheme.dangerColor.withOpacity(0.06),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: met
              ? AppTheme.successColor.withOpacity(0.3)
              : AppTheme.dangerColor.withOpacity(0.15),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            met ? Icons.check_circle_rounded : Icons.circle_outlined,
            size: 12,
            color: met ? AppTheme.successColor : AppTheme.textHint,
          ),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w500,
              color: met ? AppTheme.successColor : AppTheme.textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  // ── Helper: Input Decoration ────────────────────────────────────
  InputDecoration _buildInputDecoration({required String hint, required IconData icon}) {
    return InputDecoration(
      hintText: hint,
      hintStyle: TextStyle(color: AppTheme.textHint, fontSize: 14),
      prefixIcon: Padding(
        padding: const EdgeInsets.all(12),
        child: Icon(icon, size: 20, color: AppTheme.textSecondary),
      ),
      filled: true,
      fillColor: Colors.white,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        borderSide: BorderSide(color: AppTheme.dividerColor, width: 1.5),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        borderSide: BorderSide(color: AppTheme.dividerColor, width: 1.5),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        borderSide: const BorderSide(color: AppTheme.primaryColor, width: 2),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        borderSide: const BorderSide(color: AppTheme.dangerColor, width: 1.5),
      ),
      focusedErrorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        borderSide: const BorderSide(color: AppTheme.dangerColor, width: 2),
      ),
    );
  }

  // ── Helper: Field Label ─────────────────────────────────────────
  Widget _buildFieldLabel(String label, bool required) {
    return RichText(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: AppTheme.textPrimary,
        ),
        children: required
            ? const [
                TextSpan(
                  text: ' *',
                  style: TextStyle(
                    color: AppTheme.dangerColor,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ]
            : null,
      ),
    );
  }
}
