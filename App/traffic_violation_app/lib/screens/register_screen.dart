import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/auth_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen>
    with TickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final _fullNameController = TextEditingController();
  final _cccdController = TextEditingController();
  final _phoneController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _auth = AuthService();
  final _settings = AppSettings();
  bool _isPasswordVisible = false;
  bool _isConfirmVisible = false;
  bool _isLoading = false;
  String? _errorMessage;

  late AnimationController _animController;
  late Animation<double> _fadeAnim;
  late Animation<Offset> _slideAnim;

  late AnimationController _staggerController;

  @override
  void initState() {
    super.initState();
    _settings.addListener(_onSettingsChanged);

    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _fadeAnim = CurvedAnimation(parent: _animController, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(
      begin: const Offset(0, 0.1),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _animController, curve: Curves.easeOutCubic));

    _staggerController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1600),
    );

    _animController.forward();
    Future.delayed(const Duration(milliseconds: 300), () {
      if (mounted) _staggerController.forward();
    });
  }

  @override
  void dispose() {
    _animController.dispose();
    _staggerController.dispose();
    _fullNameController.dispose();
    _cccdController.dispose();
    _phoneController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _settings.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    if (mounted) setState(() {});
  }

  Animation<double> _staggeredFade(double start, double end) {
    return CurvedAnimation(
      parent: _staggerController,
      curve: Interval(start, end, curve: Curves.easeOutCubic),
    );
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
    if (strength < 0.3) return _settings.tr('Yếu', 'Weak');
    if (strength < 0.6) return _settings.tr('Trung bình', 'Medium');
    if (strength < 0.9) return _settings.tr('Mạnh', 'Strong');
    return _settings.tr('Rất mạnh', 'Very strong');
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
              FadeTransition(
                opacity: _fadeAnim,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                  child: Row(
                    children: [
                      IconButton(
                        onPressed: () => Navigator.pop(context),
                        icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
                      ),
                      const SizedBox(width: 4),
                      Text(
                        _settings.tr('Đăng ký tài khoản', 'Create Account'),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              // Form
              Expanded(
                child: SlideTransition(
                  position: _slideAnim,
                  child: FadeTransition(
                    opacity: _fadeAnim,
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
                                _buildAnimatedWidget(
                                  0.0, 0.15,
                                  child: Container(
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      color: AppTheme.dangerColor.withValues(alpha: 0.06),
                                      borderRadius: BorderRadius.circular(12),
                                      border: Border.all(color: AppTheme.dangerColor.withValues(alpha: 0.2)),
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
                                ),
                                const SizedBox(height: 16),
                              ],

                              // Full Name
                              _buildAnimatedWidget(0.0, 0.2, child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  _buildFieldLabel(_settings.tr('Họ và tên', 'Full name'), true),
                                  const SizedBox(height: 8),
                                  TextFormField(
                                    controller: _fullNameController,
                                    textCapitalization: TextCapitalization.words,
                                    decoration: _buildInputDecoration(
                                      hint: _settings.tr('Nguyễn Văn A', 'John Doe'),
                                      icon: Icons.person_outline,
                                    ),
                                    validator: (value) {
                                      if (value == null || value.trim().isEmpty) {
                                        return _settings.tr('Vui lòng nhập họ tên', 'Please enter your name');
                                      }
                                      return null;
                                    },
                                  ),
                                ],
                              )),
                              const SizedBox(height: 18),

                              // CCCD
                              _buildAnimatedWidget(0.1, 0.3, child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  _buildFieldLabel(_settings.tr('Số căn cước công dân', 'Citizen ID Number'), true),
                                  const SizedBox(height: 8),
                                  TextFormField(
                                    controller: _cccdController,
                                    keyboardType: TextInputType.number,
                                    maxLength: 12,
                                    style: const TextStyle(letterSpacing: 1),
                                    decoration: _buildInputDecoration(
                                      hint: _settings.tr('Nhập 12 số CCCD', 'Enter 12-digit ID'),
                                      icon: Icons.credit_card_rounded,
                                    ).copyWith(counterText: ''),
                                    validator: (value) {
                                      if (value == null || value.isEmpty) return _settings.tr('Vui lòng nhập số CCCD', 'Please enter your ID number');
                                      if (value.length != 12) return _settings.tr('Số CCCD phải gồm 12 chữ số', 'ID must be 12 digits');
                                      if (!RegExp(r'^[0-9]{12}$').hasMatch(value)) return _settings.tr('CCCD chỉ bao gồm chữ số', 'ID must contain only digits');
                                      return null;
                                    },
                                  ),
                                ],
                              )),
                              const SizedBox(height: 18),

                              // Phone
                              _buildAnimatedWidget(0.2, 0.4, child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  _buildFieldLabel(_settings.tr('Số điện thoại', 'Phone number'), false),
                                  const SizedBox(height: 8),
                                  TextFormField(
                                    controller: _phoneController,
                                    keyboardType: TextInputType.phone,
                                    decoration: _buildInputDecoration(
                                      hint: '0901234567',
                                      icon: Icons.phone_outlined,
                                    ),
                                  ),
                                ],
                              )),
                              const SizedBox(height: 18),

                              // Password
                              _buildAnimatedWidget(0.3, 0.5, child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  _buildFieldLabel(_settings.tr('Mật khẩu', 'Password'), true),
                                  const SizedBox(height: 8),
                                  TextFormField(
                                    controller: _passwordController,
                                    obscureText: !_isPasswordVisible,
                                    onChanged: (v) => setState(() {}),
                                    decoration: _buildInputDecoration(
                                      hint: _settings.tr('Tối thiểu 8 ký tự', 'Minimum 8 characters'),
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
                                      if (value == null || value.isEmpty) return _settings.tr('Vui lòng nhập mật khẩu', 'Please enter password');
                                      if (value.length < 8) return _settings.tr('Mật khẩu phải có ít nhất 8 ký tự', 'Password must be at least 8 characters');
                                      if (!RegExp(r'[A-Z]').hasMatch(value)) return _settings.tr('Phải có ít nhất 1 chữ hoa', 'Must contain at least 1 uppercase letter');
                                      if (!RegExp(r'[a-z]').hasMatch(value)) return _settings.tr('Phải có ít nhất 1 chữ thường', 'Must contain at least 1 lowercase letter');
                                      if (!RegExp(r'[0-9]').hasMatch(value)) return _settings.tr('Phải có ít nhất 1 chữ số', 'Must contain at least 1 digit');
                                      if (!RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(value)) return _settings.tr('Phải có ít nhất 1 ký tự đặc biệt', 'Must contain at least 1 special character');
                                      return null;
                                    },
                                  ),
                                ],
                              )),
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
                              _buildAnimatedWidget(0.4, 0.6, child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  _buildFieldLabel(_settings.tr('Xác nhận mật khẩu', 'Confirm password'), true),
                                  const SizedBox(height: 8),
                                  TextFormField(
                                    controller: _confirmPasswordController,
                                    obscureText: !_isConfirmVisible,
                                    decoration: _buildInputDecoration(
                                      hint: _settings.tr('Nhập lại mật khẩu', 'Re-enter password'),
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
                                        return _settings.tr('Mật khẩu không khớp', 'Passwords do not match');
                                      }
                                      return null;
                                    },
                                  ),
                                ],
                              )),
                              const SizedBox(height: 28),

                              // Register Button
                              _buildAnimatedWidget(0.5, 0.7, child: Container(
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
                                      : Text(
                                          _settings.tr('Đăng ký', 'Register'),
                                          style: const TextStyle(
                                            fontSize: 16,
                                            fontWeight: FontWeight.w700,
                                            color: Colors.white,
                                          ),
                                        ),
                                ),
                              )),
                              const SizedBox(height: 20),

                              // Already have account
                              _buildAnimatedWidget(0.6, 0.8, child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    _settings.tr('Đã có tài khoản? ', 'Already have an account? '),
                                    style: const TextStyle(color: AppTheme.textSecondary, fontSize: 13),
                                  ),
                                  GestureDetector(
                                    onTap: () => Navigator.pop(context),
                                    child: Text(
                                      _settings.tr('Đăng nhập', 'Sign In'),
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w700,
                                        color: AppTheme.primaryColor,
                                        fontSize: 14,
                                      ),
                                    ),
                                  ),
                                ],
                              )),
                              const SizedBox(height: 16),
                            ],
                          ),
                        ),
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

  // ── Animated field wrapper ─────────────────────────────────────
  Widget _buildAnimatedWidget(double start, double end, {required Widget child}) {
    final anim = _staggeredFade(start, end);
    return FadeTransition(
      opacity: anim,
      child: SlideTransition(
        position: Tween<Offset>(
          begin: const Offset(0, 0.2),
          end: Offset.zero,
        ).animate(anim),
        child: child,
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
            child: TweenAnimationBuilder<double>(
              tween: Tween(begin: 0, end: strength),
              duration: const Duration(milliseconds: 400),
              curve: Curves.easeOutCubic,
              builder: (context, value, _) {
                return LinearProgressIndicator(
                  value: value,
                  backgroundColor: AppTheme.dividerColor,
                  valueColor: AlwaysStoppedAnimation(color),
                  minHeight: 6,
                );
              },
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
        _buildReqChip(_settings.tr('≥ 8 ký tự', '≥ 8 chars'), pwd.length >= 8),
        _buildReqChip(_settings.tr('Chữ hoa', 'Uppercase'), RegExp(r'[A-Z]').hasMatch(pwd)),
        _buildReqChip(_settings.tr('Chữ thường', 'Lowercase'), RegExp(r'[a-z]').hasMatch(pwd)),
        _buildReqChip(_settings.tr('Chữ số', 'Digit'), RegExp(r'[0-9]').hasMatch(pwd)),
        _buildReqChip(_settings.tr('Ký tự ĐB', 'Special'), RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(pwd)),
      ],
    );
  }

  Widget _buildReqChip(String label, bool met) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: met
            ? AppTheme.successColor.withValues(alpha: 0.08)
            : AppTheme.dangerColor.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: met
              ? AppTheme.successColor.withValues(alpha: 0.3)
              : AppTheme.dangerColor.withValues(alpha: 0.15),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 250),
            child: Icon(
              met ? Icons.check_circle_rounded : Icons.circle_outlined,
              key: ValueKey(met),
              size: 12,
              color: met ? AppTheme.successColor : AppTheme.textHint,
            ),
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
      hintStyle: const TextStyle(color: AppTheme.textHint, fontSize: 14),
      prefixIcon: Padding(
        padding: const EdgeInsets.all(12),
        child: Icon(icon, size: 20, color: AppTheme.textSecondary),
      ),
      filled: true,
      fillColor: Colors.white,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        borderSide: const BorderSide(color: AppTheme.dividerColor, width: 1.5),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        borderSide: const BorderSide(color: AppTheme.dividerColor, width: 1.5),
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
