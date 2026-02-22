import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/services/auth_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen>
    with TickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final _cccdController = TextEditingController();
  final _passwordController = TextEditingController();
  final _auth = AuthService();
  final _settings = AppSettings();
  bool _isPasswordVisible = false;
  bool _isLoading = false;
  String? _errorMessage;

  late AnimationController _animController;
  late Animation<double> _fadeAnim;
  late Animation<Offset> _slideAnim;

  // Staggered entry animations
  late AnimationController _staggerController;
  late Animation<double> _field1Anim;
  late Animation<double> _field2Anim;
  late Animation<double> _buttonAnim;

  // Pulsing logo
  late AnimationController _pulseController;
  late Animation<double> _pulseAnim;

  @override
  void initState() {
    super.initState();
    _settings.addListener(_onSettingsChanged);

    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _fadeAnim = CurvedAnimation(parent: _animController, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(
      begin: const Offset(0, 0.12),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _animController, curve: Curves.easeOutCubic));

    _staggerController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    );
    _field1Anim = CurvedAnimation(
      parent: _staggerController,
      curve: const Interval(0.0, 0.4, curve: Curves.easeOutCubic),
    );
    _field2Anim = CurvedAnimation(
      parent: _staggerController,
      curve: const Interval(0.2, 0.6, curve: Curves.easeOutCubic),
    );
    _buttonAnim = CurvedAnimation(
      parent: _staggerController,
      curve: const Interval(0.4, 0.8, curve: Curves.easeOutCubic),
    );

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 1.0, end: 1.08).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _animController.forward();
    Future.delayed(const Duration(milliseconds: 400), () {
      if (mounted) _staggerController.forward();
    });
  }

  @override
  void dispose() {
    _animController.dispose();
    _staggerController.dispose();
    _pulseController.dispose();
    _cccdController.dispose();
    _passwordController.dispose();
    _settings.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    if (mounted) setState(() {});
  }

  Future<void> _handleLogin() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final cccd = _cccdController.text.trim();
      final email = '$cccd@vnetraffic.vn';
      await _auth.signIn(email, _passwordController.text);

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

  Future<void> _handleForgotPassword() async {
    final cccd = _cccdController.text.trim();
    if (cccd.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              const Icon(Icons.info_outline, color: Colors.white, size: 18),
              const SizedBox(width: 8),
              Text(_settings.tr('Vui lòng nhập số CCCD trước', 'Please enter your ID number first')),
            ],
          ),
          backgroundColor: AppTheme.infoColor,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
      );
      return;
    }

    try {
      final email = '$cccd@vnetraffic.vn';
      await _auth.resetPassword(email);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.white, size: 18),
                const SizedBox(width: 8),
                Text(_settings.tr('Đã gửi email đặt lại mật khẩu', 'Password reset email sent')),
              ],
            ),
            backgroundColor: AppTheme.successColor,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.toString()),
            backgroundColor: AppTheme.dangerColor,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        );
      }
    }
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
              // ── Red Header with Animated Logo ──────────
              FadeTransition(
                opacity: _fadeAnim,
                child: Padding(
                  padding: const EdgeInsets.only(top: 24, bottom: 18),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Pulsing shield logo
                      ScaleTransition(
                        scale: _pulseAnim,
                        child: Container(
                          width: 100,
                          height: 100,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            gradient: LinearGradient(
                              colors: [
                                Colors.white.withValues(alpha: 0.25),
                                Colors.white.withValues(alpha: 0.10),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            border: Border.all(
                              color: Colors.white.withValues(alpha: 0.3),
                              width: 2,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withValues(alpha: 0.15),
                                blurRadius: 20,
                                offset: const Offset(0, 8),
                              ),
                            ],
                          ),
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              Icon(
                                Icons.shield_rounded,
                                size: 52,
                                color: Colors.white.withValues(alpha: 0.9),
                              ),
                              const Positioned(
                                bottom: 22,
                                child: Icon(
                                  Icons.local_police_rounded,
                                  size: 20,
                                  color: Colors.amber,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 14),
                      const Text(
                        'VNeTraffic',
                        style: TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.w800,
                          color: Colors.white,
                          letterSpacing: 1.0,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // ── White Card Form ───────────────────────────
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
                              // ── Title ────────────────────────
                              Center(
                                child: Text(
                                  _settings.tr('Đăng nhập', 'Sign In'),
                                  style: const TextStyle(
                                    fontSize: 24,
                                    fontWeight: FontWeight.w800,
                                    color: AppTheme.textPrimary,
                                  ),
                                ),
                              ),
                              const SizedBox(height: 24),

                              // ── Error Message ────────────────
                              if (_errorMessage != null) ...[
                                _buildErrorBanner(_errorMessage!),
                                const SizedBox(height: 16),
                              ],

                              // ── CCCD Field (animated) ──────
                              FadeTransition(
                                opacity: _field1Anim,
                                child: SlideTransition(
                                  position: Tween<Offset>(
                                    begin: const Offset(0, 0.3),
                                    end: Offset.zero,
                                  ).animate(_field1Anim),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      _buildFieldLabel(
                                        _settings.tr('Số căn cước công dân', 'Citizen ID Number'),
                                        true,
                                      ),
                                      const SizedBox(height: 8),
                                      TextFormField(
                                        controller: _cccdController,
                                        keyboardType: TextInputType.number,
                                        maxLength: 12,
                                        style: const TextStyle(fontSize: 15, letterSpacing: 1),
                                        decoration: InputDecoration(
                                          hintText: _settings.tr('Nhập 12 số CCCD', 'Enter 12-digit ID'),
                                          hintStyle: const TextStyle(
                                            color: AppTheme.textHint,
                                            fontSize: 14,
                                            letterSpacing: 0,
                                          ),
                                          counterText: '',
                                          prefixIcon: const Padding(
                                            padding: EdgeInsets.all(12),
                                            child: Icon(Icons.credit_card_rounded, size: 20, color: AppTheme.textSecondary),
                                          ),
                                          filled: true,
                                          fillColor: Colors.white,
                                          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                                          border: OutlineInputBorder(
                                            borderRadius: BorderRadius.circular(AppTheme.radiusM),
                                            borderSide: const BorderSide(color: AppTheme.dangerColor, width: 1.5),
                                          ),
                                          enabledBorder: OutlineInputBorder(
                                            borderRadius: BorderRadius.circular(AppTheme.radiusM),
                                            borderSide: const BorderSide(color: AppTheme.dangerColor, width: 1.5),
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
                                        ),
                                        validator: (value) {
                                          if (value == null || value.isEmpty) {
                                            return _settings.tr('Vui lòng nhập số CCCD', 'Please enter your ID number');
                                          }
                                          if (value.length != 12) {
                                            return _settings.tr('Số CCCD phải gồm 12 chữ số', 'ID must be 12 digits');
                                          }
                                          if (!RegExp(r'^[0-9]{12}$').hasMatch(value)) {
                                            return _settings.tr('CCCD chỉ bao gồm chữ số', 'ID must contain only digits');
                                          }
                                          return null;
                                        },
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              const SizedBox(height: 18),

                              // ── Password Field (animated) ──
                              FadeTransition(
                                opacity: _field2Anim,
                                child: SlideTransition(
                                  position: Tween<Offset>(
                                    begin: const Offset(0, 0.3),
                                    end: Offset.zero,
                                  ).animate(_field2Anim),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      _buildFieldLabel(
                                        _settings.tr('Mật khẩu', 'Password'),
                                        true,
                                      ),
                                      const SizedBox(height: 8),
                                      TextFormField(
                                        controller: _passwordController,
                                        obscureText: !_isPasswordVisible,
                                        style: const TextStyle(fontSize: 15),
                                        decoration: InputDecoration(
                                          hintText: _settings.tr('Nhập mật khẩu', 'Enter password'),
                                          hintStyle: const TextStyle(
                                            color: AppTheme.textHint,
                                            fontSize: 14,
                                          ),
                                          prefixIcon: const Padding(
                                            padding: EdgeInsets.all(12),
                                            child: Icon(Icons.lock_outline_rounded, size: 20, color: AppTheme.textSecondary),
                                          ),
                                          suffixIcon: IconButton(
                                            icon: Icon(
                                              _isPasswordVisible
                                                  ? Icons.visibility_outlined
                                                  : Icons.visibility_off_outlined,
                                              size: 20,
                                              color: AppTheme.textSecondary,
                                            ),
                                            onPressed: () => setState(
                                              () => _isPasswordVisible = !_isPasswordVisible,
                                            ),
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
                                        ),
                                        validator: (value) {
                                          if (value == null || value.isEmpty) {
                                            return _settings.tr('Vui lòng nhập mật khẩu', 'Please enter password');
                                          }
                                          if (value.length < 8) {
                                            return _settings.tr('Mật khẩu phải có ít nhất 8 ký tự', 'Password must be at least 8 characters');
                                          }
                                          if (!RegExp(r'[A-Z]').hasMatch(value)) {
                                            return _settings.tr('Phải có ít nhất 1 chữ hoa', 'Must contain at least 1 uppercase letter');
                                          }
                                          if (!RegExp(r'[a-z]').hasMatch(value)) {
                                            return _settings.tr('Phải có ít nhất 1 chữ thường', 'Must contain at least 1 lowercase letter');
                                          }
                                          if (!RegExp(r'[0-9]').hasMatch(value)) {
                                            return _settings.tr('Phải có ít nhất 1 chữ số', 'Must contain at least 1 digit');
                                          }
                                          if (!RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(value)) {
                                            return _settings.tr('Phải có ít nhất 1 ký tự đặc biệt', 'Must contain at least 1 special character');
                                          }
                                          return null;
                                        },
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              const SizedBox(height: 8),

                              // ── Switch/Forgot Row ──
                              FadeTransition(
                                opacity: _field2Anim,
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    TextButton(
                                      onPressed: () => Navigator.pushNamed(context, '/register'),
                                      style: TextButton.styleFrom(
                                        padding: EdgeInsets.zero,
                                        minimumSize: const Size(0, 36),
                                        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                      ),
                                      child: Text(
                                        _settings.tr('Đổi tài khoản', 'Switch account'),
                                        style: const TextStyle(
                                          color: AppTheme.primaryColor,
                                          fontWeight: FontWeight.w600,
                                          fontSize: 13,
                                        ),
                                      ),
                                    ),
                                    TextButton(
                                      onPressed: _handleForgotPassword,
                                      style: TextButton.styleFrom(
                                        padding: EdgeInsets.zero,
                                        minimumSize: const Size(0, 36),
                                        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                      ),
                                      child: Text(
                                        _settings.tr('Quên mật khẩu ?', 'Forgot password?'),
                                        style: const TextStyle(
                                          color: AppTheme.primaryColor,
                                          fontWeight: FontWeight.w600,
                                          fontSize: 13,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(height: 18),

                              // ── Login Button Row (animated) ──
                              FadeTransition(
                                opacity: _buttonAnim,
                                child: SlideTransition(
                                  position: Tween<Offset>(
                                    begin: const Offset(0, 0.4),
                                    end: Offset.zero,
                                  ).animate(_buttonAnim),
                                  child: Row(
                                    children: [
                                      Expanded(
                                        child: SizedBox(
                                          height: 52,
                                          child: Container(
                                            decoration: BoxDecoration(
                                              gradient: AppTheme.primaryGradient,
                                              borderRadius: BorderRadius.circular(AppTheme.radiusM),
                                              boxShadow: AppTheme.redShadow,
                                            ),
                                            child: ElevatedButton(
                                              onPressed: _isLoading ? null : _handleLogin,
                                              style: ElevatedButton.styleFrom(
                                                backgroundColor: Colors.transparent,
                                                shadowColor: Colors.transparent,
                                                shape: RoundedRectangleBorder(
                                                  borderRadius: BorderRadius.circular(AppTheme.radiusM),
                                                ),
                                              ),
                                              child: _isLoading
                                                  ? const SizedBox(
                                                      height: 22,
                                                      width: 22,
                                                      child: CircularProgressIndicator(
                                                        strokeWidth: 2.5,
                                                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                                      ),
                                                    )
                                                  : Text(
                                                      _settings.tr('Đăng nhập', 'Sign In'),
                                                      style: const TextStyle(
                                                        fontSize: 16,
                                                        fontWeight: FontWeight.w700,
                                                        color: Colors.white,
                                                      ),
                                                    ),
                                            ),
                                          ),
                                        ),
                                      ),
                                      const SizedBox(width: 12),
                                      // Pulsing biometric button
                                      ScaleTransition(
                                        scale: _pulseAnim,
                                        child: SizedBox(
                                          height: 52,
                                          width: 52,
                                          child: Container(
                                            decoration: BoxDecoration(
                                              color: AppTheme.surfaceColor,
                                              borderRadius: BorderRadius.circular(AppTheme.radiusM),
                                              border: Border.all(color: AppTheme.dividerColor),
                                            ),
                                            child: IconButton(
                                              onPressed: () {
                                                ScaffoldMessenger.of(context).showSnackBar(
                                                  SnackBar(
                                                    content: Text(_settings.tr('Tính năng đang phát triển', 'Feature in development')),
                                                    behavior: SnackBarBehavior.floating,
                                                    shape: RoundedRectangleBorder(
                                                      borderRadius: BorderRadius.circular(12),
                                                    ),
                                                  ),
                                                );
                                              },
                                              icon: const Icon(
                                                Icons.fingerprint_rounded,
                                                size: 26,
                                                color: AppTheme.primaryColor,
                                              ),
                                            ),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              const SizedBox(height: 24),

                              // ── "Or sign in with" Divider ──
                              FadeTransition(
                                opacity: _buttonAnim,
                                child: Row(
                                  children: [
                                    Expanded(child: Container(height: 1, color: AppTheme.dividerColor)),
                                    Padding(
                                      padding: const EdgeInsets.symmetric(horizontal: 14),
                                      child: Text(
                                        _settings.tr('Hoặc đăng nhập bằng', 'Or sign in with'),
                                        style: const TextStyle(
                                          color: AppTheme.textSecondary,
                                          fontSize: 13,
                                        ),
                                      ),
                                    ),
                                    Expanded(child: Container(height: 1, color: AppTheme.dividerColor)),
                                  ],
                                ),
                              ),
                              const SizedBox(height: 20),

                              // ── eID Button ──
                              FadeTransition(
                                opacity: _buttonAnim,
                                child: SizedBox(
                                  height: 50,
                                  child: OutlinedButton(
                                    onPressed: () {
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        SnackBar(
                                          content: Text(_settings.tr('Tính năng đang phát triển', 'Feature in development')),
                                          behavior: SnackBarBehavior.floating,
                                          shape: RoundedRectangleBorder(
                                            borderRadius: BorderRadius.circular(12),
                                          ),
                                        ),
                                      );
                                    },
                                    style: OutlinedButton.styleFrom(
                                      backgroundColor: const Color(0xFFFFF3E0),
                                      side: const BorderSide(color: Color(0xFFFF8F00), width: 1.5),
                                      shape: RoundedRectangleBorder(
                                        borderRadius: BorderRadius.circular(AppTheme.radiusRound),
                                      ),
                                    ),
                                    child: Row(
                                      mainAxisAlignment: MainAxisAlignment.center,
                                      children: [
                                        Text(
                                          _settings.tr('Tài khoản định danh điện tử', 'Electronic ID account'),
                                          style: const TextStyle(
                                            color: Color(0xFFE65100),
                                            fontWeight: FontWeight.w700,
                                            fontSize: 15,
                                          ),
                                        ),
                                        const SizedBox(width: 10),
                                        Container(
                                          width: 30,
                                          height: 30,
                                          decoration: BoxDecoration(
                                            color: const Color(0xFF2E7D32),
                                            borderRadius: BorderRadius.circular(8),
                                          ),
                                          child: const Icon(
                                            Icons.verified_user_rounded,
                                            color: Colors.white,
                                            size: 18,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(height: 18),

                              // ── Version ──
                              Center(
                                child: Text(
                                  _settings.tr('Phiên bản: 1.0.0 (1)', 'Version: 1.0.0 (1)'),
                                  style: const TextStyle(
                                    color: AppTheme.textHint,
                                    fontSize: 12,
                                  ),
                                ),
                              ),
                              const SizedBox(height: 8),

                              // ── Register link ──
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    _settings.tr('Chưa có tài khoản? ', "Don't have an account? "),
                                    style: const TextStyle(
                                      color: AppTheme.textSecondary,
                                      fontSize: 13,
                                    ),
                                  ),
                                  GestureDetector(
                                    onTap: () => Navigator.pushNamed(context, '/register'),
                                    child: Text(
                                      _settings.tr('Đăng ký ngay', 'Register now'),
                                      style: const TextStyle(
                                        color: AppTheme.primaryColor,
                                        fontWeight: FontWeight.w700,
                                        fontSize: 14,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),

              // ── Bottom Bar ────────────────────────────────
              Container(
                color: Colors.white,
                padding: const EdgeInsets.only(bottom: 12, top: 6),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _buildBottomAction(
                      Icons.menu_book_rounded,
                      _settings.tr('Hướng dẫn\nsử dụng', 'User\nGuide'),
                      AppTheme.primaryColor,
                      () {},
                    ),
                    _buildBottomAction(
                      Icons.help_outline_rounded,
                      _settings.tr('Câu hỏi\nthường gặp', 'Frequently\nAsked'),
                      AppTheme.secondaryColor,
                      () {},
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Helper: Field Label ──────────────────────────────────────────
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

  // ── Helper: Error Banner ─────────────────────────────────────────
  Widget _buildErrorBanner(String message) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: AppTheme.dangerColor.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        border: Border.all(color: AppTheme.dangerColor.withValues(alpha: 0.2)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: AppTheme.dangerColor.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.error_outline_rounded, color: AppTheme.dangerColor, size: 18),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(
                color: AppTheme.dangerColor,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          GestureDetector(
            onTap: () => setState(() => _errorMessage = null),
            child: const Icon(Icons.close, color: AppTheme.dangerColor, size: 18),
          ),
        ],
      ),
    );
  }

  // ── Helper: Bottom Action ────────────────────────────────────────
  Widget _buildBottomAction(
    IconData icon,
    String label,
    Color color,
    VoidCallback onTap,
  ) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: color, size: 24),
          ),
          const SizedBox(height: 6),
          Text(
            label,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: AppTheme.textSecondary,
              height: 1.3,
            ),
          ),
        ],
      ),
    );
  }
}
