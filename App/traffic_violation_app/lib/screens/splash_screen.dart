import 'package:flutter/material.dart';
import 'package:traffic_violation_app/services/auth_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'dart:async';
import 'dart:math' as math;

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  final _settings = AppSettings();

  // Main entrance animation
  late AnimationController _mainController;
  late Animation<double> _logoScale;
  late Animation<double> _logoFade;
  late Animation<double> _logoRotate;

  // Text stagger animation
  late AnimationController _textController;
  late Animation<double> _titleSlide;
  late Animation<double> _titleFade;
  late Animation<double> _subtitleSlide;
  late Animation<double> _subtitleFade;

  // Pulsing glow rings
  late AnimationController _ringController;
  late Animation<double> _ring1Scale;
  late Animation<double> _ring2Scale;
  late Animation<double> _ring3Scale;

  // Shimmer gradient rotation
  late AnimationController _shimmerController;

  // Progress bar
  late AnimationController _progressController;
  late Animation<double> _progressAnim;

  // Particle system
  late AnimationController _particleController;

  // Loading text animation
  int _loadingStep = 0;
  late Timer _loadingTimer;

  final List<_Particle> _particles = [];
  final math.Random _rng = math.Random();

  @override
  void initState() {
    super.initState();
    _initParticles();
    _initAnimations();
    _startSequence();
  }

  void _initParticles() {
    for (int i = 0; i < 20; i++) {
      _particles.add(_Particle(
        x: _rng.nextDouble(),
        y: _rng.nextDouble(),
        size: _rng.nextDouble() * 3 + 1,
        speed: _rng.nextDouble() * 0.3 + 0.1,
        opacity: _rng.nextDouble() * 0.4 + 0.1,
        delay: _rng.nextDouble(),
      ));
    }
  }

  void _initAnimations() {
    // Logo entrance: scale + fade + slight rotation
    _mainController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _logoScale = Tween<double>(begin: 0.3, end: 1.0).animate(
      CurvedAnimation(parent: _mainController, curve: Curves.elasticOut),
    );
    _logoFade = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _mainController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeOut),
      ),
    );
    _logoRotate = Tween<double>(begin: -0.08, end: 0.0).animate(
      CurvedAnimation(parent: _mainController, curve: Curves.easeOutCubic),
    );

    // Text stagger: title then subtitle
    _textController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _titleSlide = Tween<double>(begin: 40, end: 0).animate(
      CurvedAnimation(
        parent: _textController,
        curve: const Interval(0.0, 0.6, curve: Curves.easeOutCubic),
      ),
    );
    _titleFade = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _textController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeOut),
      ),
    );
    _subtitleSlide = Tween<double>(begin: 30, end: 0).animate(
      CurvedAnimation(
        parent: _textController,
        curve: const Interval(0.3, 0.8, curve: Curves.easeOutCubic),
      ),
    );
    _subtitleFade = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _textController,
        curve: const Interval(0.3, 0.7, curve: Curves.easeOut),
      ),
    );

    // Pulsing glow rings (3 rings with stagger)
    _ringController = AnimationController(
      duration: const Duration(milliseconds: 2500),
      vsync: this,
    )..repeat();
    _ring1Scale = Tween<double>(begin: 1.0, end: 1.8).animate(
      CurvedAnimation(parent: _ringController, curve: Curves.easeOut),
    );
    _ring2Scale = Tween<double>(begin: 1.0, end: 2.2).animate(
      CurvedAnimation(
        parent: _ringController,
        curve: const Interval(0.2, 1.0, curve: Curves.easeOut),
      ),
    );
    _ring3Scale = Tween<double>(begin: 1.0, end: 2.6).animate(
      CurvedAnimation(
        parent: _ringController,
        curve: const Interval(0.4, 1.0, curve: Curves.easeOut),
      ),
    );

    // Shimmer rotation
    _shimmerController = AnimationController(
      duration: const Duration(milliseconds: 3000),
      vsync: this,
    )..repeat();

    // Particles
    _particleController = AnimationController(
      duration: const Duration(seconds: 10),
      vsync: this,
    )..repeat();

    // Progress bar (fills in 3s)
    _progressController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    _progressAnim = CurvedAnimation(
      parent: _progressController,
      curve: Curves.easeInOut,
    );
  }

  void _startSequence() {
    // Step 1: Logo entrance
    _mainController.forward();

    // Step 2: Text appears after logo
    Future.delayed(const Duration(milliseconds: 300), () {
      if (mounted) _textController.forward();
    });

    // Step 3: Progress bar starts
    Future.delayed(const Duration(milliseconds: 400), () {
      if (mounted) _progressController.forward();
    });

    // Loading text cycling
    _loadingTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      if (mounted) {
        setState(() {
          _loadingStep = (_loadingStep + 1) % 4;
        });
      }
    });

    // Navigate after splash — load profile from Firestore if already logged in
    Timer(const Duration(milliseconds: 1800), () async {
      if (!mounted) return;

      try {
        // Auto-discover server IP from Firebase config
        final ds = await FirebaseFirestore.instance.collection('server').doc('config').get();
        if (ds.exists && ds.data() != null) {
          final ip = ds.data()!['ip'] as String?;
          if (ip != null && ip.isNotEmpty) {
            ApiService().setServerAddress(ip);
            debugPrint('✅ Auto-discovered Server IP from Firebase: $ip');
          }
        }
      } catch (e) {
        debugPrint('⚠️ Failed to auto-discover Server IP: $e');
      }

      final auth = AuthService();
      if (auth.isLoggedIn) {
        // Load user profile & settings from Firestore before going to home
        await _settings.loadFromFirestore(auth.currentUser!.uid);
        if (mounted) Navigator.pushReplacementNamed(context, '/home');
      } else {
        if (mounted) Navigator.pushReplacementNamed(context, '/login');
      }
    });
  }

  @override
  void dispose() {
    _mainController.dispose();
    _textController.dispose();
    _ringController.dispose();
    _shimmerController.dispose();
    _particleController.dispose();
    _progressController.dispose();
    _loadingTimer.cancel();
    super.dispose();
  }

  String get _loadingText {
    final vi = _settings.isVietnamese;
    switch (_loadingStep) {
      case 0: return vi ? 'Khởi tạo hệ thống...' : 'Initializing system...';
      case 1: return vi ? 'Đang tải mô hình AI...' : 'Loading AI models...';
      case 2: return vi ? 'Kết nối server...' : 'Connecting to server...';
      case 3: return vi ? 'Sắp xong...' : 'Almost ready...';
      default: return vi ? 'Đang tải...' : 'Loading...';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Background gradient ────────────────────────────────
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Color(0xFFD32F2F),
                  Color(0xFFB71C1C),
                  Color(0xFF880E4F),
                ],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
          ),

          // ── Subtle pattern overlay ─────────────────────────────
          Opacity(
            opacity: 0.04,
            child: Image.asset(
              'assets/images/app_icon.png',
              fit: BoxFit.none,
              repeat: ImageRepeat.repeat,
              width: double.infinity,
              height: double.infinity,
              errorBuilder: (_, __, ___) => const SizedBox.shrink(),
            ),
          ),

          // ── Animated particles ─────────────────────────────────
          AnimatedBuilder(
            animation: _particleController,
            builder: (context, _) {
              return CustomPaint(
                painter: _ParticlePainter(
                  particles: _particles,
                  progress: _particleController.value,
                ),
                size: Size.infinite,
              );
            },
          ),

          // ── Main content ───────────────────────────────────────
          SafeArea(
            child: Column(
              children: [
                const Spacer(flex: 3),

                // ── Logo with glow rings ───────────────────────
                SizedBox(
                  width: 260,
                  height: 260,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      // Ring 3 (outermost)
                      _buildRing(_ring3Scale, 0.12, 160),
                      // Ring 2
                      _buildRing(_ring2Scale, 0.18, 160),
                      // Ring 1 (innermost)
                      _buildRing(_ring1Scale, 0.25, 160),
                      
                      // Shimmer border around logo
                      AnimatedBuilder(
                        animation: _shimmerController,
                        builder: (context, child) {
                          return Transform.rotate(
                            angle: _shimmerController.value * 2 * math.pi,
                            child: Container(
                              width: 172,
                              height: 172,
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(32),
                                gradient: SweepGradient(
                                  colors: [
                                    Colors.white.withValues(alpha: 0.0),
                                    Colors.white.withValues(alpha: 0.3),
                                    Colors.amber.withValues(alpha: 0.4),
                                    Colors.white.withValues(alpha: 0.0),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),

                      // Main logo
                      ScaleTransition(
                        scale: _logoScale,
                        child: FadeTransition(
                          opacity: _logoFade,
                          child: AnimatedBuilder(
                            animation: _logoRotate,
                            builder: (context, child) {
                              return Transform.rotate(
                                angle: _logoRotate.value,
                                child: child,
                              );
                            },
                            child: Container(
                              width: 160,
                              height: 160,
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(34),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withValues(alpha: 0.3),
                                    blurRadius: 30,
                                    offset: const Offset(0, 12),
                                  ),
                                  BoxShadow(
                                    color: const Color(0xFFFF6F00).withValues(alpha: 0.2),
                                    blurRadius: 40,
                                    spreadRadius: 5,
                                  ),
                                ],
                              ),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(34),
                                child: Image.asset(
                                  'assets/images/app_icon.png',
                                  fit: BoxFit.cover,
                                  errorBuilder: (_, __, ___) => Container(
                                    color: Colors.white.withValues(alpha: 0.15),
                                    child: const Icon(
                                      Icons.shield_rounded,
                                      size: 54,
                                      color: Colors.white,
                                    ),
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

                const SizedBox(height: 28),

                // ── App title ───────────────────────────────────
                AnimatedBuilder(
                  animation: _textController,
                  builder: (context, _) {
                    return Column(
                      children: [
                        // Title
                        Opacity(
                          opacity: _titleFade.value,
                          child: Transform.translate(
                            offset: Offset(0, _titleSlide.value),
                            child: ShaderMask(
                              shaderCallback: (bounds) => const LinearGradient(
                                colors: [Colors.white, Color(0xFFFFD54F)],
                                begin: Alignment.topCenter,
                                end: Alignment.bottomCenter,
                              ).createShader(bounds),
                              child: const Text(
                                'VNeTraffic',
                                style: TextStyle(
                                  fontSize: 36,
                                  fontWeight: FontWeight.w900,
                                  color: Colors.white,
                                  letterSpacing: 1.5,
                                  height: 1.2,
                                ),
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 10),

                        // Subtitle
                        Opacity(
                          opacity: _subtitleFade.value,
                          child: Transform.translate(
                            offset: Offset(0, _subtitleSlide.value),
                            child: Text(
                              _settings.tr(
                                'Hệ thống phạt nguội giao thông',
                                'Traffic Violation Detection System',
                              ),
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.white.withValues(alpha: 0.8),
                                fontWeight: FontWeight.w400,
                                letterSpacing: 0.5,
                              ),
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),

                const Spacer(flex: 2),

                // ── Progress section ────────────────────────────
                FadeTransition(
                  opacity: _logoFade,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 56),
                    child: Column(
                      children: [
                        // Animated progress bar
                        AnimatedBuilder(
                          animation: _progressAnim,
                          builder: (context, _) {
                            return Container(
                              height: 4,
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(2),
                                color: Colors.white.withValues(alpha: 0.12),
                              ),
                              child: Align(
                                alignment: Alignment.centerLeft,
                                child: FractionallySizedBox(
                                  widthFactor: _progressAnim.value,
                                  child: Container(
                                    decoration: BoxDecoration(
                                      borderRadius: BorderRadius.circular(2),
                                      gradient: const LinearGradient(
                                        colors: [
                                          Colors.white,
                                          Color(0xFFFFD54F),
                                          Colors.white,
                                        ],
                                      ),
                                      boxShadow: [
                                        BoxShadow(
                                          color: Colors.white.withValues(alpha: 0.5),
                                          blurRadius: 8,
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            );
                          },
                        ),

                        const SizedBox(height: 16),

                        // Loading text with fade transition
                        AnimatedSwitcher(
                          duration: const Duration(milliseconds: 400),
                          transitionBuilder: (child, animation) {
                            return FadeTransition(
                              opacity: animation,
                              child: SlideTransition(
                                position: Tween<Offset>(
                                  begin: const Offset(0, 0.3),
                                  end: Offset.zero,
                                ).animate(animation),
                                child: child,
                              ),
                            );
                          },
                          child: Text(
                            _loadingText,
                            key: ValueKey<int>(_loadingStep),
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.white.withValues(alpha: 0.6),
                              fontWeight: FontWeight.w500,
                              letterSpacing: 0.3,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 40),

                // ── Footer ──────────────────────────────────────
                FadeTransition(
                  opacity: _logoFade,
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 24),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.security_rounded,
                              size: 14,
                              color: Colors.white.withValues(alpha: 0.4),
                            ),
                            const SizedBox(width: 6),
                            Text(
                              'Powered by YOLOv12 & Deep Learning',
                              style: TextStyle(
                                fontSize: 11,
                                color: Colors.white.withValues(alpha: 0.4),
                                fontWeight: FontWeight.w400,
                                letterSpacing: 0.5,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 6),
                        Text(
                          '© 2025 AI Research Project',
                          style: TextStyle(
                            fontSize: 10,
                            color: Colors.white.withValues(alpha: 0.3),
                            letterSpacing: 0.3,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRing(Animation<double> scaleAnim, double maxOpacity, double size) {
    return AnimatedBuilder(
      animation: _ringController,
      builder: (context, child) {
        final scale = scaleAnim.value;
        final normalized = (scale - 1.0) / (scaleAnim.status == AnimationStatus.forward 
            ? 1.6 : 0.8);
        final opacity = (maxOpacity * (1.0 - normalized.clamp(0.0, 1.0)));
        return Transform.scale(
          scale: scale,
          child: Opacity(
            opacity: opacity.clamp(0.0, 1.0),
            child: Container(
              width: size,
              height: size,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: Colors.white,
                  width: 1.5,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

// ── Particle model ────────────────────────────────────────────
class _Particle {
  final double x, y, size, speed, opacity, delay;
  _Particle({
    required this.x,
    required this.y,
    required this.size,
    required this.speed,
    required this.opacity,
    required this.delay,
  });
}

// ── Particle painter ──────────────────────────────────────────
class _ParticlePainter extends CustomPainter {
  final List<_Particle> particles;
  final double progress;

  _ParticlePainter({required this.particles, required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    for (final p in particles) {
      final prog = ((progress + p.delay) % 1.0);
      final y = size.height * (1.0 - prog * p.speed * 3).clamp(0.0, 1.0);
      final x = size.width * p.x + math.sin(prog * math.pi * 4 + p.delay * 10) * 20;
      final fade = (math.sin(prog * math.pi) * p.opacity).clamp(0.0, 1.0);

      final paint = Paint()
        ..color = Colors.white.withValues(alpha: fade)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, p.size * 0.5);

      canvas.drawCircle(Offset(x, y), p.size, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _ParticlePainter oldDelegate) => true;
}
