import 'package:flutter/material.dart';
import 'package:traffic_violation_app/services/auth_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'dart:async';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  final _settings = AppSettings();

  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _fadeAnimation;
  late Animation<double> _slideAnimation;

  // Ring pulse animation
  late AnimationController _ringController;
  late Animation<double> _ring1Anim;
  late Animation<double> _ring2Anim;

  // Progress bar
  late AnimationController _progressController;
  late Animation<double> _progressAnim;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.elasticOut),
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeIn),
    );

    _slideAnimation = Tween<double>(begin: 30, end: 0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic),
    );

    // Pulsing rings
    _ringController = AnimationController(
      duration: const Duration(milliseconds: 2000),
      vsync: this,
    )..repeat();
    _ring1Anim = Tween<double>(begin: 1.0, end: 1.6).animate(
      CurvedAnimation(parent: _ringController, curve: Curves.easeOut),
    );
    _ring2Anim = Tween<double>(begin: 1.0, end: 1.9).animate(
      CurvedAnimation(
        parent: _ringController,
        curve: const Interval(0.3, 1.0, curve: Curves.easeOut),
      ),
    );

    // Progress bar fills over 2.8s
    _progressController = AnimationController(
      duration: const Duration(milliseconds: 2800),
      vsync: this,
    );
    _progressAnim = CurvedAnimation(parent: _progressController, curve: Curves.easeInOut);

    _controller.forward();
    Future.delayed(const Duration(milliseconds: 600), () {
      if (mounted) _progressController.forward();
    });

    Timer(const Duration(seconds: 3), () {
      if (mounted) {
        final route = AuthService().isLoggedIn ? '/home' : '/login';
        Navigator.pushReplacementNamed(context, route);
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _ringController.dispose();
    _progressController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFD32F2F), Color(0xFFB71C1C), Color(0xFF880E4F)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // ── Logo with pulsing rings ──────────────────
              SizedBox(
                width: 180,
                height: 180,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    // Ring 2 (outer)
                    AnimatedBuilder(
                      animation: _ringController,
                      builder: (context, child) {
                        return Transform.scale(
                          scale: _ring2Anim.value,
                          child: Opacity(
                            opacity: (1.0 - (_ring2Anim.value - 1.0) / 0.9).clamp(0.0, 0.3),
                            child: Container(
                              width: 110,
                              height: 110,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                border: Border.all(
                                  color: Colors.white,
                                  width: 2,
                                ),
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                    // Ring 1 (inner)
                    AnimatedBuilder(
                      animation: _ringController,
                      builder: (context, child) {
                        return Transform.scale(
                          scale: _ring1Anim.value,
                          child: Opacity(
                            opacity: (1.0 - (_ring1Anim.value - 1.0) / 0.6).clamp(0.0, 0.4),
                            child: Container(
                              width: 110,
                              height: 110,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                border: Border.all(
                                  color: Colors.white,
                                  width: 2,
                                ),
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                    // Main icon
                    ScaleTransition(
                      scale: _scaleAnimation,
                      child: FadeTransition(
                        opacity: _fadeAnimation,
                        child: Container(
                          width: 110,
                          height: 110,
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(30),
                            border: Border.all(
                              color: Colors.white.withValues(alpha: 0.25),
                              width: 2,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withValues(alpha: 0.15),
                                blurRadius: 30,
                                offset: const Offset(0, 10),
                              ),
                            ],
                          ),
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              Icon(
                                Icons.shield_rounded,
                                size: 54,
                                color: Colors.white.withValues(alpha: 0.95),
                              ),
                              const Positioned(
                                bottom: 20,
                                child: Icon(
                                  Icons.local_police_rounded,
                                  size: 18,
                                  color: Colors.amber,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // ── App Name ───────────────────────────────────
              AnimatedBuilder(
                animation: _fadeAnimation,
                builder: (context, child) {
                  return Opacity(
                    opacity: _fadeAnimation.value,
                    child: Transform.translate(
                      offset: Offset(0, _slideAnimation.value),
                      child: child,
                    ),
                  );
                },
                child: Column(
                  children: [
                    const Text(
                      'VNeTraffic',
                      style: TextStyle(
                        fontSize: 30,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _settings.tr(
                        'Hệ thống phạt nguội giao thông',
                        'Traffic Violation Detection System',
                      ),
                      style: TextStyle(
                        fontSize: 15,
                        color: Colors.white.withValues(alpha: 0.85),
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 50),

              // ── Progress Bar ───────────────────────────────
              FadeTransition(
                opacity: _fadeAnimation,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 60),
                  child: Column(
                    children: [
                      AnimatedBuilder(
                        animation: _progressAnim,
                        builder: (context, _) {
                          return ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: _progressAnim.value,
                              backgroundColor: Colors.white.withValues(alpha: 0.15),
                              valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                              minHeight: 4,
                            ),
                          );
                        },
                      ),
                      const SizedBox(height: 14),
                      Text(
                        _settings.tr('Đang tải...', 'Loading...'),
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.white.withValues(alpha: 0.7),
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
