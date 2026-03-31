import 'dart:async';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/models/vehicle.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

class VehiclesScreen extends StatefulWidget {
  const VehiclesScreen({super.key});

  @override
  State<VehiclesScreen> createState() => _VehiclesScreenState();
}

class _VehiclesScreenState extends State<VehiclesScreen> {
  final AppSettings _s = AppSettings();
  List<Vehicle> _vehicles = [];
  bool _isLoading = true;
  StreamSubscription? _vehicleSub;
  String? _boundUid;

  @override
  void initState() {
    super.initState();
    _s.addListener(_onSettingsChanged);
    _loadVehicles();
  }

  void _loadVehicles() {
    final uidFromSettings = _s.uid?.trim();
    final uid = (uidFromSettings != null && uidFromSettings.isNotEmpty)
        ? uidFromSettings
        : fb.FirebaseAuth.instance.currentUser?.uid.trim();
    if (uid == null || uid.isEmpty) {
      _boundUid = null;
      _vehicleSub?.cancel();
      _vehicleSub = null;
      if (mounted) {
        setState(() {
          _vehicles = [];
          _isLoading = false;
        });
      }
      return;
    }

    if (_boundUid == uid && _vehicleSub != null) return;
    _boundUid = uid;
    _vehicleSub?.cancel();
    if (mounted) setState(() => _isLoading = true);

    _vehicleSub = FirestoreService().vehiclesStream(uid).listen(
      (vehicles) {
        if (mounted) {
          setState(() {
            _vehicles = vehicles;
            _isLoading = false;
          });
        }
      },
      onError: (error, stackTrace) {
        debugPrint('❌ Vehicles stream error: $error');
        if (mounted) {
          setState(() {
            _vehicles = [];
            _isLoading = false;
          });
        }
      },
    );
  }

  void _onSettingsChanged() {
    _loadVehicles();
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _vehicleSub?.cancel();
    _boundUid = null;
    _s.removeListener(_onSettingsChanged);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.surfaceColor,
      body: Column(
        children: [
          // ── Header ──────────────────────────────────────
          Container(
            decoration: const BoxDecoration(
              gradient: AppTheme.headerGradient,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(24),
                bottomRight: Radius.circular(24),
              ),
            ),
            child: SafeArea(
              bottom: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
                child: Row(
                  children: [
                    if (Navigator.canPop(context))
                      IconButton(
                        icon: const Icon(Icons.arrow_back_rounded,
                            color: Colors.white),
                        onPressed: () => Navigator.pop(context),
                      ),
                    Expanded(
                      child: Text(
                        _s.tr('Phương tiện', 'Vehicles'),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: IconButton(
                        icon: const Icon(Icons.add_rounded,
                            color: Colors.white, size: 22),
                        onPressed: () {
                          _showAddVehicleDialog();
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // ── Vehicle List ────────────────────────────────
          Expanded(
            child: _isLoading
                ? const Center(
                    child:
                        CircularProgressIndicator(color: AppTheme.primaryColor))
                : _vehicles.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Container(
                              width: 72,
                              height: 72,
                              decoration: BoxDecoration(
                                color: AppTheme.primaryColor.withOpacity(0.08),
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(Icons.directions_car_outlined,
                                  size: 36, color: AppTheme.primaryColor),
                            ),
                            const SizedBox(height: 16),
                            Text(
                              _s.tr(
                                  'Chưa có phương tiện nào', 'No vehicles yet'),
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                                color: AppTheme.textPrimary,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              _s.tr('Thêm phương tiện để tra cứu vi phạm',
                                  'Add vehicles to look up violations'),
                              style: const TextStyle(
                                fontSize: 13,
                                color: AppTheme.textSecondary,
                              ),
                            ),
                            const SizedBox(height: 20),
                            ElevatedButton.icon(
                              onPressed: () {
                                _showAddVehicleDialog();
                              },
                              icon: const Icon(Icons.add_rounded, size: 18),
                              label: Text(
                                  _s.tr('Thêm phương tiện', 'Add vehicle')),
                            ),
                          ],
                        ),
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.all(16),
                        itemCount: _vehicles.length,
                        itemBuilder: (context, index) =>
                            _buildVehicleCard(_vehicles[index], index),
                      ),
          ),
        ],
      ),
    );
  }

  Widget _buildVehicleCard(Vehicle vehicle, int index) {
    final isMotorcycle = vehicle.vehicleType.contains('máy');

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: Duration(milliseconds: 500 + index * 150),
      curve: Curves.easeOutCubic,
      builder: (context, anim, child) {
        return Opacity(
          opacity: anim,
          child: Transform.translate(
            offset: Offset(0, 20 * (1 - anim)),
            child: child,
          ),
        );
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(AppTheme.radiusXL),
          boxShadow: AppTheme.cardShadow,
        ),
        child: Column(
          children: [
            // Vehicle header with gradient
            Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: isMotorcycle
                      ? [const Color(0xFFD32F2F), const Color(0xFFB71C1C)]
                      : [const Color(0xFF1565C0), const Color(0xFF0D47A1)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(20)),
              ),
              padding: const EdgeInsets.all(20),
              child: Row(
                children: [
                  Container(
                    width: 56,
                    height: 56,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Icon(
                      isMotorcycle
                          ? Icons.two_wheeler_rounded
                          : Icons.directions_car_rounded,
                      color: Colors.white,
                      size: 30,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          vehicle.vehicleType,
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.7),
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            vehicle.licensePlate,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 20,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 2,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            // Details
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _buildDetailRow(Icons.branding_watermark_rounded,
                      _s.tr('Hãng xe', 'Brand'), vehicle.brand),
                  const SizedBox(height: 10),
                  _buildDetailRow(Icons.directions_car_filled_rounded,
                      _s.tr('Dòng xe', 'Model'), vehicle.model),
                  const SizedBox(height: 10),
                  _buildDetailRow(Icons.palette_rounded,
                      _s.tr('Màu sắc', 'Color'), vehicle.color),
                  const SizedBox(height: 10),
                  _buildDetailRow(
                    Icons.person_rounded,
                    _s.tr('Chủ sở hữu', 'Owner'),
                    vehicle.ownerName,
                  ),
                ],
              ),
            ),
            // Action buttons
            Container(
              decoration: BoxDecoration(
                border: Border(
                    top: BorderSide(color: Colors.grey.withOpacity(0.1))),
              ),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  Expanded(
                    child: TextButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.search_rounded,
                          size: 16, color: AppTheme.infoColor),
                      label: Text(
                        _s.tr('Tra cứu vi phạm', 'Look up violations'),
                        style: const TextStyle(
                            color: AppTheme.infoColor,
                            fontWeight: FontWeight.w600,
                            fontSize: 13),
                      ),
                    ),
                  ),
                  Container(width: 1, height: 20, color: AppTheme.dividerColor),
                  Expanded(
                    child: TextButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.edit_outlined,
                          size: 16, color: AppTheme.textSecondary),
                      label: Text(
                        _s.tr('Chỉnh sửa', 'Edit'),
                        style: const TextStyle(
                            color: AppTheme.textSecondary,
                            fontWeight: FontWeight.w600,
                            fontSize: 13),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            color: AppTheme.primaryColor.withOpacity(0.06),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, size: 16, color: AppTheme.primaryColor),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 13,
                  color: AppTheme.textSecondary,
                ),
              ),
              Text(
                value,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                  color: AppTheme.textPrimary,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  void _showAddVehicleDialog() {
    final plateController = TextEditingController();
    final typeController = TextEditingController(text: 'Xe máy');
    final brandController = TextEditingController();
    final modelController = TextEditingController();
    final colorController = TextEditingController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
        ),
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                _s.tr('Thêm phương tiện mới', 'Add new vehicle'),
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.textPrimary,
                ),
              ),
              const SizedBox(height: 20),
              TextField(
                controller: plateController,
                decoration: InputDecoration(
                  labelText: _s.tr('Biển số xe', 'License Plate'),
                  hintText: 'VD: 29A-123.45',
                  border: const OutlineInputBorder(),
                ),
                textCapitalization: TextCapitalization.characters,
              ),
              const SizedBox(height: 12),
              TextField(
                controller: typeController,
                decoration: InputDecoration(
                  labelText: _s.tr(
                      'Loại xe (Xe máy/Ô tô)', 'Vehicle Type (Motorcycle/Car)'),
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: brandController,
                decoration: InputDecoration(
                  labelText: _s.tr('Hãng xe', 'Brand'),
                  hintText: 'VD: Honda, Toyota',
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: modelController,
                decoration: InputDecoration(
                  labelText: _s.tr('Dòng xe', 'Model'),
                  hintText: 'VD: SH 150i, Vios',
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: colorController,
                decoration: InputDecoration(
                  labelText: _s.tr('Màu sắc', 'Color'),
                  hintText: 'VD: Trắng, Đen',
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                height: 50,
                child: ElevatedButton.icon(
                  onPressed: () async {
                    if (plateController.text.trim().isEmpty) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(_s.tr('Vui lòng nhập biển số xe',
                              'Please enter license plate')),
                          backgroundColor: AppTheme.dangerColor,
                        ),
                      );
                      return;
                    }

                    final uid = _s.uid;
                    if (uid == null) return;

                    showDialog(
                      context: context,
                      barrierDismissible: false,
                      builder: (context) =>
                          const Center(child: CircularProgressIndicator()),
                    );

                    try {
                      final vehicle = Vehicle(
                        id: '',
                        licensePlate: plateController.text.trim().toUpperCase(),
                        vehicleType: typeController.text.trim(),
                        brand: brandController.text.trim(),
                        model: modelController.text.trim(),
                        color: colorController.text.trim(),
                        ownerName:
                            _s.userName.isNotEmpty ? _s.userName : 'Người dùng',
                        ownerId: uid,
                        registrationDate: DateTime.now(),
                      );

                      await FirestoreService().addVehicle(vehicle);

                      if (context.mounted) {
                        Navigator.pop(context); // loading
                        Navigator.pop(context); // bottom sheet
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Row(
                              children: [
                                const Icon(Icons.check_circle,
                                    color: Colors.white, size: 18),
                                const SizedBox(width: 8),
                                Text(_s.tr('Đã thêm phương tiện thành công',
                                    'Vehicle added successfully')),
                              ],
                            ),
                            backgroundColor: AppTheme.successColor,
                            behavior: SnackBarBehavior.floating,
                          ),
                        );
                      }
                    } catch (e) {
                      if (context.mounted) {
                        Navigator.pop(context); // loading
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(_s.tr('Lỗi thêm phương tiện',
                                'Error adding vehicle')),
                            backgroundColor: AppTheme.dangerColor,
                          ),
                        );
                      }
                    }
                  },
                  icon: const Icon(Icons.check_rounded, size: 18),
                  label: Text(_s.tr('Lưu phương tiện', 'Save vehicle')),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryColor,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(AppTheme.radiusM),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 10),
            ],
          ),
        ),
      ),
    );
  }
}
