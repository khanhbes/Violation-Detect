import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:url_launcher/url_launcher.dart';

class QrScanScreen extends StatefulWidget {
  const QrScanScreen({super.key});

  @override
  State<QrScanScreen> createState() => _QrScanScreenState();
}

class _QrScanScreenState extends State<QrScanScreen> {
  final AppSettings _s = AppSettings();
  final FirestoreService _firestore = FirestoreService();
  final MobileScannerController _scannerController = MobileScannerController(
    formats: const [BarcodeFormat.qrCode],
    detectionSpeed: DetectionSpeed.noDuplicates,
  );

  bool _isResolving = false;
  bool _torchEnabled = false;
  String _hintText = '';
  String? _lastHandledRaw;

  @override
  void initState() {
    super.initState();
    _hintText = _s.tr(
      'Đưa QR vào khung để tự động nhận diện',
      'Place QR inside the frame to scan',
    );
  }

  @override
  void dispose() {
    _scannerController.dispose();
    super.dispose();
  }

  String? _extractViolationId(String raw) {
    final text = raw.trim();
    if (text.isEmpty) return null;

    String? fromJsonValue(dynamic value) {
      if (value is! String) return null;
      final normalized = value.trim();
      return normalized.isEmpty ? null : normalized;
    }

    try {
      final decoded = jsonDecode(text);
      if (decoded is Map) {
        final id = fromJsonValue(decoded['violationId']) ??
            fromJsonValue(decoded['violation_id']) ??
            fromJsonValue(decoded['id']) ??
            fromJsonValue(decoded['ref']) ??
            fromJsonValue(decoded['code']);
        if (id != null) return id;
      } else if (decoded is String) {
        return decoded.trim().isEmpty ? null : decoded.trim();
      }
    } catch (_) {}

    final uri = Uri.tryParse(text);
    if (uri != null) {
      final fromQuery = uri.queryParameters['violationId'] ??
          uri.queryParameters['violation_id'] ??
          uri.queryParameters['id'] ??
          uri.queryParameters['ref'] ??
          uri.queryParameters['code'];
      if (fromQuery != null && fromQuery.trim().isNotEmpty) {
        return fromQuery.trim();
      }
      if (uri.pathSegments.isNotEmpty) {
        final last = uri.pathSegments.last.trim();
        if (last.isNotEmpty) return last;
      }
    }

    final upper = text.toUpperCase();
    if (RegExp(r'^VIO_[A-Z0-9_-]{4,}$').hasMatch(upper)) {
      return upper;
    }
    // Firestore auto IDs in this project are often uppercase alphanumeric.
    if (RegExp(r'^[A-Z0-9]{16,32}$').hasMatch(upper)) {
      return upper;
    }
    return null;
  }

  Future<void> _handleRawValue(String raw) async {
    final normalizedRaw = raw.trim();
    if (normalizedRaw.isEmpty || _isResolving) return;
    if (_lastHandledRaw == normalizedRaw) return;

    setState(() {
      _isResolving = true;
      _lastHandledRaw = normalizedRaw;
      _hintText =
          _s.tr('Đang kiểm tra dữ liệu QR...', 'Resolving QR payload...');
    });

    await _scannerController.stop();

    try {
      final violationId = _extractViolationId(normalizedRaw);
      if (violationId == null) {
        final uri = Uri.tryParse(normalizedRaw);
        if (uri != null && uri.scheme.isNotEmpty) {
          await _showExternalLinkDialog(uri);
        } else {
          await _showGenericPayloadDialog(normalizedRaw);
        }
        return;
      }

      final violation = await _firestore.getViolationById(violationId);
      if (!mounted) return;
      if (violation == null) {
        _showErrorSnackBar(_s.tr(
          'Không tìm thấy vi phạm tương ứng với mã QR',
          'No violation found for this QR code',
        ));
        return;
      }

      await _showViolationActionSheet(violation);
    } finally {
      if (mounted) {
        setState(() {
          _isResolving = false;
          _hintText = _s.tr(
            'Đưa QR vào khung để tự động nhận diện',
            'Place QR inside the frame to scan',
          );
        });
        await _scannerController.start();
      }
    }
  }

  Future<void> _showExternalLinkDialog(Uri uri) async {
    await showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        return Container(
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(22)),
          ),
          child: SafeArea(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _s.tr(
                      'QR chứa liên kết ngoài', 'QR contains an external link'),
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  uri.toString(),
                  style: const TextStyle(
                      fontSize: 13, color: AppTheme.textSecondary),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: () => Navigator.pop(ctx),
                        child: Text(_s.tr('Hủy', 'Cancel')),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: () async {
                          Navigator.pop(ctx);
                          if (!await launchUrl(uri,
                              mode: LaunchMode.externalApplication)) {
                            _showErrorSnackBar(
                              _s.tr('Không mở được liên kết',
                                  'Unable to open link'),
                            );
                          }
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.primaryColor,
                          foregroundColor: Colors.white,
                        ),
                        child: Text(_s.tr('Mở link', 'Open link')),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _showGenericPayloadDialog(String payload) async {
    await showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        return Container(
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(22)),
          ),
          child: SafeArea(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _s.tr('Nội dung QR', 'QR content'),
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 10),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF4F4F5),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    payload,
                    style: const TextStyle(
                      fontSize: 13,
                      color: AppTheme.textSecondary,
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: () {
                          Clipboard.setData(ClipboardData(text: payload));
                          Navigator.pop(ctx);
                          if (!mounted) return;
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(
                                _s.tr(
                                  'Đã sao chép nội dung QR',
                                  'QR content copied',
                                ),
                              ),
                            ),
                          );
                        },
                        child: Text(_s.tr('Sao chép', 'Copy')),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: () => Navigator.pop(ctx),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.primaryColor,
                          foregroundColor: Colors.white,
                        ),
                        child: Text(_s.tr('Đã hiểu', 'Got it')),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _showViolationActionSheet(Violation violation) async {
    final isPending = violation.canPay;
    final isComplaintPending = violation.isComplaintPending;
    final formatter = violation.fineAmount.toStringAsFixed(0);

    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        return Container(
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 20),
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(22)),
          ),
          child: SafeArea(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _s.tr('Đã quét thành công', 'Scan successful'),
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 10),
                _infoRow(_s.tr('Mã vi phạm', 'Violation ID'), violation.id),
                _infoRow(_s.tr('Lỗi vi phạm', 'Violation type'),
                    violation.violationType),
                _infoRow(
                    _s.tr('Biển số', 'License plate'), violation.licensePlate),
                _infoRow(
                    _s.tr('Trạng thái', 'Status'),
                    isComplaintPending
                        ? _s.tr('Chờ phản hồi', 'Awaiting response')
                        : (isPending
                            ? _s.tr('Chưa nộp', 'Unpaid')
                            : _s.tr('Đã nộp', 'Paid'))),
                _infoRow(_s.tr('Số tiền', 'Fine amount'), '$formatter ₫'),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: () {
                          Navigator.pop(ctx);
                          Navigator.pushNamed(
                            context,
                            '/violation-detail',
                            arguments: violation,
                          );
                        },
                        child: Text(_s.tr('Xem chi tiết', 'View detail')),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: () {
                          Navigator.pop(ctx);
                          if (isComplaintPending) {
                            _showErrorSnackBar(_s.tr(
                              'Lỗi này đang chờ phản hồi khiếu nại, tạm thời không thể nộp phạt.',
                              'This violation is awaiting complaint response, payment is temporarily disabled.',
                            ));
                            return;
                          }
                          if (!isPending) {
                            _showErrorSnackBar(_s.tr(
                              'Lỗi này đã nộp phạt',
                              'This violation is already paid',
                            ));
                            return;
                          }
                          Navigator.pushNamed(
                            context,
                            '/payment',
                            arguments: violation,
                          );
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: isComplaintPending
                              ? AppTheme.warningColor
                              : (isPending
                                  ? AppTheme.primaryColor
                                  : AppTheme.textHint),
                          foregroundColor: Colors.white,
                        ),
                        child: Text(
                          isComplaintPending
                              ? _s.tr('Chờ phản hồi', 'Awaiting response')
                              : (isPending
                                  ? _s.tr('Nộp phạt', 'Pay now')
                                  : _s.tr('Đã nộp', 'Paid')),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 115,
            child: Text(
              '$label:',
              style: const TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 13,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                color: AppTheme.textPrimary,
                fontWeight: FontWeight.w600,
                fontSize: 13,
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: AppTheme.dangerColor,
      ),
    );
  }

  Future<void> _showManualInputDialog() async {
    final controller = TextEditingController();
    await showDialog(
      context: context,
      builder: (ctx) {
        return AlertDialog(
          title: Text(_s.tr('Nhập mã QR thủ công', 'Manual QR input')),
          content: TextField(
            controller: controller,
            decoration: InputDecoration(
              hintText:
                  _s.tr('Dán nội dung QR vào đây', 'Paste QR payload here'),
            ),
            minLines: 1,
            maxLines: 3,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: Text(_s.tr('Hủy', 'Cancel')),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.pop(ctx);
                final raw = controller.text.trim();
                if (raw.isNotEmpty) _handleRawValue(raw);
              },
              child: Text(_s.tr('Xác nhận', 'Confirm')),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: AppTheme.primaryColor,
        foregroundColor: Colors.white,
        title: Text(_s.tr('Quét QR Code', 'Scan QR Code')),
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: MobileScanner(
              controller: _scannerController,
              onDetect: (capture) {
                for (final barcode in capture.barcodes) {
                  final raw = barcode.rawValue;
                  if (raw != null && raw.trim().isNotEmpty) {
                    _handleRawValue(raw);
                    break;
                  }
                }
              },
            ),
          ),
          Positioned.fill(
            child: IgnorePointer(
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.45),
                ),
                child: Center(
                  child: Container(
                    width: 255,
                    height: 255,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(color: Colors.white, width: 2.6),
                    ),
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            left: 18,
            right: 18,
            top: 24,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.58),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                _hintText,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white, fontSize: 13),
              ),
            ),
          ),
          Positioned(
            left: 0,
            right: 0,
            bottom: 26,
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                child: Row(
                  children: [
                    Expanded(
                      child: _actionButton(
                        icon: _torchEnabled
                            ? Icons.flash_on_rounded
                            : Icons.flash_off_rounded,
                        label: _s.tr('Đèn', 'Torch'),
                        onTap: () async {
                          await _scannerController.toggleTorch();
                          if (!mounted) return;
                          setState(() => _torchEnabled = !_torchEnabled);
                        },
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: _actionButton(
                        icon: Icons.cameraswitch_rounded,
                        label: _s.tr('Đổi cam', 'Switch'),
                        onTap: () async {
                          await _scannerController.switchCamera();
                        },
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: _actionButton(
                        icon: Icons.keyboard_alt_rounded,
                        label: _s.tr('Nhập tay', 'Manual'),
                        onTap: _showManualInputDialog,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          if (_isResolving)
            Positioned.fill(
              child: Container(
                color: Colors.black.withOpacity(0.64),
                child: const Center(
                  child: CircularProgressIndicator(color: Colors.white),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _actionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.white.withOpacity(0.14),
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 10),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: Colors.white, size: 22),
              const SizedBox(height: 4),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
