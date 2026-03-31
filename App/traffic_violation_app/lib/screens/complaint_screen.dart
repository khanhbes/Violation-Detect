import 'dart:async';
import 'dart:io';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

class ComplaintScreen extends StatefulWidget {
  const ComplaintScreen({super.key});

  @override
  State<ComplaintScreen> createState() => _ComplaintScreenState();
}

class _ComplaintScreenState extends State<ComplaintScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final FirestoreService _firestore = FirestoreService();
  final AppSettings _s = AppSettings();
  List<Violation> _violations = [];
  List<Map<String, dynamic>> _complaints = [];
  bool _isLoading = true;
  StreamSubscription? _sub;
  StreamSubscription? _complaintsSub;
  String? _boundViolationsUid;
  String? _boundComplaintsUid;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _s.addListener(_onSettingsChanged);
    _loadViolations();
    _loadComplaints();
  }

  void _loadViolations() {
    final uid = _resolveUid();
    if (uid == null) {
      _boundViolationsUid = null;
      _sub?.cancel();
      _sub = null;
      if (mounted) {
        setState(() {
          _violations = [];
          _isLoading = false;
        });
      }
      return;
    }

    if (_boundViolationsUid == uid && _sub != null) return;
    _boundViolationsUid = uid;

    _sub?.cancel();
    _sub = _firestore.violationsStream(userId: uid).listen((violations) {
      if (mounted) {
        setState(() {
          _violations = violations;
          _isLoading = false;
        });
      }
    });
  }

  void _loadComplaints() {
    final uid = _resolveUid();
    if (uid == null) {
      _boundComplaintsUid = null;
      _complaintsSub?.cancel();
      _complaintsSub = null;
      if (mounted) {
        setState(() {
          _complaints = [];
        });
      }
      return;
    }

    if (_boundComplaintsUid == uid && _complaintsSub != null) return;
    _boundComplaintsUid = uid;

    _complaintsSub?.cancel();
    _complaintsSub = _firestore.complaintsStream(uid).listen(
      (complaints) {
        if (mounted) {
          setState(() {
            _complaints = complaints;
          });
        }
      },
      onError: (error) {
        debugPrint('⚠️ Complaints stream error: $error');
        // Keep existing data on transient errors
      },
    );
  }

  String? _resolveUid() {
    // Prefer FirebaseAuth uid to avoid UID mismatch with Firestore rules
    final authUid = fb.FirebaseAuth.instance.currentUser?.uid.trim();
    if (authUid != null && authUid.isNotEmpty) return authUid;
    final settingsUid = _s.uid?.trim();
    if (settingsUid != null && settingsUid.isNotEmpty) return settingsUid;
    return null;
  }

  void _onSettingsChanged() {
    _loadViolations();
    _loadComplaints();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _sub?.cancel();
    _complaintsSub?.cancel();
    _s.removeListener(_onSettingsChanged);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.surfaceColor,
      appBar: AppBar(
        title: Text(_s.tr('Khiếu nại vi phạm', 'File Complaint')),
        backgroundColor: AppTheme.primaryColor,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: Column(
        children: [
          // Header
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
            decoration: const BoxDecoration(
              color: AppTheme.primaryColor,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(24),
                bottomRight: Radius.circular(24),
              ),
            ),
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(AppTheme.radiusL),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 44,
                        height: 44,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(Icons.info_outline_rounded,
                            color: Colors.white, size: 22),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          _s.tr(
                              'Bạn có quyền khiếu nại trong vòng 30 ngày kể từ ngày xử phạt.',
                              'You have the right to file a complaint within 30 days of the penalty.'),
                          style: TextStyle(
                              color: Colors.white.withOpacity(0.9),
                              fontSize: 13,
                              height: 1.4),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          // Tab Bar
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.grey.withOpacity(0.1),
                borderRadius: BorderRadius.circular(AppTheme.radiusM),
              ),
              child: TabBar(
                controller: _tabController,
                indicator: BoxDecoration(
                  borderRadius: BorderRadius.circular(AppTheme.radiusM),
                  color: AppTheme.primaryColor,
                ),
                indicatorSize: TabBarIndicatorSize.tab,
                labelColor: Colors.white,
                unselectedLabelColor: AppTheme.textSecondary,
                dividerColor: Colors.transparent,
                tabs: [
                  Tab(text: _s.tr('Có thể khiếu nại', 'Complainable')),
                  Tab(text: _s.tr('Đã khiếu nại', 'Complained')),
                ],
              ),
            ),
          ),

          Expanded(
            child: RefreshIndicator(
              onRefresh: () async {
                _loadViolations();
                _loadComplaints();
                setState(() {});
              },
              color: AppTheme.primaryColor,
              child: TabBarView(
                controller: _tabController,
                children: [
                  _buildComplainableList(),
                  _buildComplainedList(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildComplainableList() {
    final pendingComplaintViolationIds = _complaints
        .where((c) => (c['status'] ?? '').toString().toLowerCase() == 'pending')
        .map((c) => (c['violationId'] ?? '').toString().trim())
        .where((id) => id.isNotEmpty)
        .toSet();
    final complainable = _violations
        .where((v) =>
            v.canComplain && !pendingComplaintViolationIds.contains(v.id))
        .toList();

    if (_isLoading)
      return const Center(
          child: CircularProgressIndicator(color: AppTheme.primaryColor));
    if (complainable.isEmpty) {
      return _buildEmptyState(
        icon: Icons.verified_rounded,
        title: _s.tr('Không có vi phạm nào', 'No violations'),
        subtitle: _s.tr(
          'Hiện không có vi phạm nào có thể khiếu nại (hoặc đang chờ phản hồi)',
          'No violations are currently complainable (or they are awaiting review)',
        ),
      );
    }
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: complainable.length,
      itemBuilder: (context, index) =>
          _buildComplaintCard(complainable[index], index),
    );
  }

  Widget _buildComplainedList() {
    if (_complaints.isEmpty) {
      return _buildEmptyState(
        icon: Icons.fact_check_outlined,
        title: _s.tr('Chưa có khiếu nại', 'No complaints yet'),
        subtitle: _s.tr('Các khiếu nại đã gửi sẽ hiển thị ở đây',
            'Submitted complaints will appear here'),
      );
    }
    final df = DateFormat('HH:mm — dd/MM/yyyy');
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _complaints.length,
      itemBuilder: (context, index) {
        final c = _complaints[index];
        final createdAt = c['createdAt'];
        DateTime timestamp = DateTime.now();
        if (createdAt is DateTime) {
          timestamp = createdAt;
        } else if (createdAt != null) {
          try {
            timestamp = createdAt.toDate();
          } catch (_) {}
        }
        final status = c['status'] ?? 'pending';
        Color statusColor = AppTheme.warningColor;
        String statusText = _s.tr('Chờ phản hồi', 'Awaiting response');
        IconData statusIcon = Icons.hourglass_top_rounded;
        if (status == 'approved') {
          statusColor = AppTheme.successColor;
          statusText = _s.tr('Đã chấp nhận', 'Approved');
          statusIcon = Icons.check_circle_rounded;
        } else if (status == 'rejected') {
          statusColor = AppTheme.dangerColor;
          statusText = _s.tr('Đã từ chối', 'Rejected');
          statusIcon = Icons.cancel_rounded;
        }
        return Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            boxShadow: AppTheme.cardShadow,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Flexible(
                    child: Text(df.format(timestamp),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                            fontSize: 12, color: AppTheme.textSecondary)),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: statusColor.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(statusIcon, size: 12, color: statusColor),
                        const SizedBox(width: 4),
                        ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 120),
                          child: Text(statusText,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                  color: statusColor)),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                '${_s.tr('Lý do', 'Reason')}: ${c['reason'] ?? ''}',
                style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                    color: AppTheme.textPrimary),
              ),
              const SizedBox(height: 6),
              Text(c['description'] ?? '',
                  style: const TextStyle(
                      fontSize: 13, color: AppTheme.textSecondary)),
              // Show rejection reason if rejected
              if (status == 'rejected' &&
                  (c['adminNote'] ?? '').isNotEmpty) ...[
                const SizedBox(height: 10),
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AppTheme.dangerColor.withOpacity(0.08),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(
                        color: AppTheme.dangerColor.withOpacity(0.2)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline,
                          color: AppTheme.dangerColor, size: 16),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          '${_s.tr('Lý do từ chối', 'Rejection reason')}: ${c['adminNote']}',
                          style: const TextStyle(
                              fontSize: 12, color: AppTheme.dangerColor),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              // Delete button for approved complaints
              if (status == 'approved') ...[                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: () => _confirmDeleteComplaint(c),
                    icon: const Icon(Icons.delete_outline_rounded,
                        size: 16, color: AppTheme.dangerColor),
                    label: Text(
                      _s.tr('Xóa khiếu nại', 'Delete complaint'),
                      style: const TextStyle(
                          color: AppTheme.dangerColor, fontSize: 13),
                    ),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: AppTheme.dangerColor),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8)),
                      padding: const EdgeInsets.symmetric(vertical: 8),
                    ),
                  ),
                ),
              ],
              // Show evidence image if available
              if ((c['evidenceUrl'] ?? '').isNotEmpty) ...[
                const SizedBox(height: 10),
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.network(c['evidenceUrl'],
                      height: 120, width: double.infinity, fit: BoxFit.cover),
                ),
              ],
            ],
          ),
        );
      },
    );
  }

  Widget _buildEmptyState(
      {required IconData icon,
      required String title,
      required String subtitle}) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
                color: AppTheme.primaryColor.withOpacity(0.08),
                shape: BoxShape.circle),
            child: Icon(icon, size: 36, color: AppTheme.primaryColor),
          ),
          const SizedBox(height: 16),
          Text(title,
              style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: AppTheme.textPrimary)),
          const SizedBox(height: 6),
          Text(subtitle,
              style:
                  const TextStyle(fontSize: 13, color: AppTheme.textSecondary)),
        ],
      ),
    );
  }

  void _confirmDeleteComplaint(Map<String, dynamic> complaint) {
    final complaintId = complaint['id'] as String? ?? '';
    if (complaintId.isEmpty) return;

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(_s.tr('Xác nhận xóa', 'Confirm Delete')),
        content: Text(_s.tr(
          'Bạn có chắc muốn xóa khiếu nại đã được chấp nhận này?',
          'Are you sure you want to delete this approved complaint?',
        )),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(_s.tr('Hủy', 'Cancel')),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              try {
                await _firestore.deleteApprovedComplaint(complaintId);
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                    content: Text(_s.tr(
                        'Đã xóa khiếu nại', 'Complaint deleted')),
                    backgroundColor: AppTheme.successColor,
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(AppTheme.radiusM)),
                  ));
                }
              } catch (e) {
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                    content: Text(_s.tr(
                        'Không thể xóa khiếu nại',
                        'Failed to delete complaint')),
                    backgroundColor: AppTheme.dangerColor,
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(AppTheme.radiusM)),
                  ));
                }
              }
            },
            style: TextButton.styleFrom(foregroundColor: AppTheme.dangerColor),
            child: Text(_s.tr('Xóa', 'Delete')),
          ),
        ],
      ),
    );
  }

  Widget _buildComplaintCard(Violation v, int index) {
    final df = DateFormat('HH:mm — dd/MM/yyyy');
    final formatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: Duration(milliseconds: 350 + index * 80),
      curve: Curves.easeOutCubic,
      builder: (context, anim, child) => Opacity(
        opacity: anim,
        child: Transform.translate(
            offset: Offset(0, 15 * (1 - anim)), child: child),
      ),
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          boxShadow: AppTheme.cardShadow,
        ),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                        color: AppTheme.primaryColor.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(12)),
                    child: const Icon(Icons.warning_amber_rounded,
                        color: AppTheme.primaryColor, size: 24),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(v.violationType,
                            style: const TextStyle(
                                fontWeight: FontWeight.w600,
                                fontSize: 14,
                                color: AppTheme.textPrimary)),
                        const SizedBox(height: 3),
                        Text(df.format(v.timestamp),
                            style: const TextStyle(
                                fontSize: 12, color: AppTheme.textSecondary)),
                      ],
                    ),
                  ),
                  Text(formatter.format(v.fineAmount),
                      style: const TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 15,
                          color: AppTheme.primaryColor)),
                ],
              ),
            ),
            Container(height: 1, color: AppTheme.dividerColor),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  Expanded(
                    child: GestureDetector(
                      onTap: () => Navigator.pushNamed(
                          context, '/violation-detail',
                          arguments: v),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.visibility_outlined,
                              size: 16, color: AppTheme.infoColor),
                          const SizedBox(width: 6),
                          Text(_s.tr('Xem chi tiết', 'View detail'),
                              style: const TextStyle(
                                  color: AppTheme.infoColor,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600)),
                        ],
                      ),
                    ),
                  ),
                  Container(width: 1, height: 20, color: AppTheme.dividerColor),
                  Expanded(
                    child: GestureDetector(
                      onTap: () async {
                        final success =
                            await _showComplaintDialog(v) ?? false;
                        if (success && mounted) {
                          showDialog(
                            context: context,
                            useRootNavigator: true,
                            builder: (ctx) => AlertDialog(
                              icon: const Icon(
                                  Icons.check_circle_rounded,
                                  color: AppTheme.successColor,
                                  size: 48),
                              title: Text(_s.tr(
                                  'Gửi khiếu nại thành công',
                                  'Complaint Submitted')),
                              content: Text(_s.tr(
                                'Khiếu nại của bạn đã được gửi đi. Vi phạm này đang chờ phản hồi từ cơ quan chức năng.',
                                'Your complaint has been submitted. This violation is now awaiting review.',
                              )),
                              actions: [
                                TextButton(
                                  onPressed: () => Navigator.pop(ctx),
                                  child: Text(_s.tr('OK', 'OK')),
                                ),
                              ],
                            ),
                          );
                        }
                      },
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.rate_review_outlined,
                              size: 16, color: AppTheme.primaryColor),
                          const SizedBox(width: 6),
                          Text(_s.tr('Gửi khiếu nại', 'Submit complaint'),
                              style: const TextStyle(
                                  color: AppTheme.primaryColor,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600)),
                        ],
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

  Future<bool?> _showComplaintDialog(Violation v) {
    final controller = TextEditingController();
    String? selectedReason;
    File? evidenceImage;
    bool isUploading = false;

    final reasons = [
      _s.tr('Nhầm lẫn biển số xe', 'Wrong license plate'),
      _s.tr('Không phải phương tiện của tôi', 'Not my vehicle'),
      _s.tr('Thông tin vi phạm không chính xác', 'Incorrect violation info'),
      _s.tr('Hình ảnh không rõ ràng', 'Unclear image'),
      _s.tr('Tín hiệu giao thông hỏng', 'Broken traffic signal'),
      _s.tr('Lý do khác', 'Other reason'),
    ];

    return showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) {
          Future<void> pickImage() async {
            final picker = ImagePicker();
            final picked = await picker.pickImage(
                source: ImageSource.gallery, imageQuality: 70);
            if (picked != null) {
              setModalState(() => evidenceImage = File(picked.path));
            }
          }

          return Container(
            padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom),
            decoration: const BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
            ),
            child: SingleChildScrollView(
              child: Padding(
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
                            borderRadius: BorderRadius.circular(2)),
                      ),
                    ),
                    const SizedBox(height: 20),
                    Text(_s.tr('Gửi khiếu nại', 'Submit Complaint'),
                        style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w700,
                            color: AppTheme.textPrimary)),
                    const SizedBox(height: 6),
                    Text('${_s.tr('Vi phạm', 'Violation')}: ${v.violationType}',
                        style: const TextStyle(
                            fontSize: 14, color: AppTheme.textSecondary)),
                    const SizedBox(height: 20),

                    // Reason selector
                    RichText(
                      text: TextSpan(
                        style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: AppTheme.textPrimary),
                        children: [
                          TextSpan(
                              text: _s.tr(
                                  'Lý do khiếu nại ', 'Complaint reason ')),
                          const TextSpan(
                              text: '*',
                              style: TextStyle(color: AppTheme.dangerColor)),
                        ],
                      ),
                    ),
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: reasons.map((reason) {
                        final isSelected = selectedReason == reason;
                        return GestureDetector(
                          onTap: () =>
                              setModalState(() => selectedReason = reason),
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 200),
                            padding: const EdgeInsets.symmetric(
                                horizontal: 14, vertical: 8),
                            decoration: BoxDecoration(
                              color: isSelected
                                  ? AppTheme.primaryColor
                                  : Colors.white,
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(
                                  color: isSelected
                                      ? AppTheme.primaryColor
                                      : Colors.grey[300]!),
                            ),
                            child: Text(
                              reason,
                              style: TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w500,
                                  color: isSelected
                                      ? Colors.white
                                      : AppTheme.textPrimary),
                            ),
                          ),
                        );
                      }).toList(),
                    ),
                    const SizedBox(height: 20),

                    // Description
                    RichText(
                      text: TextSpan(
                        style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: AppTheme.textPrimary),
                        children: [
                          TextSpan(
                              text: _s.tr(
                                  'Mô tả chi tiết ', 'Detailed description ')),
                          const TextSpan(
                              text: '*',
                              style: TextStyle(color: Colors.red)),
                        ],
                      ),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: controller,
                      maxLines: 4,
                      decoration: InputDecoration(
                        hintText: _s.tr(
                            'Nhập mô tả chi tiết lý do khiếu nại...',
                            'Enter detailed complaint description...'),
                        hintStyle: const TextStyle(color: AppTheme.textHint),
                        filled: true,
                        fillColor: AppTheme.surfaceColor,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(AppTheme.radiusM),
                          borderSide: BorderSide.none,
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),

                    // Evidence image picker
                    Text(
                        _s.tr('Ảnh bằng chứng (tùy chọn)',
                            'Evidence photo (optional)'),
                        style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: AppTheme.textPrimary)),
                    const SizedBox(height: 10),
                    GestureDetector(
                      onTap: pickImage,
                      child: Container(
                        height: 120,
                        width: double.infinity,
                        decoration: BoxDecoration(
                          color: AppTheme.surfaceColor,
                          borderRadius: BorderRadius.circular(AppTheme.radiusM),
                          border: Border.all(
                              color: Colors.grey[300]!,
                              style: BorderStyle.solid),
                        ),
                        child: evidenceImage != null
                            ? Stack(
                                fit: StackFit.expand,
                                children: [
                                  ClipRRect(
                                    borderRadius:
                                        BorderRadius.circular(AppTheme.radiusM),
                                    child: Image.file(evidenceImage!,
                                        fit: BoxFit.contain),
                                  ),
                                  Positioned(
                                    top: 6,
                                    right: 6,
                                    child: GestureDetector(
                                      onTap: () => setModalState(
                                          () => evidenceImage = null),
                                      child: Container(
                                        padding: const EdgeInsets.all(4),
                                        decoration: const BoxDecoration(
                                            color: Colors.black54,
                                            shape: BoxShape.circle),
                                        child: const Icon(Icons.close,
                                            size: 16, color: Colors.white),
                                      ),
                                    ),
                                  ),
                                ],
                              )
                            : Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(Icons.add_photo_alternate_rounded,
                                      size: 32, color: Colors.grey[400]),
                                  const SizedBox(height: 8),
                                  Text(
                                      _s.tr('Nhấn để tải ảnh bằng chứng',
                                          'Tap to add evidence photo'),
                                      style: TextStyle(
                                          fontSize: 13,
                                          color: Colors.grey[500])),
                                ],
                              ),
                      ),
                    ),
                    const SizedBox(height: 24),

                    // Submit button
                    SizedBox(
                      width: double.infinity,
                      height: 50,
                      child: ElevatedButton.icon(
                        onPressed: isUploading
                            ? null
                            : () async {
                                // Validation
                                if (selectedReason == null) {
                                  ScaffoldMessenger.of(context)
                                      .showSnackBar(SnackBar(
                                    content: Text(_s.tr(
                                        'Vui lòng chọn lý do khiếu nại',
                                        'Please select a complaint reason')),
                                    backgroundColor: AppTheme.dangerColor,
                                  ));
                                  return;
                                }
                                if (controller.text.trim().isEmpty) {
                                  ScaffoldMessenger.of(context)
                                      .showSnackBar(SnackBar(
                                    content: Text(_s.tr(
                                        'Vui lòng nhập mô tả chi tiết',
                                        'Please enter a detailed description')),
                                    backgroundColor: AppTheme.dangerColor,
                                  ));
                                  return;
                                }

                                final uid = _resolveUid();
                                if (uid == null) return;
                                setModalState(() => isUploading = true);

                                try {
                                  await _firestore.submitComplaint(
                                    userId: uid,
                                    violationId: v.id,
                                    reason: selectedReason!,
                                    description: controller.text.trim(),
                                    evidenceFile: evidenceImage,
                                  );

                                  if (context.mounted) {
                                    Navigator.pop(context, true);
                                  }
                                } catch (e) {
                                  setModalState(() => isUploading = false);
                                  if (context.mounted) {
                                    final raw = e.toString();
                                    final readable =
                                        raw.startsWith('Exception:')
                                            ? raw
                                                .replaceFirst('Exception:', '')
                                                .trim()
                                            : raw;
                                    ScaffoldMessenger.of(context)
                                        .showSnackBar(SnackBar(
                                      content: Row(
                                        children: [
                                          const Icon(Icons.error_outline,
                                              color: Colors.white, size: 20),
                                          const SizedBox(width: 8),
                                          Expanded(
                                            child: Text(
                                              readable.isNotEmpty
                                                  ? readable
                                                  : _s.tr(
                                                      'Lỗi gửi khiếu nại',
                                                      'Error submitting complaint'),
                                            ),
                                          ),
                                        ],
                                      ),
                                      backgroundColor: AppTheme.dangerColor,
                                      duration: const Duration(seconds: 5),
                                    ));
                                  }
                                }
                              },
                        icon: isUploading
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(
                                    color: Colors.white, strokeWidth: 2))
                            : const Icon(Icons.send_rounded, size: 18),
                        label: Text(isUploading
                            ? _s.tr('Đang gửi...', 'Sending...')
                            : _s.tr('Gửi khiếu nại', 'Submit complaint')),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.primaryColor,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                              borderRadius:
                                  BorderRadius.circular(AppTheme.radiusM)),
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
