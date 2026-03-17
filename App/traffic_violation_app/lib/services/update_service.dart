import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:traffic_violation_app/services/api_service.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:open_file/open_file.dart';
import 'package:permission_handler/permission_handler.dart';

/// OTA Update Service — checks for updates from Server API and downloads APK.
///
/// Flow:
/// 1. App starts → checkForUpdate() calls http://SERVER/api/app/latest-version
/// 2. Compares local version (1.0.0) vs server version (1.0.2)
/// 3. If newer version exists → shows premium popup dialog
/// 4. User taps "Update" → downloads APK with progress bar → installs
class UpdateService {
  // ── Singleton ──────────────────────────────────────────────────
  static final UpdateService _instance = UpdateService._internal();
  factory UpdateService() => _instance;
  UpdateService._internal();

  /// Current app version dynamically loaded via package_info_plus
  String currentVersion = '1.0.0';
  int currentBuildNumber = 1;

  // Initialize and load version info
  Future<void> init() async {
    try {
      final  info = await PackageInfo.fromPlatform();
      currentVersion = info.version;
      currentBuildNumber = int.tryParse(info.buildNumber) ?? 1;
      debugPrint('📱 App Version loaded: $currentVersion ($currentBuildNumber)');
    } catch (e) {
      debugPrint('❌ Error loading version info: $e');
    }
  }

  // Download state
  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  bool get isDownloading => _isDownloading;
  double get downloadProgress => _downloadProgress;

  // Callbacks for UI updates
  VoidCallback? onDownloadProgressChanged;

  /// Server base URL (uses ApiService's configured IP)
  static String get _serverBaseUrl => ApiService.baseUrl;

  // ═══════════════════════════════════════════════════════════════
  //  CHECK FOR UPDATE — calls server API directly
  // ═══════════════════════════════════════════════════════════════

  /// Check for update by calling http://SERVER:PORT/api/app/latest-version
  /// Returns update info map if a newer version is available, null otherwise.
  Future<Map<String, dynamic>?> checkForUpdate() async {
    try {
      debugPrint('📱 Checking for update from: $_serverBaseUrl/api/app/latest-version');

      final response = await http
          .get(
            Uri.parse('$_serverBaseUrl/api/app/latest-version'),
            headers: {'ngrok-skip-browser-warning': 'true'},
          )
          .timeout(const Duration(seconds: 5));

      if (response.statusCode != 200) {
        debugPrint('📱 Update check: server returned ${response.statusCode}');
        return null;
      }

      final data = json.decode(response.body) as Map<String, dynamic>;

      await init(); // Ensure we have latest values before checking

      final latestVersion = data['version'] as String? ?? currentVersion;
      final latestBuild = data['buildNumber'] as int? ?? currentBuildNumber;
      final rawDownloadUrl = data['downloadUrl'] as String? ?? '';
      final changelog = data['changelog'] as String? ?? '';
      final forceUpdate = data['forceUpdate'] as bool? ?? false;

      debugPrint('📱 Update check: current=$currentVersion ($currentBuildNumber), '
          'latest=$latestVersion ($latestBuild)');

      if (_isNewerVersion(latestVersion, latestBuild)) {
        // Build full download URL
        // If downloadUrl is a relative path (e.g. /api/app/download-apk/xxx.apk),
        // prepend the server base URL.
        // If it's already a full URL (e.g. https://storage.googleapis.com/...),
        // use it as-is.
        String fullDownloadUrl = rawDownloadUrl;
        if (rawDownloadUrl.isNotEmpty && !rawDownloadUrl.startsWith('http')) {
          fullDownloadUrl = '$_serverBaseUrl$rawDownloadUrl';
        }

        debugPrint('📱 ✅ Update available! v$latestVersion');
        debugPrint('📱 Download URL: $fullDownloadUrl');

        return {
          'version': latestVersion,
          'buildNumber': latestBuild,
          'downloadUrl': fullDownloadUrl,
          'changelog': changelog,
          'forceUpdate': forceUpdate,
        };
      }

      debugPrint('📱 App is up to date.');
      return null;
    } catch (e) {
      debugPrint('❌ Update check failed: $e');
      return null;
    }
  }

  /// Compare version strings (e.g., "1.0.2" > "1.0.1")
  bool _isNewerVersion(String latestVersion, int latestBuild) {
    try {
      final currentParts = currentVersion.split('.').map(int.parse).toList();
      final latestParts = latestVersion.split('.').map(int.parse).toList();

      // Pad to same length
      while (currentParts.length < 3) currentParts.add(0);
      while (latestParts.length < 3) latestParts.add(0);

      for (int i = 0; i < 3; i++) {
        if (latestParts[i] > currentParts[i]) return true;
        if (latestParts[i] < currentParts[i]) return false;
      }

      // Same version string — compare build numbers
      return latestBuild > currentBuildNumber;
    } catch (e) {
      debugPrint('❌ Version comparison error: $e');
      return false;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  DOWNLOAD APK with progress tracking
  // ═══════════════════════════════════════════════════════════════

  /// Download APK from the given URL and trigger install.
  Future<void> downloadAndInstallApk(String downloadUrl, {
    Function(double)? onProgress,
  }) async {
    if (_isDownloading) return;

    _isDownloading = true;
    _downloadProgress = 0.0;
    onDownloadProgressChanged?.call();

    try {
      // Không cần xin quyền Storage (Permission.storage) vì getExternalStorageDirectory() 
      // là bộ nhớ nội bộ của app, từ Android 10+ không cần xin quyền. 
      // Hàm xin quyền tự động bị từ chối trên Android 13+ gây lỗi.

      final uri = Uri.parse(downloadUrl);

      // Get storage directory to save APK
      final dir = await getExternalStorageDirectory();
      if (dir == null) throw Exception('Cannot access storage directory');

      final filePath = '${dir.path}/app-update.apk';
      final file = File(filePath);

      // Delete old APK if exists
      if (await file.exists()) {
        await file.delete();
      }

      debugPrint('📱 Downloading APK from: $downloadUrl');
      debugPrint('📱 Saving to: $filePath');

      // Download with progress tracking using streamed response
      final request = http.Request('GET', uri);
      // Giả lập browser header để tắt nén chunked nếu có thể
      request.headers['Accept-Encoding'] = 'identity'; 
      request.headers['ngrok-skip-browser-warning'] = 'true';
      // Tiết lộ lỗi mạng cụ thể thay vì ẩn đi 
      http.StreamedResponse response;
      try {
        response = await http.Client().send(request).timeout(const Duration(seconds: 15));
      } catch (e) {
        throw Exception('Không thể kết nối đến máy chủ: $e');
      }

      if (response.statusCode != 200) {
        throw Exception('Máy chủ gửi dữ liệu lỗi (Mã HTTP: ${response.statusCode})');
      }

      // Nếu Content-Length null (thường do transfer-encoding: chunked), dùng Header 'content-length' tự lấy
      int contentLength = response.contentLength ?? 0;
      if (contentLength == 0 && response.headers.containsKey('content-length')) {
        contentLength = int.tryParse(response.headers['content-length']!) ?? 0;
      }
      
      int bytesReceived = 0;

      final sink = file.openWrite();

      try {
        await response.stream.listen(
          (chunk) {
            sink.add(chunk);
            bytesReceived += chunk.length;
            if (contentLength > 0) {
              _downloadProgress = bytesReceived / contentLength;
              onProgress?.call(_downloadProgress);
            } else {
              onProgress?.call(-(bytesReceived.toDouble()));
            }
            onDownloadProgressChanged?.call();
          },
        ).asFuture();
      } catch (error) {
        await sink.close();
        debugPrint('❌ Download error: $error');
        throw Exception('Hỏng kết nối đang tải: $error');
      }

      await sink.close();
      debugPrint('📱 ✅ APK download complete: ${bytesReceived ~/ 1024} KB');

      // Trigger APK install
      await _installApk(filePath);

      // We don't set isDownloading to false here, because we want the UI
      // to stay disabled while the package installer is running.
      // But we call progress 1.0 to show 'Đang cài đặt...'
      onProgress?.call(1.0);
    } catch (e) {
      debugPrint('❌ Download failed: $e');
      _isDownloading = false;
      _downloadProgress = 0.0;
      onDownloadProgressChanged?.call();
      throw Exception('Lỗi tải xuống: $e');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  INSTALL APK on Android
  // ═══════════════════════════════════════════════════════════════

  /// Trigger Android APK installation using open_file.
  Future<void> _installApk(String filePath) async {
    try {
      debugPrint('📱 Triggering APK install: $filePath');

      if (Platform.isAndroid) {
        final result = await OpenFile.open(
          filePath,
          type: 'application/vnd.android.package-archive',
        );
        debugPrint('📱 Install intent result: ${result.type} - ${result.message}');
        if (result.type != ResultType.done) {
          throw Exception(result.message);
        }
      }
    } catch (e) {
      debugPrint('❌ APK install trigger failed: $e');
      throw Exception('Lỗi mở file cài đặt: $e');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  SHOW UPDATE DIALOG
  // ═══════════════════════════════════════════════════════════════

  /// Show a beautiful update dialog with changelog and download progress.
  static Future<void> showUpdateDialog(
    BuildContext context, {
    required Map<String, dynamic> updateInfo,
  }) async {
    final settings = AppSettings();
    final version = updateInfo['version'] as String;
    final changelog = updateInfo['changelog'] as String? ?? '';
    final forceUpdate = updateInfo['forceUpdate'] as bool? ?? false;
    final downloadUrl = updateInfo['downloadUrl'] as String? ?? '';

    await showDialog(
      context: context,
      barrierDismissible: !forceUpdate,
      builder: (ctx) => _UpdateDialog(
        version: version,
        changelog: changelog,
        forceUpdate: forceUpdate,
        downloadUrl: downloadUrl,
        settings: settings,
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  UPDATE DIALOG WIDGET — Premium, animated UI
// ═══════════════════════════════════════════════════════════════════════════

class _UpdateDialog extends StatefulWidget {
  final String version;
  final String changelog;
  final bool forceUpdate;
  final String downloadUrl;
  final AppSettings settings;

  const _UpdateDialog({
    required this.version,
    required this.changelog,
    required this.forceUpdate,
    required this.downloadUrl,
    required this.settings,
  });

  @override
  State<_UpdateDialog> createState() => _UpdateDialogState();
}

class _UpdateDialogState extends State<_UpdateDialog>
    with SingleTickerProviderStateMixin {
  late AnimationController _animController;
  late Animation<double> _scaleAnim;
  late Animation<double> _fadeAnim;

  bool _isDownloading = false;
  double _progress = 0.0;
  String _statusText = '';
  bool _downloadComplete = false;
  String _errorText = '';

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );
    _scaleAnim = CurvedAnimation(parent: _animController, curve: Curves.easeOutBack);
    _fadeAnim = CurvedAnimation(parent: _animController, curve: Curves.easeIn);
    _animController.forward();
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  Future<void> _startDownload() async {
    if (widget.downloadUrl.isEmpty) {
      setState(() {
        _errorText = widget.settings.tr(
          'Không tìm thấy link tải. Vui lòng thử lại sau.',
          'Download link not found. Please try again later.',
        );
      });
      return;
    }

    setState(() {
      _isDownloading = true;
      _errorText = '';
      _statusText = widget.settings.tr('Đang tải xuống...', 'Downloading...');
      _progress = 0.0;
    });

    try {
      await UpdateService().downloadAndInstallApk(
        widget.downloadUrl,
        onProgress: (progress) {
          if (mounted) {
            setState(() {
              _progress = progress;
              if (progress >= 0.0) {
                final percent = (progress * 100).toInt();
                _statusText = widget.settings.tr(
                  'Đang tải xuống... $percent%',
                  'Downloading... $percent%',
                );
                if (progress >= 1.0) {
                  _downloadComplete = true;
                  _statusText = widget.settings.tr(
                    'Tải hoàn tất! Đang mở trình cài đặt...',
                    'Download complete! Opening installer...',
                  );
                }
              } else {
                // Indeterminate (chunked encoding, no content_length)
                final mb = (-progress) / (1024 * 1024);
                _statusText = widget.settings.tr(
                  'Đang tải xuống... ${mb.toStringAsFixed(2)} MB', 
                  'Downloading... ${mb.toStringAsFixed(2)} MB'
                );
              }
            });
          }
        },
      );
      // Wait a moment for the installer to pop up
      if (mounted) {
        setState(() {
          _statusText = widget.settings.tr(
            'Đang mở màn hình cài đặt...',
            'Opening installation screen...',
          );
          // Cho phép ấn cập nhật lại nếu install bị huỷ
          _isDownloading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isDownloading = false;
          _errorText = e.toString().contains('Cleartext') 
            ? 'Cần bật Cleartext networking, đóng app và build lại.' 
            : widget.settings.tr('Lỗi: ', 'Error: ') + e.toString().replaceAll('Exception:', '').trim();
          _statusText = widget.settings.tr('Cập nhật thất bại.', 'Update failed.');
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _scaleAnim,
      child: FadeTransition(
        opacity: _fadeAnim,
        child: Dialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
          elevation: 16,
          child: Container(
            constraints: const BoxConstraints(maxWidth: 340),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(24),
              gradient: const LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Colors.white, Color(0xFFF8F9FA)],
              ),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // ── Header with icon ──────────────────────────
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.fromLTRB(24, 28, 24, 20),
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Color(0xFFE53935), Color(0xFFD32F2F)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(24),
                      topRight: Radius.circular(24),
                    ),
                  ),
                  child: Column(
                    children: [
                      Container(
                        width: 64,
                        height: 64,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.system_update_rounded,
                          color: Colors.white,
                          size: 36,
                        ),
                      ),
                      const SizedBox(height: 14),
                      Text(
                        widget.settings.tr(
                          'Cập nhật mới!',
                          'New Update Available!',
                        ),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          'v${UpdateService().currentVersion} → v${widget.version}',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                // ── Changelog ──────────────────────────────────
                if (widget.changelog.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(24, 20, 24, 0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.new_releases_rounded,
                                size: 16, color: Colors.orange[700]),
                            const SizedBox(width: 6),
                            Text(
                              widget.settings.tr('Có gì mới:', 'What\'s new:'),
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w700,
                                color: Colors.grey[800],
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: const Color(0xFFF5F5F5),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            widget.changelog,
                            style: TextStyle(
                              fontSize: 13,
                              height: 1.5,
                              color: Colors.grey[700],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                // ── Download Progress ──────────────────────────
                if (_isDownloading)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(24, 20, 24, 0),
                    child: Column(
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: LinearProgressIndicator(
                            value: _progress < 0 ? null : _progress,
                            minHeight: 8,
                            backgroundColor: Colors.grey[200],
                            valueColor: AlwaysStoppedAnimation<Color>(
                              _downloadComplete
                                  ? Colors.green
                                  : const Color(0xFFE53935),
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            SizedBox(
                              width: 16,
                              height: 16,
                              child: _downloadComplete
                                  ? const Icon(Icons.check_circle,
                                      color: Colors.green, size: 16)
                                  : const CircularProgressIndicator(
                                      strokeWidth: 2,
                                      valueColor: AlwaysStoppedAnimation<Color>(
                                          Color(0xFFE53935)),
                                    ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _statusText,
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.grey[600],
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                              if (_progress >= 0)
                                Text(
                                  '${(_progress * 100).toInt()}%',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.grey[800],
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                          ],
                        ),
                      ],
                    ),
                  ),

                // ── Error message ──────────────────────────────
                if (_errorText.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(24, 12, 24, 0),
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: Colors.red.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: Colors.red.withOpacity(0.3)),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.error_outline, color: Colors.red, size: 16),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _errorText,
                              style: const TextStyle(
                                fontSize: 12,
                                color: Colors.red,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                // ── Action Buttons ──────────────────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
                  child: Column(
                    children: [
                      // Update button
                      SizedBox(
                        width: double.infinity,
                        height: 48,
                        child: ElevatedButton(
                          onPressed: _isDownloading ? null : _startDownload,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFFE53935),
                            disabledBackgroundColor: Colors.grey[300],
                            foregroundColor: Colors.white,
                            elevation: 4,
                            shadowColor: const Color(0xFFE53935).withOpacity(0.4),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(14),
                            ),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                _isDownloading
                                    ? Icons.downloading_rounded
                                    : Icons.download_rounded,
                                size: 20,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                _isDownloading
                                    ? widget.settings.tr(
                                        'Đang tải...', 'Downloading...')
                                    : widget.settings.tr(
                                        'Cập nhật ngay', 'Update Now'),
                                style: const TextStyle(
                                  fontSize: 15,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),

                      // Skip button (only if not force update)
                      if (!widget.forceUpdate && !_isDownloading)
                        Padding(
                          padding: const EdgeInsets.only(top: 10),
                          child: TextButton(
                            onPressed: () => Navigator.of(context).pop(),
                            child: Text(
                              widget.settings.tr('Để sau', 'Later'),
                              style: TextStyle(
                                color: Colors.grey[500],
                                fontSize: 14,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
