import 'package:flutter/material.dart';
import 'package:traffic_violation_app/services/api_service.dart';

/// Normalize a violation image URL so it's always absolute and usable.
///
/// - Already absolute (`http://` / `https://`) → returned as-is.
/// - Relative path (`/snapshots/...` or `snapshots/...`) → prefixed with [ApiService.baseUrl].
/// - Empty, local OS path, or other invalid value → returns empty string.
String normalizeViolationImageUrl(String? raw) {
  if (raw == null) return '';
  final url = raw.trim();
  if (url.isEmpty) return '';

  // Already a full URL
  if (url.startsWith('http://') || url.startsWith('https://')) return url;

  // Relative web path (e.g. /snapshots/redlight/img.jpg)
  if (url.startsWith('/snapshots') || url.startsWith('snapshots')) {
    final path = url.startsWith('/') ? url : '/$url';
    return '${ApiService.baseUrl}$path';
  }

  // Anything else (local OS path like C:\..., /data/..., etc.) is invalid
  return '';
}

/// Whether a normalized URL is displayable via [Image.network].
bool isValidImageUrl(String url) {
  return url.isNotEmpty &&
      (url.startsWith('http://') || url.startsWith('https://'));
}

/// Unified widget for displaying violation images with loading / error /
/// placeholder handling.  Used across violations list, detail, home popup,
/// and zoom viewer so behaviour is consistent everywhere.
class ViolationImage extends StatelessWidget {
  final String imageUrl;
  final BoxFit fit;
  final double? width;
  final double? height;

  const ViolationImage({
    super.key,
    required this.imageUrl,
    this.fit = BoxFit.cover,
    this.width,
    this.height,
  });

  @override
  Widget build(BuildContext context) {
    final normalized = normalizeViolationImageUrl(imageUrl);
    final isDark = Theme.of(context).brightness == Brightness.dark;

    // Guard: non-finite sizes (e.g. double.infinity inside AlertDialog)
    // cause 'input.isFinite' asserts — fall back to null (unconstrained).
    final safeWidth = (width != null && width!.isFinite) ? width : null;
    final safeHeight = (height != null && height!.isFinite) ? height : null;

    if (!isValidImageUrl(normalized)) {
      return _placeholder(isDark);
    }

    return Image.network(
      normalized,
      fit: fit,
      width: safeWidth,
      height: safeHeight,
      loadingBuilder: (_, child, progress) {
        if (progress == null) return child;
        return _loadingIndicator(isDark);
      },
      errorBuilder: (_, __, ___) => _placeholder(isDark),
    );
  }

  Widget _placeholder(bool isDark) {
    final safeWidth = (width != null && width!.isFinite) ? width : null;
    final safeHeight = (height != null && height!.isFinite) ? height : null;
    return Container(
      width: safeWidth,
      height: safeHeight,
      color: isDark ? const Color(0xFF24324A) : Colors.grey[100],
      child: Icon(
        Icons.image_not_supported_rounded,
        color: isDark ? const Color(0xFF94A3B8) : Colors.grey[400],
        size: 32,
      ),
    );
  }

  Widget _loadingIndicator(bool isDark) {
    final safeWidth = (width != null && width!.isFinite) ? width : null;
    final safeHeight = (height != null && height!.isFinite) ? height : null;
    return Container(
      width: safeWidth,
      height: safeHeight,
      color: isDark ? const Color(0xFF24324A) : Colors.grey[50],
      child: const Center(
        child: SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
      ),
    );
  }
}
