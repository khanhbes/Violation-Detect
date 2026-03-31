import 'package:flutter/material.dart';
import 'package:traffic_violation_app/widgets/violation_image.dart';

class ZoomImageViewerScreen extends StatefulWidget {
  final String imageUrl;
  final String heroTag;

  const ZoomImageViewerScreen({
    super.key,
    required this.imageUrl,
    required this.heroTag,
  });

  @override
  State<ZoomImageViewerScreen> createState() => _ZoomImageViewerScreenState();
}

class _ZoomImageViewerScreenState extends State<ZoomImageViewerScreen> {
  final TransformationController _controller = TransformationController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _resetTransform() {
    _controller.value = Matrix4.identity();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: GestureDetector(
        onDoubleTap: _resetTransform,
        child: Center(
          child: Hero(
            tag: widget.heroTag,
            child: InteractiveViewer(
              transformationController: _controller,
              minScale: 1,
              maxScale: 5,
              panEnabled: true,
              scaleEnabled: true,
              child: ViolationImage(
                imageUrl: widget.imageUrl,
                fit: BoxFit.contain,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
