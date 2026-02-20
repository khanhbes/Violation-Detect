import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;
import 'package:flutter/foundation.dart';
import 'package:traffic_violation_app/models/violation.dart';

/// Service to communicate with the Detection Web backend.
/// Uses WebSocket for real-time push + HTTP polling as fallback.
class ApiService {
  // ── Configuration ──────────────────────────────────────────────
  // Change this to your backend server IP (same WiFi network)
  static String serverIp = '192.168.1.93';
  static int serverPort = 8000;
  static String get baseUrl => 'http://$serverIp:$serverPort';
  static String get wsUrl => 'ws://$serverIp:$serverPort/ws/app';

  // Polling (fallback)
  Timer? _pollTimer;
  DateTime? _lastFetchTime;

  // Data
  final List<Violation> _violations = [];
  final StreamController<List<Violation>> _violationStream =
      StreamController<List<Violation>>.broadcast();
  final StreamController<Violation> _newViolationStream =
      StreamController<Violation>.broadcast();

  // WebSocket
  WebSocketChannel? _wsChannel;
  bool _wsConnected = false;
  Timer? _pingTimer;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  static const int _maxReconnectAttempts = 5;
  bool _intentionalDisconnect = false;
  bool _isReconnecting = false;

  // Connection state
  final StreamController<bool> _connectionStream =
      StreamController<bool>.broadcast();

  // Singleton
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  // ── Streams ────────────────────────────────────────────────────
  Stream<List<Violation>> get violationsStream => _violationStream.stream;
  Stream<Violation> get newViolationStream => _newViolationStream.stream;
  Stream<bool> get connectionStream => _connectionStream.stream;
  List<Violation> get violations => List.unmodifiable(_violations);

  // ── Connection ─────────────────────────────────────────────────
  bool _isConnected = false;
  bool get isConnected => _isConnected;
  bool get isWebSocketConnected => _wsConnected;

  Future<bool> testConnection() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/app/stats'))
          .timeout(const Duration(seconds: 5));
      _isConnected = response.statusCode == 200;
      _connectionStream.add(_isConnected);
      return _isConnected;
    } catch (e) {
      _isConnected = false;
      _connectionStream.add(false);
      return false;
    }
  }

  // ══════════════════════════════════════════════════════════════
  // WEBSOCKET — Real-time push from backend
  // ══════════════════════════════════════════════════════════════

  /// Connect to backend WebSocket for real-time violation push.
  void connectWebSocket() {
    // Prevent multiple simultaneous connection attempts
    if (_isReconnecting) return;

    _intentionalDisconnect = false;
    _disconnectWebSocket(intentional: false);

    try {
      debugPrint('📱 Connecting WebSocket: $wsUrl');
      _isReconnecting = true;

      _wsChannel = WebSocketChannel.connect(
        Uri.parse(wsUrl),
      );

      _wsChannel!.stream.listen(
        (message) {
          _isReconnecting = false;
          _handleWsMessage(message);
        },
        onDone: () {
          _isReconnecting = false;
          if (!_intentionalDisconnect) {
            debugPrint('📱 WebSocket closed unexpectedly');
            _onWsDisconnected();
          }
        },
        onError: (error) {
          _isReconnecting = false;
          debugPrint('📱 WebSocket error: $error');
          _onWsDisconnected();
        },
      );

      // Start ping timer to keep connection alive (every 25s)
      _pingTimer?.cancel();
      _pingTimer = Timer.periodic(const Duration(seconds: 25), (_) {
        if (_wsConnected) {
          _sendWsMessage({'action': 'ping'});
        }
      });
    } catch (e) {
      _isReconnecting = false;
      debugPrint('📱 WebSocket connect error: $e');
      _scheduleReconnect();
    }
  }

  void _handleWsMessage(dynamic rawMessage) {
    try {
      final msg = json.decode(rawMessage as String) as Map<String, dynamic>;
      final type = msg['type'] as String? ?? '';

      switch (type) {
        case 'connected':
          // Welcome message from server — connection is stable now
          _wsConnected = true;
          _isConnected = true;
          _reconnectAttempts = 0; // Reset reconnect counter
          _connectionStream.add(true);
          debugPrint('📱 ✅ WebSocket connected! '
              'Pending: ${msg['pending_violations']}, '
              'Total: ${msg['total_violations']}');

          // Fetch existing violations via WebSocket
          _sendWsMessage({'action': 'get_violations'});
          break;

        case 'new_violation':
          // 🚨 Real-time violation push from backend!
          final data = msg['data'] as Map<String, dynamic>?;
          if (data != null) {
            final violation = _violationFromApi(data);
            _violations.insert(0, violation);
            _newViolationStream.add(violation);
            _violationStream.add(List.unmodifiable(_violations));
            debugPrint('🚨 Real-time violation: ${violation.violationType}');
          }
          break;

        case 'violations_list':
          // Full list of violations
          final dataList = msg['data'] as List<dynamic>? ?? [];
          _violations.clear();
          for (var item in dataList.reversed) {
            _violations.add(_violationFromApi(item as Map<String, dynamic>));
          }
          _violationStream.add(List.unmodifiable(_violations));
          break;

        case 'pong':
          // Heartbeat response — connection alive, no log needed
          break;

        case 'stats':
          // Stats response (can be used by UI)
          break;
      }
    } catch (e) {
      debugPrint('📱 WS message parse error: $e');
    }
  }

  void _sendWsMessage(Map<String, dynamic> data) {
    try {
      if (_wsChannel != null) {
        _wsChannel!.sink.add(json.encode(data));
      }
    } catch (e) {
      // Silently handle send errors — connection might be closing
    }
  }

  void _onWsDisconnected() {
    if (_intentionalDisconnect) return;

    final wasConnected = _wsConnected;
    _wsConnected = false;
    _isConnected = false;
    _pingTimer?.cancel();

    // Only notify UI if state actually changed
    if (wasConnected) {
      _connectionStream.add(false);
      debugPrint('📱 Connection lost, will attempt reconnect');
    }

    _scheduleReconnect();
  }

  void _scheduleReconnect() {
    if (_intentionalDisconnect) return;

    if (_reconnectAttempts >= _maxReconnectAttempts) {
      debugPrint('📱 Max reconnect attempts reached ($_maxReconnectAttempts), '
          'falling back to polling. Tap "Reconnect" to try again.');
      _connectionStream.add(false);
      startPolling();
      return;
    }

    _reconnectTimer?.cancel();
    // Exponential backoff: 2s, 4s, 8s, 16s, 30s
    final delaySec = (2 << _reconnectAttempts).clamp(2, 30);
    _reconnectAttempts++;

    if (_reconnectAttempts <= 2) {
      // Only log first 2 attempts to avoid spam
      debugPrint('📱 Reconnecting in ${delaySec}s (attempt $_reconnectAttempts/$_maxReconnectAttempts)');
    }

    _reconnectTimer = Timer(Duration(seconds: delaySec), () {
      if (!_intentionalDisconnect) {
        connectWebSocket();
      }
    });
  }

  void _disconnectWebSocket({bool intentional = true}) {
    _pingTimer?.cancel();
    _reconnectTimer?.cancel();
    if (intentional) {
      _intentionalDisconnect = true;
    }
    try {
      _wsChannel?.sink.close(ws_status.goingAway);
    } catch (_) {}
    _wsChannel = null;
    _wsConnected = false;
    _isReconnecting = false;
  }

  /// Manually trigger a reconnect (e.g., from UI button)
  void reconnect() {
    _reconnectAttempts = 0;
    _intentionalDisconnect = false;
    stopPolling();
    connectWebSocket();
  }

  // ══════════════════════════════════════════════════════════════
  // HTTP POLLING — Fallback when WebSocket fails
  // ══════════════════════════════════════════════════════════════

  Future<List<Violation>> fetchViolations() async {
    try {
      String url = '$baseUrl/api/app/violations';
      if (_lastFetchTime != null) {
        url += '?since=${_lastFetchTime!.toIso8601String()}';
      }

      final response = await http
          .get(Uri.parse(url))
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        _isConnected = true;
        _connectionStream.add(true);
        final data = json.decode(response.body);
        final List<dynamic> items = data['violations'] ?? [];

        final newViolations = items.map((j) => _violationFromApi(j)).toList();

        if (_lastFetchTime != null) {
          for (var v in newViolations) {
            _violations.insert(0, v);
            _newViolationStream.add(v);
          }
        } else {
          _violations.clear();
          _violations.addAll(newViolations.reversed);
        }

        _lastFetchTime = DateTime.now();
        _violationStream.add(List.unmodifiable(_violations));
        return _violations;
      }
    } catch (e) {
      _isConnected = false;
      _connectionStream.add(false);
    }
    return _violations;
  }

  Future<Map<String, dynamic>?> fetchStats() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/app/stats'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      }
    } catch (_) {}
    return null;
  }

  // ── Polling ────────────────────────────────────────────────────
  void startPolling({Duration interval = const Duration(seconds: 15)}) {
    stopPolling();
    fetchViolations();
    _pollTimer = Timer.periodic(interval, (_) => fetchViolations());
    debugPrint('📱 Started polling mode (every ${interval.inSeconds}s)');
  }

  void stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  // ── Dispose ────────────────────────────────────────────────────
  void dispose() {
    stopPolling();
    _disconnectWebSocket(intentional: true);
    _violationStream.close();
    _newViolationStream.close();
    _connectionStream.close();
  }

  // ── Helpers ────────────────────────────────────────────────────
  Violation _violationFromApi(Map<String, dynamic> json) {
    return Violation(
      id: json['id'] ?? '',
      licensePlate: json['licensePlate'] ?? 'Đang xác minh',
      violationType: json['violationType'] ?? 'Vi phạm',
      violationCode: json['violationCode'] ?? '',
      description: json['description'] ?? '',
      timestamp: DateTime.tryParse(json['timestamp'] ?? '') ?? DateTime.now(),
      location: json['location'] ?? '',
      imageUrl: json['imageUrl'] != null
          ? '$baseUrl${json['imageUrl']}'
          : 'https://images.unsplash.com/photo-1449034446853-66c86144b0ad?w=800',
      fineAmount: (json['fineAmount'] ?? 0).toDouble(),
      status: json['status'] ?? 'pending',
      lawReference: json['lawReference'] ?? '',
    );
  }

  /// Update server IP at runtime
  void setServerAddress(String ip, {int port = 8000}) {
    serverIp = ip;
    serverPort = port;
    _lastFetchTime = null;
    _violations.clear();
    // Reconnect WebSocket with new address
    reconnect();
  }
}
