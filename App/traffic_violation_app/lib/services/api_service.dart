import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;
import 'package:traffic_violation_app/models/violation.dart';

/// Service to communicate with the Detection Web backend.
/// Uses WebSocket for real-time push + HTTP polling as fallback.
class ApiService {
  // ── Configuration ──────────────────────────────────────────────
  // Change this to your backend server IP (same WiFi network)
  static String serverIp = '192.168.1.12';
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
  static const int _maxReconnectAttempts = 20;

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
    _disconnectWebSocket();

    try {
      print('📱 Connecting WebSocket: $wsUrl');
      _wsChannel = WebSocketChannel.connect(Uri.parse(wsUrl));

      _wsChannel!.stream.listen(
        (message) => _handleWsMessage(message),
        onDone: () {
          print('📱 WebSocket closed');
          _onWsDisconnected();
        },
        onError: (error) {
          print('📱 WebSocket error: $error');
          _onWsDisconnected();
        },
      );

      // Start ping timer to keep connection alive (every 25s)
      _pingTimer?.cancel();
      _pingTimer = Timer.periodic(const Duration(seconds: 25), (_) {
        _sendWsMessage({'action': 'ping'});
      });
    } catch (e) {
      print('📱 WebSocket connect error: $e');
      _scheduleReconnect();
    }
  }

  void _handleWsMessage(dynamic rawMessage) {
    try {
      final msg = json.decode(rawMessage as String) as Map<String, dynamic>;
      final type = msg['type'] as String? ?? '';

      switch (type) {
        case 'connected':
          // Welcome message from server
          _wsConnected = true;
          _isConnected = true;
          _reconnectAttempts = 0;
          _connectionStream.add(true);
          print('📱 WebSocket connected! '
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
            print('🚨 Real-time violation: ${violation.violationType}');
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
          // Heartbeat response — connection alive
          break;

        case 'stats':
          // Stats response (can be used by UI)
          break;
      }
    } catch (e) {
      print('📱 WS message parse error: $e');
    }
  }

  void _sendWsMessage(Map<String, dynamic> data) {
    try {
      _wsChannel?.sink.add(json.encode(data));
    } catch (e) {
      print('📱 WS send error: $e');
    }
  }

  void _onWsDisconnected() {
    _wsConnected = false;
    _isConnected = false;
    _pingTimer?.cancel();
    _connectionStream.add(false);
    _scheduleReconnect();
  }

  void _scheduleReconnect() {
    if (_reconnectAttempts >= _maxReconnectAttempts) {
      print('📱 Max reconnect attempts reached, falling back to polling');
      startPolling();
      return;
    }

    _reconnectTimer?.cancel();
    // Exponential backoff: 1s, 2s, 4s, 8s, ... max 30s
    final delay = Duration(
        seconds: (1 << _reconnectAttempts).clamp(1, 30));
    _reconnectAttempts++;
    print('📱 Reconnecting in ${delay.inSeconds}s (attempt $_reconnectAttempts)');

    _reconnectTimer = Timer(delay, () {
      connectWebSocket();
    });
  }

  void _disconnectWebSocket() {
    _pingTimer?.cancel();
    _reconnectTimer?.cancel();
    try {
      _wsChannel?.sink.close(ws_status.goingAway);
    } catch (_) {}
    _wsChannel = null;
    _wsConnected = false;
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
  void startPolling({Duration interval = const Duration(seconds: 10)}) {
    stopPolling();
    fetchViolations();
    _pollTimer = Timer.periodic(interval, (_) => fetchViolations());
  }

  void stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  // ── Dispose ────────────────────────────────────────────────────
  void dispose() {
    stopPolling();
    _disconnectWebSocket();
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
    connectWebSocket();
  }
}
