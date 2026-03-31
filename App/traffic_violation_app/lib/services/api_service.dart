import 'dart:async';
import 'dart:convert';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:traffic_violation_app/models/violation.dart';
import 'package:traffic_violation_app/services/app_settings.dart';
import 'package:traffic_violation_app/services/firestore_service.dart';
import 'package:web_socket_channel/status.dart' as ws_status;
import 'package:web_socket_channel/web_socket_channel.dart';

class ServerAddressConfig {
  final String ip;
  final int port;

  const ServerAddressConfig({
    required this.ip,
    required this.port,
  });
}

class CoreRefreshReport {
  final Map<String, String> taskStatus;
  final bool timedOut;

  const CoreRefreshReport({
    required this.taskStatus,
    required this.timedOut,
  });

  bool get isSuccess =>
      !timedOut && taskStatus.values.every((status) => status == 'success');
}

/// Three-state connection indicator used by the UI.
enum ConnectionStatus { disconnected, connecting, connected }

/// Service to communicate with the Detection Web backend.
/// Uses WebSocket for real-time push + HTTP polling as fallback.
class ApiService {
  // Update these when backend address changes.
  static String serverIp = '192.168.1.13';
  static int serverPort = 8000;

  static bool get _isNgrok => serverIp.contains('ngrok-free.dev');
  static String get _cleanIp =>
      _isNgrok ? serverIp.replaceAll(RegExp(r':\d+'), '') : serverIp;

  static String get baseUrl =>
      _isNgrok ? 'https://$_cleanIp' : 'http://$_cleanIp:$serverPort';
  static String get wsUrl =>
      _isNgrok ? 'wss://$_cleanIp/ws/app' : 'ws://$_cleanIp:$serverPort/ws/app';

  static const int _maxReconnectAttempts = 5;
  static const Duration _pingInterval = Duration(seconds: 25);
  static const Duration _sessionHeartbeatInterval = Duration(seconds: 30);

  // Polling (fallback)
  Timer? _pollTimer;
  DateTime? _lastFetchTime;

  // Data
  final List<Violation> _violations = <Violation>[];
  final StreamController<List<Violation>> _violationStream =
      StreamController<List<Violation>>.broadcast();
  final StreamController<Violation> _newViolationStream =
      StreamController<Violation>.broadcast();

  // WebSocket
  WebSocketChannel? _wsChannel;
  bool _wsConnected = false;
  bool _isConnected = false;
  bool _intentionalDisconnect = false;
  bool _isReconnecting = false;
  int _reconnectAttempts = 0;
  Timer? _pingTimer;
  Timer? _reconnectTimer;
  Timer? _pollWsRetryTimer;
  Timer? _serverConfigDebounceTimer;

  // Session heartbeat
  final String _sessionId = _buildRuntimeSessionId();
  late final String _deviceId = _sessionId;
  Timer? _sessionHeartbeatTimer;
  bool _sessionUpsertInFlight = false;
  bool _sessionClearInFlight = false;

  // Connection state
  final StreamController<bool> _connectionStream =
      StreamController<bool>.broadcast();
  final StreamController<ServerAddressConfig> _serverAddressStream =
      StreamController<ServerAddressConfig>.broadcast();
  StreamSubscription<DocumentSnapshot<Map<String, dynamic>>>? _serverConfigSub;
  bool _serverConfigAutoSyncStarted = false;

  // Singleton
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  // Streams
  Stream<List<Violation>> get violationsStream => _violationStream.stream;
  Stream<Violation> get newViolationStream => _newViolationStream.stream;
  Stream<bool> get connectionStream => _connectionStream.stream;
  Stream<ServerAddressConfig> get serverAddressStream =>
      _serverAddressStream.stream;
  List<Violation> get violations => List<Violation>.unmodifiable(_violations);

  bool get isConnected => _isConnected;
  bool get isWebSocketConnected => _wsConnected;
  bool get isReconnecting => _isReconnecting;

  /// Simplified 3-state connection status for UI banners.
  ConnectionStatus get connectionStatus {
    if (_isConnected) return ConnectionStatus.connected;
    if (_isReconnecting) return ConnectionStatus.connecting;
    return ConnectionStatus.disconnected;
  }

  ServerAddressConfig? _normalizeServerAddressInput(
    String rawInput, {
    int fallbackPort = 8000,
  }) {
    var input = rawInput.trim();
    if (input.isEmpty) return null;

    input = input.replaceAll(RegExp(r'[\\/]+$'), '');

    Uri? uri;
    if (input.contains('://')) {
      uri = Uri.tryParse(input);
    } else {
      uri = Uri.tryParse('http://$input');
    }

    String host = '';
    int? parsedPort;

    if (uri != null && uri.host.trim().isNotEmpty) {
      host = uri.host.trim();
      if (uri.hasPort) {
        parsedPort = uri.port;
      }
    } else {
      var sanitized = input.replaceFirst(RegExp(r'^https?://'), '');
      sanitized = sanitized.split('/').first.trim();
      final lastColon = sanitized.lastIndexOf(':');
      if (lastColon > 0 && lastColon < sanitized.length - 1) {
        host = sanitized.substring(0, lastColon).trim();
        parsedPort = int.tryParse(sanitized.substring(lastColon + 1).trim());
      } else {
        host = sanitized.trim();
      }
    }

    host = host.replaceAll('[', '').replaceAll(']', '').trim();
    if (host.isEmpty) return null;

    final effectivePort =
        (parsedPort ?? fallbackPort) <= 0 ? 8000 : (parsedPort ?? fallbackPort);

    return ServerAddressConfig(ip: host, port: effectivePort);
  }

  void startServerConfigAutoSync() {
    if (_serverConfigAutoSyncStarted) return;
    // Guard: only listen when user is authenticated (server/config requires auth)
    if (fb.FirebaseAuth.instance.currentUser == null) {
      debugPrint('â³ startServerConfigAutoSync skipped: no authenticated user');
      return;
    }
    _serverConfigAutoSyncStarted = true;
    bool firstSync = true;

    _serverConfigSub = FirebaseFirestore.instance
        .collection('server')
        .doc('config')
        .snapshots()
        .listen(
      (snapshot) {
        final data = snapshot.data();
        if (data == null) return;

        final ip = (data['ip'] ?? '').toString().trim();
        if (ip.isEmpty) return;

        final rawPort = data['port'];
        final nextPort = rawPort is num
            ? rawPort.toInt()
            : int.tryParse((rawPort ?? '').toString()) ?? serverPort;

        final parsed = _normalizeServerAddressInput(ip, fallbackPort: nextPort);
        if (parsed == null) return;

        final ipChanged = serverIp != parsed.ip || serverPort != parsed.port;
        if (!ipChanged && !firstSync) return;
        firstSync = false;

        _serverConfigDebounceTimer?.cancel();
        _serverConfigDebounceTimer = Timer(const Duration(milliseconds: 900),
            () async {
          final ok = await pingServerAddress(parsed.ip, port: parsed.port);
          if (!ok) {
            debugPrint(
              'âš ï¸ Ignored auto-sync server switch: unreachable ${parsed.ip}:${parsed.port}',
            );
            return;
          }

          final changed = setServerAddress(parsed.ip, port: parsed.port);
          debugPrint(
            'ðŸ”„ Auto-synced server: ${parsed.ip}:${parsed.port} (changed=$changed)',
          );
        });
      },
      onError: (Object error) {
        final errStr = error.toString();
        if (errStr.contains('permission-denied') ||
            errStr.contains('PERMISSION_DENIED')) {
          debugPrint(
            'ðŸ”’ Server config auto-sync: permission-denied (user may be signed out). Stopping listener.',
          );
          stopServerConfigAutoSync();
        } else {
          debugPrint('âš ï¸ Server config auto-sync listener error: $error');
        }
      },
    );
  }

  void stopServerConfigAutoSync() {
    _serverConfigSub?.cancel();
    _serverConfigSub = null;
    _serverConfigDebounceTimer?.cancel();
    _serverConfigDebounceTimer = null;
    _serverConfigAutoSyncStarted = false;
  }

  static String _buildRuntimeSessionId() {
    final micros = DateTime.now().microsecondsSinceEpoch;
    final millis = DateTime.now().millisecondsSinceEpoch;
    final suffix = (micros ^ millis).toRadixString(16);
    return 'sess_${micros}_$suffix';
  }

  String? _currentUid() {
    final uid = fb.FirebaseAuth.instance.currentUser?.uid.trim();
    if (uid == null || uid.isEmpty) return null;
    return uid;
  }

  Future<String?> _currentIdToken() async {
    final user = fb.FirebaseAuth.instance.currentUser;
    if (user == null) return null;
    try {
      final token = await user.getIdToken();
      if (token == null || token.trim().isEmpty) return null;
      return token;
    } catch (_) {
      return null;
    }
  }

  /// Public accessor for Firebase ID token (used by other services).
  Future<String?> currentIdToken() => _currentIdToken();

  Map<String, String> _buildJsonHeaders({String? idToken}) {
    final headers = <String, String>{
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'ngrok-skip-browser-warning': 'true',
    };
    final token = (idToken ?? '').trim();
    if (token.isNotEmpty) {
      headers['Authorization'] = 'Bearer $token';
    }
    return headers;
  }

  Map<String, String> _buildPlainHeaders() {
    return const <String, String>{
      'Accept': 'application/json',
      'ngrok-skip-browser-warning': 'true',
    };
  }

  Future<Uri> _buildWsUri() async {
    final baseUri = Uri.parse(wsUrl);
    final uid = _currentUid();
    if (uid == null || uid.isEmpty) {
      return baseUri;
    }

    final query = <String, String>{
      ...baseUri.queryParameters,
      'user_id': uid,
      'session_id': _sessionId,
      'device_id': _deviceId,
    };
    final idToken = await _currentIdToken();
    if (idToken != null && idToken.isNotEmpty) {
      query['id_token'] = idToken;
    }
    return baseUri.replace(queryParameters: query);
  }

  bool _shouldAcceptViolation(Violation violation) {
    if (violation.userId.trim().isEmpty) return true;
    final uid = _currentUid();
    if (uid == null || uid.isEmpty) return false;
    return violation.userId.trim() == uid;
  }

  Future<void> _upsertSessionHeartbeat({required String trigger}) async {
    final uid = _currentUid();
    if (uid == null || uid.isEmpty || _sessionUpsertInFlight) return;

    _sessionUpsertInFlight = true;
    try {
      final idToken = await _currentIdToken();
      final uri = Uri.parse('$baseUrl/api/app/session/upsert');
      final payload = <String, dynamic>{
        'user_id': uid,
        'session_id': _sessionId,
        'device_id': _deviceId,
        'last_seen_at': DateTime.now().toUtc().toIso8601String(),
      };
      final response = await http
          .post(
            uri,
            headers: _buildJsonHeaders(idToken: idToken),
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 8));

      if (response.statusCode >= 400) {
        debugPrint(
          'Session upsert failed ($trigger): ${response.statusCode} ${response.body}',
        );
      }
    } catch (e) {
      debugPrint('Session upsert error ($trigger): $e');
    } finally {
      _sessionUpsertInFlight = false;
    }
  }

  Future<void> clearSession({bool includeAuthToken = true}) async {
    if (_sessionClearInFlight) return;
    _sessionClearInFlight = true;
    try {
      final uid = _currentUid();
      final idToken = includeAuthToken ? await _currentIdToken() : null;
      final uri = Uri.parse('$baseUrl/api/app/session/clear');
      final payload = <String, dynamic>{
        'session_id': _sessionId,
        'device_id': _deviceId,
      };
      if (uid != null && uid.isNotEmpty) {
        payload['user_id'] = uid;
      }

      final response = await http
          .post(
            uri,
            headers: _buildJsonHeaders(idToken: idToken),
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 8));

      if (response.statusCode >= 400) {
        debugPrint(
          'Session clear failed: ${response.statusCode} ${response.body}',
        );
      }
    } catch (e) {
      debugPrint('Session clear error: $e');
    } finally {
      // Reset incremental fetch time after logout â†’ next login will do full sync
      _lastFetchTime = null;
      _sessionClearInFlight = false;
    }
  }

  void _startSessionHeartbeatTimer() {
    _sessionHeartbeatTimer?.cancel();
    _sessionHeartbeatTimer =
        Timer.periodic(_sessionHeartbeatInterval, (_) async {
      await _upsertSessionHeartbeat(trigger: 'timer');
    });
  }

  Future<bool> testConnection() async {
    final uid = _currentUid();
    final uri = Uri.parse('$baseUrl/api/app/stats').replace(
      queryParameters: uid == null || uid.isEmpty
          ? null
          : <String, String>{
              'user_id': uid,
            },
    );
    try {
      final response =
          await http.get(uri, headers: _buildPlainHeaders()).timeout(
                const Duration(seconds: 5),
              );
      _isConnected = response.statusCode == 200;
      _connectionStream.add(_isConnected);
      return _isConnected;
    } catch (_) {
      _isConnected = false;
      _connectionStream.add(false);
      return false;
    }
  }

  /// Connect to backend WebSocket for real-time violation push.
  void connectWebSocket() {
    unawaited(_connectWebSocket());
  }

  Future<void> _connectWebSocket() async {
    if (_isReconnecting) return;

    _intentionalDisconnect = false;
    _disconnectWebSocket(intentional: false, clearSessionOnServer: false);

    try {
      final uri = await _buildWsUri();
      debugPrint('Connecting WebSocket: $uri');
      _isReconnecting = true;
      _startSessionHeartbeatTimer();

      _wsChannel = WebSocketChannel.connect(uri);

      _wsChannel!.stream.listen(
        (message) {
          _isReconnecting = false;
          _handleWsMessage(message);
        },
        onDone: () {
          _isReconnecting = false;
          if (!_intentionalDisconnect) {
            debugPrint('WebSocket closed unexpectedly');
            _onWsDisconnected();
          }
        },
        onError: (error) {
          _isReconnecting = false;
          debugPrint('WebSocket error: $error');
          _onWsDisconnected();
        },
      );

      _pingTimer?.cancel();
      _pingTimer = Timer.periodic(_pingInterval, (_) {
        if (!_wsConnected) return;
        _sendWsMessage(<String, dynamic>{
          'action': 'ping',
          'session_id': _sessionId,
          'device_id': _deviceId,
          'last_seen_at': DateTime.now().toUtc().toIso8601String(),
        });
      });

      unawaited(_upsertSessionHeartbeat(trigger: 'ws_connect'));
    } catch (e) {
      _isReconnecting = false;
      debugPrint('WebSocket connect error: $e');
      _scheduleReconnect();
    }
  }

  void _handleWsMessage(dynamic rawMessage) {
    try {
      final msg = json.decode(rawMessage as String) as Map<String, dynamic>;
      final type = msg['type'] as String? ?? '';

      switch (type) {
        case 'connected':
          _wsConnected = true;
          _isConnected = true;
          _reconnectAttempts = 0;
          _connectionStream.add(true);
          stopPolling();

          _sendWsMessage(<String, dynamic>{
            'action': 'get_violations',
          });
          break;

        case 'new_violation':
          final data = msg['data'] as Map<String, dynamic>?;
          if (data == null) return;
          final violation = _violationFromApi(data);
          if (!_shouldAcceptViolation(violation)) {
            return;
          }
          _violations.removeWhere((v) => v.id == violation.id);
          _violations.insert(0, violation);
          _newViolationStream.add(violation);
          _violationStream.add(List<Violation>.unmodifiable(_violations));
          break;

        case 'violations_list':
          final dataList = msg['data'] as List<dynamic>? ?? <dynamic>[];
          final next = <Violation>[];
          for (final item in dataList) {
            if (item is! Map<String, dynamic>) continue;
            final violation = _violationFromApi(item);
            if (_shouldAcceptViolation(violation)) {
              next.add(violation);
            }
          }
          next.sort((a, b) => b.timestamp.compareTo(a.timestamp));
          _violations
            ..clear()
            ..addAll(next);
          _violationStream.add(List<Violation>.unmodifiable(_violations));
          break;

        case 'pong':
          break;

        case 'error':
          final message = (msg['message'] ?? '').toString();
          if (message.isNotEmpty) {
            debugPrint('WS error from server: $message');
          }
          break;

        default:
          break;
      }
    } catch (e) {
      debugPrint('WS message parse error: $e');
    }
  }

  void _sendWsMessage(Map<String, dynamic> data) {
    try {
      _wsChannel?.sink.add(json.encode(data));
    } catch (_) {
      // Ignore send errors while socket is closing/reconnecting.
    }
  }

  void _onWsDisconnected() {
    if (_intentionalDisconnect) return;

    final wasConnected = _wsConnected;
    _wsConnected = false;
    _isConnected = false;
    _pingTimer?.cancel();

    if (wasConnected) {
      _connectionStream.add(false);
      debugPrint('Connection lost, scheduling reconnect');
    }

    _scheduleReconnect();
  }

  void _scheduleReconnect() {
    if (_intentionalDisconnect) return;

    if (_reconnectAttempts >= _maxReconnectAttempts) {
      debugPrint(
        'Max reconnect attempts reached ($_maxReconnectAttempts), fallback to polling',
      );
      _connectionStream.add(false);
      startPolling();
      return;
    }

    _reconnectTimer?.cancel();
    final delaySec = (2 << _reconnectAttempts).clamp(2, 30);
    _reconnectAttempts++;

    _reconnectTimer = Timer(Duration(seconds: delaySec), () {
      if (_intentionalDisconnect) return;
      connectWebSocket();
    });
  }

  void _disconnectWebSocket({
    bool intentional = true,
    bool clearSessionOnServer = true,
  }) {
    _pingTimer?.cancel();
    _reconnectTimer?.cancel();
    _wsConnected = false;
    _isReconnecting = false;

    if (intentional) {
      _intentionalDisconnect = true;
      if (_isConnected) {
        _isConnected = false;
        _connectionStream.add(false);
      }
      _sessionHeartbeatTimer?.cancel();
      if (clearSessionOnServer) {
        unawaited(clearSession());
      }
    }

    try {
      _wsChannel?.sink.close(ws_status.normalClosure);
    } catch (_) {}
    _wsChannel = null;
  }

  /// Manually trigger a reconnect (e.g., from UI button).
  void reconnect() {
    _reconnectAttempts = 0;
    _intentionalDisconnect = false;
    stopPolling();
    connectWebSocket();
  }

  /// Called after user logs in: reconnect WebSocket so the handshake includes
  /// the correct uid & id_token for the signed-in user.
  void reconnectWithNewUser() {
    debugPrint('Reconnecting WebSocket after login with new user identity');
    _reconnectAttempts = 0;
    _intentionalDisconnect = false;
    _lastFetchTime = null;
    stopPolling();
    // Slightly longer delay so Firebase Auth state fully propagates before building WS URI.
    Future.delayed(const Duration(milliseconds: 800), connectWebSocket);
  }

  /// Reconnect after auth state changes and wait for a WS 'connected' ack.
  /// Returns true if WS confirms within [timeout], false otherwise.
  /// Does NOT short-circuit on stale _wsConnected/_isConnected state Ã¢â‚¬â€
  /// always waits for the new handshake to complete.
  Future<bool> reconnectWithNewUserAndWait({
    Duration timeout = const Duration(seconds: 10),
  }) async {
    // Reset connected flags so we don't return early from a stale session.
    _wsConnected = false;
    _isConnected = false;

    reconnectWithNewUser();

    try {
      await connectionStream
          .where((connected) => connected)
          .first
          .timeout(timeout);
      return true;
    } catch (e) {
      // Fallback: verify connectivity via HTTP health check.
      // WS timeout is not fatal when polling works fine â€” log as info.
      final httpOk = await testConnection();
      if (httpOk) {
        debugPrint('â„¹ï¸ reconnectWithNewUserAndWait: WS timeout but HTTP OK');
      } else {
        debugPrint('âš ï¸ reconnectWithNewUserAndWait: both WS and HTTP failed');
      }
      return httpOk;
    }
  }

  Future<List<Violation>> fetchViolations() async {
    final query = <String, String>{};
    
    // Incremental sync: only fetch new violations since last successful fetch
    // On fresh login/reconnect: _lastFetchTime = null â†’ full sync
    if (_lastFetchTime != null) {
      query['since'] = _lastFetchTime!.toIso8601String();
    }
    
    final uid = _currentUid();
    if (uid != null && uid.isNotEmpty) {
      query['user_id'] = uid;
    }

    final uri = Uri.parse('$baseUrl/api/app/violations').replace(
      queryParameters: query.isEmpty ? null : query,
    );

    try {
      final response = await http
          .get(uri, headers: _buildPlainHeaders())
          .timeout(const Duration(seconds: 10));

      if (response.statusCode != 200) {
        throw Exception('HTTP ${response.statusCode}');
      }

      final body = json.decode(response.body) as Map<String, dynamic>;
      final list = (body['violations'] as List<dynamic>? ?? <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .map(_violationFromApi)
          .where(_shouldAcceptViolation)
          .toList()
        ..sort((a, b) => b.timestamp.compareTo(a.timestamp));

      // MERGE logic: instead of clear + replace, merge by id
      // This prevents losing old data when polling is incremental
      final existingIds = _violations.map((v) => v.id).toSet();
      
      for (final newViolation in list) {
        if (!existingIds.contains(newViolation.id)) {
          // New violation: add and trigger newViolationStream
          _violations.add(newViolation);
          _newViolationStream.add(newViolation);
        } else {
          // Existing violation: update it if timestamp is newer (for status changes)
          final idx = _violations.indexWhere((v) => v.id == newViolation.id);
          if (idx >= 0 && newViolation.timestamp.isAfter(_violations[idx].timestamp)) {
            // Preserve valid image if new record lost it
            final merged = (newViolation.imageUrl.isEmpty &&
                    _violations[idx].imageUrl.isNotEmpty)
                ? Violation(
                    id: newViolation.id,
                    userId: newViolation.userId,
                    licensePlate: newViolation.licensePlate,
                    violationType: newViolation.violationType,
                    violationCode: newViolation.violationCode,
                    description: newViolation.description,
                    timestamp: newViolation.timestamp,
                    location: newViolation.location,
                    imageUrl: _violations[idx].imageUrl,
                    fineAmount: newViolation.fineAmount,
                    status: newViolation.status,
                    lawReference: newViolation.lawReference,
                    complaintStatus: newViolation.complaintStatus,
                    paymentLocked: newViolation.paymentLocked,
                    complaintLocked: newViolation.complaintLocked,
                    paymentDueDate: newViolation.paymentDueDate,
                    paidAt: newViolation.paidAt,
                  )
                : newViolation;
            _violations[idx] = merged;
            // Don't trigger newViolationStream for updates
          }
        }
      }
      
      // Sort by timestamp descending
      _violations.sort((a, b) => b.timestamp.compareTo(a.timestamp));
      
      _violationStream.add(List<Violation>.unmodifiable(_violations));
      _lastFetchTime = DateTime.now();
      return List<Violation>.unmodifiable(_violations);
    } catch (e) {
      debugPrint('Fetch violations error: $e');
      return List<Violation>.unmodifiable(_violations);
    }
  }

  Future<Map<String, dynamic>?> fetchStats() async {
    final uid = _currentUid();
    final uri = Uri.parse('$baseUrl/api/app/stats').replace(
      queryParameters: uid == null || uid.isEmpty
          ? null
          : <String, String>{
              'user_id': uid,
            },
    );
    try {
      final response =
          await http.get(uri, headers: _buildPlainHeaders()).timeout(
                const Duration(seconds: 10),
              );
      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      }
    } catch (e) {
      debugPrint('Fetch stats error: $e');
    }
    return null;
  }

  void startPolling({Duration interval = const Duration(seconds: 15)}) {
    stopPolling();
    fetchViolations();
    _pollTimer = Timer.periodic(interval, (_) => fetchViolations());
    _pollWsRetryTimer = Timer.periodic(const Duration(seconds: 25), (_) {
      if (_intentionalDisconnect || _wsConnected || _isReconnecting) return;
      connectWebSocket();
    });
  }

  void stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
    _pollWsRetryTimer?.cancel();
    _pollWsRetryTimer = null;
  }

  void dispose() {
    stopPolling();
    stopServerConfigAutoSync();
    _sessionHeartbeatTimer?.cancel();
    _disconnectWebSocket(intentional: true);
    _violationStream.close();
    _newViolationStream.close();
    _connectionStream.close();
    _serverAddressStream.close();
  }

  Violation _violationFromApi(Map<String, dynamic> json) {
    // Pre-resolve relative image URLs so the result has an absolute URL.
    for (final key in ['imageUrl', 'image_url', 'snapshotPath']) {
      final raw = (json[key] ?? '').toString().trim();
      if (raw.isNotEmpty &&
          !raw.startsWith('http://') &&
          !raw.startsWith('https://')) {
        final path = raw.startsWith('/') ? raw : '/$raw';
        json[key] = '$baseUrl$path';
      }
    }

    final fineRaw = json['fineAmount'];
    final fineAmount = fineRaw is num
        ? fineRaw.toDouble()
        : double.tryParse((fineRaw ?? '').toString()) ?? 0.0;

    final timestampRaw =
        (json['createdAt'] ?? json['timestamp'] ?? '').toString().trim();
    final timestamp =
        DateTime.tryParse(timestampRaw)?.toLocal() ?? DateTime.now();

    // Pick best image URL from multiple possible keys
    final imageUrl = (json['imageUrl'] ?? json['image_url'] ?? json['snapshotPath'] ?? '').toString().trim();

    return Violation(
      id: (json['id'] ?? '').toString(),
      userId: (json['userId'] ?? '').toString(),
      licensePlate: (json['licensePlate'] ?? '').toString(),
      violationType: (json['violationType'] ?? '').toString(),
      violationCode: (json['violationCode'] ?? '').toString(),
      description: (json['description'] ?? '').toString(),
      timestamp: timestamp,
      location: (json['location'] ?? '').toString(),
      imageUrl: imageUrl,
      fineAmount: fineAmount,
      status: (json['status'] ?? 'pending').toString().toLowerCase(),
      lawReference: (json['lawReference'] ?? '').toString(),
      complaintStatus: (json['complaintStatus'] ?? '').toString().toLowerCase(),
      paymentLocked: json['paymentLocked'] == true,
      complaintLocked: json['complaintLocked'] == true,
      paymentDueDate: DateTime.tryParse(
        (json['paymentDueDate'] ?? '').toString(),
      ),
      paidAt: DateTime.tryParse((json['paidAt'] ?? '').toString()),
    );
  }

  /// Update server address at runtime.
  bool setServerAddress(String ip, {int port = 8000}) {
    final parsed = _normalizeServerAddressInput(ip, fallbackPort: port);
    if (parsed == null) {
      debugPrint('ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Ignored invalid server address input: "$ip"');
      return false;
    }

    final normalizedIp = parsed.ip;
    final normalizedPort = parsed.port;
    final hasChanged = serverIp != normalizedIp || serverPort != normalizedPort;
    if (!hasChanged) {
      return false;
    }

    serverIp = normalizedIp;
    serverPort = normalizedPort;
    _lastFetchTime = null;

    debugPrint('ðŸ” Server address changed -> soft reconnect: $serverIp:$serverPort');
    _serverAddressStream.add(
      ServerAddressConfig(
        ip: serverIp,
        port: serverPort,
      ),
    );
    reconnect();
    return true;
  }

  Future<bool> pingServerAddress(
    String ip, {
    int port = 8000,
    Duration timeout = const Duration(seconds: 4),
  }) async {
    final parsed = _normalizeServerAddressInput(ip, fallbackPort: port);
    if (parsed == null) return false;
    final endpoint = parsed.ip.contains('ngrok-free.dev')
        ? Uri.parse('https://${parsed.ip}/api/app/stats')
        : Uri.parse('http://${parsed.ip}:${parsed.port}/api/app/stats');
    try {
      final response = await http
          .get(endpoint, headers: _buildPlainHeaders())
          .timeout(timeout);
      return response.statusCode >= 200 && response.statusCode < 500;
    } catch (_) {
      return false;
    }
  }

  Future<CoreRefreshReport> refreshCoreData(
    String uid, {
    Duration taskTimeout = const Duration(seconds: 6),
    Duration hardTimeout = const Duration(seconds: 14),
  }) async {
    final taskStatus = <String, String>{
      'profile_settings': 'pending',
      'violations': 'pending',
      'notifications_badge': 'pending',
      'connection': 'pending',
    };

    Future<void> runTask(String name, Future<void> Function() action) async {
      try {
        await action().timeout(taskTimeout);
        taskStatus[name] = 'success';
      } on TimeoutException {
        taskStatus[name] = 'timeout';
      } catch (_) {
        taskStatus[name] = 'error';
      }
    }

    final jobs = <Future<void>>[
      runTask('profile_settings', () async {
        await AppSettings().loadFromFirestore(uid);
      }),
      runTask('violations', () async {
        await fetchViolations();
      }),
      runTask('notifications_badge', () async {
        final notifs = await FirestoreService().notificationsStream(uid).first;
        final unread = notifs.where((n) => !n.isRead).length;
        AppSettings().setNotificationCount(unread);
      }),
      runTask('connection', () async {
        await testConnection();
      }),
    ];

    var timedOut = false;
    try {
      await Future.wait(jobs).timeout(hardTimeout);
    } on TimeoutException {
      timedOut = true;
      for (final entry in taskStatus.entries) {
        if (entry.value == 'pending') {
          taskStatus[entry.key] = 'timeout';
        }
      }
    }

    return CoreRefreshReport(taskStatus: taskStatus, timedOut: timedOut);
  }
}
