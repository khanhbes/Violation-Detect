#!/bin/bash
# ===================================================================
#  VNeTraffic - Full Auto Deploy (Mac/Linux/Git Bash)
#  Tu dong: Tim IP WiFi - Sua code Flutter - Build APK - Upload
# ===================================================================
#
#  Cach dung:
#    1. chmod +x deploy.sh   (chi can lan dau)
#    2. ./deploy.sh
# ===================================================================

# -- CAU HINH (CHI CAN SUA PHAN NAY) ------------------------------
SERVER_PORT="8000"
CHANGELOG="Auto update version and build"
FORCE_UPDATE="false"
# -----------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$SCRIPT_DIR/App/traffic_violation_app"
APK_PATH="$APP_DIR/build/app/outputs/flutter-apk/app-release.apk"
API_SERVICE="$APP_DIR/lib/services/api_service.dart"

# ===================================================================
#  BUOC 0: Tu dong tim IP WiFi
# ===================================================================
echo ""
echo "[0/5] Dang tim IP WiFi cua may tinh..."

WIFI_IP=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    WIFI_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
else
    # Linux
    WIFI_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -z "$WIFI_IP" ]; then
        WIFI_IP=$(ip route get 1 2>/dev/null | awk '{print $(NF-2);exit}')
    fi
fi

if [ -z "$WIFI_IP" ]; then
    echo "[FAIL] Khong tim thay IP WiFi! Kiem tra ket noi mang."
    exit 1
fi

SERVER_IP="$WIFI_IP"
SERVER_URL="http://$SERVER_IP:$SERVER_PORT"

echo "       IP tim thay: $WIFI_IP"
echo "       Server URL : $SERVER_URL"
echo ""

# ===================================================================
#  BUOC 1: Ghi IP vao code Flutter (api_service.dart)
# ===================================================================
echo "[1/5] Dang cap nhat IP $WIFI_IP vao api_service.dart..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/static String serverIp = '[^']*'/static String serverIp = '$WIFI_IP'/g" "$API_SERVICE"
else
    sed -i "s/static String serverIp = '[^']*'/static String serverIp = '$WIFI_IP'/g" "$API_SERVICE"
fi

echo "       OK!"
echo ""

# ===================================================================
#  BUOC 2: Dong bo va Tu dong tang Version
# ===================================================================
echo "[2/5] Tu dong tang version trong pubspec.yaml..."

cd "$APP_DIR" || { echo "[FAIL] Khong tim thay thu muc app: $APP_DIR"; exit 1; }

CURRENT_VER_LINE=$(grep '^version:' pubspec.yaml)
if [[ $CURRENT_VER_LINE =~ version:\ ([0-9]+)\.([0-9]+)\.([0-9]+)\+([0-9]+) ]]; then
    MAJOR="${BASH_REMATCH[1]}"
    MINOR="${BASH_REMATCH[2]}"
    PATCH="${BASH_REMATCH[3]}"
    BUILD_NUMBER="${BASH_REMATCH[4]}"
    
    PATCH=$((PATCH + 1))
    BUILD_NUMBER=$((BUILD_NUMBER + 1))
    APP_VERSION="$MAJOR.$MINOR.$PATCH"
elif [[ $CURRENT_VER_LINE =~ version:\ ([0-9\.]+)\+([0-9]+) ]]; then
    APP_VERSION="${BASH_REMATCH[1]}"
    BUILD_NUMBER="${BASH_REMATCH[2]}"
    BUILD_NUMBER=$((BUILD_NUMBER + 1))
else
    APP_VERSION="1.0.0"
    BUILD_NUMBER="1"
fi

echo "       New Version : $APP_VERSION+$BUILD_NUMBER"

if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/^version: .*/version: $APP_VERSION+$BUILD_NUMBER/g" pubspec.yaml
else
    sed -i "s/^version: .*/version: $APP_VERSION+$BUILD_NUMBER/g" pubspec.yaml
fi

echo "       OK!"
echo ""

# ===================================================================
#  BUOC 3: Build APK
# ===================================================================
echo "=========================================================="
echo "  VNeTraffic - Auto Deploy v$APP_VERSION"
echo "----------------------------------------------------------"
echo "  Server  : $SERVER_URL"
echo "  Version : $APP_VERSION (build $BUILD_NUMBER)"
echo "  WiFi IP : $WIFI_IP (tu dong phat hien)"
echo "=========================================================="
echo ""
echo "[3/5] Building APK..."
echo ""
flutter build apk --release
if [ $? -ne 0 ]; then
    echo ""
    echo "[FAIL] Build that bai! Kiem tra loi o tren."
    exit 1
fi

if [ ! -f "$APK_PATH" ]; then
    echo "[FAIL] Khong tim thay file APK: $APK_PATH"
    exit 1
fi

APK_SIZE=$(du -h "$APK_PATH" | cut -f1)
echo ""
echo "[OK] Build thanh cong! APK: $APK_SIZE"
echo ""

# ===================================================================
#  BUOC 4: Upload APK len Server
# ===================================================================
echo "[4/5] Uploading APK to $SERVER_URL..."
echo ""
curl -X POST "$SERVER_URL/api/app/upload-apk" \
    -F "file=@$APK_PATH" \
    -F "version=$APP_VERSION" \
    -F "build_number=$BUILD_NUMBER" \
    -F "changelog=$CHANGELOG" \
    -F "force_update=$FORCE_UPDATE"

if [ $? -ne 0 ]; then
    echo ""
    echo "[FAIL] Upload that bai! Kiem tra server co dang chay khong."
    echo "   Thu: curl $SERVER_URL/api/app/stats"
    exit 1
fi

echo ""
echo ""

# ===================================================================
#  BUOC 5: Xac nhan
# ===================================================================
echo "[5/5] Kiem tra version tren server..."
echo ""
curl -s "$SERVER_URL/api/app/latest-version" | python3 -m json.tool 2>/dev/null || \
curl -s "$SERVER_URL/api/app/latest-version"
echo ""
echo ""
echo "=========================================================="
echo "  DEPLOY HOAN TAT!"
echo ""
echo "  App v$APP_VERSION da duoc day len server."
echo "  WiFi IP: $WIFI_IP (da tu dong cap nhat vao code)"
echo "  Nguoi dung mo app se thay popup cap nhat."
echo "=========================================================="
echo ""
