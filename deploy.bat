@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM ===================================================================
REM  VNeTraffic - Full Auto Deploy (Windows)
REM  Tu dong: Tim IP WiFi - Sua code Flutter - Build APK - Upload
REM ===================================================================

REM -- CAU HINH (CHI CAN SUA PHAN NAY) ------------------------------
set SERVER_PORT=8000
set FORCE_UPDATE=false
REM -----------------------------------------------------------------

REM ===================================================================
REM  NHAP WHAT'S NEW (Changelog)
REM ===================================================================
echo.
echo ===================================================================
echo   NHAP NOI DUNG CAP NHAT (What's New)
echo -------------------------------------------------------------------
echo   VD: Sua loi thanh toan, them tinh nang QR
echo   De trong va nhan Enter de dung mac dinh.
echo ===================================================================
set /p USER_CHANGELOG="  Noi dung: "
if "%USER_CHANGELOG%"=="" (
    set CHANGELOG=Bug fixes and improvements
) else (
    set CHANGELOG=%USER_CHANGELOG%
)
echo       Changelog: %CHANGELOG%
echo.

set SCRIPT_DIR=%~dp0
set APP_DIR=%SCRIPT_DIR%App\traffic_violation_app
set APK_PATH=%APP_DIR%\build\app\outputs\flutter-apk\app-release.apk
set HELPER=%SCRIPT_DIR%deploy_helper.ps1

REM ===================================================================
REM  BUOC 0: Tu dong tim IP WiFi
REM ===================================================================
echo.
echo [0/5] Dang tim IP WiFi cua may tinh...

for /f "tokens=*" %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%HELPER%" -Action get-ip') do set WIFI_IP=%%i

if "%WIFI_IP%"=="" set WIFI_IP=0.0.0.0
if "%WIFI_IP%"=="0.0.0.0" (
    echo [FAIL] Khong tim thay IP WiFi! Kiem tra ket noi mang.
    pause
    exit /b 1
)

set SERVER_IP=%WIFI_IP%
set SERVER_URL=http://%SERVER_IP%:%SERVER_PORT%

echo       IP tim thay : %WIFI_IP%
echo       Server URL  : %SERVER_URL%
echo.

REM ===================================================================
REM  BUOC 1: Ghi IP vao code Flutter
REM ===================================================================
echo [1/5] Cap nhat IP %WIFI_IP% vao api_service.dart...

powershell -NoProfile -ExecutionPolicy Bypass -File "%HELPER%" -Action set-ip -AppDir "%APP_DIR%" -Version "%WIFI_IP%"

echo       OK!
echo.

REM ===================================================================
REM  BUOC 2: Dong bo va Tu dong tang Version
REM ===================================================================
echo [2/5] Tu dong tang version trong pubspec.yaml...

for /f "tokens=1,2 delims=_" %%a in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%HELPER%" -Action increment-version -AppDir "%APP_DIR%"') do (
    set APP_VERSION=%%a
    set BUILD_NUMBER=%%b
)

echo       New Version : %APP_VERSION%+%BUILD_NUMBER%
echo.

REM ===================================================================
REM  BUOC 3: Build APK
REM ===================================================================
echo ===================================================================
echo   VNeTraffic - Auto Deploy v%APP_VERSION%
echo -------------------------------------------------------------------
echo   Server  : %SERVER_URL%
echo   Version : %APP_VERSION% (build %BUILD_NUMBER%)
echo   WiFi IP : %WIFI_IP% (tu dong phat hien)
echo ===================================================================
echo.
echo [3/5] Building APK...
echo.
cd /d "%APP_DIR%"
call flutter build apk --release
if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAIL] Build that bai! Kiem tra loi o tren.
    pause
    exit /b 1
)

if not exist "%APK_PATH%" (
    echo [FAIL] Khong tim thay file APK: %APK_PATH%
    pause
    exit /b 1
)

for %%A in ("%APK_PATH%") do set APK_SIZE=%%~zA
set /a APK_SIZE_MB=%APK_SIZE% / 1048576
echo.
echo [OK] Build thanh cong! APK: %APK_SIZE_MB% MB
echo.

REM ===================================================================
REM  BUOC 4: Upload APK len Server
REM ===================================================================
echo [4/5] Uploading APK to %SERVER_URL%...
echo.
curl -X POST "%SERVER_URL%/api/app/upload-apk" ^
    -F "file=@%APK_PATH%" ^
    -F "version=%APP_VERSION%" ^
    -F "build_number=%BUILD_NUMBER%" ^
    -F "changelog=%CHANGELOG%" ^
    -F "force_update=%FORCE_UPDATE%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAIL] Upload that bai! Kiem tra server co dang chay khong.
    echo    Thu: curl %SERVER_URL%/api/app/stats
    pause
    exit /b 1
)

echo.
echo.

REM ===================================================================
REM  BUOC 5: Xac nhan
REM ===================================================================
echo [5/5] Kiem tra version tren server...
echo.
curl -s "%SERVER_URL%/api/app/latest-version"
echo.
echo.
echo ===================================================================
echo   DEPLOY HOAN TAT!
echo.
echo   App v%APP_VERSION% da duoc day len server.
echo   WiFi IP: %WIFI_IP% (da tu dong cap nhat vao code)
echo   Nguoi dung mo app se thay popup cap nhat.
echo ===================================================================
echo.
pause
