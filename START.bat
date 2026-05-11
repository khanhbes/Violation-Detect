@echo off
chcp 65001 >nul
title VNeTraffic - Full System
color 0A

echo.
echo   VNeTraffic - KHOI DONG HE THONG DAY DU          
echo   Server   : http://localhost:8000                        
echo   Ngrok    : Tu dong kich hoat trong app.py              
echo   Webhook  : .../api/webhook/sepay                      
echo   App WS   : ws://localhost:8000/ws/app                   
echo.

cd /d "%~dp0"

REM Kiem tra .venv
if not exist ".venv\Scripts\python.exe" (
    echo [FAIL] Khong tim thay .venv! Hay chay:
    echo        python -m venv .venv
    echo        .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Kich hoat venv va chay server
echo [*] Dang khoi dong...
echo     Server + Ngrok + WebSocket + FCM
echo.
call .venv\Scripts\activate.bat
python "Detection Web\Web\app.py"

pause
