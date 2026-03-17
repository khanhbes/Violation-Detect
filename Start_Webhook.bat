@echo off
echo =======================================================
echo    KHOI DONG NGROK WEBHOOK CHO SEPAY (THANH TOAN)
echo =======================================================
echo.
echo Server: http://localhost:8000
echo Domain: https://unhealthier-cibarial-lannie.ngrok-free.dev
echo Webhook URL: https://unhealthier-cibarial-lannie.ngrok-free.dev/api/webhook/sepay
echo.
echo Dang ket noi Internet...

cd /d "%~dp0"
.\ngrok_bin\ngrok.exe http --url=unhealthier-cibarial-lannie.ngrok-free.dev 8000

pause
