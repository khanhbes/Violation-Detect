@echo off
chcp 65001 >nul
cd /d "%~dp0"
title VNeTraffic Launcher
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0VNeTraffic.ps1"
if errorlevel 1 (
  echo.
  echo Launcher ket thuc voi loi. Xem thong bao phia tren.
  pause
)
