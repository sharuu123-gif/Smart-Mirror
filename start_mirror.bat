@echo off
title SmartMirror-Vox Launcher
echo ==================================================
echo   Starting SmartMirror-Vox Services
echo ==================================================

echo [*] Starting n8n service...
start "n8n Service" /min cmd /c "n8n start"

timeout /t 3 /nobreak >nul

echo [*] Starting Flask backend server...
start "Flask Backend" /min cmd /c "cd /d %~dp0 && python app.py"

timeout /t 3 /nobreak >nul

echo [*] Opening Smart Mirror in default browser...
start http://localhost:5000

echo ==================================================
echo   SmartMirror-Vox is online!
echo   Mirror UI: http://localhost:5000
echo   n8n:       http://localhost:5678
echo ==================================================
