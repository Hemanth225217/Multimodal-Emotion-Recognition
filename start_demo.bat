@echo off
rem One-click launcher for the Multimodal Emotion AI web demo (serves model B on real MELD test clips).
rem Double-click it, or run "start_demo.bat nobrowser" to start the server without opening the browser.
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Cannot find venv\Scripts\python.exe - keep this file in the project folder.
  pause
  exit /b 1
)
echo Starting the demo server (the model takes about 20 seconds to load)...
start "Emotion demo server - close this window to stop it" cmd /k "venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000"
if /i "%~1"=="nobrowser" exit /b 0
timeout /t 25 /nobreak >nul
start "" "http://127.0.0.1:8000"
