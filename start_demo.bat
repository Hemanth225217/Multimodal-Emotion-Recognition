@echo off
rem One-click launcher for the Multimodal Emotion AI web demo (serves model B on real MELD test clips).
rem Double-click it, or run "start_demo.bat nobrowser" to start the server without opening the browser.
cd /d "%~dp0"
set "PY=venv\Scripts\python.exe"
rem The final-review worktree shares the Python environment of the main project folder next to it.
if not exist "%PY%" set "PY=..\Multimodal_Emotion_Recognition\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Cannot find venv\Scripts\python.exe - keep this file in the project folder.
  pause
  exit /b 1
)
echo Starting the demo server (the model takes about 20 seconds to load)...
start "Emotion demo server - close this window to stop it" cmd /k %PY% -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
if /i "%~1"=="nobrowser" exit /b 0
timeout /t 25 /nobreak >nul
start "" "http://127.0.0.1:8000"
