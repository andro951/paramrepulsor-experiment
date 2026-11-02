@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project Python environment is missing. See WINDOWS-README.txt.
  pause
  exit /b 1
)
echo ParamRepulsor actual-training viewer
echo Open http://127.0.0.1:8765 manually in your browser.
echo Keep this window open. Press Ctrl+C to stop the server.
echo.
".venv\Scripts\python.exe" experiment\simulator\server.py
echo.
echo The server has stopped.
pause
