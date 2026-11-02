@echo off
setlocal
cd /d "%~dp0"
python setup_windows.py
if errorlevel 1 (
  echo Setup stopped. Read the error above; use Python 3.10 and the C++ build tools required by Annoy.
  pause
  exit /b 1
)
pause
