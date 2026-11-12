@echo off
cd /d "%~dp0"
set "PY="
if defined YUCHE_PYTHON if exist "%YUCHE_PYTHON%" set "PY=%YUCHE_PYTHON%"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY set "PY=%~dp0ai\terrainFlowSim\ai_model\.venv\Scripts\python.exe"
"%PY%" backend\show_users.py
pause
