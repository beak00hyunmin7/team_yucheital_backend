@echo off
rem ==========================================================================
rem  Prediction server launcher (Windows, WSL MariaDB version; MySQL80 version: run_server.bat)
rem   - python : YUCHE_PYTHON > prediction_server\.venv > ai submodule's ai_model\.venv
rem   - DB     : MariaDB inside WSL Ubuntu (accounts / login / history)
rem   - model  : AI model code comes from the submodule prediction_server\ai (fetched if missing)
rem  NOTE: keep this file ASCII only. A UTF-8 batch file with "chcp 65001" is misread by cmd
rem        (Korean text breaks the parser: "... is not recognized", ") was unexpected").
rem ==========================================================================

setlocal
cd /d "%~dp0"

set "PY="
if defined YUCHE_PYTHON if exist "%YUCHE_PYTHON%" set "PY=%YUCHE_PYTHON%"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "ai\terrainFlowSim\ai_model\.venv\Scripts\python.exe" set "PY=%~dp0ai\terrainFlowSim\ai_model\.venv\Scripts\python.exe"
if not defined PY (
  echo [ERROR] python venv not found.
  echo         1^) create prediction_server\.venv and install packages as in README, or
  echo         2^) set YUCHE_PYTHON to a python.exe path.
  pause
  exit /b 1
)

if not defined AI_MODEL_DIR if not exist "ai\terrainFlowSim\ai_model\src\inference_api.py" (
  echo [INFO] fetching AI model code ^(submodule^)...
  git -C "%~dp0.." submodule update --init prediction_server/ai
)

echo [1/4] starting WSL MariaDB...
wsl -d Ubuntu-22.04 -u root -- service mariadb start 2>nul
if errorlevel 1 (
  echo   ^(no WSL/MariaDB - prediction works without login^)
  goto :db_done
)
rem MariaDB takes a few seconds to start. Wait up to 20 s so the server does not start without DB.
rem (labels must stay outside ( ) blocks - a label inside a block breaks cmd's parser)
echo   waiting for MariaDB...
for /L %%i in (1,1,20) do (
  wsl -d Ubuntu-22.04 -u root -- mysqladmin --silent ping >nul 2>&1 && goto :db_done
  timeout /t 1 /nobreak >nul
)
echo   (warning: MariaDB did not answer within 20 s - login may not work)
:db_done

rem The WSL2 VM shuts itself down when idle and MariaDB dies with it. The Windows -> WSL
rem 127.0.0.1:3306 forwarding can also drop on its own. Running any wsl command from Windows
rem revives both, so while the server runs a minimized window runs "wsl ... true" every 45 s.
rem (the server also wakes WSL once and reconnects when a request is refused: auth.connect_with_recovery)
echo [2/4] starting WSL keepalive...
start "yuche-wsl-keepalive" /min cmd /c "for /l %%k in (1,0,2) do (wsl -d Ubuntu-22.04 -u root -- true & timeout /t 45 /nobreak >nul)"

echo [3/4] checking server packages...
"%PY%" -m pip install -q -r backend\requirements.txt

rem Bind all interfaces so other PCs/phones on the same router can connect.
rem For this PC only, change BIND below to 127.0.0.1.
rem Firewall rule (once, admin PowerShell):
rem   New-NetFirewallRule -DisplayName "yuche server 8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
set BIND=0.0.0.0

echo [4/4] server start
echo       this PC    : http://127.0.0.1:8000
"%PY%" backend\lan_ip.py
echo.
if "%BIND%"=="0.0.0.0" echo       * anyone on the same network can connect.
echo.
"%PY%" -m uvicorn app:app --app-dir backend --host %BIND% --port 8000

rem Close the keepalive window when the server stops (otherwise the WSL VM keeps its memory).
echo stopping WSL keepalive...
taskkill /fi "WINDOWTITLE eq yuche-wsl-keepalive" /f >nul 2>&1

endlocal
