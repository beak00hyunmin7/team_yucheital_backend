@echo off
REM yuche server launcher (this PC = server), MySQL80 version. (WSL MariaDB version: run_windows.bat)
REM   - DB      : MySQL80 on this PC, database yuche_ai_db (accounts + analysis history)
REM   - files   : prediction_server\data\history (history images), prediction_server\.yuche_secret (login key)
REM   - screen  : serves the team frontend (vtkjs.html) at http://<this PC>:8000
REM   - python  : YUCHE_PYTHON > prediction_server\.venv > ai submodule's ai_model\.venv
setlocal
cd /d "%~dp0"

set "PY="
if defined YUCHE_PYTHON if exist "%YUCHE_PYTHON%" set "PY=%YUCHE_PYTHON%"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "ai\terrainFlowSim\ai_model\.venv\Scripts\python.exe" set "PY=%~dp0ai\terrainFlowSim\ai_model\.venv\Scripts\python.exe"
if not defined PY (
  echo [ERROR] python venv not found. Make prediction_server\.venv ^(see README^) or set YUCHE_PYTHON.
  pause
  exit /b 1
)

if not defined AI_MODEL_DIR if not exist "ai\terrainFlowSim\ai_model\src\inference_api.py" (
  echo [INFO] fetching AI model code ^(submodule^)...
  git -C "%~dp0.." submodule update --init prediction_server/ai
)

set "YUCHE_DB_URL=mysql+pymysql://root:1234@127.0.0.1:3306/yuche_ai_db?charset=utf8mb4"
REM frontend folder. Set YUCHE_SITE_DIR before running to override.
REM default: the frontend repo cloned next to this backend repo (team_yucheital_frontend or sejong_yucheital_project)
if not defined YUCHE_SITE_DIR if exist "%~dp0..\..\team_yucheital_frontend\project_yucheital\yuche_frontend\vtkjs.html" set "YUCHE_SITE_DIR=%~dp0..\..\team_yucheital_frontend\project_yucheital\yuche_frontend"
if not defined YUCHE_SITE_DIR set "YUCHE_SITE_DIR=%~dp0..\..\sejong_yucheital_project\project_yucheital\yuche_frontend"
if not exist "%YUCHE_SITE_DIR%\vtkjs.html" echo [WARN] frontend folder not found: %YUCHE_SITE_DIR%

echo [1/3] MySQL80 database...
"%PY%" -c "import pymysql; c=pymysql.connect(host='127.0.0.1',user='root',password='1234'); c.cursor().execute('CREATE DATABASE IF NOT EXISTS yuche_ai_db CHARACTER SET utf8mb4'); c.commit(); print('  DB ready: yuche_ai_db')"
if errorlevel 1 (
  echo [ERROR] Cannot connect to MySQL. Check that the MySQL80 service is running.
  pause
  exit /b 1
)

echo [2/3] packages...
"%PY%" -m pip install -q -r backend\requirements.txt


echo [3/3] server start
echo.
echo       this PC    : http://127.0.0.1:8000
"%PY%" backend\lan_ip.py
echo.
echo       (Ctrl+C to stop)
echo.
"%PY%" -m uvicorn app:app --app-dir backend --host 0.0.0.0 --port 8000

endlocal
pause
