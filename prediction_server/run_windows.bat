@echo off
chcp 65001 >nul
rem ==========================================================================
rem  예측 서버 실행 (Windows)
rem   - 파이썬: YUCHE_PYTHON > prediction_server\.venv > AI 서브모듈의 ai_model\.venv 순서로 찾는다
rem   - 회원가입/로그인/기록용 MySQL(MariaDB)은 WSL Ubuntu 안에서 실행한다
rem   - AI 모델 코드는 서브모듈 prediction_server\ai 에서 가져온다 (없으면 받는다)
rem ==========================================================================

setlocal
cd /d "%~dp0"

set "PY="
if defined YUCHE_PYTHON if exist "%YUCHE_PYTHON%" set "PY=%YUCHE_PYTHON%"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "ai\terrainFlowSim\ai_model\.venv\Scripts\python.exe" set "PY=%~dp0ai\terrainFlowSim\ai_model\.venv\Scripts\python.exe"
if not defined PY (
  echo [오류] 파이썬 가상환경을 찾을 수 없습니다.
  echo        1^) prediction_server\.venv 를 만들고 README 의 설치 순서대로 패키지를 설치하거나
  echo        2^) 환경변수 YUCHE_PYTHON 에 python.exe 경로를 지정하세요.
  pause
  exit /b 1
)

if not defined AI_MODEL_DIR if not exist "ai\terrainFlowSim\ai_model\src\inference_api.py" (
  echo [안내] AI 모델 코드^(서브모듈^)를 받습니다...
  git -C "%~dp0.." submodule update --init prediction_server/ai
)

echo [1/4] WSL MariaDB 시작...
wsl -d Ubuntu-22.04 -u root -- service mariadb start 2>nul
if errorlevel 1 (
  echo   ^(WSL/MariaDB 없음 - 로그인 없이 예측만 가능합니다^)
) else (
  rem MariaDB 는 기동에 몇 초 걸린다. 바로 서버를 띄우면 DB 없이 뜬 것처럼 보일 수 있어 최대 20초 기다린다.
  echo   MariaDB 준비 대기 중...
  for /L %%i in (1,1,20) do (
    wsl -d Ubuntu-22.04 -u root -- mysqladmin --silent ping >nul 2>&1 && goto :db_ready
    timeout /t 1 /nobreak >nul
  )
  echo   ^(경고: MariaDB 가 20초 안에 응답하지 않았습니다 - 로그인이 안 될 수 있습니다^)
  :db_ready
)

rem WSL2 VM 은 유휴 상태면 스스로 종료되고, 그때 안의 MariaDB 도 같이 죽는다. 또 VM 이 살아 있어도
rem Windows -> WSL 의 127.0.0.1:3306 포워딩만 따로 끊길 수 있다. 둘 다 Windows 에서 wsl 명령을 한 번
rem 날리면 되살아나므로, 서버가 도는 동안 45초마다 wsl 명령을 날리는 유지용 창을 띄운다.
rem (서버 코드도 요청 중 연결이 거부되면 auth.connect_with_recovery 가 WSL 을 한 번 깨우고 다시 연결한다)
echo [2/4] WSL 유지용 프로세스 시작...
start "yuche-wsl-keepalive" /min cmd /c "for /l %%k in (1,0,2) do (wsl -d Ubuntu-22.04 -u root -- true & timeout /t 45 /nobreak >nul)"

echo [3/4] 웹 서버 의존성 확인...
"%PY%" -m pip install -q -r backend\requirements.txt

rem 같은 공유기에 있는 다른 PC/폰에서도 들어올 수 있게 모든 인터페이스에 바인딩한다.
rem 이 PC 에서만 쓰려면 아래 BIND 를 127.0.0.1 로 바꾼다.
rem 방화벽 규칙은 한 번만 관리자 PowerShell 로 추가하면 된다:
rem   New-NetFirewallRule -DisplayName "yuche server 8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
set BIND=0.0.0.0

for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 ^| Where-Object { $_.PrefixOrigin -eq 'Dhcp' -and $_.IPAddress -notlike '169.254.*' } ^| Select-Object -First 1).IPAddress"`) do set LANIP=%%i

echo [4/4] 서버 시작
echo       이 PC        : http://127.0.0.1:8000
if defined LANIP echo       같은 공유기  : http://%LANIP%:8000
if not defined LANIP echo       ^(랜 주소를 못 찾았습니다 - ipconfig 로 확인하세요^)
echo.
if "%BIND%"=="0.0.0.0" echo       * 같은 네트워크에 있는 사람은 누구나 들어올 수 있습니다.
echo.
"%PY%" -m uvicorn app:app --app-dir backend --host %BIND% --port 8000

rem 서버를 끄면 유지용 창도 닫는다 (안 닫으면 WSL VM 이 계속 메모리를 잡는다).
echo 종료 - WSL 유지용 프로세스 정리...
taskkill /fi "WINDOWTITLE eq yuche-wsl-keepalive" /f >nul 2>&1

endlocal
