@echo off
setlocal EnableDelayedExpansion

if not exist .venv\Scripts\python.exe (
    set "PY_CMD="
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3.12"
    if not defined PY_CMD (
        py -3.11 --version >nul 2>&1
        if not errorlevel 1 set "PY_CMD=py -3.11"
    )
    if not defined PY_CMD (
        python --version >nul 2>&1
        if not errorlevel 1 set "PY_CMD=python"
    )
    if not defined PY_CMD (
        echo Python 3.11 or newer is required.
        pause
        exit /b 1
    )
    !PY_CMD! -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
