@echo off
setlocal

where docker >nul 2>&1
if errorlevel 1 (
    echo Docker Desktop is required. Install and start Docker Desktop first.
    pause
    exit /b 1
)

if not exist .env copy .env.example .env >nul
docker compose up --build
