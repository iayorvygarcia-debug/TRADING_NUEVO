@echo off
title AI Quant Trading Platform Server
cd /d "%~dp0"
echo ========================================================
echo   Iniciando AI Quant Trading Platform...
echo ========================================================
echo.

if not exist ".venv\Scripts\uvicorn.exe" (
    echo [!] No se encontro el entorno .venv. Instalando con uv...
    uv sync
    if errorlevel 1 (
        echo [-] Error al sincronizar dependencias con uv.
        pause
        exit /b 1
    )
)

if not exist ".env" (
    if exist ".env.example" (
        echo [+] Creando archivo .env inicial desde .env.example...
        copy .env.example .env >nul
    )
)

echo [+] Servidor Uvicorn iniciando...
echo [+] API URL:       http://127.0.0.1:8000
echo [+] Dashboard Web: http://localhost:8000/dashboard
echo [+] Documentacion: http://localhost:8000/docs
echo.
.venv\Scripts\uvicorn.exe apps.api.main:app --reload --host 127.0.0.1 --port 8000
pause
