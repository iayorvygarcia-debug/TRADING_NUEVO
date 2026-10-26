@echo off
title Actualizar Servidor Hostinger - mysmartdomains.com
color 0B
echo ==============================================================
echo   ACTUALIZAR CODIGO EN HOSTINGER (mysmartdomains.com)
echo ==============================================================
echo.
echo  Sincronizando archivos y reiniciando servidor con credenciales .env...
echo.
echo ==============================================================
echo.

uv run python "%~dp0scripts\deploy_hostinger.py"
if %ERRORLEVEL% NEQ 0 (
    python "%~dp0scripts\deploy_hostinger.py"
)

echo.
echo ==============================================================
echo   Proceso finalizado con exito!
echo   Abre ahora: https://mysmartdomains.com/trading/dashboard
echo ==============================================================
echo.
pause
