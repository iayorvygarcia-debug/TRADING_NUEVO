@echo off
title Quant OS - Servidor Local de Trading (Sin 429!)
color 0A
echo.
echo ============================================================
echo   QUANT OS - MODO LOCAL (Bypass de rate-limit Hostinger)
echo ============================================================
echo.
echo  Este script inicia un servidor proxy local que conecta
echo  tu PC directamente al motor de trading en el servidor,
echo  sin pasar por el firewall de Hostinger.
echo.
echo  El dashboard se abrira en: http://localhost:8765/dashboard
echo.
echo  MANTENLO ABIERTO mientras usas el dashboard.
echo  Cierra esta ventana para detener el proxy.
echo.
echo ============================================================
echo.

REM Try UV python first (fast Python manager)
uv run python "%~dp0local_proxy.py" 2>nul
if %ERRORLEVEL% EQU 0 goto end

REM Try standard python
python "%~dp0local_proxy.py" 2>nul
if %ERRORLEVEL% EQU 0 goto end

REM Try python3
python3 "%~dp0local_proxy.py" 2>nul
if %ERRORLEVEL% EQU 0 goto end

echo.
echo ERROR: Python no encontrado.
echo Instala Python desde https://python.org o ejecuta:
echo   uv run python local_proxy.py
echo.

:end
pause

