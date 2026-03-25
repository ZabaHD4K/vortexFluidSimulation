@echo off
title Vortex Fluid Simulation
cd /d "%~dp0"

echo ============================================
echo   Vortex - Real-Time Fluid Simulation
echo ============================================
echo.

:: Try Node.js (npx serve) first
where node >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Node.js found. Starting server...
    echo.
    echo  Open http://localhost:3000 in your browser
    echo  Press Ctrl+C to stop
    echo.
    start "" "http://localhost:3000"
    npx -y serve -l 3000 -s .
    goto :end
)

:: Try Python 3
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Python found. Starting server...
    echo.
    echo  Open http://localhost:3000 in your browser
    echo  Press Ctrl+C to stop
    echo.
    start "" "http://localhost:3000"
    python -m http.server 3000
    goto :end
)

:: Try Python via py launcher
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Python found. Starting server...
    echo.
    echo  Open http://localhost:3000 in your browser
    echo  Press Ctrl+C to stop
    echo.
    start "" "http://localhost:3000"
    py -m http.server 3000
    goto :end
)

echo [ERROR] No se encontro Node.js ni Python.
echo.
echo Instala uno de estos para servir la app:
echo   - Node.js: https://nodejs.org
echo   - Python:  https://python.org
echo.
pause

:end
