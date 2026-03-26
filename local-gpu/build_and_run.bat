@echo off
title Vortex GPU - Build and Run
echo.
echo  ========================================
echo    Vortex GPU - CUDA SPH Simulator
echo  ========================================
echo.

:: Setup MSVC + Windows SDK environment
set "MSVC_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
set "WINSDK_DIR=C:\Program Files (x86)\Windows Kits\10"
set "WINSDK_VER=10.0.26100.0"

set "PATH=%MSVC_DIR%\bin\HostX64\x64;%WINSDK_DIR%\bin\%WINSDK_VER%\x64;C:\Program Files\CMake\bin;C:\Users\aleja\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set "INCLUDE=%MSVC_DIR%\include;%WINSDK_DIR%\Include\%WINSDK_VER%\ucrt;%WINSDK_DIR%\Include\%WINSDK_VER%\shared;%WINSDK_DIR%\Include\%WINSDK_VER%\winrt;%WINSDK_DIR%\Include\%WINSDK_VER%\um"
set "LIB=%MSVC_DIR%\lib\x64;%WINSDK_DIR%\Lib\%WINSDK_VER%\ucrt\x64;%WINSDK_DIR%\Lib\%WINSDK_VER%\um\x64"

:: Create build directory
if not exist build mkdir build
cd build

:: Configure (only if not already configured)
if not exist build.ninja (
    echo [1/3] Configuring with CMake...
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_RC_COMPILER=rc -DCMAKE_MT=mt
    if %errorlevel% neq 0 (
        echo [ERROR] CMake configure failed.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Already configured, skipping...
)

:: Build
echo.
echo [2/3] Building Release...
cmake --build .
if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

:: Run
echo.
echo [3/3] Launching Vortex GPU...
echo.
vortex_gpu.exe
cd ..
