@echo off
setlocal EnableExtensions
title CCTV Inteligente - Videovigilancia y Auditoria Forense
cd /d "%~dp0"

echo ======================================================
echo   CCTV Inteligente - Videovigilancia y Auditoria
echo   Lanzador automatico (doble clic)
echo ======================================================
echo.

rem --- 1) Verificar que Python este instalado y en el PATH ---
set "PY=python"
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if not errorlevel 1 (
        set "PY=py"
    ) else (
        echo [ERROR] Python no esta instalado o no esta en el PATH.
        echo.
        echo Para instalarlo:
        echo   1. Descarga Python desde https://www.python.org/downloads/
        echo   2. Al instalar, marca la casilla "Add Python to PATH"
        echo      ANTES de pulsar "Install Now".
        echo   3. Vuelve a ejecutar iniciar.bat
        goto :fail
    )
)
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo Detectado %%v
echo.

rem --- 2) Crear el entorno virtual e instalar dependencias ---
if exist "venv\Scripts\python.exe" goto :deps_ok
echo Preparando entorno por primera vez...
echo (Solo ocurre una vez; puede tardar varios minutos segun tu conexion)
echo.
%PY% -m venv venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual "venv".
    goto :fail
)
echo Instalando dependencias desde requirements.txt...
echo.
"venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Fallo la instalacion de dependencias.
    echo Verifica tu conexion a internet y vuelve a ejecutar iniciar.bat
    goto :fail
)
echo.
echo Entorno listo.
:deps_ok

rem --- 3) Activar el entorno virtual ---
call "venv\Scripts\activate.bat"

rem --- 4) Descargar el modelo YOLO si no esta presente ---
if exist "models\yolov8s.pt" goto :model_ok
echo Descargando el modelo de IA yolov8s.pt (~22 MB)...
python -c "import os,urllib.request;os.makedirs('models',exist_ok=True);u='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt';t='models'+os.sep+'yolov8s.pt.download';urllib.request.urlretrieve(u,t);os.replace(t,'models'+os.sep+'yolov8s.pt')"
if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo descargar el modelo.
    echo Verifica tu conexion a internet y vuelve a ejecutar iniciar.bat
    goto :fail
)
echo Modelo descargado en models\yolov8s.pt
:model_ok

rem --- 4b) Verificar el banco de videos locales de prueba ---
rem Los clips vienen en el repo/ZIP; si falta alguno se sintetiza uno
rem local con OpenCV. Nunca se descarga nada (funciona 100% offline).
if exist "data\videos\prueba.mp4" if exist "data\videos\retail.mp4" goto :videos_ok
echo Preparando videos de demostracion locales...
python -c "from src.config import load_settings;from src.demo_video import ensure_demo_clips;ensure_demo_clips(load_settings().video_presets)"
:videos_ok

rem --- 5) Lanzar la aplicacion ---
echo.
echo Iniciando la aplicacion...
echo (No cierres esta ventana mientras uses el sistema)
echo.
python main.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] La aplicacion termino con un error. Revisa el mensaje anterior.
    goto :fail
)
exit /b 0

:fail
echo.
echo ------------------------------------------------------
echo Presiona cualquier tecla para cerrar esta ventana...
pause >nul
exit /b 1
