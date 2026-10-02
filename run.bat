@echo off
REM ---------------------------------------------------------------------------
REM  ChatDFT launcher
REM
REM  Starts the local web application and opens it in your browser.
REM
REM  Usage:   run.bat            (port 8000)
REM           run.bat 9000       (custom port)
REM
REM  The application needs a Python environment with PySCF and RDKit. The
REM  conda environment below is used when it exists; otherwise the script
REM  falls back to a .venv in this folder.
REM ---------------------------------------------------------------------------
setlocal

set "CONDA_PY=%USERPROFILE%\miniconda3\envs\biochar-dft\python.exe"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

if exist "%CONDA_PY%" (
  set "PY=%CONDA_PY%"
  echo [ChatDFT] Python: conda environment biochar-dft
) else if exist "%VENV_PY%" (
  set "PY=%VENV_PY%"
  echo [ChatDFT] Python: bundled virtual environment
) else (
  echo [ChatDFT] No Python environment found.
  echo          Expected one of:
  echo            %CONDA_PY%
  echo            %VENV_PY%
  echo          Install the dependencies first:  pip install -r requirements.txt
  pause
  exit /b 1
)

cd /d "%~dp0"
set "CHATDFT_HOST=127.0.0.1"
set "CHATDFT_PORT=%PORT%"

echo [ChatDFT] Starting server on http://127.0.0.1:%PORT%
echo [ChatDFT] Opening your browser. Close this window to stop the server.
echo.

REM Wait for the server to bind the port, then open the browser.
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:%PORT%"

"%PY%" -W ignore -m backend.server
