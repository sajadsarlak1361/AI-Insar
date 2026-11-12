@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
    echo and enable "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -e .
if errorlevel 1 (
    echo Install failed. Check the messages above.
    pause
    exit /b 1
)

python -c "from ai_insar.dependencies import require_runtime; require_runtime()"
if errorlevel 1 (
    echo A package required for SAFE processing is missing.
    if not defined AI_INSAR_NONINTERACTIVE pause
    exit /b 1
)

if defined AI_INSAR_NONINTERACTIVE (
    echo Dependencies are installed. The GUI was not started.
    exit /b 0
)

echo Starting AI Insar...
python -m ai_insar
if errorlevel 1 pause
endlocal
