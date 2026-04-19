@echo off
REM Horse ID Pipeline Runner (Windows)
REM
REM This script automatically detects and activates the Python virtual environment,
REM then runs the unified horse identification pipeline.
REM
REM Usage:
REM   run_pipeline.bat                                    # Directory ingestion (interactive)
REM   run_pipeline.bat --email                            # Email ingestion  
REM   run_pipeline.bat --dir --path C:\path\to\horses --date 20240315  # Non-interactive directory
REM   run_pipeline.bat --force                            # Force override locks

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Handle help request
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help

echo 🐴 Horse ID Pipeline Runner (Windows)
echo =========================================

REM Change to script directory
cd /d "%SCRIPT_DIR%"

REM Function to find and activate virtual environment
call :activate_venv
if !errorlevel! neq 0 exit /b !errorlevel!

REM Determine Python command
call :get_python_cmd
if !errorlevel! neq 0 exit /b !errorlevel!

REM Check if run_pipeline.py exists
if not exist "run_pipeline.py" (
    echo ❌ run_pipeline.py not found in %SCRIPT_DIR%
    echo Please ensure you're running this script from the horse-id directory.
    exit /b 1
)

echo.
echo 🚀 Starting pipeline with arguments: %*
echo.

REM Execute the Python pipeline script with all provided arguments
%PYTHON_CMD% run_pipeline.py %*
exit /b %errorlevel%

:activate_venv
echo 🔍 Searching for Python virtual environment...

REM Check common virtual environment locations
set "venv_paths[0]=%SCRIPT_DIR%\venv"
set "venv_paths[1]=%SCRIPT_DIR%\.venv"
set "venv_paths[2]=%SCRIPT_DIR%\..\venv"  
set "venv_paths[3]=%SCRIPT_DIR%\..\.venv"
set "venv_paths[4]=.\venv"
set "venv_paths[5]=.\.venv"

for /l %%i in (0,1,5) do (
    if exist "!venv_paths[%%i]!\Scripts\activate.bat" (
        echo ✅ Found virtual environment: !venv_paths[%%i]!
        call "!venv_paths[%%i]!\Scripts\activate.bat"
        python --version >nul 2>&1 && (
            for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo 🐍 Virtual environment activated: %%v
        )
        exit /b 0
    )
)

echo ⚠️  No virtual environment found. Trying system Python...

REM Try to get Python version
python --version >nul 2>&1 && (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo 🐍 Using system Python: %%v
) || (
    python3 --version >nul 2>&1 && (
        for /f "tokens=*" %%v in ('python3 --version 2^>^&1') do echo 🐍 Using system Python: %%v
    )
)

REM Check if required modules are available
python -c "import pandas, yaml" >nul 2>&1 || python3 -c "import pandas, yaml" >nul 2>&1 || (
    echo ❌ Required Python packages not found.
    echo.
    echo Please either:
    echo 1. Create and activate a virtual environment with required packages
    echo 2. Install required packages globally: pip install pandas pyyaml pillow streamlit ultralytics
    echo.
    echo Recommended setup:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt  # if you have one
    exit /b 1
)

exit /b 0

:get_python_cmd
REM Determine which Python command to use
python --version >nul 2>&1 && (
    set "PYTHON_CMD=python"
    exit /b 0
)

python3 --version >nul 2>&1 && (
    set "PYTHON_CMD=python3"  
    exit /b 0
)

py --version >nul 2>&1 && (
    set "PYTHON_CMD=py"
    exit /b 0
)

echo ❌ Python not found. Please install Python 3.7 or later.
echo Make sure Python is added to your PATH environment variable.
exit /b 1

:show_help
echo 🐴 Horse ID Pipeline Runner (Windows)
echo.
echo Usage:
echo   %~nx0 [OPTIONS]
echo.
echo Ingestion modes:
echo   (default)                           Directory ingestion (interactive)
echo   --email                             Email ingestion
echo   --dir --path PATH --date YYYYMMDD   Non-interactive directory ingestion
echo.
echo Options:
echo   --force                             Force override existing locks
echo   --check-lock                        Check lock status and exit
echo   -h, --help                          Show this help message
echo.
echo Examples:
echo   %~nx0                                        # Interactive directory ingestion
echo   %~nx0 --email                                # Email ingestion
echo   %~nx0 --dir --path C:\photos --date 20240315  # Non-interactive
echo   %~nx0 --force                                # Override locks and run
echo.
exit /b 0