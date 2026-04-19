@echo off
REM Horse Management App Runner (Windows)
REM
REM This script automatically detects and activates the Python virtual environment,
REM then runs the Streamlit-based horse management application.
REM
REM Usage:
REM   manage_horses.bat [streamlit-options]

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Handle help request
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help

echo 🐴 Horse Management App Runner (Windows)
echo ============================================

REM Change to script directory
cd /d "%SCRIPT_DIR%"

REM Function to find and activate virtual environment
call :activate_venv
if !errorlevel! neq 0 exit /b !errorlevel!

REM Determine Python command
call :get_python_cmd
if !errorlevel! neq 0 exit /b !errorlevel!

REM Check if manage_horses.py exists
if not exist "manage_horses.py" (
    echo ❌ manage_horses.py not found in %SCRIPT_DIR%
    echo Please ensure you're running this script from the horse-id directory.
    exit /b 1
)

REM Check if streamlit is available
%PYTHON_CMD% -c "import streamlit" >nul 2>&1 || (
    echo ❌ Streamlit not found. Please install it:
    echo    pip install streamlit
    exit /b 1
)

echo.
echo 🚀 Starting horse management application...
echo 📊 This will open a web interface in your browser
echo.

REM Execute Streamlit with manage_horses.py and any additional arguments
%PYTHON_CMD% -m streamlit run manage_horses.py %*
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
python -c "import streamlit, pandas, yaml" >nul 2>&1 || python3 -c "import streamlit, pandas, yaml" >nul 2>&1 || (
    echo ❌ Required Python packages not found.
    echo.
    echo Please either:
    echo 1. Create and activate a virtual environment with required packages
    echo 2. Install required packages globally: pip install streamlit pandas pyyaml pillow
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
echo 🐴 Horse Management App Runner (Windows)
echo.
echo Usage:
echo   %~nx0 [streamlit-options]
echo.
echo Description:
echo   Automatically activates the Python virtual environment and starts
echo   the Streamlit-based horse management application.
echo.
echo Examples:
echo   %~nx0                          # Start with default settings
echo   %~nx0 --server.port 8502       # Start on custom port
echo   %~nx0 --server.headless true   # Start in headless mode
echo.
echo Features:
echo   - Multi-user safety with pipeline lock detection
echo   - Visual horse image management and review
echo   - Status management (Active, EXCLUDE, REVIEW)
echo   - Canonical ID assignment and merging
echo   - Horse name normalization
echo   - Detection status override
echo.
echo The application will open in your default web browser.
echo Use Ctrl+C to stop the application.
echo.
exit /b 0