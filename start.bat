@echo off
setlocal
title Local RAG Engine Launcher

REM Define the name for the virtual environment directory
set VENV_DIR=.venv

ECHO.
ECHO --- Local RAG Engine Launcher ---
ECHO.

REM Check if Python is available
python --version >nul 2>nul
if %errorlevel% neq 0 (
    ECHO ERROR: Python is not found in your system's PATH.
    ECHO Please install Python 3.8-3.11 and ensure it's added to PATH.
    pause
    exit /b
)

REM Step 1: Check for and create the virtual environment
ECHO [1/4] Checking for virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    ECHO Virtual environment not found. Creating one...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        ECHO ERROR: Failed to create virtual environment.
        pause
        exit /b
    )
) ELSE (
    ECHO Virtual environment found.
)

REM Step 2: Activate environment and install dependencies
ECHO.
ECHO [2/4] Activating virtual environment and installing dependencies...
call "%VENV_DIR%\Scripts\activate.bat"
pip install -r requirements.txt
if %errorlevel% neq 0 (
    ECHO ERROR: Failed to install dependencies from requirements.txt.
    pause
    exit /b
)

REM Step 3: Check for and download models
ECHO.
ECHO [3/4] Checking for local models...
if not exist ".\models\BAAI_bge-m3" (
    ECHO Models not found. Running download script (this may take a while)...
    python download_models.py
    if %errorlevel% neq 0 (
        ECHO ERROR: Failed to download models. Check your internet connection.
        pause
        exit /b
    )
) ELSE (
    ECHO Models found locally.
)

REM Step 4: Launch the application
ECHO.
ECHO [4/4] Starting the RAG Application...
ECHO.
python app.py

endlocal
ECHO.
ECHO Application closed.
pause