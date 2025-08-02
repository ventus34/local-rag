
@echo off
setlocal
title Local RAG Engine Launcher

REM --- Configuration ---
set PYTHON_VERSION=3.12.4
set PYTHON_DIR=.\python-embed
set VENV_DIR=.\.venv
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
set PYTHON_ZIP_NAME=python-embed.zip
set PIP_URL=https://bootstrap.pypa.io/get-pip.py

ECHO.
ECHO --- Self-Contained Local RAG Engine Launcher (Python %PYTHON_VERSION%) ---
ECHO.

REM --- Step 1: Check for and Download Portable Python ---
if exist "%PYTHON_DIR%\python.exe" (
    ECHO Found local Python %PYTHON_VERSION%.
) ELSE (
    ECHO Local Python not found. Downloading version %PYTHON_VERSION%...
    ECHO This will only happen once.
    powershell -Command "Invoke-WebRequest -Uri %PYTHON_URL% -OutFile %PYTHON_ZIP_NAME%"
    if %errorlevel% neq 0 (
        ECHO ERROR: Failed to download Python. Check your internet connection.
        del %PYTHON_ZIP_NAME% >nul 2>nul
        pause
        exit /b
    )

    ECHO Unpacking Python...
    powershell -Command "Expand-Archive -Path .\%PYTHON_ZIP_NAME% -DestinationPath .\%PYTHON_DIR% -Force"
    if %errorlevel% neq 0 (
        ECHO ERROR: Failed to unpack Python.
        del %PYTHON_ZIP_NAME% >nul 2>nul
        pause
        exit /b
    )
    del %PYTHON_ZIP_NAME%
)

REM --- Step 2: Ensure pip is installed in the embedded Python ---
if not exist "%PYTHON_DIR%\Scripts\pip.exe" (
    ECHO Pip not found. Installing pip...

    REM The embeddable package needs a ._pth file to enable site-packages for pip
    ECHO import site > "%PYTHON_DIR%\python312._pth"

    powershell -Command "Invoke-WebRequest -Uri %PIP_URL% -OutFile get-pip.py"
    if %errorlevel% neq 0 (
        ECHO ERROR: Failed to download get-pip.py.
        del get-pip.py >nul 2>nul
        pause
        exit /b
    )

    "%PYTHON_DIR%\python.exe" get-pip.py
    if %errorlevel% neq 0 (
        ECHO ERROR: Failed to install pip.
        del get-pip.py >nul 2>nul
        pause
        exit /b
    )
    del get-pip.py
)

REM --- Step 3: Create Virtual Environment using the embedded Python ---
ECHO [1/4] Checking for virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    ECHO Virtual environment not found. Creating one...
    "%PYTHON_DIR%\python.exe" -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        ECHO ERROR: Failed to create virtual environment.
        pause
        exit /b
    )
) ELSE (
    ECHO Virtual environment found.
)

REM --- Step 4: Activate and Install Dependencies ---
ECHO.
ECHO [2/4] Activating virtual environment and installing dependencies...
call "%VENV_DIR%\Scripts\activate.bat"
pip install -r requirements.txt
if %errorlevel% neq 0 (
    ECHO ERROR: Failed to install dependencies from requirements.txt. Please ensure PyTorch for your CUDA version is correctly specified or removed.
    pause
    exit /b
)

REM --- Step 5: Download Models ---
ECHO.
ECHO [3/4] Checking for local models...
if not exist ".\models\nomic-ai_CodeRankEmbed" (
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

REM --- Step 6: Launch Application ---
ECHO.
ECHO [4/4] Starting the RAG Application...
ECHO.
python app.py

endlocal
ECHO.
ECHO Application closed.
pause