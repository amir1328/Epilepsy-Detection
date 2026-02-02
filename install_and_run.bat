@echo off
setlocal
title Epilepsy App - Auto Setup & Run

echo ==================================================
echo      Epilepsy Prediction App - One-Click Run
echo ==================================================
echo.

:: 1. Check if Python is installed
echo [Checking Python]
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found! 
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b
)
python --version
echo.

:: 2. Upgrade pip (good practice to avoid install errors)
echo [Updating Setup Tools]
python -m pip install --upgrade pip
echo.

:: 3. Install Requirements
echo [Installing Dependencies]
echo This may take a few minutes on the first run...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    echo Please check your internet connection and try again.
    pause
    exit /b
)
echo Dependencies are ready.
echo.

:: 4. Run the App
echo [Launching App]
echo Opening browser...
streamlit run app.py

:: 5. Pause only if app crashes or closes
if %errorlevel% neq 0 (
    echo.
    echo [App Closed with Error]
    pause
)
