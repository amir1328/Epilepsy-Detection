@echo off
title Epilepsy Prediction Dashboard Launcher
echo ==================================================
echo      Epilepsy Prediction & Localization App
echo ==================================================
echo.

echo [1/2] Checking and installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies. Please check your Python installation.
    pause
    exit /b
)

echo.
echo [2/2] Launching Streamlit App...
echo.
streamlit run app.py

pause
