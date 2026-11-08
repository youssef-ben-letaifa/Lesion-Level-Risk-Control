@echo off
title KiTS23 Segmentation Testing Suite
color 0A

echo ==============================================================================
echo        KiTS23 SEGMENTATION TESTING SUITE ^& RISK-CONTROLLED VIEWER
echo ==============================================================================
echo.
echo Initializing environment and launching local interactive viewer...
echo Models loaded from 'Training Result' and 'models' junction.
echo.

cd /d "%~dp0"
python run_interface.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] An error occurred while launching the interface.
    echo Please verify Python is installed and added to PATH.
    pause
)

