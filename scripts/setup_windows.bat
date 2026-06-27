@echo off
setlocal

cd /d "%~dp0\.."

python -m venv .venv
if errorlevel 1 exit /b 1

call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

pip install -r requirements.txt
if errorlevel 1 exit /b 1

pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1

python -m pytest
if errorlevel 1 exit /b 1

echo Setup complete.
