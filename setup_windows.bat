@echo off
REM One-time setup. Double-click or run from the project folder.
py -3.11 -m venv venv || py -3 -m venv venv
if errorlevel 1 ( echo Could not create venv. Install Python 3.11 and tick "Add to PATH". & exit /b 1 )
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 ( echo pip install failed. Paste the error and debug it. & exit /b 1 )
if not exist .env copy .env.example .env
echo.
echo Setup done. Edit .env and put your PostgreSQL password in PGPASSWORD.
