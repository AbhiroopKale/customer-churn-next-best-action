@echo off
REM Needs PostgreSQL running and PGPASSWORD set in .env. Run AFTER run_pipeline.bat
call venv\Scripts\activate
python -m scripts.setup_database || exit /b 1
python -m scripts.load_to_postgres || exit /b 1
echo Database ready (churn_nba). Connect Power BI to the vw_* views.
