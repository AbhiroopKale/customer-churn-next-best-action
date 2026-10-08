@echo off
REM Runs the full ML pipeline (no database needed): train -> EDA -> figures -> batch scoring -> results summary
call venv\Scripts\activate
python -m src.train_models || exit /b 1
python -m scripts.run_eda || exit /b 1
python -m scripts.make_report_figures || exit /b 1
python -m scripts.run_batch_scoring || exit /b 1
python -m scripts.imbalance_experiment || exit /b 1
python -m scripts.generate_results_md || exit /b 1
echo.
echo Pipeline finished. See models\model_comparison.csv, reports\ and data\processed\
