@echo off
REM ============================================================
REM  GRU arm of the EAAI experiments (optional, run after run_all.bat).
REM  Needs PyTorch, CPU build, installed once:
REM     pip install torch --index-url https://download.pytorch.org/whl/cpu
REM  The GRU is tuned on the same TimeSeriesSplit folds and ROC AUC
REM  criterion as LR, RF and HGB (cross-validation AUC in best.json).
REM  72 cells: vol x (HAR, HAR+HCF, HAR+BASE+HCF, HCF) + dir x (BASE, BASE+HCF)
REM  for 4 markets x 3 horizons. Finished cells are skipped on re-run.
REM ============================================================
cd /d "%~dp0.."
echo Project folder: %CD%
python -c "import torch; print('PyTorch', torch.__version__)"
if errorlevel 1 goto :notorch

python EAAI_experiments\eaai_experiments.py --models GRU --targets vol --featuresets HAR HAR+HCF HAR+BASE+HCF HCF
if errorlevel 1 goto :error
python EAAI_experiments\eaai_experiments.py --models GRU --targets dir --featuresets BASE BASE+HCF
if errorlevel 1 goto :error
python EAAI_experiments\eaai_analysis.py
if errorlevel 1 goto :error
python EAAI_experiments\eaai_figures.py
echo.
echo ===== GRU DONE =====
pause
exit /b 0

:notorch
echo.
echo PyTorch is not installed in this Python. Run:
echo     pip install torch --index-url https://download.pytorch.org/whl/cpu
pause
exit /b 1

:error
echo.
echo ===== STOPPED WITH AN ERROR - see the messages above and Results_EAAI\run_*.log =====
pause
exit /b 1
