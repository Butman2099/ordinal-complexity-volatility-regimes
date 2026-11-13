@echo off
REM ============================================================
REM  Full EAAI run: stage 1 (LR, HGB), stage 2 (RF), analysis, figures.
REM  Run from an Anaconda Prompt (or any terminal) where "python"
REM  is the interpreter used for the Jupyter notebooks.
REM  Finished cells are skipped, so the file can be re-run after an
REM  interruption and will continue where it stopped.
REM ============================================================
cd /d "%~dp0.."
echo Project folder: %CD%

python EAAI_experiments\eaai_experiments.py --models LR HGB
if errorlevel 1 goto :error
python EAAI_experiments\eaai_experiments.py --models RF
if errorlevel 1 goto :error
python EAAI_experiments\eaai_analysis.py
if errorlevel 1 goto :error
python EAAI_experiments\eaai_figures.py
echo.
echo ===== ALL DONE =====
pause
exit /b 0

:error
echo.
echo ===== STOPPED WITH AN ERROR - see the messages above and Results_EAAI\run_*.log =====
pause
exit /b 1
