@echo off
REM ============================================================
REM  Resume the EAAI run after a power cut or a restart.
REM  1) checks Results_EAAI and sets aside any file damaged by the cut,
REM  2) continues run_all.bat; finished cells are skipped.
REM  Start it from the same kind of terminal as before (e.g. Anaconda
REM  Prompt), so that "python" is the right interpreter.
REM ============================================================
cd /d "%~dp0.."
echo Project folder: %CD%
python EAAI_experiments\eaai_check.py --fix
if errorlevel 1 goto :error
call "%~dp0run_all.bat"
exit /b 0

:error
echo.
echo ===== The check could not run. Is this the terminal with the right Python? =====
pause
exit /b 1
