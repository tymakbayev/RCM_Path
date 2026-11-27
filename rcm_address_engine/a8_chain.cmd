@echo off
rem A8 chain (protocol §8): selfcheck -> N10 run; any error stops the chain
cd /d "%~dp0"
rem PY = the project venv interpreter (README); the reference stand used .venv-pg\Scripts\python.exe
if not defined PY set "PY=python"
rem PY holds a bare path without quotes; it is quoted below
"%PY%" -u a8_run.py selfcheck > ..\results\a8_selfcheck.log 2>&1 || exit /b 1
"%PY%" -u a8_run.py run > ..\results\a8_n10.log 2>&1 || exit /b 2
echo chain done
