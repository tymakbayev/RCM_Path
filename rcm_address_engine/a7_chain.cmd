@echo off
rem A7 chain (protocol §3): selfcheck -> N1 r1 -> N1 r2 -> N10; any error stops the chain
cd /d "%~dp0"
rem PY = the project venv interpreter (README); the reference stand used .venv-pg\Scripts\python.exe
if not defined PY set "PY=python"
rem PY holds a bare path without quotes; it is quoted below
"%PY%" -u a7_run.py selfcheck > ..\results\a7_selfcheck.log 2>&1 || exit /b 1
"%PY%" -u a7_run.py dynamics N1 r1 > ..\results\a7_n1_r1.log 2>&1 || exit /b 2
"%PY%" -u a7_run.py dynamics N1 r2 > ..\results\a7_n1_r2.log 2>&1 || exit /b 3
"%PY%" -u a7_run.py dynamics N10 r1 > ..\results\a7_n10_r1.log 2>&1 || exit /b 4
echo chain done
