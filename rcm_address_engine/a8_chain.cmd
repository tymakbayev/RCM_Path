@echo off
rem A8 chain (protocol §8): selfcheck -> N10 run; any error stops the chain
cd /d C:\gams\test\rcm_address_engine
set PY=C:\gams\test\rcm_address_engine\.venv-pg\Scripts\python.exe
%PY% -u a8_run.py selfcheck > results\a8_selfcheck.log 2>&1 || exit /b 1
%PY% -u a8_run.py run > results\a8_n10.log 2>&1 || exit /b 2
echo chain done
