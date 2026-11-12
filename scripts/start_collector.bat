@echo off
rem Keeps the live-data collector running (saves Delhi's readings and logs forecasts every 15 minutes).
rem Double-click this file and leave the window open. Close the window to stop it.
rem If the collector ever stops, it restarts itself after 30 seconds.
cd /d "%~dp0.."
title Delhi air quality collector - leave this window open
:loop
echo [%date% %time%] Starting collector...
".venv\Scripts\python.exe" -u src\live.py --loop --every 15
echo [%date% %time%] Collector stopped. Restarting in 30 seconds. Close this window to stop.
timeout /t 30 /nobreak >nul
goto loop
