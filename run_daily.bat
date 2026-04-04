@echo off
:: Orcastrading — Daily end-of-day automation script
:: Schedule this with Windows Task Scheduler to run Mon-Fri after market close
:: Suggested time: 22:00 local time (after US market close + data refresh)
::
:: Task Scheduler setup (run once in an elevated PowerShell):
::   schtasks /create /tn "Orcastrading Daily" /tr "\"C:\Users\admin\Desktop\Orcastrading\run_daily.bat\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 22:00 /f

cd /d "%~dp0"

:: Load .env variables
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
)

:: Run the daily routine — scan all assets, check open trades, generate HTML
"C:\Users\admin\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m p4_live daily --html >> "%~dp0logs\daily.log" 2>&1

:: Log the exit code
echo [%date% %time%] Exit code: %errorlevel% >> "%~dp0logs\daily.log"
