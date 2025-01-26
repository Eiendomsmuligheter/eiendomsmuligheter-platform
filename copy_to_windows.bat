@echo off
echo Kopierer filer til Windows...
if not exist "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter" mkdir "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter"

robocopy "%~dp0" "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter" /E /XD "backup" "sync_logs" /XF copy_to_windows.bat

if errorlevel 8 (
    echo Det oppstod en feil under kopiering.
    pause
    exit /b 1
) else (
    echo Kopiering fullfort!
    pause
)