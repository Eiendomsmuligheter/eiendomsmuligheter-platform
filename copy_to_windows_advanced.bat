@echo off
echo Velg hvordan du vil kopiere filene:
echo 1. Kopier og overskriv alle filer
echo 2. Kopier kun nye og endrede filer (behold nyere filer i malmappen)
echo 3. Kopier kun filer som ikke finnes fra for
echo.
choice /C 123 /N /M "Velg alternativ (1-3): "

set COPY_MODE=%ERRORLEVEL%

set "DESTINATION=C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter_new"

if not exist "%DESTINATION%" mkdir "%DESTINATION%"

if %COPY_MODE%==1 (
    echo Kopierer og overskriver alle filer...
    robocopy "%~dp0." "%DESTINATION%" /E /XD backup sync_logs /XF copy_to_windows.bat copy_to_windows_advanced.bat /IS /IT
) else if %COPY_MODE%==2 (
    echo Kopierer kun nye og endrede filer...
    robocopy "%~dp0." "%DESTINATION%" /E /XD backup sync_logs /XF copy_to_windows.bat copy_to_windows_advanced.bat
) else (
    echo Kopierer kun nye filer...
    robocopy "%~dp0." "%DESTINATION%" /E /XD backup sync_logs /XF copy_to_windows.bat copy_to_windows_advanced.bat /XO
)

if errorlevel 8 (
    echo Det oppstod en feil under kopiering.
    pause
    exit /b 1
) else (
    echo.
    echo Kopiering fullfort!
    echo.
    echo Robocopy returnerte status: %ERRORLEVEL%
    echo 0 = Ingen filer kopiert
    echo 1 = Filer kopiert
    echo 2 = Ekstra filer eller mapper oppdaget
    echo 4 = Enkelte filer eller mapper oppdaterte
    echo.
    echo Alle filer er na kopiert til: %DESTINATION%
    echo Du kan na trygt flytte filene til den endelige plasseringen hvis du onsker.
    echo.
    pause
)