@echo off
echo Velg hvordan du vil kopiere filene:
echo 1. Kopier og overskriv alle filer
echo 2. Kopier kun nye og endrede filer (behold nyere filer i malmappen)
echo 3. Kopier kun filer som ikke finnes fra for
echo.
choice /C 123 /N /M "Velg alternativ (1-3): "

set COPY_MODE=%ERRORLEVEL%

set "DESTINATION=C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter"

if not exist "%DESTINATION%" mkdir "%DESTINATION%"

echo.
echo Oppdaterer hovedfiler...

if %COPY_MODE%==1 (
    echo Kopierer og overskriver alle filer...
    robocopy "%~dp0ai_modules" "%DESTINATION%\ai_modules" /E /IS /IT
    robocopy "%~dp0checklists" "%DESTINATION%\checklists" /E /IS /IT
    copy /Y "%~dp0app.py" "%DESTINATION%\app.py"
    copy /Y "%~dp0regulations_handler.py" "%DESTINATION%\regulations_handler.py"
    copy /Y "%~dp0rental_analyzer.py" "%DESTINATION%\rental_analyzer.py"
) else if %COPY_MODE%==2 (
    echo Kopierer kun nye og endrede filer...
    robocopy "%~dp0ai_modules" "%DESTINATION%\ai_modules" /E
    robocopy "%~dp0checklists" "%DESTINATION%\checklists" /E
    copy "%~dp0app.py" "%DESTINATION%\app.py"
    copy "%~dp0regulations_handler.py" "%DESTINATION%\regulations_handler.py"
    copy "%~dp0rental_analyzer.py" "%DESTINATION%\rental_analyzer.py"
) else (
    echo Kopierer kun nye filer...
    robocopy "%~dp0ai_modules" "%DESTINATION%\ai_modules" /E /XO
    robocopy "%~dp0checklists" "%DESTINATION%\checklists" /E /XO
    copy "%~dp0app.py" "%DESTINATION%\app.py" /N
    copy "%~dp0regulations_handler.py" "%DESTINATION%\regulations_handler.py" /N
    copy "%~dp0rental_analyzer.py" "%DESTINATION%\rental_analyzer.py" /N
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
    echo Husk a sjekke at alle filene ble kopiert riktig.
    echo.
    pause
)