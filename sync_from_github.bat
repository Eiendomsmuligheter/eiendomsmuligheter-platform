@echo off
setlocal EnableDelayedExpansion
color 0A

:: Sett mappene
set TARGET_DIR=C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter
set LOG_DIR=%TARGET_DIR%\sync_logs
set REPO_URL=https://github.com/Eiendomsmuligheter/eiendomsmuligheter-platform.git
set TEMP_DIR=%TEMP%\eiendomsmuligheter_temp

:: Opprett dagens dato i format YYYY-MM-DD
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set LOGDATE=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%
set LOGFILE=%LOG_DIR%\sync_log_%LOGDATE%.txt

:: Opprett loggmappe hvis den ikke eksisterer
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Slett logger eldre enn 7 dager
forfiles /P "%LOG_DIR%" /M *.txt /D -7 /C "cmd /c del @path" 2>nul

:: Start ny loggfil med tidsstempel
echo Sync started at %TIME% on %DATE% > "%LOGFILE%"
echo Target: %TARGET_DIR% >> "%LOGFILE%"
echo GitHub Repository: %REPO_URL% >> "%LOGFILE%"
echo. >> "%LOGFILE%"

:: Vis header
cls
echo ========================================
echo    Eiendomsmuligheter GitHub Sync Tool
echo ========================================
echo.
echo GitHub Repository: %REPO_URL%
echo Maal: %TARGET_DIR%
echo Loggfil: %LOGFILE%
echo.

:: Opprett målmappen hvis den ikke eksisterer
if not exist "%TARGET_DIR%" (
    mkdir "%TARGET_DIR%"
    echo Opprettet maalmappe: %TARGET_DIR%
    echo Opprettet maalmappe: %TARGET_DIR% >> "%LOGFILE%"
)

:: Slett temp-mappen hvis den eksisterer og opprett på nytt
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
mkdir "%TEMP_DIR%"

:: Klon repository
echo Henter filer fra GitHub...
echo [%TIME%] Starter GitHub sync >> "%LOGFILE%"
git clone %REPO_URL% "%TEMP_DIR%" 2>> "%LOGFILE%"

:: Flytt til frontend-mappen
set "FRONTEND_DIR=%TEMP_DIR%\frontend"
if not exist "%FRONTEND_DIR%" (
    echo Feil: Frontend-mappen ble ikke funnet!
    echo [%TIME%] Feil: Frontend-mappen ble ikke funnet >> "%LOGFILE%"
    goto :cleanup
)

if errorlevel 1 (
    echo Feil ved kloning av repository!
    echo [%TIME%] Feil ved GitHub sync >> "%LOGFILE%"
    goto :cleanup
)

:: Tell totalt antall filer som skal kopieres
set "totalFiles=0"
for /f "tokens=*" %%F in ('dir /s /b /a-d "%FRONTEND_DIR%"') do set /a totalFiles+=1

:: Initialiser teller for kopierte filer
set "copiedFiles=0"

:: Vis progress bar funksjon
set "progressbar="
for /L %%i in (1,1,50) do set "progressbar=!progressbar!="

echo.
echo Kopierer filer...
echo.

:: Kopier alle filer og mapper med deres struktur
for /R "%FRONTEND_DIR%" %%F in (*) do (
    if not "%%F"=="%TEMP_DIR%\.git" (
        set "relativePath=%%~pF"
        set "relativePath=!relativePath:%FRONTEND_DIR%=!"

        :: Sjekk om filen er nyere eller ikke eksisterer i målmappen
        if not exist "%TARGET_DIR%!relativePath!%%~nxF" (
            echo Kopierer: %%~nxF
            echo [%TIME%] Ny fil kopiert: %%~nxF >> "%LOGFILE%"

            :: Opprett målmappen hvis den ikke eksisterer
            if not exist "%TARGET_DIR%!relativePath!" mkdir "%TARGET_DIR%!relativePath!"

            copy "%%F" "%TARGET_DIR%!relativePath!%%~nxF" >nul
        ) else (
            fc "%%F" "%TARGET_DIR%!relativePath!%%~nxF" >nul
            if errorlevel 1 (
                echo Oppdaterer: %%~nxF
                echo [%TIME%] Fil oppdatert: %%~nxF >> "%LOGFILE%"
                copy "%%F" "%TARGET_DIR%!relativePath!%%~nxF" >nul
            )
        )

        :: Oppdater progress bar
        set /a copiedFiles+=1
        set /a percent=copiedFiles*100/totalFiles
        set /a bars=percent/2

        :: Vis progress bar
        set "progress="
        for /L %%i in (1,1,!bars!) do set "progress=!progress!="
        echo.[!progress!%progressbar:~!bars!% ] !percent!%%
    )
)

:cleanup
:: Rydd opp temp-mappen
rmdir /s /q "%TEMP_DIR%"

echo.
echo ========================================
echo Synkronisering fullfort!
echo Totalt antall filer behandlet: %copiedFiles%
echo Loggfil lagret: %LOGFILE%
echo ========================================
echo [%TIME%] Sync fullfort. Totalt %copiedFiles% filer behandlet >> "%LOGFILE%"

timeout /t 5
exit