@echo off 
:backup_loop 
timeout /t 60 /nobreak 
cd "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter" 
git add . 
git commit -m "Auto-backup %date% %time%" 
git push origin main 
if %ERRORLEVEL% neq 0 ( 
    echo Git push feilet, prøver igjen om 1 minutt... 
    timeout /t 60 /nobreak 
) 
goto backup_loop 
