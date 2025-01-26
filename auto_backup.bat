@echo off 
:backup_loop 
timeout /t 60 /nobreak 
cd "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter" 
git add . 
git commit -m "Auto-backup %date% %time%" 
git push origin main 
goto backup_loop 
