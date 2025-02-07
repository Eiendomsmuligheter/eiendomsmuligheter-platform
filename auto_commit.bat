@echo off 
:commit_loop 
timeout /t 300 /nobreak 
cd "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter" 
git add . 
git commit -m "Auto-commit: %date% %time%" 
git push origin main 
echo [%date% %time%] Endringer lagret 
goto commit_loop 
