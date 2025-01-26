@echo off 
:backup_loop 
timeout /t 60 /nobreak 
git -C "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter" add . 
git -C "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter" commit -m "Auto-backup %date% %time%" 
git -C "C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter" push 
goto backup_loop 
