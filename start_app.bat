@echo off
echo Starting EiendomsAI Platform...

REM Start streamlit application
start /B streamlit run app.py --server.port 8501

REM Start the web server for the main interface
start /B python -m http.server 8080

echo Platform started!
echo Please open your browser and go to:
echo Main interface: http://localhost:8080
echo Streamlit app: http://localhost:8501

pause