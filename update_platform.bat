@echo off
echo.
echo ================================
echo    EiendomsAI Platform Update
echo ================================
echo.
echo Starter oppdatering...

:: Sett målmappen
set TARGET_DIR=C:\Users\Ahmad\OneDrive\plattform\eiendomsmuligheter

:: Opprett backup av eksisterende filer
echo Oppretter backup av eksisterende filer...
if not exist "%TARGET_DIR%\backup" mkdir "%TARGET_DIR%\backup"
if exist "%TARGET_DIR%\index.html" copy "%TARGET_DIR%\index.html" "%TARGET_DIR%\backup\index.html.bak"
if exist "%TARGET_DIR%\app.py" copy "%TARGET_DIR%\app.py" "%TARGET_DIR%\backup\app.py.bak"

:: Opprett nye filer
echo Oppretter nye filer...

:: index.html
echo Creating index.html...
(
echo ^<!DOCTYPE html^>
echo ^<html lang="no"^>
echo ^<head^>
echo     ^<meta charset="UTF-8"^>
echo     ^<meta name="viewport" content="width=device-width, initial-scale=1.0"^>
echo     ^<title^>EiendomsAI - Din Digitale Eiendomsrådgiver^</title^>
echo     ^<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"^>
echo     ^<style^>
echo         :root {
echo             --primary-color: #2c3e50;
echo             --secondary-color: #3498db;
echo             --accent-color: #e74c3c;
echo         }
echo         body {
echo             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
echo             margin: 0;
echo             padding: 0;
echo             background-color: #f8f9fa;
echo         }
echo         .navbar {
echo             background-color: var(--primary-color^);
echo             padding: 1rem;
echo         }
echo         .navbar-brand {
echo             color: white !important;
echo             font-weight: bold;
echo             font-size: 1.5rem;
echo         }
echo         .main-container {
echo             display: flex;
echo             height: calc(100vh - 76px^);
echo         }
echo         .sidebar {
echo             width: 300px;
echo             background-color: white;
echo             border-right: 1px solid #dee2e6;
echo             padding: 1rem;
echo             display: flex;
echo             flex-direction: column;
echo         }
echo         .content-area {
echo             flex: 1;
echo             padding: 2rem;
echo             overflow-y: auto;
echo         }
echo         .chat-container {
echo             position: fixed;
echo             bottom: 20px;
echo             right: 20px;
echo             width: 350px;
echo             height: 500px;
echo             background: white;
echo             border-radius: 10px;
echo             box-shadow: 0 0 10px rgba(0,0,0,0.1^);
echo             display: flex;
echo             flex-direction: column;
echo         }
echo         .chat-header {
echo             background: var(--secondary-color^);
echo             color: white;
echo             padding: 1rem;
echo             border-radius: 10px 10px 0 0;
echo         }
echo         .chat-messages {
echo             flex: 1;
echo             overflow-y: auto;
echo             padding: 1rem;
echo         }
echo         .chat-input {
echo             padding: 1rem;
echo             border-top: 1px solid #dee2e6;
echo         }
echo         .feature-card {
echo             background: white;
echo             border-radius: 10px;
echo             padding: 1.5rem;
echo             margin-bottom: 1rem;
echo             box-shadow: 0 2px 4px rgba(0,0,0,0.1^);
echo             transition: transform 0.2s;
echo         }
echo         .feature-card:hover {
echo             transform: translateY(-5px^);
echo         }
echo         .ai-badge {
echo             background: var(--accent-color^);
echo             color: white;
echo             padding: 0.25rem 0.5rem;
echo             border-radius: 20px;
echo             font-size: 0.8rem;
echo         }
echo     ^</style^>
echo ^</head^>
echo ^<body^>
echo     ^<nav class="navbar"^>
echo         ^<div class="container-fluid"^>
echo             ^<a class="navbar-brand" href="#"^>
echo                 ^<i class="fas fa-home"^>^</i^> EiendomsAI
echo                 ^<span class="ai-badge"^>AI-Drevet^</span^>
echo             ^</a^>
echo         ^</div^>
echo     ^</nav^>
echo     ^<div class="main-container"^>
echo         ^<div class="sidebar"^>
echo             ^<h5 class="mb-3"^>Verktøy^</h5^>
echo             ^<div class="mb-3"^>
echo                 ^<input type="text" class="form-control" placeholder="Lim inn FINN.no URL"^>
echo             ^</div^>
echo             ^<button class="btn btn-primary mb-3"^>Analyser Eiendom^</button^>
echo             ^<div class="feature-card"^>
echo                 ^<h6^>Priskalkulator^</h6^>
echo                 ^<p class="small"^>Beregn kostnader for dine ønskede endringer^</p^>
echo             ^</div^>
echo             ^<div class="feature-card"^>
echo                 ^<h6^>Regelverk Sjekk^</h6^>
echo                 ^<p class="small"^>Få oversikt over gjeldende lover og regler^</p^>
echo             ^</div^>
echo         ^</div^>
echo         ^<div class="content-area"^>
echo             ^<div class="row"^>
echo                 ^<div class="col-md-8"^>
echo                     ^<h2^>Velkommen til EiendomsAI^</h2^>
echo                     ^<p class="lead"^>Din intelligente partner for eiendomsutvikling og boligforbedring^</p^>
echo                     ^<div class="feature-card"^>
echo                         ^<h4^>Hvordan kan vi hjelpe deg?^</h4^>
echo                         ^<ul^>
echo                             ^<li^>Analysere boligmuligheter fra FINN.no^</li^>
echo                             ^<li^>Vurdere utleiedeler og tomteutnyttelse^</li^>
echo                             ^<li^>Beregne kostnader for oppgraderinger^</li^>
echo                             ^<li^>Sjekke lokale byggeforskrifter^</li^>
echo                         ^</ul^>
echo                     ^</div^>
echo                 ^</div^>
echo             ^</div^>
echo         ^</div^>
echo     ^</div^>
echo     ^<div class="chat-container"^>
echo         ^<div class="chat-header"^>
echo             ^<h5 class="m-0"^>EiendomsAI Assistent^</h5^>
echo         ^</div^>
echo         ^<div class="chat-messages" id="chatMessages"^>^</div^>
echo         ^<div class="chat-input"^>
echo             ^<div class="input-group"^>
echo                 ^<input type="text" class="form-control" placeholder="Skriv din melding her..."^>
echo                 ^<button class="btn btn-primary"^>Send^</button^>
echo             ^</div^>
echo         ^</div^>
echo     ^</div^>
echo     ^<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"^>^</script^>
echo     ^<script src="https://kit.fontawesome.com/your-fontawesome-kit.js"^>^</script^>
echo ^</body^>
echo ^</html^>
) > "%TARGET_DIR%\index.html"

:: app.py
echo Creating app.py...
(
echo import streamlit as st
echo import requests
echo from bs4 import BeautifulSoup
echo import json
echo import re
echo from datetime import datetime
echo from rental_analyzer import RentalAnalyzer, BuildingType
echo import pandas as pd
echo import plotly.express as px
echo.
echo # Konfigurasjon for sideoppsett
echo st.set_page_config^(
echo     page_title="EiendomsAI",
echo     page_icon="🏠",
echo     layout="wide"
echo ^)
echo.
echo # Initialiser rental analyzer
echo rental_analyzer = RentalAnalyzer^(^)
) > "%TARGET_DIR%\app.py"

:: rental_analyzer.py
echo Creating rental_analyzer.py...
(
echo import requests
echo from dataclasses import dataclass
echo from typing import List, Dict, Optional
echo from enum import Enum
echo import json
echo import re
echo from pathlib import Path
echo import logging
echo.
echo # Konfigurer logging
echo logging.basicConfig^(level=logging.INFO^)
echo logger = logging.getLogger^(__name__^)
echo.
echo class BuildingType^(Enum^):
echo     APARTMENT = "leilighet"
echo     HOUSE = "enebolig"
echo     ROW_HOUSE = "rekkehus"
echo     BASEMENT = "kjellerleilighet"
echo     ATTIC = "loftsleilighet"
echo     ANNEXE = "anneks"
) > "%TARGET_DIR%\rental_analyzer.py"

:: regulations_handler.py
echo Creating regulations_handler.py...
(
echo import requests
echo from typing import Dict, List, Optional
echo import json
echo import logging
echo from pathlib import Path
echo.
echo logger = logging.getLogger^(__name__^)
echo.
echo class RegulationsHandler:
echo     def __init__^(self^):
echo         self.load_regulations^(^)
echo         
echo     def load_regulations^(self^):
echo         """Last inn reguleringsplaner og forskrifter"""
echo         self.regulations = {
echo             "Oslo": {
echo                 "utnyttelsesgrad": 0.24,
echo                 "max_hoyde": 9.0,
echo                 "min_avstand_nabo": 4.0,
echo                 "parkering_krav": {"bil": 1, "sykkel": 2}
echo             }
echo         }
) > "%TARGET_DIR%\regulations_handler.py"

:: requirements.txt
echo Creating requirements.txt...
(
echo streamlit==1.24.0
echo requests==2.31.0
echo beautifulsoup4==4.12.2
echo pandas==2.0.3
echo numpy==1.24.3
echo plotly==5.18.0
) > "%TARGET_DIR%\requirements.txt"

echo.
echo Installasjonen er fullført! Filene er kopiert til:
echo %TARGET_DIR%
echo.
echo En backup av de gamle filene er lagret i:
echo %TARGET_DIR%\backup
echo.
echo For å installere nødvendige Python-pakker, kjør:
echo pip install -r requirements.txt
echo.
echo For å starte applikasjonen, kjør:
echo streamlit run app.py
echo.
pause