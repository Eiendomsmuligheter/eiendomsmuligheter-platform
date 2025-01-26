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
echo.
echo Oppretter backup av eksisterende filer...
if not exist "%TARGET_DIR%\backup" mkdir "%TARGET_DIR%\backup"
if exist "%TARGET_DIR%\index.html" copy "%TARGET_DIR%\index.html" "%TARGET_DIR%\backup\index.html.bak"

:: Opprett den nye moderne index.html
echo.
echo Oppretter ny moderne index.html...
(
echo ^<!DOCTYPE html^>
echo ^<html lang="no"^>
echo ^<head^>
echo     ^<meta charset="UTF-8"^>
echo     ^<meta name="viewport" content="width=device-width, initial-scale=1.0"^>
echo     ^<title^>EiendomsAI ^| Fremtidens Eiendomsanalyse^</title^>
echo     ^<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"^>
echo     ^<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/^>
echo     ^<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"^>^</script^>
echo     ^<style^>
echo         :root {
echo             --glass-bg: rgba(255, 255, 255, 0.1^);
echo             --glass-border: rgba(255, 255, 255, 0.2^);
echo             --primary-gradient: linear-gradient(135deg, #00ff87 0%%, #60efff 100%%^);
echo             --secondary-gradient: linear-gradient(135deg, #FF0080 0%%, #7928CA 100%%^);
echo             --accent-gradient: linear-gradient(135deg, #3B82F6 0%%, #10B981 100%%^);
echo             --dark-bg: #0a0a0a;
echo             --text-primary: #ffffff;
echo             --text-secondary: rgba(255, 255, 255, 0.7^);
echo         }
echo         body {
echo             margin: 0;
echo             padding: 0;
echo             background: var(--dark-bg^);
echo             color: var(--text-primary^);
echo             font-family: 'Segoe UI', sans-serif;
echo             min-height: 100vh;
echo             overflow-x: hidden;
echo         }
echo         #particles-js {
echo             position: fixed;
echo             top: 0;
echo             left: 0;
echo             width: 100%%;
echo             height: 100%%;
echo             z-index: 0;
echo         }
echo         .glass-effect {
echo             background: var(--glass-bg^);
echo             backdrop-filter: blur(10px^);
echo             -webkit-backdrop-filter: blur(10px^);
echo             border: 1px solid var(--glass-border^);
echo             border-radius: 20px;
echo         }
echo         .navbar {
echo             background: transparent;
echo             padding: 1.5rem;
echo             position: relative;
echo             z-index: 1000;
echo         }
echo         .navbar-brand {
echo             font-size: 2rem;
echo             font-weight: 700;
echo             background: var(--primary-gradient^);
echo             -webkit-background-clip: text;
echo             -webkit-text-fill-color: transparent;
echo             position: relative;
echo         }
echo         .main-container {
echo             position: relative;
echo             z-index: 1;
echo             padding: 2rem;
echo             display: grid;
echo             grid-template-columns: 300px 1fr;
echo             gap: 2rem;
echo             height: calc(100vh - 100px^);
echo         }
echo         .feature-card {
echo             background: rgba(255, 255, 255, 0.05^);
echo             border-radius: 15px;
echo             padding: 1.5rem;
echo             margin-bottom: 1rem;
echo             transition: all 0.3s ease;
echo             border: 1px solid rgba(255, 255, 255, 0.1^);
echo             position: relative;
echo             overflow: hidden;
echo         }
echo         .feature-card:hover {
echo             transform: translateY(-5px^) scale(1.02^);
echo             box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2^);
echo         }
echo         .btn-analyze {
echo             background: var(--primary-gradient^);
echo             border: none;
echo             border-radius: 15px;
echo             padding: 1rem 2rem;
echo             color: white;
echo             font-weight: 600;
echo             transition: all 0.3s ease;
echo             width: 100%%;
echo             position: relative;
echo             overflow: hidden;
echo         }
echo     ^</style^>
echo ^</head^>
echo ^<body^>
echo     ^<div id="particles-js"^>^</div^>
echo     ^<nav class="navbar"^>
echo         ^<div class="container-fluid"^>
echo             ^<a class="navbar-brand" href="#"^>EiendomsAI^</a^>
echo             ^<span class="ai-badge"^>AI Aktiv^</span^>
echo         ^</div^>
echo     ^</nav^>
echo     ^<div class="main-container"^>
echo         ^<aside class="sidebar glass-effect"^>
echo             ^<h4^>AI Verktøy^</h4^>
echo             ^<input type="text" class="form-control mb-3" placeholder="Lim inn FINN.no URL..."^>
echo             ^<button class="btn-analyze mb-4"^>Analyser Eiendom^</button^>
echo             ^<div class="feature-card"^>
echo                 ^<h5^>Smart Analyse^</h5^>
echo                 ^<p^>Automatisk vurdering av utleiepotensial^</p^>
echo             ^</div^>
echo             ^<div class="feature-card"^>
echo                 ^<h5^>3D Visualisering^</h5^>
echo                 ^<p^>Se mulige endringer i 3D^</p^>
echo             ^</div^>
echo         ^</aside^>
echo         ^<main class="content-area glass-effect"^>
echo             ^<h2^>Velkommen til Fremtidens Eiendomsanalyse^</h2^>
echo             ^<p class="lead"^>La vår AI hjelpe deg med å maksimere din eiendoms potensial^</p^>
echo             ^<div class="row mt-4"^>
echo                 ^<div class="col-md-4"^>
echo                     ^<div class="feature-card"^>
echo                         ^<h5^>98%% Nøyaktighet^</h5^>
echo                         ^<p^>AI-drevet analyse^</p^>
echo                     ^</div^>
echo                 ^</div^>
echo                 ^<div class="col-md-4"^>
echo                     ^<div class="feature-card"^>
echo                         ^<h5^>5 Min Analysetid^</h5^>
echo                         ^<p^>Rask prosessering^</p^>
echo                     ^</div^>
echo                 ^</div^>
echo                 ^<div class="col-md-4"^>
echo                     ^<div class="feature-card"^>
echo                         ^<h5^>24/7 Tilgjengelig^</h5^>
echo                         ^<p^>Alltid klar til å hjelpe^</p^>
echo                     ^</div^>
echo                 ^</div^>
echo             ^</div^>
echo         ^</main^>
echo     ^</div^>
echo     ^<script^>
echo         // Initialiser particles.js
echo         particlesJS('particles-js',
echo             {
echo                 "particles": {
echo                     "number": {
echo                         "value": 80,
echo                         "density": {
echo                             "enable": true,
echo                             "value_area": 800
echo                         }
echo                     },
echo                     "color": {
echo                         "value": "#ffffff"
echo                     },
echo                     "opacity": {
echo                         "value": 0.5,
echo                         "random": false
echo                     },
echo                     "size": {
echo                         "value": 3,
echo                         "random": true
echo                     },
echo                     "line_linked": {
echo                         "enable": true,
echo                         "distance": 150,
echo                         "color": "#ffffff",
echo                         "opacity": 0.4,
echo                         "width": 1
echo                     },
echo                     "move": {
echo                         "enable": true,
echo                         "speed": 2,
echo                         "direction": "none",
echo                         "random": false,
echo                         "straight": false,
echo                         "out_mode": "out"
echo                     }
echo                 },
echo                 "interactivity": {
echo                     "detect_on": "canvas",
echo                     "events": {
echo                         "onhover": {
echo                             "enable": true,
echo                             "mode": "repulse"
echo                         },
echo                         "onclick": {
echo                             "enable": true,
echo                             "mode": "push"
echo                         },
echo                         "resize": true
echo                     }
echo                 }
echo             }
echo         ^);
echo     ^</script^>
echo ^</body^>
echo ^</html^>
) > "%TARGET_DIR%\index.html"

:: Installer nødvendige pakker
echo.
echo Installerer nødvendige pakker...
pip install --quiet streamlit pandas plotly

echo.
echo ================================
echo    Oppdatering fullført!
echo ================================
echo.
echo Den nye moderne versjonen er nå installert i:
echo %TARGET_DIR%
echo.
echo En backup av den gamle versjonen finnes i:
echo %TARGET_DIR%\backup
echo.
echo Trykk en tast for å avslutte...
pause