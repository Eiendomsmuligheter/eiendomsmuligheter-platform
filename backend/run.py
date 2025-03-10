#!/usr/bin/env python
"""
Eiendomsmuligheter Platform Kjøreskript
---------------------------------------

Dette skriptet starter backend-tjenesten uten å bruke Docker.
Det er nyttig for utvikling og testing.

Bruk:
    python run.py [--port PORT] [--host HOST] [--reload] [--workers WORKERS] [--init-db]

Alternativer:
    --port PORT       Port som API-et skal kjøres på (standard: 8000)
    --host HOST       Host som API-et skal kjøres på (standard: 127.0.0.1)
    --reload          Aktiver automatisk opplasting når kode endres (kun for utvikling)
    --workers WORKERS Antall uvicorn workers (standard: 1)
    --init-db         Initialiser databasen før oppstart
    --use-sqlite      Bruk SQLite istedenfor PostgreSQL
    --help            Vis denne hjelpen

Eksempel:
    # Kjør med standard innstillinger (port 8000 på localhost)
    python run.py
    
    # Kjør på port 5000 med automatisk opplasting
    python run.py --port 5000 --reload
    
    # Initialiser databasen og kjør med SQLite
    python run.py --init-db --use-sqlite
"""
import os
import sys
import argparse
import logging
import subprocess
import platform
import traceback
import time
import warnings

# Sett opp logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("eiendomsmuligheter")

# Ignorer advarsler som kan forstyrre oppstarten
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*model_url.*has conflict with protected namespace.*")
warnings.filterwarnings("ignore", message=".*The 'extra' field.*")

# Legg til prosjektets rotmappe i PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    """Parseargumenter for skriptet"""
    parser = argparse.ArgumentParser(description="Kjør Eiendomsmuligheter Platform Backend")
    parser.add_argument('--port', type=int, default=8000, help="Port å kjøre serveren på")
    parser.add_argument('--host', type=str, default="127.0.0.1", help="Host å binde til")
    parser.add_argument('--reload', action='store_true', help="Aktiver automatisk opplasting")
    parser.add_argument('--workers', type=int, default=1, help="Antall uvicorn workers")
    parser.add_argument('--init-db', action='store_true', help="Initialiser databasen før oppstart")
    parser.add_argument('--use-sqlite', action='store_true', help="Bruk SQLite istedenfor PostgreSQL")
    parser.add_argument('--debug', action='store_true', help="Aktiver debug-modus")
    parser.add_argument('--no-lazy-load', action='store_true', help="Deaktiver lazy-loading av modeller")
    
    return parser.parse_args()

def check_requirements(use_sqlite=False):
    """Sjekk om alle nødvendige pakker er installert"""
    logger.info("Sjekker avhengigheter...")
    
    try:
        import pkg_resources
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        
        if not os.path.exists(requirements_path):
            logger.warning(f"Finner ikke requirements.txt i {requirements_path}")
            return True
            
        with open(requirements_path, 'r') as f:
            required_packages = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
        
        # Pakker som er valgfrie
        optional_packages = []
        if use_sqlite:
            optional_packages.append('psycopg2-binary')  # Ikke nødvendig for SQLite
        else:
            optional_packages.append('aiosqlite')  # Ikke nødvendig for PostgreSQL
        
        # For maskinlæring, kun sjekk en av dem
        ml_packages = ['tensorflow', 'torch', 'onnxruntime']
        ml_found = False
            
        missing_packages = []
        for package in required_packages:
            package_name = package.split('==')[0].split('>=')[0].split('[')[0].strip()
            
            # Hopp over valgfrie pakker hvis de er i listen
            if package_name in optional_packages:
                continue
                
            # Sjekk maskinlæringspakker spesielt
            if package_name in ml_packages:
                try:
                    pkg_resources.require(package_name)
                    ml_found = True
                except (pkg_resources.DistributionNotFound, Exception):
                    pass
                continue
            
            try:
                pkg_resources.require(package_name)
            except (pkg_resources.DistributionNotFound, Exception) as e:
                missing_packages.append(package_name)
                logger.warning(f"Manglende pakke: {package_name} - {str(e)}")
        
        # Sjekk at minst en ML-pakke er tilgjengelig
        if not ml_found and any(pkg in required_packages for pkg in ml_packages):
            logger.warning("Ingen maskinlæringspakker funnet (tensorflow/torch/onnxruntime). Installerer minst én.")
            missing_packages.append("onnxruntime")  # Legg til den letteste
        
        if missing_packages:
            logger.warning(f"Manglende pakker: {', '.join(missing_packages)}")
            logger.info("Forsøker å installere manglende pakker...")
            
            try:
                # Installasjonskommando
                cmd = [sys.executable, "-m", "pip", "install"]
                for pkg in missing_packages:
                    # Finn originalversjonen fra requirements.txt
                    for req in required_packages:
                        if req.startswith(pkg):
                            cmd.append(req)
                            break
                    else:
                        cmd.append(pkg)
                
                logger.info(f"Kjører: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Feil ved installasjon: {result.stderr}")
                    return False
                else:
                    logger.info("Pakker installert.")
                    return True
            except Exception as e:
                logger.error(f"Feil ved installasjon av pakker: {str(e)}")
                return False
        else:
            logger.info("Alle nødvendige avhengigheter er installert.")
            return True
    except Exception as e:
        logger.error(f"Feil ved sjekking av avhengigheter: {str(e)}")
        return False

def init_database(use_sqlite=False):
    """Initialiser databasen"""
    logger.info("Initialiserer database...")
    
    try:
        # Sett miljøvariabel
        if use_sqlite:
            logger.info("Bruker SQLite database")
            os.environ["USE_SQLITE"] = "true"
            
        # Importer database-modulen
        try:
            from backend.database.database import init_db
            init_db()
            logger.info("Database initialisert vellykket!")
        except ImportError:
            try:
                # Prøv alternativ import
                from database.database import init_db
                init_db()
                logger.info("Database initialisert vellykket! (alternativ import)")
            except ImportError as e:
                logger.error(f"Kunne ikke importere database-modulen: {str(e)}")
                logger.error(traceback.format_exc())
                sys.exit(1)
    except Exception as e:
        logger.error(f"Feil ved initialisering av database: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def check_database_connection(use_sqlite=False):
    """Sjekk om databasetilkoblingen fungerer"""
    logger.info("Sjekker databasetilkobling...")
    
    try:
        # Sett miljøvariabel
        if use_sqlite:
            os.environ["USE_SQLITE"] = "true"
            
        # Importer database-modulen og test tilkoblingen
        try:
            from backend.database.database import engine
        except ImportError:
            try:
                # Prøv alternativ import
                from database.database import engine
            except ImportError as e:
                logger.error(f"Kunne ikke importere database-engine: {str(e)}")
                logger.error(traceback.format_exc())
                sys.exit(1)
        
        # Test tilkoblingen
        conn = engine.connect()
        conn.close()
        
        logger.info("Databasetilkobling vellykket!")
    except Exception as e:
        logger.error(f"Kunne ikke koble til database: {str(e)}")
        logger.error(traceback.format_exc())
        if not use_sqlite:
            logger.info("Du kan bruke --use-sqlite for å kjøre med SQLite istedenfor PostgreSQL")
        sys.exit(1)

def run_server(host, port, reload=False, workers=1):
    """Start API-server med uvicorn"""
    logger.info(f"Starter server på http://{host}:{port}...")
    
    try:
        # Verifiser at vi kan importere app-modulen
        try:
            # Legg til prosjektets rotmappe i PYTHONPATH
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
            sys.path.insert(0, PROJECT_ROOT)
            sys.path.insert(0, SCRIPT_DIR)
            
            # Forsøk å importere app direkte
            try:
                import app
                logger.info("Importert app.py vellykket (direkte import)")
                app_module = "app:app"
            except ImportError as e:
                logger.warning(f"Kunne ikke importere app direkte: {e}")
                
                # Forsøk å importere fra backend-pakken
                try:
                    from backend.app import app
                    logger.info("Importert app-modulen vellykket (fra backend)")
                    app_module = "backend.app:app"
                except ImportError as e2:
                    logger.warning(f"Kunne ikke importere fra backend.app: {e2}")
                    
                    # Dette er et siste forsøk der vi bruker den relative stien
                    app_module = "app:app"
                    logger.info(f"Bruker {app_module} som fallback")
            
        except Exception as e:
            logger.error(f"Problem med import av app: {e}")
            logger.error(traceback.format_exc())
            app_module = "app:app"  # Fallback
        
        import uvicorn
        logger.info("Importert uvicorn vellykket")
        
        # Sett miljøvariabler som uvicorn trenger
        os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
        
        # Logg detaljert konfigurasjon
        logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
        logger.info(f"Kjører app-modul: {app_module}")
        logger.info(f"Host: {host}, Port: {port}")
        logger.info(f"Reload: {reload}, Workers: {workers}")
        
        if os.name == 'nt' and reload:
            logger.warning("Hot reload kan være ustabilt på Windows. Vurder å bruke --reload=False hvis du opplever problemer.")
        
        # Lag config-objekt for uvicorn
        config = uvicorn.Config(
            app_module,
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,  # Kun 1 worker med reload
            log_level="info"
        )
        
        logger.info("Starter uvicorn server...")
        server = uvicorn.Server(config)
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server avsluttet av bruker")
    except Exception as e:
        logger.error(f"Feil ved oppstart av server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def pre_initialize_services(args):
    """Pre-initialiser viktige tjenester for å unngå oppstartsproblemer"""
    logger.info("Pre-initialiserer viktige tjenester...")
    
    # Unngå circular import
    from importlib import import_module
    import threading
    
    # Liste over tjenester å initialisere
    services_to_init = []
    
    # Bare legg til tjenester som vi vet kan ta tid å starte
    if not args.no_lazy_load:
        try:
            # CommuneConnect
            services_to_init.append(("backend.api.CommuneConnect", "CommuneConnect", "get_instance"))
            
            # AlterraML
            services_to_init.append(("ai_modules.AlterraML", "AlterraML", None))
        except Exception as e:
            logger.warning(f"Kunne ikke definere tjenester for pre-initialisering: {e}")
    
    # Initialiser i separate tråder
    init_threads = []
    for module_path, class_name, method_name in services_to_init:
        def initialize_service(module_path, class_name, method_name):
            try:
                logger.info(f"Initialiserer {class_name}...")
                module = import_module(module_path)
                class_ = getattr(module, class_name)
                
                if method_name:
                    # Hvis vi har en metode, kall den (f.eks. singleton-pattern)
                    instance = getattr(class_, method_name)()
                else:
                    # Ellers initialiser direkte
                    instance = class_()
                    
                logger.info(f"{class_name} initialisert vellykket i bakgrunnen")
                return instance
            except Exception as e:
                logger.warning(f"Kunne ikke initialisere {class_name}: {e}")
                if args.debug:
                    logger.debug(traceback.format_exc())
                return None
        
        # Start tråd
        thread = threading.Thread(
            target=initialize_service, 
            args=(module_path, class_name, method_name),
            daemon=True  # Daemon-tråder avsluttes når hovedprogrammet avsluttes
        )
        thread.start()
        init_threads.append(thread)
    
    # Ikke vent på trådene - la dem kjøre i bakgrunnen
    logger.info(f"Startet {len(init_threads)} initialiseringstråder i bakgrunnen")

def main():
    """Hovedfunksjon"""
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("Eiendomsmuligheter Platform Backend")
    logger.info("=" * 50)
    
    # Sett debug-modus
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug-modus aktivert")
    
    # Sjekk avhengigheter
    if not check_requirements(args.use_sqlite):
        logger.error("Feil ved sjekk av avhengigheter. Forsøker å fortsette likevel.")
    
    # Sett miljøvariabler
    if args.use_sqlite:
        os.environ["USE_SQLITE"] = "true"
    
    # Pre-initialiser viktige tjenester
    pre_initialize_services(args)
    
    # Initialiser database om nødvendig
    if args.init_db:
        init_database(args.use_sqlite)
    
    # Sjekk databasetilkobling
    try:
        check_database_connection(args.use_sqlite)
    except Exception as e:
        logger.error(f"Databasefeil: {e}")
        if args.debug:
            logger.debug(traceback.format_exc())
        sys.exit(1)
    
    # Start serveren
    try:
        run_server(args.host, args.port, args.reload, args.workers)
    except Exception as e:
        logger.error(f"Serverfeil: {e}")
        if args.debug:
            logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program avsluttet av bruker")
    except Exception as e:
        logger.error(f"Uventet feil: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 