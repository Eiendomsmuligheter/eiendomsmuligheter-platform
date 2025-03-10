"""
Database-konfigurasjon for Eiendomsmuligheter Platform

Denne modulen setter opp SQLAlchemy-tilkobling til databasen
og definerer Base-klassen for alle modeller.
"""
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from typing import Generator

# Konfigurer logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hent databasekonfigurasjon fra miljøvariabler
DB_USER = os.getenv("POSTGRES_USER", "eiendom_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")  # I produksjon bør denne være sikker
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "eiendomsmuligheter")
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
ECHO_SQL = os.getenv("ECHO_SQL", "false").lower() == "true"

# Sett opp databasen URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# For SQLite (kan brukes for lokal utvikling uten PostgreSQL)
SQLITE_URL = os.getenv("SQLITE_URL", "sqlite:///./eiendomsmuligheter.db")

# Velg URL basert på miljø
def get_database_url():
    """Hent database URL basert på miljøvariabel"""
    use_sqlite = os.getenv("USE_SQLITE", "false").lower() == "true"
    return SQLITE_URL if use_sqlite else DATABASE_URL

# Opprett engine og session factory
def create_db_engine():
    """Opprett SQLAlchemy engine med passende innstillinger"""
    url = get_database_url()
    
    is_sqlite = url.startswith("sqlite")
    
    logger.info(f"Kobler til database via: {url.split('@')[-1]}")  # Logg bare host/path, ikke passord
    
    engine_args = {
        "echo": ECHO_SQL
    }
    
    # PostgreSQL-spesifikke innstillinger
    if not is_sqlite:
        engine_args.update({
            "pool_size": POOL_SIZE,
            "max_overflow": MAX_OVERFLOW,
            "pool_pre_ping": True
        })
    
    return create_engine(url, **engine_args)

# Opprett engine
engine = create_db_engine()

# Opprett session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Opprett scoped session for thread-safety
db_session = scoped_session(SessionLocal)

# Declarative base class
Base = declarative_base()
Base.query = db_session.query_property()

# Context manager for database sessions
@contextmanager
def get_db() -> Generator:
    """
    Context manager som gir en database session som
    automatisk avsluttes og rulles tilbake ved unntak.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()

def init_db():
    """Initialiser databasen og opprett tabeller"""
    # Import modeller slik at de er registrert med Base
    from backend.database.models import property_model, user_model, analysis_model
    
    # Opprett tabeller
    logger.info("Oppretter databasetabeller...")
    Base.metadata.create_all(bind=engine)
    logger.info("Databasetabeller opprettet!")

def reset_db():
    """Reset databasen (fjern alle tabeller og opprett dem på nytt)"""
    logger.warning("Fjerner alle tabeller i databasen!")
    Base.metadata.drop_all(bind=engine)
    init_db()
    logger.info("Database resatt og tabeller opprettet!")

# Initialiser direkte hvis skriptet kjøres
if __name__ == "__main__":
    if os.getenv("RESET_DB", "false").lower() == "true":
        reset_db()
    else:
        init_db()