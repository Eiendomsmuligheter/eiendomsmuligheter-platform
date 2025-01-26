#!/bin/bash

# Eiendomsmuligheter Platform - AI Context Loader
# Dette skriptet kjøres ved start av hver ny AI-samtale

# Farger for bedre lesbarhet
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Prosjektmappe
PROJECT_DIR="/home/computeruse/eiendomsmuligheter"

# Funksjon for å generere tidsstempel
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Oppdater timestamps og git-informasjon
update_timestamps() {
    # Oppdater siste commit info
    LAST_COMMIT=$(git -C "$PROJECT_DIR" log -1 --format="%h - %s (%cr)")
    sed -i "s/- Siste commit: \[DATO\]/- Siste commit: $LAST_COMMIT/" "$PROJECT_DIR/ai_startup_instructions.md"
}

# Hovedfunksjon
main() {
    echo -e "${GREEN}Laster AI kontekst for Eiendomsmuligheter Platform...${NC}"
    
    # Oppdater tidsstempler og git-info
    update_timestamps
    
    # Vis oppstartsinstruksjoner
    echo -e "${BLUE}AI Assistant oppstartsinstruksjoner er lastet.${NC}"
    echo -e "${BLUE}Generer statusrapport og fortsett arbeidet.${NC}"
    
    # Last instruksjonene
    cat "$PROJECT_DIR/ai_startup_instructions.md"
}

# Kjør hovedfunksjonen
main