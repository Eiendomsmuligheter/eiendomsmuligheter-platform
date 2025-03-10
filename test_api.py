import requests
import json
import time
import os
import traceback
from pprint import pprint

# Farger for terminal-output
class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Base URL for API
BASE_URL = "http://127.0.0.1:8000/api"

def analyze_property():
    """Tester property/analyze endepunktet med en spesifikk adresse"""
    print(f"{TermColors.HEADER}Analyserer eiendom: Moreneveien 37, 3058 Solbergmoen{TermColors.ENDC}")
    
    # Request data
    property_data = {
        "address": "Moreneveien 37, 3058 Solbergmoen",
        "lot_size": 750.0,
        "current_utilization": 0.25,
        "building_height": 6.5,
        "floor_area_ratio": 0.6,
        "zoning_category": "residential",
        "municipality_id": "3025"  # Asker kommune
    }
    
    # Send analysis request
    try:
        print(f"Sender forespørsel til {BASE_URL}/property/analyze")
        print(f"Data: {json.dumps(property_data, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/property/analyze",
            json=property_data,
            timeout=10  # Setter en timeout på 10 sekunder
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Vis hovedresultater
            print(f"\n{TermColors.OKGREEN}✓ Analyse fullført{TermColors.ENDC}")
            print(f"\n{TermColors.BOLD}=== EIENDOMSDETALJER ==={TermColors.ENDC}")
            print(f"Adresse: {result['address']}")
            print(f"Eiendoms-ID: {result['property_id']}")
            
            # Byggepotensial
            print(f"\n{TermColors.BOLD}=== BYGGEPOTENSIAL ==={TermColors.ENDC}")
            bp = result['building_potential']
            print(f"Maksimalt byggbart areal: {bp.get('max_buildable_area')} m²")
            print(f"Maksimal byggehøyde: {bp.get('max_height')} m")
            print(f"Maksimalt antall enheter: {bp.get('max_units')}")
            print(f"Optimal konfigurasjon: {bp.get('optimal_configuration')}")
            
            # Reguleringer
            print(f"\n{TermColors.BOLD}=== REGULERINGSREGLER ==={TermColors.ENDC}")
            for reg in result['regulations']:
                print(f"• {reg.get('description')}: {reg.get('value')} {reg.get('unit', '')}")
            
            # ROI og risiko
            print(f"\n{TermColors.BOLD}=== ØKONOMI OG RISIKO ==={TermColors.ENDC}")
            roi = result.get('roi_estimate', 0) * 100
            print(f"Estimert ROI: {roi:.1f}%")
            
            risk = result.get('risk_assessment', {})
            for risk_type, level in risk.items():
                risk_name = risk_type.replace('_', ' ').title()
                level_color = TermColors.OKGREEN if level == 'low' else TermColors.WARNING if level == 'medium' else TermColors.FAIL
                print(f"• {risk_name}: {level_color}{level.upper()}{TermColors.ENDC}")
            
            # Energiprofil
            if 'energy_profile' in result:
                print(f"\n{TermColors.BOLD}=== ENERGIPROFIL ==={TermColors.ENDC}")
                ep = result['energy_profile']
                print(f"Energiklasse: {ep.get('energy_class')}")
                print(f"Oppvarmingsbehov: {ep.get('heating_demand')} kWh/m²")
                print(f"Kjølebehov: {ep.get('cooling_demand')} kWh/m²")
                print(f"Primær energikilde: {ep.get('primary_energy_source')}")
            
            # Anbefalinger
            print(f"\n{TermColors.BOLD}=== ANBEFALINGER ==={TermColors.ENDC}")
            for rec in result.get('recommendations', []):
                print(f"• {rec}")
                
            # Nødvendige skjemaer
            print(f"\n{TermColors.BOLD}=== NØDVENDIGE SKJEMAER ==={TermColors.ENDC}")
            forms = get_required_forms(result)
            for form_name, form_url in forms.items():
                print(f"• {form_name}: {form_url}")
            
            return result
        else:
            print(f"{TermColors.FAIL}Feil: {response.status_code} - {response.text}{TermColors.ENDC}")
            return None
    except Exception as e:
        print(f"{TermColors.FAIL}Feil: {str(e)}{TermColors.ENDC}")
        print("Detaljer:")
        traceback.print_exc()
        return None

def generate_terrain_visualization(property_id):
    """Tester visualization/terrain/generate endepunktet"""
    print(f"\n{TermColors.HEADER}Genererer terrengvisualisering for eiendom {property_id}{TermColors.ENDC}")
    
    # Request data
    terrain_data = {
        "property_id": property_id,
        "width": 100.0,
        "depth": 100.0,
        "resolution": 128,
        "include_surroundings": True,
        "include_buildings": True,
        "texture_type": "satellite"
    }
    
    # Send terrain generation request
    try:
        response = requests.post(
            f"{BASE_URL}/visualization/terrain/generate",
            json=terrain_data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n{TermColors.OKGREEN}✓ Terrengvisualisering fullført{TermColors.ENDC}")
            print(f"\n{TermColors.BOLD}=== TERRENGDATA ==={TermColors.ENDC}")
            print(f"Høydekart URL: {result.get('heightmap_url')}")
            print(f"Tekstur URL: {result.get('texture_url')}")
            
            # Geografiske grenser
            bounds = result.get('bounds', {})
            print(f"\n{TermColors.BOLD}=== GEOGRAFISKE GRENSER ==={TermColors.ENDC}")
            print(f"Nord: {bounds.get('north')}")
            print(f"Sør: {bounds.get('south')}")
            print(f"Øst: {bounds.get('east')}")
            print(f"Vest: {bounds.get('west')}")
            
            return result
        else:
            print(f"{TermColors.FAIL}Feil: {response.status_code} - {response.text}{TermColors.ENDC}")
            return None
    except Exception as e:
        print(f"{TermColors.FAIL}Feil: {str(e)}{TermColors.ENDC}")
        return None

def generate_building_visualization(property_id):
    """Tester visualization/building/generate endepunktet"""
    print(f"\n{TermColors.HEADER}Genererer bygningsvisualisering for eiendom {property_id}{TermColors.ENDC}")
    
    # Request data
    building_data = {
        "property_id": property_id,
        "building_type": "residential",
        "floors": 2,
        "width": 12.0,
        "depth": 10.0,
        "height": 7.5,
        "roof_type": "pitched",
        "style": "modern"
    }
    
    # Send building generation request
    try:
        response = requests.post(
            f"{BASE_URL}/visualization/building/generate",
            json=building_data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n{TermColors.OKGREEN}✓ Bygningsvisualisering fullført{TermColors.ENDC}")
            print(f"\n{TermColors.BOLD}=== BYGNINGSMODELL ==={TermColors.ENDC}")
            print(f"3D modell URL: {result.get('model_url')}")
            print(f"Miniatyr URL: {result.get('thumbnail_url')}")
            
            # Metadata
            metadata = result.get('metadata', {})
            print(f"\n{TermColors.BOLD}=== BYGNINGSDETALJER ==={TermColors.ENDC}")
            for key, value in metadata.items():
                if key != 'generated_at':
                    print(f"{key}: {value}")
            
            return result
        else:
            print(f"{TermColors.FAIL}Feil: {response.status_code} - {response.text}{TermColors.ENDC}")
            return None
    except Exception as e:
        print(f"{TermColors.FAIL}Feil: {str(e)}{TermColors.ENDC}")
        return None

def get_required_forms(analysis_result):
    """Bestemmer hvilke skjemaer som er nødvendige basert på analyseresultatet"""
    forms = {
        "Byggesøknad (søknadspliktig tiltak)": "https://dibk.no/soknad-og-skjema/alle-byggesoknader/",
        "Nabovarsel": "https://dibk.no/soknad-og-skjema/alle-byggesoknader/Nabovarsel-og-nabomerknader/",
        "Ansvarsrett (gjennomføringsplan)": "https://dibk.no/soknad-og-skjema/alle-byggesoknader/Ansvar-og-kontroll/",
    }
    
    # Legg til skjemaer basert på analyseresultatet
    bp = analysis_result.get('building_potential', {})
    regs = analysis_result.get('regulations', [])
    
    # Hvis over 4 enheter, legg til søknad om deling
    if bp.get('max_units', 0) > 4:
        forms["Søknad om deling og seksjonering"] = "https://dibk.no/soknad-og-skjema/soknad-om-seksjonering/"
    
    # Sjekk reguleringsregler for spesielle forhold
    for reg in regs:
        if reg.get('category') == 'heritage' or (reg.get('description') and 'verneverdig' in reg.get('description').lower()):
            forms["Dispensasjonssøknad (kulturminnevern)"] = "https://dibk.no/soknad-og-skjema/alle-byggesoknader/Dispensasjon/"
    
    # Legg til VA-søknad hvis stor utbygging
    if bp.get('max_buildable_area', 0) > 300:
        forms["Søknad om VA-tilkobling"] = "https://www.asker.kommune.no/vann-og-avlop/soknader-og-skjemaer/"
    
    return forms

def main():
    """Hovedfunksjon som kjører testing av API-et"""
    print(f"{TermColors.BOLD}{TermColors.HEADER}=== TESTING AV EIENDOMSMULIGHETER API ==={TermColors.ENDC}")
    
    # Steg 1: Analyser eiendom
    analysis_result = analyze_property()
    
    if analysis_result:
        property_id = analysis_result.get('property_id')
        
        # Steg 2: Generer terrengvisualisering
        terrain_result = generate_terrain_visualization(property_id)
        
        # Steg 3: Generer bygningsvisualisering
        building_result = generate_building_visualization(property_id)
        
        # Oppsummering
        if terrain_result and building_result:
            print(f"\n{TermColors.OKGREEN}{TermColors.BOLD}✓ Alle tester fullført vellykket!{TermColors.ENDC}")
            print(f"\nDu kan nå gå til http://localhost:3000/property/{property_id} for å se fullstendig visualisering og analyse.")
    else:
        print(f"\n{TermColors.FAIL}Testing avbrutt på grunn av feil.{TermColors.ENDC}")

if __name__ == "__main__":
    main() 