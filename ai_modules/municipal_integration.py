from typing import Dict, List, Optional, Any
import requests
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

@dataclass
class MunicipalRequirement:
    code: str
    description: str
    required_documents: List[str]
    approval_process: str
    processing_time: str
    fees: float
    priority: str = "normal"
    deadline: Optional[str] = None
    responsible_department: Optional[str] = None
    legal_references: List[str] = None
    
@dataclass
class Municipality:
    code: str
    name: str
    county: str
    population: int
    contact_info: Dict[str, str]
    digital_services: List[str]
    processing_times: Dict[str, str]
    fees_table: Dict[str, float]
    
class MunicipalIntegration:
    def __init__(self):
        self.base_url = "https://api.kommune.no"
        self.municipalities = self._load_municipalities()
        self.cache = {}
        self.session = requests.Session()
        self._initialize_session()
        
    def _initialize_session(self):
        """Initialiser HTTP session med nødvendige headers"""
        self.session.headers.update({
            "User-Agent": "EiendomsAI/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        
    def _load_municipalities(self) -> Dict[str, Municipality]:
        """Last inn informasjon om kommuner"""
        try:
            # Last fra lokal cache hvis tilgjengelig
            cache_path = Path("data/municipalities.json")
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        code: Municipality(**info) 
                        for code, info in data.items()
                    }
            
            # Hvis ikke i cache, hent fra API
            response = self.session.get(f"{self.base_url}/municipalities")
            if response.status_code == 200:
                data = response.json()
                
                # Lagre til cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                return {
                    code: Municipality(**info) 
                    for code, info in data.items()
                }
                
        except Exception as e:
            logger.error(f"Feil ved lasting av kommunedata: {str(e)}")
            
        # Returner dummy-data hvis alt annet feiler
        return {
            "0301": Municipality(
                code="0301",
                name="Oslo",
                county="Oslo",
                population=693494,
                contact_info={
                    "email": "postmottak@oslo.kommune.no",
                    "phone": "21 80 21 80",
                    "address": "Rådhuset, 0037 Oslo"
                },
                digital_services=["byggesak", "reguleringsplan", "eiendomsinfo"],
                processing_times={
                    "byggesak": "4-6 uker",
                    "bruksendring": "6-8 uker",
                    "reguleringsplan": "12-16 uker"
                },
                fees_table={
                    "byggesak": 5000.0,
                    "bruksendring": 3500.0,
                    "reguleringsplan": 15000.0
                }
            )
        }
        
    def get_requirements(self, municipality: str, building_type: str) -> List[MunicipalRequirement]:
        """Hent krav for spesifikk kommune og byggtype"""
        try:
            # Sjekk cache først
            cache_key = f"{municipality}_{building_type}_requirements"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Hent kommune-informasjon
            muni = self.municipalities.get(municipality)
            if not muni:
                raise ValueError(f"Ukjent kommune: {municipality}")
            
            # Hent krav fra kommunens API
            response = self.session.get(
                f"{self.base_url}/requirements",
                params={
                    "municipality": municipality,
                    "building_type": building_type
                }
            )
            
            if response.status_code == 200:
                requirements_data = response.json()
                requirements = []
                
                for req_data in requirements_data:
                    requirement = MunicipalRequirement(
                        code=req_data["code"],
                        description=req_data["description"],
                        required_documents=req_data["required_documents"],
                        approval_process=req_data["approval_process"],
                        processing_time=req_data["processing_time"],
                        fees=req_data["fees"],
                        priority=req_data.get("priority", "normal"),
                        deadline=req_data.get("deadline"),
                        responsible_department=req_data.get("responsible_department"),
                        legal_references=req_data.get("legal_references", [])
                    )
                    requirements.append(requirement)
                
                # Lagre i cache
                self.cache[cache_key] = requirements
                return requirements
                
        except requests.RequestException as e:
            logger.error(f"Nettverksfeil ved henting av krav: {str(e)}")
        except Exception as e:
            logger.error(f"Feil ved henting av kommunale krav: {str(e)}")
        
        # Returner standard krav hvis API-kall feiler
        return [
            MunicipalRequirement(
                code="REQ001",
                description="Krav til brannsikring",
                required_documents=["Branntegning", "Risikoanalyse"],
                approval_process="Standard",
                processing_time="4-6 uker",
                fees=5000.0,
                priority="høy",
                responsible_department="Brann- og redningsetaten",
                legal_references=["TEK17 §11-1", "TEK17 §11-2"]
            ),
            MunicipalRequirement(
                code="REQ002",
                description="Krav til lydisolasjon",
                required_documents=["Lydteknisk rapport"],
                approval_process="Enkel",
                processing_time="2-3 uker",
                fees=2500.0,
                priority="normal",
                responsible_department="Byggesaksavdelingen",
                legal_references=["TEK17 §13-6"]
            ),
            MunicipalRequirement(
                code="REQ003",
                description="Krav til ventilasjon",
                required_documents=["Ventilasjonstegninger", "Teknisk beskrivelse"],
                approval_process="Standard",
                processing_time="3-4 uker",
                fees=3500.0,
                priority="normal",
                responsible_department="Byggesaksavdelingen",
                legal_references=["TEK17 §13-1", "TEK17 §13-2", "TEK17 §13-3"]
            )
        ]
            
    def generate_application(self, property_data: Dict[str, Any], municipality: str) -> Dict[str, Any]:
        """Generer søknadsskjema for kommunen"""
        try:
            # Valider inndata
            required_fields = ["address", "owner", "purpose", "building_type"]
            for field in required_fields:
                if field not in property_data:
                    raise ValueError(f"Manglende påkrevd felt: {field}")
            
            # Hent kommunespesifikke krav
            requirements = self.get_requirements(
                municipality,
                property_data["building_type"]
            )
            
            # Generer søknadsdokumenter
            documents = self._generate_documents(property_data, requirements)
            
            # Verifiser at alle krav er oppfylt
            requirements_met = self._verify_requirements(
                property_data,
                requirements,
                documents
            )
            
            # Opprett søknad
            application = {
                "application_id": self._generate_application_id(municipality),
                "status": "draft",
                "submission_date": datetime.now().isoformat(),
                "municipality": municipality,
                "property_data": property_data,
                "documents": documents,
                "requirements_met": requirements_met,
                "estimated_processing_time": self.estimate_processing_time(
                    "bruksendring",
                    municipality
                ),
                "fees": sum(req.fees for req in requirements),
                "contact_person": property_data.get("owner"),
                "notifications_enabled": True
            }
            
            # Lagre søknad i systemet
            self._save_application(application)
            
            return application
            
        except ValueError as e:
            logger.error(f"Valideringsfeil: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Feil ved generering av søknad: {str(e)}")
            return {"error": str(e)}
            
    def _generate_application_id(self, municipality: str) -> str:
        """Generer unik søknads-ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = "".join(re.findall(r"\d+", str(hash(timestamp))))[:4]
        return f"{municipality}-{timestamp}-{random_suffix}"
    
    def _generate_documents(self, property_data: Dict[str, Any],
                          requirements: List[MunicipalRequirement]) -> List[Dict[str, Any]]:
        """Generer nødvendige dokumenter for søknaden"""
        documents = []
        
        # Hovedsøknadsskjema
        documents.append({
            "type": "application_form",
            "title": "Søknad om bruksendring",
            "content": self._generate_application_form(property_data),
            "format": "pdf"
        })
        
        # Generer dokumenter basert på krav
        for req in requirements:
            for doc_type in req.required_documents:
                documents.append({
                    "type": doc_type.lower().replace(" ", "_"),
                    "title": doc_type,
                    "content": self._generate_document_content(
                        doc_type,
                        property_data,
                        req
                    ),
                    "format": "pdf",
                    "requirement_code": req.code
                })
        
        return documents
    
    def _verify_requirements(self, property_data: Dict[str, Any],
                           requirements: List[MunicipalRequirement],
                           documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verifiser at alle krav er oppfylt"""
        verification_results = []
        
        for req in requirements:
            # Sjekk om alle påkrevde dokumenter er generert
            required_docs = set(req.required_documents)
            submitted_docs = set(
                doc["type"] for doc in documents
                if doc.get("requirement_code") == req.code
            )
            
            missing_docs = required_docs - submitted_docs
            
            verification_results.append({
                "requirement_code": req.code,
                "status": "fulfilled" if not missing_docs else "missing_documents",
                "missing_documents": list(missing_docs),
                "notes": []
            })
        
        return verification_results
            
    def check_zoning_regulations(self, address: str, municipality: str) -> Dict[str, Any]:
        """Sjekk reguleringsplan for adressen"""
        try:
            # Formater adresse
            formatted_address = self._format_address(address)
            
            # Hent reguleringsdata fra kommunens API
            response = self.session.get(
                f"{self.base_url}/zoning",
                params={
                    "municipality": municipality,
                    "address": formatted_address
                }
            )
            
            if response.status_code == 200:
                zoning_data = response.json()
                
                # Analyser reguleringsdata
                analysis = self._analyze_zoning_data(zoning_data)
                
                return {
                    "zone_type": analysis["zone_type"],
                    "allowed_usage": analysis["allowed_usage"],
                    "restrictions": analysis["restrictions"],
                    "property_details": {
                        "area": zoning_data.get("area"),
                        "floor_area_ratio": zoning_data.get("floor_area_ratio"),
                        "height_restrictions": zoning_data.get("height_restrictions")
                    },
                    "special_considerations": analysis["special_considerations"],
                    "nearby_services": self._get_nearby_services(
                        formatted_address,
                        municipality
                    ),
                    "historical_data": self._get_historical_data(
                        formatted_address,
                        municipality
                    )
                }
                
        except requests.RequestException as e:
            logger.error(f"Nettverksfeil ved sjekk av reguleringsplan: {str(e)}")
        except Exception as e:
            logger.error(f"Feil ved sjekk av reguleringsplan: {str(e)}")
        
        # Returner standard data hvis API-kall feiler
        return {
            "zone_type": "residential",
            "allowed_usage": [
                "housing",
                "home_office",
                "small_business"
            ],
            "restrictions": [
                {
                    "type": "noise",
                    "description": "Støygrense 55 dB på dagtid",
                    "requirement": "Must comply with T-1442"
                },
                {
                    "type": "parking",
                    "description": "Minimum 1 parkeringsplass per boenhet",
                    "requirement": "Required"
                }
            ],
            "property_details": {
                "area": 150,
                "floor_area_ratio": 0.35,
                "height_restrictions": "Max 8m"
            },
            "special_considerations": [
                "Verneverdig område",
                "Nær kollektivknutepunkt"
            ],
            "nearby_services": {
                "public_transport": ["buss", "trikk"],
                "schools": ["Barneskole (500m)", "Ungdomsskole (1km)"],
                "healthcare": ["Legesenter (800m)"]
            }
        }
            
    def estimate_processing_time(self, application_type: str,
                               municipality: str) -> Dict[str, Any]:
        """Estimer behandlingstid for søknaden"""
        try:
            # Hent historiske data
            historical_data = self._get_historical_processing_times(
                application_type,
                municipality
            )
            
            # Beregn estimat basert på historiske data
            base_time = historical_data["average_time"]
            confidence = historical_data["confidence"]
            
            # Juster for sesongvariasjoner
            current_month = datetime.now().month
            seasonal_factor = self._calculate_seasonal_factor(current_month)
            
            # Juster for arbeidsbelastning
            workload_factor = self._get_current_workload_factor(municipality)
            
            # Beregn endelig estimat
            estimated_weeks = base_time * seasonal_factor * workload_factor
            
            return {
                "estimated_weeks": round(estimated_weeks, 1),
                "confidence": confidence,
                "factors": [
                    {
                        "name": "Sesongvariasjon",
                        "impact": f"{(seasonal_factor-1)*100:+.1f}%"
                    },
                    {
                        "name": "Arbeidsbelastning",
                        "impact": f"{(workload_factor-1)*100:+.1f}%"
                    }
                ],
                "historical_data": {
                    "average": historical_data["average_time"],
                    "min": historical_data["min_time"],
                    "max": historical_data["max_time"]
                },
                "notes": [
                    "Estimatet er basert på historiske data",
                    "Faktisk behandlingstid kan variere"
                ]
            }
            
        except Exception as e:
            logger.error(f"Feil ved estimering av behandlingstid: {str(e)}")
            
            # Returner standardestimater hvis beregning feiler
            return {
                "estimated_weeks": 6,
                "confidence": 0.8,
                "factors": [
                    {"name": "Kompleksitet", "impact": "0%"},
                    {"name": "Sesong", "impact": "0%"}
                ],
                "notes": ["Standardestimater brukt grunnet mangel på data"]
            }
    
    def _calculate_seasonal_factor(self, month: int) -> float:
        """Beregn sesongbasert justeringsfaktor"""
        # Høyere faktor i travle perioder (vår/høst)
        seasonal_factors = {
            1: 0.9,   # Januar
            2: 0.9,   # Februar
            3: 1.1,   # Mars
            4: 1.2,   # April
            5: 1.3,   # Mai
            6: 1.1,   # Juni
            7: 0.8,   # Juli
            8: 0.9,   # August
            9: 1.2,   # September
            10: 1.1,  # Oktober
            11: 1.0,  # November
            12: 0.9   # Desember
        }
        return seasonal_factors.get(month, 1.0)
    
    def _get_current_workload_factor(self, municipality: str) -> float:
        """Hent current arbeidsbelastningsfaktor for kommunen"""
        try:
            response = self.session.get(
                f"{self.base_url}/workload/{municipality}"
            )
            if response.status_code == 200:
                data = response.json()
                return float(data.get("workload_factor", 1.0))
        except:
            pass
        return 1.0
        
    def _format_address(self, address: str) -> str:
        """Formater adresse til standard format"""
        # Fjern unødvendige mellomrom og spesialtegn
        formatted = re.sub(r'\s+', ' ', address).strip()
        
        # Konverter til store bokstaver for gatenavn
        parts = formatted.split(' ')
        if len(parts) > 1:
            street_name = ' '.join(parts[:-1]).title()
            number = parts[-1]
            formatted = f"{street_name} {number}"
            
        return formatted
        
    def _analyze_zoning_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyser reguleringsdata"""
        analysis = {
            "zone_type": data.get("zone_type", "unknown"),
            "allowed_usage": [],
            "restrictions": [],
            "special_considerations": []
        }
        
        # Analyser tillatt bruk
        if "allowed_usage" in data:
            analysis["allowed_usage"] = data["allowed_usage"]
        
        # Analyser restriksjoner
        if "restrictions" in data:
            analysis["restrictions"] = [
                {
                    "type": r["type"],
                    "description": r["description"],
                    "requirement": r.get("requirement", "Required")
                }
                for r in data["restrictions"]
            ]
        
        # Spesielle hensyn
        if "special_considerations" in data:
            analysis["special_considerations"] = data["special_considerations"]
            
        return analysis
        
    def _get_nearby_services(self, address: str, municipality: str) -> Dict[str, List[str]]:
        """Hent informasjon om nærliggende tjenester"""
        try:
            response = self.session.get(
                f"{self.base_url}/services/nearby",
                params={
                    "address": address,
                    "municipality": municipality,
                    "radius": 1000  # meter
                }
            )
            
            if response.status_code == 200:
                return response.json()
        except:
            pass
            
        # Returner dummy-data hvis API-kall feiler
        return {
            "public_transport": ["buss", "trikk"],
            "schools": ["Barneskole (500m)", "Ungdomsskole (1km)"],
            "healthcare": ["Legesenter (800m)"]
        }
        
    def _get_historical_data(self, address: str, municipality: str) -> Dict[str, Any]:
        """Hent historiske data for eiendommen"""
        try:
            response = self.session.get(
                f"{self.base_url}/property/history",
                params={
                    "address": address,
                    "municipality": municipality
                }
            )
            
            if response.status_code == 200:
                return response.json()
        except:
            pass
            
        # Returner dummy-data hvis API-kall feiler
        return {
            "previous_applications": [],
            "zoning_changes": [],
            "property_value_history": []
        }
        
    def _get_historical_processing_times(self, application_type: str,
                                     municipality: str) -> Dict[str, Any]:
        """Hent historiske behandlingstider"""
        try:
            response = self.session.get(
                f"{self.base_url}/processing-times",
                params={
                    "type": application_type,
                    "municipality": municipality
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "average_time": float(data["average"]),
                    "min_time": float(data["min"]),
                    "max_time": float(data["max"]),
                    "confidence": float(data["confidence"])
                }
        except:
            pass
            
        # Returner dummy-data hvis API-kall feiler
        return {
            "average_time": 6.0,
            "min_time": 4.0,
            "max_time": 8.0,
            "confidence": 0.8
        }
        
    def _generate_application_form(self, property_data: Dict[str, Any]) -> bytes:
        """Generer hovedsøknadsskjema"""
        try:
            # Her ville vi normalt bruke en dokumentgenerator (f.eks. ReportLab)
            # for å lage et faktisk PDF-dokument
            
            # For demonstrasjon, returner en enkel tekstrepresentasjon
            content = f"""
            SØKNAD OM BRUKSENDRING
            =====================
            
            EIENDOMSINFORMASJON
            ------------------
            Adresse: {property_data['address']}
            Eier: {property_data['owner']}
            Formål: {property_data['purpose']}
            Byggtype: {property_data['building_type']}
            
            KONTAKTINFORMASJON
            -----------------
            {property_data.get('contact_info', 'Ikke spesifisert')}
            
            BESKRIVELSE AV TILTAKET
            ----------------------
            {property_data.get('description', 'Ikke spesifisert')}
            
            Generert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return content.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Feil ved generering av søknadsskjema: {str(e)}")
            return b""
            
    def _generate_document_content(self, doc_type: str, 
                                property_data: Dict[str, Any],
                                requirement: MunicipalRequirement) -> bytes:
        """Generer innhold for et spesifikt dokument"""
        try:
            # Her ville vi normalt bruke en dokumentgenerator for å lage
            # faktiske dokumenter basert på maler
            
            content = f"""
            {doc_type.upper()}
            {'=' * len(doc_type)}
            
            EIENDOM
            -------
            Adresse: {property_data['address']}
            
            KRAV
            ----
            Kode: {requirement.code}
            Beskrivelse: {requirement.description}
            
            TEKNISK INFORMASJON
            ------------------
            {self._generate_technical_content(doc_type, property_data)}
            
            ANSVARLIG
            ---------
            Avdeling: {requirement.responsible_department}
            
            RELEVANTE FORSKRIFTER
            -------------------
            {', '.join(requirement.legal_references)}
            
            Generert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return content.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Feil ved generering av dokument {doc_type}: {str(e)}")
            return b""
            
    def _generate_technical_content(self, doc_type: str,
                                 property_data: Dict[str, Any]) -> str:
        """Generer teknisk innhold basert på dokumenttype"""
        if doc_type.lower() == "branntegning":
            return """
            BRANNSIKRINGSTILTAK
            ------------------
            1. Rømningsveier
            2. Brannalarmsystem
            3. Sprinklersystem
            4. Brannskiller
            """
        elif doc_type.lower() == "lydteknisk rapport":
            return """
            LYDTEKNISKE KRAV
            ---------------
            1. Luftlydisolasjon
            2. Trinnlydisolasjon
            3. Etterklangstid
            """
        elif doc_type.lower() == "ventilasjonstegninger":
            return """
            VENTILASJONSKRAV
            ---------------
            1. Luftmengder
            2. Ventilasjonskanaler
            3. Luftbehandlingsanlegg
            """
        else:
            return "Teknisk innhold ikke spesifisert for denne dokumenttypen."
            
    def _save_application(self, application: Dict[str, Any]) -> None:
        """Lagre søknad i systemet"""
        try:
            # Her ville vi normalt lagre i en database
            
            # For demonstrasjon, lagre til fil
            save_path = Path(f"applications/{application['application_id']}.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(application, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Feil ved lagring av søknad: {str(e)}")
            raise

