from typing import Dict, List, Optional, Any, Union, Tuple
import requests
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from urllib.parse import urljoin
import hashlib
import asyncio
import aiohttp
from google.cloud import vision
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import emoji

logger = logging.getLogger(__name__)

@dataclass
class MunicipalRequirement:
    """Utvidet dataklasse for kommunale krav"""
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
    regulations: List[Dict[str, Any]] = None
    historical_precedents: List[Dict[str, Any]] = None
    alternative_solutions: List[Dict[str, Any]] = None
    exemption_criteria: List[str] = None
    environmental_impact: Dict[str, Any] = None

@dataclass
class Municipality:
    """Utvidet kommunedataklasse med rik informasjon"""
    code: str
    name: str
    county: str
    population: int
    contact_info: Dict[str, str]
    digital_services: List[str]
    processing_times: Dict[str, str]
    fees_table: Dict[str, float]
    zoning_regulations: Dict[str, Any]
    building_requirements: Dict[str, Any]
    environmental_requirements: Dict[str, Any]
    historical_decisions: List[Dict[str, Any]]
    contact_persons: Dict[str, Dict[str, str]]
    api_endpoints: Dict[str, str]
    document_templates: Dict[str, str]

class MunicipalIntegration:
    """Avansert kommuneintegrasjon med AI-støtte"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialiserer kommuneintegrasjon med avanserte funksjoner"""
        self.base_url = "https://api.kommune.no"
        self.municipalities = self._load_enhanced_municipalities()
        self.cache = self._initialize_cache()
        self.session = self._initialize_session()
        self.ai_models = self._initialize_ai_models()
        self.document_processor = self._initialize_document_processor()
        self.regulation_analyzer = self._initialize_regulation_analyzer()
        
        # Last konfigurasjon
        self.config = self._load_config(config_path)
        
    def _initialize_ai_models(self) -> Dict[str, Any]:
        """Initialiserer AI-modeller for avansert analyse"""
        return {
            "document_classification": self._load_document_classifier(),
            "requirement_analyzer": self._load_requirement_analyzer(),
            "zoning_analyzer": self._load_zoning_analyzer(),
            "text_extraction": self._initialize_text_extraction()
        }
        
    def _load_enhanced_municipalities(self) -> Dict[str, Municipality]:
        """Laster utvidet kommuneinformasjon med rik data"""
        try:
            # Last fra cache først
            cache_path = Path("data/enhanced_municipalities.json")
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {code: Municipality(**info) for code, info in data.items()}
            
            # Hent fra API med utvidet informasjon
            municipalities = {}
            async with aiohttp.ClientSession() as session:
                for code in self._get_municipality_codes():
                    muni_data = await self._fetch_municipality_data(session, code)
                    if muni_data:
                        # Berik data med tilleggsinformasjon
                        enriched_data = self._enrich_municipality_data(muni_data)
                        municipalities[code] = Municipality(**enriched_data)
            
            # Lagre til cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({k: v.__dict__ for k, v in municipalities.items()},
                         f, ensure_ascii=False, indent=2)
            
            return municipalities
            
        except Exception as e:
            logger.error(f"Feil ved lasting av kommunedata: {str(e)}")
            raise
            
    async def get_requirements(
        self,
        municipality: str,
        building_type: str,
        include_analysis: bool = True,
        include_historical: bool = True
    ) -> List[MunicipalRequirement]:
        """Henter og analyserer kommunale krav med AI-støtte"""
        try:
            # Sjekk cache
            cache_key = f"{municipality}_{building_type}_requirements"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Hent grunnleggende krav
            basic_requirements = await self._fetch_basic_requirements(
                municipality,
                building_type
            )
            
            # Berik med AI-analyse
            if include_analysis:
                enriched_requirements = await self._enrich_requirements_with_ai(
                    basic_requirements,
                    municipality,
                    building_type
                )
            else:
                enriched_requirements = basic_requirements
            
            # Legg til historiske presedens
            if include_historical:
                enriched_requirements = await self._add_historical_precedents(
                    enriched_requirements,
                    municipality
                )
            
            # Analyser miljøpåvirkning
            for req in enriched_requirements:
                req.environmental_impact = await self._analyze_environmental_impact(
                    req,
                    municipality
                )
            
            # Lagre i cache
            self.cache[cache_key] = enriched_requirements
            
            return enriched_requirements
            
        except Exception as e:
            logger.error(f"Feil ved henting av kommunale krav: {str(e)}")
            raise
            
    async def generate_complete_application(
        self,
        property_data: Dict[str, Any],
        municipality: str,
        include_ai_analysis: bool = True
    ) -> Dict[str, Any]:
        """Genererer komplett søknadspakke med AI-assistanse"""
        try:
            # Validering og forberedelse
            self._validate_property_data(property_data)
            muni_data = await self._get_municipality_details(municipality)
            
            # Hent alle relevante krav
            requirements = await self.get_requirements(
                municipality,
                property_data["building_type"]
            )
            
            # AI-analyse av prosjektet
            if include_ai_analysis:
                analysis = await self._perform_ai_project_analysis(
                    property_data,
                    requirements,
                    municipality
                )
            
            # Generer alle nødvendige dokumenter
            documents = await self._generate_application_documents(
                property_data,
                requirements,
                analysis if include_ai_analysis else None
            )
            
            # Verifiser krav og forskrifter
            compliance_check = await self._verify_compliance(
                property_data,
                requirements,
                documents
            )
            
            # Generer kostnadsoversikt
            cost_analysis = await self._generate_cost_analysis(
                property_data,
                requirements,
                municipality
            )
            
            # Lag komplett søknadspakke
            application = {
                "application_id": self._generate_secure_application_id(municipality),
                "status": "ready_for_submission",
                "submission_date": datetime.now().isoformat(),
                "municipality": municipality,
                "property_data": property_data,
                "requirements": requirements,
                "documents": documents,
                "compliance_check": compliance_check,
                "ai_analysis": analysis if include_ai_analysis else None,
                "cost_analysis": cost_analysis,
                "processing_estimate": await self._estimate_detailed_processing_time(
                    property_data,
                    municipality
                ),
                "next_steps": self._generate_next_steps(compliance_check),
                "notifications_config": self._setup_notifications(property_data)
            }
            
            # Lagre og valider
            saved_application = await self._save_and_validate_application(application)
            
            return saved_application
            
        except Exception as e:
            logger.error(f"Feil ved generering av søknad: {str(e)}")
            raise
            
    async def check_comprehensive_zoning(
        self,
        address: str,
        municipality: str,
        include_ai_analysis: bool = True
    ) -> Dict[str, Any]:
        """Utfører omfattende reguleringsanalyse med AI-støtte"""
        try:
            # Grunnleggende reguleringsdata
            basic_zoning = await self._fetch_basic_zoning_data(address, municipality)
            
            # AI-basert analyse
            if include_ai_analysis:
                zoning_analysis = await self._analyze_zoning_with_ai(
                    basic_zoning,
                    address,
                    municipality
                )
                
                # Berik med maskinlæringsbaserte innsikter
                enriched_zoning = await self._enrich_zoning_data(
                    zoning_analysis,
                    address,
                    municipality
                )
            else:
                enriched_zoning = basic_zoning
            
            # Hent tilleggsinformasjon
            historical_data = await self._get_historical_zoning_data(
                address,
                municipality
            )
            nearby_services = await self._analyze_nearby_services(
                address,
                municipality
            )
            environmental_impact = await self._assess_environmental_impact(
                address,
                municipality
            )
            
            # Kompiler komplett analyse
            comprehensive_analysis = {
                "basic_zoning": basic_zoning,
                "ai_analysis": zoning_analysis if include_ai_analysis else None,
                "historical_data": historical_data,
                "nearby_services": nearby_services,
                "environmental_impact": environmental_impact,
                "development_potential": await self._analyze_development_potential(
                    enriched_zoning,
                    address,
                    municipality
                ),
                "restrictions": await self._analyze_restrictions(
                    enriched_zoning,
                    municipality
                ),
                "recommendations": await self._generate_zoning_recommendations(
                    enriched_zoning,
                    address,
                    municipality
                )
            }
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Feil ved reguleringsanalyse: {str(e)}")
            raise
            
    async def _analyze_development_potential(
        self,
        zoning_data: Dict[str, Any],
        address: str,
        municipality: str
    ) -> Dict[str, Any]:
        """Analyserer utviklingspotensial med AI-støtte"""
        try:
            # Hent grunnleggende data
            property_data = await self._fetch_property_data(address, municipality)
            
            # Analyser med AI
            analysis = await self._ai_development_analysis(
                zoning_data,
                property_data,
                municipality
            )
            
            # Beregn potensial og muligheter
            potential = {
                "buildable_area": analysis["buildable_area"],
                "height_potential": analysis["height_potential"],
                "usage_options": analysis["usage_options"],
                "value_increase": analysis["value_increase"],
                "constraints": analysis["constraints"],
                "recommendations": analysis["recommendations"]
            }
            
            return potential
            
        except Exception as e:
            logger.error(f"Feil ved analyse av utviklingspotensial: {str(e)}")
            raise
            
    async def _generate_application_documents(
        self,
        property_data: Dict[str, Any],
        requirements: List[MunicipalRequirement],
        ai_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Genererer søknadsdokumenter med AI-assistanse"""
        try:
            documents = []
            
            # Hovedsøknadsskjema
            main_application = await self._generate_main_application(
                property_data,
                requirements,
                ai_analysis
            )
            documents.append(main_application)
            
            # Tekniske tegninger og visualiseringer
            technical_drawings = await self._generate_technical_drawings(
                property_data,
                requirements
            )
            documents.extend(technical_drawings)
            
            # Dokumentasjon og analyser
            documentation = await self._generate_documentation(
                property_data,
                requirements,
                ai_analysis
            )
            documents.extend(documentation)
            
            return documents
            
        except Exception as e:
            logger.error(f"Feil ved dokumentgenerering: {str(e)}")
            raise
            
    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        format: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Genererer omfattende rapport med AI-støttet analyse"""
        try:
            report = {
                "executive_summary": self._generate_executive_summary(analysis_results),
                "detailed_analysis": {
                    "zoning_analysis": self._format_zoning_analysis(
                        analysis_results["zoning"]
                    ),
                    "requirements_analysis": self._format_requirements_analysis(
                        analysis_results["requirements"]
                    ),
                    "development_potential": self._format_development_analysis(
                        analysis_results["development"]
                    ),
                    "cost_analysis": self._format_cost_analysis(
                        analysis_results["costs"]
                    )
                },
                "recommendations": self._generate_recommendations(analysis_results),
                "next_steps": self._generate_action_plan(analysis_results),
                "visualizations": self._generate_visualizations(analysis_results)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Feil ved rapportgenerering: {str(e)}")
            raise