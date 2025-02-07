import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from ..models.municipal_model import (
    PropertyData,
    Regulation,
    BuildingApplication,
    HistoricalCase,
    BuildingPermit,
    MunicipalFees,
    ZoningPlan,
    PropertyAnalysis,
    MunicipalityAPI
)

logger = logging.getLogger(__name__)

class MunicipalityService:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialiserer tjenesten med konfigurasjon for ulike kommuner
        """
        self.config = config
        self.session = None
        self.municipality_apis: Dict[str, MunicipalityAPI] = {}
        self._initialize_apis()

    def _initialize_apis(self):
        """
        Initialiserer API-konfigurasjoner for hver støttet kommune
        """
        for muni_code, muni_config in self.config["municipalities"].items():
            self.municipality_apis[muni_code] = MunicipalityAPI(
                municipality_code=muni_code,
                municipality_name=muni_config["name"],
                base_url=muni_config["base_url"],
                api_key=muni_config.get("api_key"),
                endpoints=muni_config["endpoints"],
                authentication_method=muni_config["auth_method"],
                rate_limits=muni_config.get("rate_limits"),
                contact_info=muni_config["contact_info"]
            )

    async def __aenter__(self):
        """
        Setter opp asynkron HTTP-sesjon
        """
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Lukker HTTP-sesjon
        """
        if self.session:
            await self.session.close()

    async def get_property_data(self, municipality_code: str, property_id: str) -> PropertyData:
        """
        Henter grunndata for en eiendom
        """
        try:
            api = self.municipality_apis[municipality_code]
            endpoint = api.endpoints["property_data"]
            url = f"{api.base_url}{endpoint}/{property_id}"

            async with self.session.get(url, headers=self._get_headers(api)) as response:
                response.raise_for_status()
                data = await response.json()
                return PropertyData(**data)

        except Exception as e:
            logger.error(f"Feil ved henting av eiendomsdata: {str(e)}")
            raise

    async def get_regulations(self, municipality_code: str, property_id: str) -> List[Regulation]:
        """
        Henter gjeldende reguleringsbestemmelser for en eiendom
        """
        try:
            api = self.municipality_apis[municipality_code]
            endpoint = api.endpoints["regulations"]
            url = f"{api.base_url}{endpoint}/{property_id}"

            async with self.session.get(url, headers=self._get_headers(api)) as response:
                response.raise_for_status()
                data = await response.json()
                return [Regulation(**reg) for reg in data]

        except Exception as e:
            logger.error(f"Feil ved henting av reguleringsbestemmelser: {str(e)}")
            raise

    async def submit_building_application(
        self,
        municipality_code: str,
        application: BuildingApplication
    ) -> Dict[str, Any]:
        """
        Sender inn byggesøknad til kommunen
        """
        try:
            api = self.municipality_apis[municipality_code]
            endpoint = api.endpoints["submit_application"]
            url = f"{api.base_url}{endpoint}"

            async with self.session.post(
                url,
                headers=self._get_headers(api),
                json=application.dict()
            ) as response:
                response.raise_for_status()
                return await response.json()

        except Exception as e:
            logger.error(f"Feil ved innsending av byggesøknad: {str(e)}")
            raise

    async def get_historical_cases(
        self,
        municipality_code: str,
        property_id: str
    ) -> List[HistoricalCase]:
        """
        Henter historiske byggesaker for en eiendom
        """
        try:
            api = self.municipality_apis[municipality_code]
            endpoint = api.endpoints["historical_cases"]
            url = f"{api.base_url}{endpoint}/{property_id}"

            async with self.session.get(url, headers=self._get_headers(api)) as response:
                response.raise_for_status()
                data = await response.json()
                return [HistoricalCase(**case) for case in data]

        except Exception as e:
            logger.error(f"Feil ved henting av historiske saker: {str(e)}")
            raise

    async def get_zoning_plan(
        self,
        municipality_code: str,
        plan_id: str
    ) -> ZoningPlan:
        """
        Henter reguleringsplan
        """
        try:
            api = self.municipality_apis[municipality_code]
            endpoint = api.endpoints["zoning_plan"]
            url = f"{api.base_url}{endpoint}/{plan_id}"

            async with self.session.get(url, headers=self._get_headers(api)) as response:
                response.raise_for_status()
                data = await response.json()
                return ZoningPlan(**data)

        except Exception as e:
            logger.error(f"Feil ved henting av reguleringsplan: {str(e)}")
            raise

    async def analyze_property_potential(
        self,
        municipality_code: str,
        property_id: str
    ) -> PropertyAnalysis:
        """
        Utfører en komplett analyse av eiendomspotensial
        """
        try:
            # Hent all nødvendig informasjon parallelt
            property_data, regulations, historical_cases = await asyncio.gather(
                self.get_property_data(municipality_code, property_id),
                self.get_regulations(municipality_code, property_id),
                self.get_historical_cases(municipality_code, property_id)
            )

            # Analyser utviklingspotensial
            development_potential = self._analyze_development_potential(
                property_data,
                regulations,
                historical_cases
            )

            # Generer anbefalinger
            recommendations = self._generate_recommendations(
                development_potential,
                regulations
            )

            return PropertyAnalysis(
                property_id=property_id,
                analysis_date=datetime.now(),
                current_regulations=regulations[0] if regulations else None,
                development_potential=development_potential,
                restrictions=self._get_restrictions(regulations, historical_cases),
                recommended_actions=recommendations,
                estimated_processing_time=self._estimate_processing_time(development_potential),
                estimated_costs=self._estimate_costs(development_potential, municipality_code),
                risk_assessment=self._assess_risks(development_potential, historical_cases)
            )

        except Exception as e:
            logger.error(f"Feil ved analyse av eiendomspotensial: {str(e)}")
            raise

    def _get_headers(self, api: MunicipalityAPI) -> Dict[str, str]:
        """
        Genererer headers for API-kall
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if api.api_key:
            headers["Authorization"] = f"Bearer {api.api_key}"
            
        return headers

    def _analyze_development_potential(
        self,
        property_data: PropertyData,
        regulations: List[Regulation],
        historical_cases: List[HistoricalCase]
    ) -> Dict[str, Any]:
        """
        Analyserer utviklingspotensial basert på all tilgjengelig data
        """
        # Implementer avansert analyse her
        return {
            "potential_types": self._identify_potential_development_types(regulations),
            "max_buildable_area": self._calculate_max_buildable_area(property_data, regulations),
            "suggested_developments": self._suggest_developments(property_data, regulations),
            "constraints": self._identify_constraints(property_data, regulations, historical_cases)
        }

    def _generate_recommendations(
        self,
        development_potential: Dict[str, Any],
        regulations: List[Regulation]
    ) -> List[Dict[str, Any]]:
        """
        Genererer konkrete anbefalinger basert på utviklingspotensial
        """
        recommendations = []
        
        # Implementer anbefalingsgenerering her
        for potential_type in development_potential["potential_types"]:
            recommendations.append({
                "type": potential_type,
                "description": self._get_development_description(potential_type),
                "requirements": self._get_development_requirements(potential_type, regulations),
                "estimated_timeline": self._estimate_timeline(potential_type),
                "next_steps": self._get_next_steps(potential_type)
            })
            
        return recommendations

    def _estimate_processing_time(self, development_potential: Dict[str, Any]) -> int:
        """
        Estimerer behandlingstid i dager
        """
        # Implementer estimering her
        base_time = 60  # Standard behandlingstid
        complexity_factor = len(development_potential["constraints"]) * 10
        return base_time + complexity_factor

    def _estimate_costs(
        self,
        development_potential: Dict[str, Any],
        municipality_code: str
    ) -> Dict[str, float]:
        """
        Estimerer kostnader for utvikling
        """
        # Implementer kostnadsestimering her
        return {
            "application_fee": self._calculate_application_fee(development_potential, municipality_code),
            "development_cost": self._calculate_development_cost(development_potential),
            "connection_fees": self._calculate_connection_fees(municipality_code),
            "other_fees": self._calculate_other_fees(development_potential)
        }

    def _assess_risks(
        self,
        development_potential: Dict[str, Any],
        historical_cases: List[HistoricalCase]
    ) -> Dict[str, Any]:
        """
        Vurderer risiko ved utviklingsprosjektet
        """
        # Implementer risikovurdering her
        return {
            "regulatory_risks": self._assess_regulatory_risks(development_potential),
            "technical_risks": self._assess_technical_risks(development_potential),
            "historical_risks": self._assess_historical_risks(historical_cases),
            "mitigation_strategies": self._generate_risk_mitigation_strategies()
        }

    # Implementer hjelpemetoder her...
