import requests
from typing import Dict, List, Optional, Any
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from ai_modules.municipal_integration import MunicipalIntegration

logger = logging.getLogger(__name__)

@dataclass
class ZoningRegulation:
    zone_type: str
    usage_types: List[str]
    max_height: float
    max_utilization: float
    min_distance_neighbor: float
    special_requirements: List[str]
    last_updated: datetime

@dataclass
class BuildingRequirement:
    category: str
    description: str
    minimum_value: Optional[float]
    maximum_value: Optional[float]
    required: bool
    reference_law: str
    exemption_possible: bool

class RegulationsHandler:
    def __init__(self):
        self.municipal_integration = MunicipalIntegration()
        self.regulations_db = self._initialize_regulations_db()
        self.building_requirements = self._load_building_requirements()
        self.last_update = datetime.now()
        
    def _initialize_regulations_db(self) -> Dict[str, Dict[str, Any]]:
        """Initialiser database med reguleringer"""
        return {
            "Oslo": {
                "bolig": {
                    "utnyttelsesgrad": 0.24,
                    "max_hoyde": 9.0,
                    "min_avstand_nabo": 4.0,
                    "parkering_krav": {
                        "bil": {"min": 1, "max": 2},
                        "sykkel": {"min": 2, "max": None}
                    },
                    "special_zones": {
                        "verneverdig": {
                            "restricted_changes": True,
                            "additional_requirements": [
                                "Bevaringsverdig fasade",
                                "Historisk tilpasning"
                            ]
                        }
                    }
                },
                "næring": {
                    "utnyttelsesgrad": 0.40,
                    "max_hoyde": 12.0,
                    "min_avstand_nabo": 5.0,
                    "parkering_krav": {
                        "bil": {"min": 2, "max": 10},
                        "sykkel": {"min": 5, "max": None}
                    }
                }
            }
        }
        
    def _load_building_requirements(self) -> Dict[str, BuildingRequirement]:
        """Last inn byggtekniske krav"""
        return {
            "ceiling_height": BuildingRequirement(
                category="ROM",
                description="Minimum takhøyde i oppholdsrom",
                minimum_value=2.4,
                maximum_value=None,
                required=True,
                reference_law="TEK17 §12-7",
                exemption_possible=True
            ),
            "window_area": BuildingRequirement(
                category="LYS",
                description="Minimum vindusareal i prosent av gulvareal",
                minimum_value=0.1,
                maximum_value=None,
                required=True,
                reference_law="TEK17 §13-7",
                exemption_possible=False
            ),
            "ventilation": BuildingRequirement(
                category="LUFT",
                description="Krav til ventilasjon",
                minimum_value=None,
                maximum_value=None,
                required=True,
                reference_law="TEK17 §13-1",
                exemption_possible=False
            )
        }
