from typing import Dict, List, Optional
import ifcopenshell
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class BIMService:
    def __init__(self):
        self.supported_formats = ['ifc', 'ifcxml', 'ifcjson']
        
    async def create_bim_model(self, property_data: Dict, analysis_results: Dict) -> str:
        """Oppretter en BIM-modell fra eiendomsdata og analyseresultater"""
        try:
            # Opprett ny IFC fil
            ifc_file = ifcopenshell.file(schema="IFC4")
            
            # Sett opp prosjekt og kontekst
            project = ifc_file.createIfcProject(
                GlobalId=ifcopenshell.guid.new(),
                Name="Eiendomsanalyse"
            )
            
            # Opprett bygning
            building = self._create_building(ifc_file, property_data)
            
            # Legg til etasjer
            for floor_data in property_data.get('floors', []):
                floor = self._create_floor(ifc_file, building, floor_data)
                
            # Legg til rom og soner
            for space_data in property_data.get('spaces', []):
                space = self._create_space(ifc_file, building, space_data)
                
            # Eksporter til IFC
            output_path = Path(f"/tmp/{property_data['id']}_model.ifc")
            ifc_file.write(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Feil ved oppretting av BIM-modell: {e}")
            raise
            
    def _create_building(self, ifc_file, property_data: Dict):
        """Oppretter bygningsobjekt i IFC"""
        return ifc_file.createIfcBuilding(
            GlobalId=ifcopenshell.guid.new(),
            Name=f"Bygning - {property_data.get('address')}",
            CompositionType="ELEMENT",
            ObjectPlacement=self._create_placement(ifc_file)
        )
        
    def _create_floor(self, ifc_file, building, floor_data: Dict):
        """Oppretter etasjeobjekt i IFC"""
        return ifc_file.createIfcBuildingStorey(
            GlobalId=ifcopenshell.guid.new(),
            Name=f"Etasje {floor_data.get('level')}",
            CompositionType="ELEMENT",
            ObjectPlacement=self._create_placement(ifc_file)
        )
        
    def _create_space(self, ifc_file, building, space_data: Dict):
        """Oppretter romobjekt i IFC"""
        return ifc_file.createIfcSpace(
            GlobalId=ifcopenshell.guid.new(),
            Name=space_data.get('name'),
            CompositionType="ELEMENT",
            ObjectPlacement=self._create_placement(ifc_file)
        )
        
    def _create_placement(self, ifc_file):
        """Hjelpefunksjon for å opprette plassering i IFC"""
        axis2placement = ifc_file.createIfcAxis2Placement3D(
            ifc_file.createIfcCartesianPoint((0.0, 0.0, 0.0))
        )
        return ifc_file.createIfcLocalPlacement(None, axis2placement)
        
    async def export_model(self, model_path: str, format: str = 'ifc') -> str:
        """Eksporterer BIM-modell til ønsket format"""
        if format not in self.supported_formats:
            raise ValueError(f"Ikke støttet format. Støttede formater: {self.supported_formats}")
            
        try:
            ifc_file = ifcopenshell.open(model_path)
            output_path = Path(model_path).with_suffix(f".{format}")
            
            if format == 'ifcxml':
                ifc_file.write(str(output_path))
            elif format == 'ifcjson':
                # Konverter til JSON
                json_data = self._ifc_to_json(ifc_file)
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Feil ved eksport av BIM-modell: {e}")
            raise
            
    def _ifc_to_json(self, ifc_file) -> Dict:
        """Konverterer IFC til JSON format"""
        json_data = {
            "project": {},
            "buildings": [],
            "stories": [],
            "spaces": []
        }
        
        # Hent prosjektinfo
        project = ifc_file.by_type("IfcProject")[0]
        json_data["project"] = {
            "id": project.GlobalId,
            "name": project.Name
        }
        
        # Hent bygninger
        for building in ifc_file.by_type("IfcBuilding"):
            json_data["buildings"].append({
                "id": building.GlobalId,
                "name": building.Name
            })
            
        # Hent etasjer
        for story in ifc_file.by_type("IfcBuildingStorey"):
            json_data["stories"].append({
                "id": story.GlobalId,
                "name": story.Name,
                "elevation": story.Elevation if hasattr(story, "Elevation") else 0
            })
            
        # Hent rom
        for space in ifc_file.by_type("IfcSpace"):
            json_data["spaces"].append({
                "id": space.GlobalId,
                "name": space.Name,
                "longName": space.LongName if hasattr(space, "LongName") else None
            })
            
        return json_data