"""
Property Analysis Service - Core component for analyzing properties.
"""
from typing import Dict, Any, List
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PropertyAnalysis:
    property_id: str
    address: str
    municipality: str
    analysis_date: datetime
    gnr_bnr: str
    regulations: Dict[str, Any]
    development_potential: Dict[str, Any]
    building_data: Dict[str, Any]
    restrictions: List[str]
    recommendations: List[Dict[str, Any]]

class PropertyAnalyzer:
    """Main property analysis service."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_analyzers()
    
    def _init_analyzers(self):
        """Initialize all sub-analyzers."""
        # Initialize AI models and services
        self.floor_plan_analyzer = FloorPlanAnalyzer()
        self.regulation_analyzer = RegulationAnalyzer()
        self.development_analyzer = DevelopmentAnalyzer()
        self.building_code_analyzer = BuildingCodeAnalyzer()
        self.energy_analyzer = EnergyAnalyzer()
    
    async def analyze_property(self, 
                             address: str = None,
                             image_url: str = None,
                             listing_url: str = None) -> PropertyAnalysis:
        """
        Main analysis method that coordinates all sub-analyses.
        """
        try:
            # Step 1: Basic property information
            property_info = await self._get_property_info(address, image_url, listing_url)
            
            # Step 2: Municipality regulations
            regulations = await self.regulation_analyzer.analyze(
                municipality=property_info['municipality'],
                gnr_bnr=property_info['gnr_bnr']
            )
            
            # Step 3: Building analysis
            building_data = await self.floor_plan_analyzer.analyze(
                floor_plans=property_info['floor_plans'],
                building_type=property_info['building_type']
            )
            
            # Step 4: Development potential
            development_potential = await self.development_analyzer.analyze(
                building_data=building_data,
                regulations=regulations,
                property_info=property_info
            )
            
            # Step 5: Building code analysis
            building_code_results = await self.building_code_analyzer.analyze(
                construction_year=property_info['construction_year'],
                municipality=property_info['municipality']
            )
            
            # Step 6: Energy analysis
            energy_analysis = await self.energy_analyzer.analyze(
                building_data=building_data,
                construction_year=property_info['construction_year']
            )
            
            # Compile recommendations
            recommendations = self._compile_recommendations(
                development_potential=development_potential,
                building_code_results=building_code_results,
                energy_analysis=energy_analysis
            )
            
            return PropertyAnalysis(
                property_id=property_info['id'],
                address=property_info['address'],
                municipality=property_info['municipality'],
                analysis_date=datetime.now(),
                gnr_bnr=property_info['gnr_bnr'],
                regulations=regulations,
                development_potential=development_potential,
                building_data=building_data,
                restrictions=building_code_results['restrictions'],
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error during property analysis: {str(e)}")
            raise
    
    async def _get_property_info(self, 
                               address: str = None,
                               image_url: str = None,
                               listing_url: str = None) -> Dict[str, Any]:
        """
        Fetches basic property information from various sources.
        """
        # Implementation for property info gathering
        pass
    
    def _compile_recommendations(self,
                               development_potential: Dict[str, Any],
                               building_code_results: Dict[str, Any],
                               energy_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compiles and prioritizes recommendations based on analysis results.
        """
        recommendations = []
        
        # Analyze development potential
        if development_potential['can_split_plot']:
            recommendations.append({
                'type': 'plot_division',
                'priority': 'high',
                'description': 'Tomten kan deles',
                'potential_value': development_potential['split_plot_value'],
                'requirements': development_potential['split_plot_requirements']
            })
        
        # Analyze basement potential
        if development_potential['basement_conversion_possible']:
            recommendations.append({
                'type': 'basement_conversion',
                'priority': 'medium',
                'description': 'Kjeller kan gjøres om til utleieenhet',
                'potential_value': development_potential['basement_rental_value'],
                'requirements': development_potential['basement_requirements']
            })
        
        # Analyze attic potential
        if development_potential['attic_conversion_possible']:
            recommendations.append({
                'type': 'attic_conversion',
                'priority': 'medium',
                'description': 'Loft kan bygges om',
                'potential_value': development_potential['attic_conversion_value'],
                'requirements': development_potential['attic_requirements']
            })
        
        # Energy improvements
        for improvement in energy_analysis['recommended_improvements']:
            recommendations.append({
                'type': 'energy_improvement',
                'priority': improvement['priority'],
                'description': improvement['description'],
                'potential_savings': improvement['annual_savings'],
                'enova_support': improvement['enova_support'],
                'requirements': improvement['requirements']
            })
        
        return sorted(recommendations, 
                     key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])