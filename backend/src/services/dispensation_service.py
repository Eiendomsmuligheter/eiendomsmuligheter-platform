from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)

class DispensationService:
    def __init__(self):
        self.regulation_types = {
            'height': self._evaluate_height_dispensation,
            'coverage': self._evaluate_coverage_dispensation,
            'distance': self._evaluate_distance_dispensation,
            'parking': self._evaluate_parking_dispensation,
            'usage': self._evaluate_usage_dispensation
        }
        
    async def evaluate_dispensation_needs(
        self,
        property_data: Dict,
        development_plan: Dict,
        regulations: Dict
    ) -> Dict:
        """Evaluerer behov for dispensasjon basert på utviklingsplanen"""
        try:
            dispensations_needed = []
            
            # Sjekk hver reguleringstype
            for reg_type, evaluator in self.regulation_types.items():
                if reg_type in regulations:
                    evaluation = await evaluator(
                        property_data,
                        development_plan,
                        regulations[reg_type]
                    )
                    if evaluation['needed']:
                        dispensations_needed.append(evaluation)
                        
            # Generer samlet vurdering
            overall_assessment = self._generate_overall_assessment(
                dispensations_needed,
                property_data,
                regulations
            )
            
            return {
                'dispensations_needed': dispensations_needed,
                'overall_assessment': overall_assessment,
                'recommendation': self._generate_recommendation(
                    dispensations_needed,
                    overall_assessment
                )
            }
            
        except Exception as e:
            logger.error(f"Error in dispensation evaluation: {str(e)}")
            raise
            
    async def _evaluate_height_dispensation(
        self,
        property_data: Dict,
        development_plan: Dict,
        regulations: Dict
    ) -> Dict:
        """Evaluerer behov for høydedispensasjon"""
        max_allowed_height = regulations.get('max_height', 0)
        planned_height = development_plan.get('planned_height', 0)
        
        needed = planned_height > max_allowed_height
        deviation = planned_height - max_allowed_height if needed else 0
        
        return {
            'type': 'height',
            'needed': needed,
            'current': max_allowed_height,
            'planned': planned_height,
            'deviation': deviation,
            'justification': self._generate_height_justification(
                deviation,
                property_data,
                regulations
            ) if needed else None
        }
        
    async def _evaluate_coverage_dispensation(
        self,
        property_data: Dict,
        development_plan: Dict,
        regulations: Dict
    ) -> Dict:
        """Evaluerer behov for dispensasjon fra utnyttelsesgrad"""
        max_coverage = regulations.get('max_coverage', 0)
        planned_coverage = development_plan.get('planned_coverage', 0)
        
        needed = planned_coverage > max_coverage
        deviation = planned_coverage - max_coverage if needed else 0
        
        return {
            'type': 'coverage',
            'needed': needed,
            'current': max_coverage,
            'planned': planned_coverage,
            'deviation': deviation,
            'justification': self._generate_coverage_justification(
                deviation,
                property_data,
                regulations
            ) if needed else None
        }
        
    def _generate_height_justification(
        self,
        deviation: float,
        property_data: Dict,
        regulations: Dict
    ) -> str:
        """Genererer begrunnelse for høydedispensasjon"""
        justification = "Søknad om dispensasjon fra høydebestemmelser begrunnes med:\n\n"
        
        if deviation <= 1.0:
            justification += "1. Minimal overskridelse som ikke påvirker områdets karakter\n"
        
        if property_data.get('terrain_challenges', False):
            justification += "2. Krevende terrengforhold som nødvendiggjør tilpasning\n"
            
        if property_data.get('surrounding_buildings_height', 0) >= deviation:
            justification += "3. Tilpasning til omkringliggende bebyggelse\n"
            
        return justification
        
    def _generate_coverage_justification(
        self,
        deviation: float,
        property_data: Dict,
        regulations: Dict
    ) -> str:
        """Genererer begrunnelse for utnyttelsesgrad-dispensasjon"""
        justification = "Søknad om dispensasjon fra utnyttelsesgrad begrunnes med:\n\n"
        
        if deviation <= 5.0:
            justification += "1. Mindre avvik som ikke vesentlig endrer områdets karakter\n"
            
        if property_data.get('public_transport_proximity', False):
            justification += "2. Nærhet til kollektivtransport støtter fortetting\n"
            
        return justification
        
    def _generate_overall_assessment(
        self,
        dispensations: List[Dict],
        property_data: Dict,
        regulations: Dict
    ) -> Dict:
        """Genererer samlet vurdering av dispensasjonsbehov"""
        total_deviations = len(dispensations)
        
        if total_deviations == 0:
            return {
                'recommendation': 'no_dispensation_needed',
                'probability_of_approval': 1.0,
                'processing_time': 'standard',
                'risk_level': 'low'
            }
            
        # Beregn sannsynlighet for godkjennelse
        probability = self._calculate_approval_probability(
            dispensations,
            property_data,
            regulations
        )
        
        # Bestem risikornivå
        risk_level = 'high' if probability < 0.5 else 'medium' if probability < 0.8 else 'low'
        
        # Estimer behandlingstid
        processing_time = self._estimate_processing_time(
            dispensations,
            risk_level
        )
        
        return {
            'recommendation': 'proceed' if probability > 0.6 else 'reconsider',
            'probability_of_approval': probability,
            'processing_time': processing_time,
            'risk_level': risk_level
        }
        
    def _calculate_approval_probability(
        self,
        dispensations: List[Dict],
        property_data: Dict,
        regulations: Dict
    ) -> float:
        """Beregner sannsynlighet for godkjennelse av dispensasjon"""
        base_probability = 1.0
        
        for dispensation in dispensations:
            # Reduser sannsynlighet basert på avvikets størrelse
            deviation_factor = 1.0 - (
                dispensation.get('deviation', 0) / 100
            )
            base_probability *= deviation_factor
            
        # Juster for andre faktorer
        if property_data.get('historical_area', False):
            base_probability *= 0.8
            
        if property_data.get('public_transport_proximity', False):
            base_probability *= 1.2
            
        # Normaliser til 0-1
        return max(min(base_probability, 1.0), 0.0)
        
    def _estimate_processing_time(
        self,
        dispensations: List[Dict],
        risk_level: str
    ) -> str:
        """Estimerer behandlingstid for dispensasjonssøknad"""
        base_time = 6  # uker
        
        # Juster for antall dispensasjoner
        time_factor = len(dispensations) * 2
        
        # Juster for risikonivå
        risk_factors = {
            'low': 1,
            'medium': 1.5,
            'high': 2
        }
        
        total_time = base_time * time_factor * risk_factors[risk_level]
        
        if total_time <= 8:
            return 'standard'
        elif total_time <= 12:
            return 'extended'
        else:
            return 'long'
            
    def _generate_recommendation(
        self,
        dispensations: List[Dict],
        assessment: Dict
    ) -> Dict:
        """Genererer konkrete anbefalinger for dispensasjonssøknad"""
        if not dispensations:
            return {
                'action': 'proceed_without_dispensation',
                'description': 'Ingen dispensasjoner nødvendig',
                'next_steps': []
            }
            
        if assessment['probability_of_approval'] < 0.4:
            return {
                'action': 'reconsider_plan',
                'description': 'Høy risiko for avslag',
                'next_steps': [
                    'Revurder utviklingsplanen',
                    'Reduser avvik fra bestemmelsene',
                    'Vurder alternative løsninger'
                ]
            }
            
        return {
            'action': 'proceed_with_dispensation',
            'description': 'Dispensasjonssøknad anbefales',
            'next_steps': [
                'Utarbeid detaljert begrunnelse',
                'Innhent uttalelser fra naboer',
                'Forbered dokumentasjon',
                f"Forventet behandlingstid: {assessment['processing_time']}"
            ]
        }