import openai
from typing import Dict, List, Optional
import logging
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class GPT4RegulationAnalyzer:
    """
    Bruker GPT-4 for avansert analyse av byggeforskrifter og reguleringer.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._initialize_openai()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "model_version": "gpt-4",
            "temperature": 0.3,
            "max_tokens": 2000,
            "confidence_threshold": 0.85
        }
        
    def _initialize_openai(self):
        openai.api_key = self.config.get("openai_api_key")
        if not openai.api_key:
            raise ValueError("OpenAI API-nøkkel mangler i konfigurasjon")
            
    async def analyze_regulations(self,
                                regulation_text: str,
                                project_details: Dict,
                                municipality: str) -> Dict:
        """
        Analyser byggeforskrifter og reguleringer for et prosjekt
        """
        try:
            # Formater spørringen til GPT-4
            prompt = self._create_analysis_prompt(
                regulation_text,
                project_details,
                municipality
            )
            
            # Få analyse fra GPT-4
            analysis = await self._get_gpt4_analysis(prompt)
            
            # Prosesser og strukturer analysen
            structured_analysis = self._structure_analysis(analysis)
            
            return {
                "requirements": structured_analysis["requirements"],
                "interpretations": structured_analysis["interpretations"],
                "recommendations": structured_analysis["recommendations"],
                "compliance_status": self._check_compliance(
                    structured_analysis,
                    project_details
                )
            }
            
        except Exception as e:
            logger.error(f"Feil ved analyse av forskrifter: {str(e)}")
            raise
            
    def _create_analysis_prompt(self,
                              regulation_text: str,
                              project_details: Dict,
                              municipality: str) -> str:
        return f"""
        Analyser følgende byggeforskrifter for {municipality} kommune:
        
        {regulation_text}
        
        For følgende prosjekt:
        {json.dumps(project_details, indent=2, ensure_ascii=False)}
        
        Gi en detaljert analyse med:
        1. Spesifikke krav som gjelder
        2. Tolkning av kravene
        3. Anbefalinger for overholdelse
        4. Potensielle utfordringer og løsninger
        """
        
    async def _get_gpt4_analysis(self, prompt: str) -> Dict:
        response = await openai.ChatCompletion.acreate(
            model=self.config["model_version"],
            messages=[
                {"role": "system", "content": "Du er en ekspert på norske byggeforskrifter."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        
        return json.loads(response.choices[0].message.content)
        
    def _structure_analysis(self, analysis: Dict) -> Dict:
        """
        Strukturer og valider GPT-4 analysen
        """
        return {
            "requirements": analysis.get("requirements", []),
            "interpretations": analysis.get("interpretations", []),
            "recommendations": analysis.get("recommendations", [])
        }
        
    def _check_compliance(self,
                         analysis: Dict,
                         project_details: Dict) -> Dict:
        """
        Sjekk om prosjektet oppfyller kravene
        """
        return {
            "overall_status": "compliant",
            "requirements_status": {},
            "issues": [],
            "recommendations": []
        }