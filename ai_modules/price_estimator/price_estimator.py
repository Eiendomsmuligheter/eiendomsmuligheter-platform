import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

class PriceEstimator:
    """
    Avansert prisestimator som kombinerer ML og deep learning
    for nøyaktig verdivurdering av eiendommer.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.ml_model = self._initialize_ml_model()
        self.deep_model = self._initialize_deep_model()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "ml_model_type": "gradient_boosting",
            "use_gpu": torch.cuda.is_available(),
            "confidence_threshold": 0.85
        }
        
    def _initialize_ml_model(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
    def _initialize_deep_model(self) -> nn.Module:
        class PriceNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(50, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
                
            def forward(self, x):
                return self.network(x)
                
        model = PriceNet()
        if self.config["use_gpu"] and torch.cuda.is_available():
            model = model.cuda()
        return model
        
    async def estimate_price(self,
                           property_data: Dict,
                           market_conditions: Dict) -> Dict:
        """
        Estimer eiendomspris basert på egenskaper og markedsforhold
        """
        try:
            features = self._prepare_features(property_data, market_conditions)
            ml_estimate = self._get_ml_estimate(features)
            deep_estimate = self._get_deep_estimate(features)
            
            final_estimate = self._combine_estimates(ml_estimate, deep_estimate)
            
            return {
                "estimated_price": final_estimate,
                "confidence_score": self._calculate_confidence(features),
                "market_factors": self._analyze_market_factors(market_conditions)
            }
            
        except Exception as e:
            logger.error(f"Feil ved prisestimering: {str(e)}")
            raise