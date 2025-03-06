import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import json
import os
import requests
import asyncio
from datetime import datetime
import joblib
import re

# Sett opp logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyDataset(Dataset):
    """Custom PyTorch dataset for property data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        else:
            self.targets = None
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class PriceNet(nn.Module):
    """Neural network for property price estimation with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers
        hidden_layers = []
        for i in range(len(hidden_dims)-1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            hidden_layers.append(nn.Dropout(0.3))
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Softmax(dim=1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Uncertainty estimation head
        self.uncertainty_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = nn.functional.relu(x)
        
        features = self.hidden_layers(x)
        
        # Apply attention to get feature importance
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights
        
        # Main output (price prediction)
        price = self.output_layer(weighted_features)
        
        # Uncertainty estimation (log variance)
        log_var = self.uncertainty_layer(features)
        
        return price, log_var, attention_weights

class MarketFactorsAnalyzer:
    """Analyzes market factors that influence property prices"""
    
    def __init__(self):
        self.interest_rate_impact = {
            "low": 1.05,      # Low interest rates boost prices
            "medium": 1.0,    # Neutral impact
            "high": 0.95      # High interest rates reduce prices
        }
        
        self.supply_demand_impact = {
            "low_supply_high_demand": 1.10,  # Seller's market
            "balanced": 1.0,                 # Balanced market
            "high_supply_low_demand": 0.92   # Buyer's market
        }
        
        self.seasonal_factors = {
            # Monthly adjustment factors
            1: 0.97,   # January
            2: 0.98,   # February
            3: 1.02,   # March
            4: 1.05,   # April
            5: 1.07,   # May
            6: 1.05,   # June
            7: 0.90,   # July
            8: 0.95,   # August
            9: 1.03,   # September
            10: 1.01,  # October
            11: 0.98,  # November
            12: 0.94   # December
        }
        
    def analyze(self, market_conditions: Dict) -> Dict:
        """Analyze market factors and their impact on property price"""
        current_month = datetime.now().month
        
        # Get interest rate impact
        interest_rate_level = market_conditions.get("interest_rate_level", "medium")
        interest_rate_factor = self.interest_rate_impact.get(interest_rate_level, 1.0)
        
        # Get supply/demand impact
        supply_demand = market_conditions.get("supply_demand_balance", "balanced")
        supply_demand_factor = self.supply_demand_impact.get(supply_demand, 1.0)
        
        # Get seasonal factor
        seasonal_factor = self.seasonal_factors.get(current_month, 1.0)
        
        # Calculate total market adjustment factor
        market_adjustment = interest_rate_factor * supply_demand_factor * seasonal_factor
        
        return {
            "interest_rate": {
                "level": interest_rate_level,
                "impact_factor": interest_rate_factor
            },
            "supply_demand": {
                "balance": supply_demand,
                "impact_factor": supply_demand_factor
            },
            "seasonal": {
                "month": current_month,
                "impact_factor": seasonal_factor
            },
            "total_market_adjustment": market_adjustment
        }

class LocationAnalyzer:
    """Analyzes location factors that influence property prices"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
        # Predefined location factors (would be more comprehensive in production)
        self.city_factors = {
            "oslo": 1.8,
            "bergen": 1.3,
            "trondheim": 1.25,
            "stavanger": 1.2,
            "drammen": 1.15,
            "tromsø": 1.2,
            "kristiansand": 1.1
        }
        
        self.area_type_factors = {
            "city_center": 1.3,
            "urban": 1.1,
            "suburban": 1.0,
            "rural": 0.8
        }
        
    async def analyze(self, location_data: Dict) -> Dict:
        """Analyze location factors and their impact on property price"""
        city = location_data.get("city", "").lower()
        area_type = location_data.get("area_type", "suburban").lower()
        
        # Get city factor
        city_factor = self.city_factors.get(city, 1.0)
        
        # Get area type factor
        area_factor = self.area_type_factors.get(area_type, 1.0)
        
        # Get additional amenities factor
        amenities_factor = self._calculate_amenities_factor(location_data)
        
        # Calculate total location adjustment factor
        location_adjustment = city_factor * area_factor * amenities_factor
        
        return {
            "city": {
                "name": city,
                "impact_factor": city_factor
            },
            "area_type": {
                "type": area_type,
                "impact_factor": area_factor
            },
            "amenities": {
                "score": amenities_factor,
                "details": self._get_amenities_details(location_data)
            },
            "total_location_adjustment": location_adjustment
        }
        
    def _calculate_amenities_factor(self, location_data: Dict) -> float:
        """Calculate factor based on nearby amenities"""
        base_factor = 1.0
        
        # Add adjustments for each amenity
        amenities = location_data.get("amenities", {})
        
        if amenities.get("public_transport", False):
            base_factor += 0.05
            
        if amenities.get("schools", False):
            base_factor += 0.03
            
        if amenities.get("shopping", False):
            base_factor += 0.02
            
        if amenities.get("parks", False):
            base_factor += 0.02
            
        if amenities.get("healthcare", False):
            base_factor += 0.01
            
        return base_factor
        
    def _get_amenities_details(self, location_data: Dict) -> List[Dict]:
        """Get details about nearby amenities"""
        amenities = location_data.get("amenities", {})
        result = []
        
        for amenity, present in amenities.items():
            if present:
                result.append({
                    "type": amenity,
                    "distance": amenities.get(f"{amenity}_distance", "unknown")
                })
                
        return result

class RentalIncomeEstimator:
    """Estimates potential rental income for a property"""
    
    def __init__(self):
        # Base rental rates per square meter per month by area type
        self.base_rates = {
            "city_center": 250,  # NOK per m²
            "urban": 200,
            "suburban": 170,
            "rural": 130
        }
        
        # Adjustment factors for property qualities
        self.quality_adjustments = {
            "excellent": 1.15,
            "good": 1.05,
            "fair": 1.0,
            "poor": 0.85
        }
        
        # Adjustment for property type
        self.type_adjustments = {
            "apartment": 1.1,
            "house": 1.0,
            "townhouse": 1.05,
            "basement_unit": 0.85
        }
        
    def estimate(self, property_data: Dict, location_data: Dict) -> Dict:
        """Estimate rental income for the property"""
        # Get area type
        area_type = location_data.get("area_type", "suburban")
        
        # Get base rate
        base_rate = self.base_rates.get(area_type, self.base_rates["suburban"])
        
        # Get property area
        area = property_data.get("area_m2", 0)
        
        # Get property quality
        quality = property_data.get("condition", "fair")
        quality_factor = self.quality_adjustments.get(quality, 1.0)
        
        # Get property type
        prop_type = property_data.get("type", "house")
        type_factor = self.type_adjustments.get(prop_type, 1.0)
        
        # Calculate monthly rental
        monthly_rental = area * base_rate * quality_factor * type_factor / 100
        
        # Calculate annual rental income (account for vacancy)
        vacancy_rate = 0.05  # 5% vacancy rate
        annual_rental = monthly_rental * 12 * (1 - vacancy_rate)
        
        # Calculate ROI percentage for rental property
        property_value = property_data.get("estimated_value", 0)
        roi_percentage = (annual_rental / property_value * 100) if property_value > 0 else 0
        
        return {
            "monthly_rental_estimate": monthly_rental,
            "annual_rental_estimate": annual_rental,
            "vacancy_rate": vacancy_rate,
            "roi_percentage": roi_percentage,
            "adjustments": {
                "area_type": area_type,
                "quality": quality,
                "property_type": prop_type
            }
        }

class RenovationPotentialAnalyzer:
    """Analyzes renovation potential and ROI for improvements"""
    
    def __init__(self):
        # Cost estimates for different renovation types (NOK per m²)
        self.renovation_costs = {
            "cosmetic": 2000,
            "moderate": 5000,
            "major": 15000,
            "kitchen": 8000,  # per kitchen
            "bathroom": 10000,  # per bathroom
            "basement_conversion": 12000,
            "attic_conversion": 15000
        }
        
        # Value increase factors for different renovation types
        self.value_increase_factors = {
            "cosmetic": 1.5,  # 50% ROI
            "moderate": 1.3,
            "major": 1.2,
            "kitchen": 1.7,
            "bathroom": 1.6,
            "basement_conversion": 1.8,
            "attic_conversion": 1.7
        }
        
    def analyze(self, property_data: Dict, opportunities: List[Dict]) -> Dict:
        """Analyze renovation potential and ROI"""
        renovation_options = []
        property_area = property_data.get("area_m2", 100)
        
        for opportunity in opportunities:
            renovation_type = opportunity.get("type", "")
            affected_area = opportunity.get("area_m2", property_area * 0.2)  # Default to 20% of property
            
            if renovation_type in self.renovation_costs:
                # Calculate cost
                unit_cost = self.renovation_costs[renovation_type]
                
                # Adjust cost based on renovation type
                if renovation_type in ["kitchen", "bathroom"]:
                    total_cost = unit_cost * opportunity.get("count", 1)
                else:
                    total_cost = unit_cost * affected_area
                
                # Calculate value increase
                value_factor = self.value_increase_factors.get(renovation_type, 1.0)
                value_increase = total_cost * value_factor
                
                # Calculate ROI
                roi = (value_increase / total_cost) - 1
                
                renovation_options.append({
                    "type": renovation_type,
                    "description": opportunity.get("description", ""),
                    "estimated_cost": total_cost,
                    "estimated_value_increase": value_increase,
                    "roi_percentage": roi * 100,
                    "payback_period_years": 1 / roi if roi > 0 else float('inf')
                })
        
        # Sort by ROI
        renovation_options.sort(key=lambda x: x["roi_percentage"], reverse=True)
        
        # Find the best overall recommendation
        best_recommendation = renovation_options[0] if renovation_options else None
        
        return {
            "renovation_options": renovation_options,
            "best_recommendation": best_recommendation,
            "total_potential_value_increase": sum(option["estimated_value_increase"] for option in renovation_options)
        }

class ProfitabilityCalculator:
    """Calculates profitability metrics for property investment"""
    
    def __init__(self):
        # Default financial parameters
        self.default_loan_rate = 0.035  # 3.5% annual interest
        self.default_loan_term = 25     # 25 years
        self.default_down_payment = 0.15  # 15% down payment
        
    def calculate(self, property_data: Dict, rental_data: Dict, 
                 financing_data: Optional[Dict] = None) -> Dict:
        """Calculate profitability metrics for the property"""
        # Get property value
        property_value = property_data.get("estimated_value", 0)
        
        # Get rental income
        annual_rental = rental_data.get("annual_rental_estimate", 0)
        
        # Get financing details (or use defaults)
        if financing_data:
            loan_rate = financing_data.get("loan_rate", self.default_loan_rate)
            loan_term = financing_data.get("loan_term", self.default_loan_term)
            down_payment_pct = financing_data.get("down_payment_percentage", self.default_down_payment)
        else:
            loan_rate = self.default_loan_rate
            loan_term = self.default_loan_term
            down_payment_pct = self.default_down_payment
        
        # Calculate loan amount
        down_payment = property_value * down_payment_pct
        loan_amount = property_value - down_payment
        
        # Calculate monthly mortgage payment
        r = loan_rate / 12  # Monthly interest rate
        n = loan_term * 12  # Total number of payments
        monthly_mortgage = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        
        # Calculate annual expenses
        property_tax = property_value * 0.003  # 0.3% property tax
        insurance = property_value * 0.005     # 0.5% insurance
        maintenance = property_value * 0.01    # 1% maintenance
        
        total_annual_expenses = (monthly_mortgage * 12) + property_tax + insurance + maintenance
        
        # Calculate cash flow and profitability metrics
        annual_cash_flow = annual_rental - total_annual_expenses
        monthly_cash_flow = annual_cash_flow / 12
        
        # Calculate cash-on-cash return
        cash_invested = down_payment + (property_value * 0.04)  # Down payment + 4% closing costs
        cash_on_cash_return = annual_cash_flow / cash_invested
        
        # Calculate cap rate
        net_operating_income = annual_rental - (property_tax + insurance + maintenance)
        cap_rate = net_operating_income / property_value
        
        return {
            "monthly_mortgage": monthly_mortgage,
            "monthly_cash_flow": monthly_cash_flow,
            "annual_cash_flow": annual_cash_flow,
            "cash_on_cash_return": cash_on_cash_return,
            "cap_rate": cap_rate,
            "expenses": {
                "mortgage": monthly_mortgage * 12,
                "property_tax": property_tax,
                "insurance": insurance,
                "maintenance": maintenance
            },
            "financing": {
                "purchase_price": property_value,
                "down_payment": down_payment,
                "loan_amount": loan_amount,
                "loan_rate": loan_rate,
                "loan_term": loan_term
            }
        }

class PriceEstimator:
    """
    Avansert prisestimator som kombinerer ML og deep learning
    for nøyaktig verdivurdering av eiendommer og utleiepotensial.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.preprocessor = self._initialize_preprocessor()
        self.ml_model = self._initialize_ml_model()
        self.deep_model = self._initialize_deep_model()
        self.market_analyzer = MarketFactorsAnalyzer()
        self.location_analyzer = LocationAnalyzer(api_key=self.config.get("api_key"))
        self.rental_estimator = RentalIncomeEstimator()
        self.renovation_analyzer = RenovationPotentialAnalyzer()
        self.profitability_calculator = ProfitabilityCalculator()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Kunne ikke laste konfigurasjon: {str(e)}")
        
        # Default configuration
        return {
            "ml_model_type": "gradient_boosting",
            "deep_model": {
                "hidden_dims": [256, 128, 64],
                "learning_rate": 0.001,
                "batch_size": 64
            },
            "use_gpu": torch.cuda.is_available(),
            "confidence_threshold": 0.85,
            "price_range_percentage": 0.1,  # +/- 10% price range
            "version": "1.0.0",
            "model_path": "models/",
            "api_key": None
        }
        
    def _initialize_preprocessor(self) -> ColumnTransformer:
        """Initialize the data preprocessor pipeline"""
        # Define categorical and numerical features
        categorical_features = ["property_type", "area_type", "condition"]
        numerical_features = ["area_m2", "bedrooms", "bathrooms", "year_built", "floor"]
        
        # Create preprocessing pipelines for each feature type
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessing steps into a single transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
        
    def _initialize_ml_model(self) -> Union[GradientBoostingRegressor, RandomForestRegressor]:
        """Initialize the ML model based on configuration"""
        model_type = self.config.get("ml_model_type", "gradient_boosting").lower()
        
        if model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        else:  # Default to gradient boosting
            return GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
    
    def _initialize_deep_model(self) -> PriceNet:
        """Initialize the deep learning model"""
        deep_config = self.config.get("deep_model", {})
        hidden_dims = deep_config.get("hidden_dims", [256, 128, 64])
        
        # Assume input dimension is 50 (will be updated during training)
        model = PriceNet(input_dim=50, hidden_dims=hidden_dims)
        
        if self.config.get("use_gpu", False) and torch.cuda.is_available():
            model = model.cuda()
            
        return model
    
    def save_models(self, path: Optional[str] = None):
        """Save trained models to disk"""
        if path is None:
            path = self.config.get("model_path", "models/")
            
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save ML model
        joblib.dump(self.ml_model, os.path.join(path, "ml_model.joblib"))
        
        # Save preprocessing pipeline
        joblib.dump(self.preprocessor, os.path.join(path, "preprocessor.joblib"))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))
        
        # Save deep model
        torch.save(self.deep_model.state_dict(), os.path.join(path, "deep_model.pt"))
        
        logger.info(f"Models saved to {path}")
        
    def load_models(self, path: Optional[str] = None):
        """Load trained models from disk"""
        if path is None:
            path = self.config.get("model_path", "models/")
            
        try:
            # Load ML model
            ml_model_path = os.path.join(path, "ml_model.joblib")
            if os.path.exists(ml_model_path):
                self.ml_model = joblib.load(ml_model_path)
                
            # Load preprocessing pipeline
            preprocessor_path = os.path.join(path, "preprocessor.joblib")
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                
            # Load scaler
            scaler_path = os.path.join(path, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            # Load deep model
            deep_model_path = os.path.join(path, "deep_model.pt")
            if os.path.exists(deep_model_path):
                # Get input dimension from the saved model
                state_dict = torch.load(deep_model_path)
                input_dim = state_dict['input_layer.weight'].shape[1]
                
                # Reinitialize with correct dimensions
                deep_config = self.config.get("deep_model", {})
                hidden_dims = deep_config.get("hidden_dims", [256, 128, 64])
                self.deep_model = PriceNet(input_dim=input_dim, hidden_dims=hidden_dims)
                
                # Load weights
                self.deep_model.load_state_dict(state_dict)
                
                # Move to GPU if available
                if self.config.get("use_gpu", False) and torch.cuda.is_available():
                    self.deep_model = self.deep_model.cuda()
                    
            logger.info(f"Models loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def _prepare_features(self, property_data: Dict, location_data: Dict) -> np.ndarray:
        """Prepare features for model input"""
        # Extract and combine features
        features = {}
        
        # Property features
        for key in ["area_m2", "bedrooms", "bathrooms", "year_built", "floor"]:
            features[key] = property_data.get(key, 0)
            
        # Property categorical features
        features["property_type"] = property_data.get("type", "house")
        features["condition"] = property_data.get("condition", "fair")
        
        # Location features
        features["area_type"] = location_data.get("area_type", "suburban")
        
        # Create DataFrame for preprocessing
        features_df = pd.DataFrame([features])
        
        # Apply preprocessing
        try:
            processed_features = self.preprocessor.transform(features_df)
            return processed_features
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            # Fallback: return standardized numerical features only
            numerical_features = np.array([[
                features["area_m2"],
                features["bedrooms"],
                features["bathrooms"],
                features["year_built"],
                features["floor"]
            ]])
            return self.scaler.fit_transform(numerical_features)
    
    def _get_ml_estimate(self, features: np.ndarray) -> float:
        """Get price estimate from ML model"""
        try:
            return float(self.ml_model.predict(features)[0])
        except Exception as e:
            logger.error(f"Error from ML model: {str(e)}")
            # Fallback to a simple heuristic
            return self._get_fallback_estimate(features)
    
    def _get_deep_estimate(self, features: np.ndarray) -> Tuple[float, float]:
        """Get price estimate and uncertainty from deep learning model"""
        try:
            # Convert to PyTorch tensor
            if self.config.get("use_gpu", False) and torch.cuda.is_available():
                tensor = torch.tensor(features, dtype=torch.float32).cuda()
            else:
                tensor = torch.tensor(features, dtype=torch.float32)
                
            # Get prediction
            self.deep_model.eval()
            with torch.no_grad():
                price, log_var, _ = self.deep_model(tensor)
                
            # Convert to numpy
            price_value = price.cpu().numpy()[0, 0]
            uncertainty = np.exp(log_var.cpu().numpy()[0, 0])
            
            return float(price_value), float(uncertainty)
            
        except Exception as e:
            logger.error(f"Error from deep model: {str(e)}")
            # Fallback
            return self._get_fallback_estimate(features), 1000000
    
    def _get_fallback_estimate(self, features: np.ndarray) -> float:
        """Simple heuristic for property price estimation"""
        # Extract area from features (assumes area is the first feature after scaling)
        # This is a simplified fallback that just uses area as the primary predictor
        scaled_area = features[0, 0]
        
        # Reverting standardization (roughly)
        area = scaled_area * 50 + 100  # Assuming mean=100, std=50
        
        # Simple price model: 50,000 NOK per m²
        return area * 50000
    
    def _combine_estimates(self, ml_estimate: float, deep_result: Tuple[float, float]) -> float:
        """Combine estimates from different models"""
        deep_estimate, uncertainty = deep_result
        
        # Calculate weights based on uncertainty
        # Lower uncertainty = higher weight
        deep_weight = 1.0 / (1.0 + uncertainty / 1000000)
        ml_weight = 1.0 - deep_weight
        
        # Weighted average
        combined = (deep_estimate * deep_weight) + (ml_estimate * ml_weight)
        
        return combined
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for the estimate"""
        # In a real implementation, this would be more sophisticated
        # Here, we'll just use a placeholder
        base_confidence = 0.8
        
        # Confidence decreases with unusual properties
        confidence_penalty = 0.1 if np.any(np.abs(features) > 2) else 0
        
        return max(0, min(1, base_confidence - confidence_penalty))
    
    def _calculate_price_range(self, estimate: float, confidence: float) -> Dict:
        """Calculate price range based on estimate and confidence"""
        # Adjust range based on confidence (lower confidence = wider range)
        range_percentage = self.config.get("price_range_percentage", 0.1)
        adjusted_range = range_percentage + (1 - confidence) * 0.15
        
        # Calculate min and max
        min_price = estimate * (1 - adjusted_range)
        max_price = estimate * (1 + adjusted_range)
        
        return {
            "min": min_price,
            "max": max_price,
            "range_percentage": adjusted_range * 100
        }
    
    async def estimate_price(self,
                           property_data: Dict,
                           location_data: Dict,
                           market_conditions: Optional[Dict] = None) -> Dict:
        """
        Estimer eiendomspris basert på egenskaper, beliggenhet og markedsforhold
        """
        try:
            # Default market conditions if not provided
            if market_conditions is None:
                market_conditions = {
                    "interest_rate_level": "medium",
                    "supply_demand_balance": "balanced"
                }
                
            # Prepare features
            features = self._prepare_features(property_data, location_data)
            
            # Get estimates from models
            ml_estimate = self._get_ml_estimate(features)
            deep_result = self._get_deep_estimate(features)
            
            # Combine estimates
            raw_estimate = self._combine_estimates(ml_estimate, deep_result)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(features)
            
            # Analyze market factors
            market_factors = self.market_analyzer.analyze(market_conditions)
            
            # Analyze location factors
            location_factors = await self.location_analyzer.analyze(location_data)
            
            # Apply adjustments
            market_adjustment = market_factors.get("total_market_adjustment", 1.0)
            location_adjustment = location_factors.get("total_location_adjustment", 1.0)
            
            # Calculate final estimate
            final_estimate = raw_estimate * market_adjustment * location_adjustment
            
            # Calculate price range
            price_range = self._calculate_price_range(final_estimate, confidence)
            
            # Add final estimate to property data for further calculations
            property_data["estimated_value"] = final_estimate
            
            # Estimate rental income potential
            rental_data = self.rental_estimator.estimate(property_data, location_data)
            
            # Analyze renovation potential
            renovation_data = self.renovation_analyzer.analyze(
                property_data, 
                property_data.get("improvement_opportunities", [])
            )
            
            # Calculate profitability metrics
            profitability = self.profitability_calculator.calculate(
                property_data,
                rental_data
            )
            
            # Return comprehensive results
            return {
                "property_valuation": {
                    "estimated_price": final_estimate,
                    "price_range": price_range,
                    "confidence_score": confidence,
                    "price_per_sqm": final_estimate / property_data.get("area_m2", 1)
                },
                "market_factors": market_factors,
                "location_factors": location_factors,
                "rental_potential": rental_data,
                "renovation_potential": renovation_data,
                "investment_analysis": profitability,
                "monetization_strategies": self._generate_monetization_strategies(
                    property_data, 
                    rental_data,
                    renovation_data,
                    profitability
                )
            }
            
        except Exception as e:
            logger.error(f"Feil ved prisestimering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _generate_monetization_strategies(self,
                                        property_data: Dict,
                                        rental_data: Dict,
                                        renovation_data: Dict,
                                        profitability: Dict) -> List[Dict]:
        """Generate monetization strategies based on property analysis"""
        strategies = []
        property_value = property_data.get("estimated_value", 0)
        
        # Strategy 1: Rental income
        monthly_rental = rental_data.get("monthly_rental_estimate", 0)
        roi_percentage = rental_data.get("roi_percentage", 0)
        
        if monthly_rental > 0:
            strategies.append({
                "type": "rental",
                "title": "Langsiktig utleie",
                "monthly_income": monthly_rental,
                "annual_income": monthly_rental * 12,
                "roi_percentage": roi_percentage,
                "pros": [
                    "Stabil månedlig inntekt",
                    "Relativ lav arbeidsinnsats ved veletablert leieforhold",
                    "Potensial for langsiktig verdiøkning"
                ],
                "cons": [
                    "Risiko for leietakervansker",
                    "Vedlikeholdskostnader",
                    "Perioder med potensielt ledig utleie"
                ]
            })
            
        # Strategy 2: Short-term rental
        if location_data := property_data.get("location", {}):
            is_tourist_area = location_data.get("is_tourist_area", False)
            area_type = location_data.get("area_type", "")
            
            if is_tourist_area or area_type in ["city_center", "urban"]:
                short_term_multiplier = 1.8  # Short-term rentals can yield 1.8x long-term
                short_term_monthly = monthly_rental * short_term_multiplier
                short_term_occupancy = 0.7  # 70% occupancy rate
                
                strategies.append({
                    "type": "short_term_rental",
                    "title": "Korttidsutleie (Airbnb/FINN Reise)",
                    "monthly_income": short_term_monthly * short_term_occupancy,
                    "annual_income": short_term_monthly * short_term_occupancy * 12,
                    "roi_percentage": roi_percentage * short_term_multiplier * short_term_occupancy,
                    "occupancy_rate": short_term_occupancy * 100,
                    "pros": [
                        "Høyere inntektspotensial enn langsiktig utleie",
                        "Fleksibilitet i bruk av eiendommen",
                        "Mulighet for sesongbasert prising"
                    ],
                    "cons": [
                        "Høyere arbeidsbelastning og administrasjon",
                        "Varierende belegg",
                        "Strengere regulatoriske krav i enkelte områder"
                    ]
                })
        
        # Strategy 3: Renovate and sell
        best_renovation = None
        if "renovation_options" in renovation_data and renovation_data["renovation_options"]:
            best_renovation = max(
                renovation_data["renovation_options"], 
                key=lambda x: x.get("roi_percentage", 0)
            )
            
        if best_renovation and best_renovation.get("roi_percentage", 0) > 20:  # Only if ROI > 20%
            strategies.append({
                "type": "renovate_sell",
                "title": "Oppgradere og selge",
                "renovation_cost": best_renovation.get("estimated_cost", 0),
                "potential_profit": best_renovation.get("estimated_value_increase", 0) - best_renovation.get("estimated_cost", 0),
                "roi_percentage": best_renovation.get("roi_percentage", 0),
                "timeframe": "3-6 måneder",
                "pros": [
                    "Rask avkastning på investering",
                    "Konkret tidsramme for prosjektet",
                    "Mindre langsiktig risiko"
                ],
                "cons": [
                    "Risiko for uforutsette kostnader",
                    "Krever ekspertise eller profesjonell bistand",
                    "Avhengig av markedsforhold ved salg"
                ]
            })
            
        # Strategy 4: House hacking (live in one part, rent out another)
        if property_data.get("area_m2", 0) > 100 or property_data.get("divisible", False):
            house_hack_rental = monthly_rental * 0.6  # Assume 60% of property can be rented
            mortgage = profitability.get("monthly_mortgage", 0)
            
            strategies.append({
                "type": "house_hack",
                "title": "Bo i en del, lei ut en del",
                "monthly_income": house_hack_rental,
                "mortgage_coverage_percentage": (house_hack_rental / mortgage * 100) if mortgage > 0 else 0,
                "pros": [
                    "Reduserer egen bokostnad betydelig",
                    "Enklere å administrere enn fullt utleie",
                    "Mulighet for skattefordeler"
                ],
                "cons": [
                    "Mindre privatliv",
                    "Potensielle konflikter med leietakere",
                    "Begrenset til visse typer eiendommer"
                ]
            })
            
        # Strategy 5: Property development (if renovation potential is high)
        if renovation_data.get("total_potential_value_increase", 0) > property_value * 0.3:
            strategies.append({
                "type": "development",
                "title": "Eiendomsutvikling",
                "development_cost": renovation_data.get("total_potential_value_increase", 0) * 0.7,
                "potential_profit": renovation_data.get("total_potential_value_increase", 0) * 0.3,
                "timeframe": "6-18 måneder",
                "pros": [
                    "Høyest potensial for verdiskapning",
                    "Mulighet for skreddersydd utvikling",
                    "Potensielt betydelig fortjeneste"
                ],
                "cons": [
                    "Høyest risiko og kompleksitet",
                    "Krever mer kapital",
                    "Lengre tidshorisont"
                ]
            })
            
        # Sort strategies by ROI/profitability
        def strategy_value(strategy):
            if "roi_percentage" in strategy:
                return strategy["roi_percentage"]
            elif "potential_profit" in strategy:
                return strategy["potential_profit"] / strategy.get("development_cost", 1) * 100
            else:
                return 0
                
        strategies.sort(key=strategy_value, reverse=True)
        
        return strategies
    
    async def train(self, training_data: pd.DataFrame, target_column: str):
        """Train the price estimation models"""
        try:
            # Split features and target
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]
            
            # Fit the preprocessor
            self.preprocessor.fit(X)
            
            # Transform features
            X_processed = self.preprocessor.transform(X)
            
            # Train ML model
            self.ml_model.fit(X_processed, y)
            
            # Scale target for deep learning model
            y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Train deep learning model
            await self._train_deep_model(X_processed, y_scaled)
            
            # Evaluate models
            ml_score = cross_val_score(self.ml_model, X_processed, y, cv=5).mean()
            
            logger.info(f"ML model trained successfully. CV score: {ml_score:.4f}")
            return {
                "success": True,
                "ml_cv_score": ml_score
            }
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _train_deep_model(self, X: np.ndarray, y: np.ndarray):
        """Train the deep learning model"""
        # Update input dimension based on processed features
        input_dim = X.shape[1]
        deep_config = self.config.get("deep_model", {})
        hidden_dims = deep_config.get("hidden_dims", [256, 128, 64])
        
        # Reinitialize with correct dimensions
        self.deep_model = PriceNet(input_dim=input_dim, hidden_dims=hidden_dims)
        
        # Move to GPU if available
        if self.config.get("use_gpu", False) and torch.cuda.is_available():
            self.deep_model = self.deep_model.cuda()
        
        # Create dataset and dataloader
        dataset = PropertyDataset(X, y)
        batch_size = deep_config.get("batch_size", 64)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create optimizer
        lr = deep_config.get("learning_rate", 0.001)
        optimizer = optim.Adam(self.deep_model.parameters(), lr=lr)
        
        # Define loss function (negative log likelihood for uncertainty)
        def gaussian_nll_loss(pred, target, log_var):
            precision = torch.exp(-log_var)
            return torch.mean(0.5 * precision * (pred - target) ** 2 + 0.5 * log_var)
        
        # Training loop
        num_epochs = deep_config.get("epochs", 100)
        self.deep_model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Move to GPU if available
                if self.config.get("use_gpu", False) and torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda()
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                pred, log_var, _ = self.deep_model(batch_X)
                
                # Calculate loss
                loss = gaussian_nll_loss(pred, batch_y, log_var)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_X.size(0)
                
            epoch_loss = running_loss / len(dataset)
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        logger.info(f"Deep learning model trained successfully")
