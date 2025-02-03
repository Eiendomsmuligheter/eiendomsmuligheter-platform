import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

class PropertyOptimizationModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(PropertyOptimizationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)  # [utnyttelsesgrad, romdistribusjon, energieffektivitet, økonomisk_potensial]
        )
        
    def forward(self, x):
        return self.network(x)

class PropertyDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class OptimizationService:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 20  # Antall input features
        self.model = PropertyOptimizationModel(self.input_size).to(self.device)
        self.scaler = StandardScaler()
        
        if model_path:
            self.load_model(model_path)
            
    def preprocess_data(self, property_data: Dict) -> np.ndarray:
        """Konverterer eiendomsdata til modell-input"""
        features = []
        
        # Tomtestørrelse og eksisterende bygg
        features.extend([
            property_data.get('plot_size', 0),
            property_data.get('existing_building_size', 0),
            property_data.get('building_coverage_ratio', 0),
            property_data.get('floor_area_ratio', 0)
        ])
        
        # Reguleringsinformasjon
        features.extend([
            property_data.get('max_height', 0),
            property_data.get('min_distance_to_neighbor', 0),
            property_data.get('max_units', 0),
            property_data.get('parking_requirements', 0)
        ])
        
        # Terreng og orientering
        features.extend([
            property_data.get('slope_angle', 0),
            property_data.get('solar_exposure', 0),
            property_data.get('ground_quality', 0)
        ])
        
        # Økonomiske faktorer
        features.extend([
            property_data.get('market_value_per_sqm', 0),
            property_data.get('construction_cost_per_sqm', 0),
            property_data.get('rental_potential', 0)
        ])
        
        # Tekniske forhold
        features.extend([
            property_data.get('energy_rating', 0),
            property_data.get('technical_condition', 0),
            property_data.get('renovation_potential', 0)
        ])
        
        # Lokasjon og tilgjengelighet
        features.extend([
            property_data.get('distance_to_center', 0),
            property_data.get('public_transport_access', 0),
            property_data.get('neighborhood_rating', 0)
        ])
        
        return np.array(features).reshape(1, -1)
        
    async def optimize_property_usage(self, property_data: Dict) -> Dict:
        """Analyserer og optimaliserer eiendomsutvikling"""
        try:
            # Preprocess data
            features = self.preprocess_data(property_data)
            scaled_features = self.scaler.transform(features)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(scaled_features).to(self.device)
            
            # Get model predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(input_tensor)
                
            # Convert predictions to usable results
            results = self._interpret_predictions(predictions[0].cpu().numpy())
            
            # Generate detailed recommendations
            recommendations = self._generate_recommendations(results, property_data)
            
            return {
                'optimization_results': results,
                'recommendations': recommendations,
                'confidence_score': self._calculate_confidence(predictions[0])
            }
            
        except Exception as e:
            logger.error(f"Error in property optimization: {str(e)}")
            raise
            
    def _interpret_predictions(self, raw_predictions: np.ndarray) -> Dict:
        """Konverterer modellprediksjoner til meningsfylte resultater"""
        return {
            'optimal_utilization': float(raw_predictions[0]),  # 0-1 score
            'room_distribution': float(raw_predictions[1]),    # Optimal fordeling av rom
            'energy_efficiency': float(raw_predictions[2]),    # Energieffektivitetsscore
            'economic_potential': float(raw_predictions[3])    # Økonomisk potensial-score
        }
        
    def _generate_recommendations(self, results: Dict, property_data: Dict) -> List[Dict]:
        """Genererer spesifikke anbefalinger basert på optimeringsresultater"""
        recommendations = []
        
        # Utnyttelsesgrad anbefalinger
        if results['optimal_utilization'] > 0.7:
            recommendations.append({
                'category': 'utilization',
                'title': 'Høyt utviklingspotensial',
                'description': 'Eiendommen har betydelig potensial for utvikling',
                'actions': [
                    'Vurder utvidelse av eksisterende bygningsmasse',
                    'Undersøk muligheter for tilbygg',
                    'Vurder oppdeling i flere enheter'
                ]
            })
            
        # Romdistribusjon anbefalinger
        if results['room_distribution'] > 0.6:
            recommendations.append({
                'category': 'layout',
                'title': 'Optimalisering av planløsning',
                'description': 'Potensial for forbedret romutnyttelse',
                'actions': [
                    'Optimaliser rominndeling',
                    'Vurder åpen planløsning',
                    'Undersøk muligheter for flere bad/soverom'
                ]
            })
            
        # Energieffektivitet anbefalinger
        if results['energy_efficiency'] < 0.5:
            recommendations.append({
                'category': 'energy',
                'title': 'Energioptimalisering',
                'description': 'Betydelig potensial for energiforbedringer',
                'actions': [
                    'Vurder etterisolering',
                    'Undersøk muligheter for varmepumpe',
                    'Vurder solcelleinstallasjon'
                ]
            })
            
        return recommendations
        
    def _calculate_confidence(self, predictions: torch.Tensor) -> float:
        """Beregner konfidensgrad for prediksjonene"""
        # Implementer konfidensberegning basert på modellens sikkerhet
        confidence = torch.mean(torch.abs(predictions)).item()
        return min(max(confidence, 0.0), 1.0)
        
    def save_model(self, path: str):
        """Lagrer modell og scaler"""
        torch.save(self.model.state_dict(), f"{path}_model.pth")
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
    def load_model(self, path: str):
        """Laster modell og scaler"""
        self.model.load_state_dict(torch.load(f"{path}_model.pth"))
        self.scaler = joblib.load(f"{path}_scaler.joblib")
        self.model.eval()