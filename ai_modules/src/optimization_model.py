import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os
import json
import asyncio
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyOptimizationConfig:
    """Konfigurasjon for eiendomsoptimeringsmodellen"""
    def __init__(self, **kwargs):
        # Modellparametre
        self.input_size = kwargs.get('input_size', 32)  # Utvidet med flere features
        self.hidden_sizes = kwargs.get('hidden_sizes', [256, 256, 128, 64])
        self.dropout_rate = kwargs.get('dropout_rate', 0.4)
        self.activation = kwargs.get('activation', 'leaky_relu')
        self.output_size = kwargs.get('output_size', 7)  # Utvidet med flere målverdier
        
        # Treningsparametre
        self.batch_size = kwargs.get('batch_size', 64)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.num_epochs = kwargs.get('num_epochs', 150)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 15)
        
        # Ensemble-oppsett
        self.use_ensemble = kwargs.get('use_ensemble', True)
        self.ensemble_models = kwargs.get('ensemble_models', ['nn', 'gbm', 'rf'])
        self.ensemble_weights = kwargs.get('ensemble_weights', {'nn': 0.5, 'gbm': 0.3, 'rf': 0.2})
        
        # Modellversjonering
        self.model_version = kwargs.get('model_version', '2.0.0')
        
        # Adaptiv læring
        self.use_lr_scheduler = kwargs.get('use_lr_scheduler', True)
        self.use_mixed_precision = kwargs.get('use_mixed_precision', True)
        
        # Utvinning av features
        self.use_feature_extraction = kwargs.get('use_feature_extraction', True)
        self.use_feature_selection = kwargs.get('use_feature_selection', True)
        
        # Regularisering
        self.use_l1_reg = kwargs.get('use_l1_reg', False)
        self.l1_lambda = kwargs.get('l1_lambda', 0.01)
        
        # Caching
        self.cache_predictions = kwargs.get('cache_predictions', True)
        self.cache_timeout = kwargs.get('cache_timeout', 3600)  # 1 time
    
    @classmethod
    def from_json(cls, json_path: str) -> 'PropertyOptimizationConfig':
        """Laster konfigurasjon fra JSON-fil"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """Lagrer konfigurasjon til JSON-fil"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

class PropertyOptimizationModel(nn.Module):
    def __init__(self, config: PropertyOptimizationConfig):
        super(PropertyOptimizationModel, self).__init__()
        self.config = config
        
        # Dynamisk oppbygging av nettverket basert på konfigurasjon
        layers = []
        in_features = config.input_size
        
        for i, hidden_size in enumerate(config.hidden_sizes):
            layers.append(nn.Linear(in_features, hidden_size))
            
            # Velg aktivering basert på konfigurasjon
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif config.activation == 'selu':
                layers.append(nn.SELU())
            elif config.activation == 'swish':
                layers.append(nn.SiLU())  # Swish aktivering
            
            layers.append(nn.Dropout(config.dropout_rate))
            in_features = hidden_size
        
        # Output-lag med tilpasset størrelse
        layers.append(nn.Linear(in_features, config.output_size))
        
        # Lag sekventiell modell
        self.network = nn.Sequential(*layers)
        
        # Batch normalisering for alle input features
        self.batch_norm = nn.BatchNorm1d(config.input_size)
        
        # Attention mekanisme for feature importance
        self.attention = nn.Sequential(
            nn.Linear(config.input_size, config.input_size),
            nn.Sigmoid()
        )
        
        # Feature transformation layers
        self.feature_transform = nn.Sequential(
            nn.Linear(config.input_size, config.input_size),
            nn.LeakyReLU(0.1)
        )
        
        # Initialiser vekter for bedre konvergens
        self._init_weights()
    
    def _init_weights(self):
        """Initialiser vekter for å unngå vanishing/exploding gradients"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Batch normalization
        x = self.batch_norm(x)
        
        # Apply attention mechanism
        if self.config.use_feature_extraction:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Apply feature transformation
        if self.config.use_feature_selection:
            x = self.feature_transform(x)
        
        # Forward pass through main network
        output = self.network(x)
        
        return output, attention_weights if self.config.use_feature_extraction else output

class PropertyDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, transform=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, target

class OptimizationService:
    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None):
        """Initialiserer optimeringstjenesten med konfigurasjon og modeller"""
        # Last eller opprett konfigurasjon
        if config_path and os.path.exists(config_path):
            self.config = PropertyOptimizationConfig.from_json(config_path)
        else:
            self.config = PropertyOptimizationConfig()
        
        # Sett opp device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Bruker {self.device} for beregninger")
        
        # Initialiser scaler
        self.scaler = StandardScaler()
        
        # Initialiser neural network modell
        self.nn_model = PropertyOptimizationModel(self.config).to(self.device)
        
        # Initialiser ensemble modeller
        self.ensemble_models = {}
        if self.config.use_ensemble:
            if 'gbm' in self.config.ensemble_models:
                self.ensemble_models['gbm'] = GradientBoostingRegressor(
                    n_estimators=200, 
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )
            
            if 'rf' in self.config.ensemble_models:
                self.ensemble_models['rf'] = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=10,
                    random_state=42
                )
        
        # Last modell hvis angitt
        if model_path:
            self.load_model(model_path)
        
        # Caching system
        self.prediction_cache = {}
        
        # Feature names and output names for better interpretability
        self.feature_names = [
            # Tomtestørrelse og eksisterende bygg
            'plot_size', 'existing_building_size', 'building_coverage_ratio', 'floor_area_ratio',
            
            # Reguleringsinformasjon
            'max_height', 'min_distance_to_neighbor', 'max_units', 'parking_requirements',
            
            # Terreng og orientering
            'slope_angle', 'solar_exposure', 'ground_quality', 'view_quality',
            
            # Økonomiske faktorer
            'market_value_per_sqm', 'construction_cost_per_sqm', 'rental_potential', 'sale_price_potential',
            
            # Tekniske forhold
            'energy_rating', 'technical_condition', 'renovation_potential', 'building_age',
            
            # Lokasjon og tilgjengelighet
            'distance_to_center', 'public_transport_access', 'neighborhood_rating', 'school_quality',
            
            # Markedsfaktorer
            'market_demand', 'price_trend', 'rental_market_strength', 'investment_attractiveness',
            
            # Bærekraft
            'sustainability_score', 'solar_potential', 'green_area_ratio', 'eco_materials_potential'
        ]
        
        self.output_names = [
            'optimal_utilization',        # Optimal utnyttelsesgrad av tomten
            'room_distribution',          # Optimal fordeling av romtyper
            'energy_efficiency',          # Energieffektivitetsscore
            'economic_potential',         # Økonomisk potensial
            'renovation_priority',        # Prioritering av renovering
            'development_complexity',     # Kompleksitet for utvikling
            'roi_timeline'                # Estimert ROI tidslinje (år)
        ]
    
    def preprocess_data(self, property_data: Dict[str, Any]) -> np.ndarray:
        """Konverterer eiendomsdata til modell-input med forbedret feature engineering"""
        features = []
        
        # Fyll ut features basert på feature_names
        for feature_name in self.feature_names:
            # Hent verdi eller bruk standard hvis ikke tilgjengelig
            value = property_data.get(feature_name, 0)
            
            # Enkel feature transformasjon for å håndtere ulike skaleringseffekter
            if feature_name in ['plot_size', 'existing_building_size']:
                # Log-transform for store areal for å redusere skew
                value = np.log1p(value) if value > 0 else 0
            
            if feature_name in ['distance_to_center']:
                # Invers transform for avstand (nærmere er bedre)
                value = 1 / (1 + value) if value > 0 else 0
            
            features.append(value)
        
        # Legg til beregnede kombinasjonsfeatures
        if 'plot_size' in property_data and property_data['plot_size'] > 0:
            # Beregn tomteutnyttelse
            if 'existing_building_size' in property_data:
                current_utilization = property_data['existing_building_size'] / property_data['plot_size']
                features.append(current_utilization)
            else:
                features.append(0)
        else:
            features.append(0)
        
        # Legg til markedspotensial basert på lokasjon og trend
        market_potential = (
            property_data.get('market_value_per_sqm', 0) * 
            property_data.get('neighborhood_rating', 0.5) * 
            (1 + property_data.get('price_trend', 0))
        )
        features.append(market_potential)
        
        # Beregn teknisk score
        technical_score = (
            (5 - property_data.get('building_age', 0) / 10) * 0.4 +  # Nyere bygg er bedre
            property_data.get('technical_condition', 0.5) * 0.4 +
            property_data.get('energy_rating', 0.5) * 0.2
        )
        features.append(max(0, min(1, technical_score)))
        
        # Normaliser features til input_size
        while len(features) < self.config.input_size:
            features.append(0)  # Padding med nuller
        
        features = features[:self.config.input_size]  # Begrens til input_size
        
        return np.array(features).reshape(1, -1)
    
    async def optimize_property_usage(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyserer og optimaliserer eiendomsutvikling med ensemble-prediksjoner"""
        try:
            # Sjekk cache først hvis aktivert
            cache_key = json.dumps(sorted(property_data.items()))
            if self.config.cache_predictions and cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.config.cache_timeout:
                    logger.info("Returning cached prediction")
                    return cache_entry['result']
            
            # Preprocess data
            features = self.preprocess_data(property_data)
            scaled_features = self.scaler.transform(features)
            
            # Konverter til tensor
            input_tensor = torch.FloatTensor(scaled_features).to(self.device)
            
            # Get neural network predictions
            self.nn_model.eval()
            with torch.no_grad():
                if self.config.use_feature_extraction:
                    nn_predictions, attention_weights = self.nn_model(input_tensor)
                    attention_weights = attention_weights.cpu().numpy()[0]
                else:
                    nn_predictions = self.nn_model(input_tensor)
                    attention_weights = None
                
                nn_predictions = nn_predictions.cpu().numpy()[0]
            
            # Ensemble predictions
            predictions = nn_predictions
            ensemble_predictions = {}
            
            if self.config.use_ensemble and self.ensemble_models:
                # Innhent prediksjoner fra andre modeller
                all_predictions = {'nn': nn_predictions}
                
                for model_name, model in self.ensemble_models.items():
                    try:
                        # Bruk sklearn-modeller for prediksjon
                        model_predictions = model.predict(scaled_features)[0]
                        
                        # Skaler til riktig område hvis nødvendig
                        if len(model_predictions) != len(self.output_names):
                            # Håndter forskjeller i output-størrelse
                            model_predictions = np.pad(
                                model_predictions, 
                                (0, len(self.output_names) - len(model_predictions)),
                                'constant',
                                constant_values=0.5
                            )
                        
                        all_predictions[model_name] = model_predictions
                        ensemble_predictions[model_name] = model_predictions
                    except Exception as e:
                        logger.warning(f"Error in {model_name} prediction: {str(e)}")
                
                # Beregn vektet ensemble
                if len(all_predictions) > 1:
                    predictions = np.zeros_like(nn_predictions)
                    total_weight = 0
                    
                    for model_name, model_preds in all_predictions.items():
                        weight = self.config.ensemble_weights.get(model_name, 0.5)
                        predictions += model_preds * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        predictions /= total_weight
            
            # Konverter prediksjoner til resultater
            results = self._interpret_predictions(predictions)
            
            # Beregn feature importance
            feature_importance = {}
            if attention_weights is not None:
                for i, feature_name in enumerate(self.feature_names[:len(attention_weights)]):
                    feature_importance[feature_name] = float(attention_weights[i])
            
            # Generer detaljerte anbefalinger
            recommendations = self._generate_recommendations(results, property_data)
            
            # Generer optimalisert utviklingsplan
            development_plan = self._generate_development_plan(results, property_data)
            
            # Beregn finansielle projektfremskrivninger
            financial_projections = self._calculate_financial_projections(results, property_data)
            
            # Analyser risiko og usikkerhet
            risk_analysis = self._analyze_risks(results, property_data)
            
            # Lag komplett analyseresultat
            analysis_result = {
                'optimization_results': results,
                'recommendations': recommendations,
                'feature_importance': feature_importance,
                'development_plan': development_plan,
                'financial_projections': financial_projections,
                'risk_analysis': risk_analysis,
                'ensemble_predictions': ensemble_predictions,
                'confidence_score': self._calculate_confidence(predictions)
            }
            
            # Oppdater cache
            if self.config.cache_predictions:
                self.prediction_cache[cache_key] = {
                    'result': analysis_result,
                    'timestamp': datetime.now()
                }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in property optimization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _interpret_predictions(self, raw_predictions: np.ndarray) -> Dict[str, float]:
        """Konverterer modellprediksjoner til meningsfylte resultater med bedre tolkning"""
        results = {}
        
        # Map each prediction to its corresponding output name
        for i, output_name in enumerate(self.output_names):
            if i < len(raw_predictions):
                # Ensure all predictions are in appropriate ranges
                if output_name in ['optimal_utilization', 'energy_efficiency', 'economic_potential']:
                    # These should be between 0 and 1
                    results[output_name] = float(np.clip(raw_predictions[i], 0, 1))
                elif output_name == 'room_distribution':
                    # Room distribution is a value from 0 to 1 representing optimality
                    results[output_name] = float(np.clip(raw_predictions[i], 0, 1))
                elif output_name == 'renovation_priority':
                    # Priority from 0 (low) to 1 (high)
                    results[output_name] = float(np.clip(raw_predictions[i], 0, 1))
                elif output_name == 'development_complexity':
                    # Complexity from 0 (simple) to 1 (complex)
                    results[output_name] = float(np.clip(raw_predictions[i], 0, 1))
                elif output_name == 'roi_timeline':
                    # ROI timeline in years (1-20)
                    results[output_name] = float(max(1, min(20, raw_predictions[i] * 20)))
        
        # Derive additional insights
        if 'optimal_utilization' in results and 'development_complexity' in results:
            # Development potential combines utilization and inverse complexity
            results['development_potential'] = float(
                results['optimal_utilization'] * (1 - results['development_complexity'] * 0.5)
            )
        
        if 'economic_potential' in results and 'roi_timeline' in results:
            # Investment attractiveness combines economic potential and ROI timeline
            results['investment_attractiveness'] = float(
                results['economic_potential'] * (1 - results['roi_timeline'] / 20)
            )
        
        return results
    
    def _generate_recommendations(self, results: Dict[str, float], property_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genererer spesifikke anbefalinger basert på optimeringsresultater med flere detaljer"""
        recommendations = []
        
        # Utnyttelsesgrad anbefalinger
        utilization_score = results.get('optimal_utilization', 0)
        current_utilization = 0
        if property_data.get('plot_size', 0) > 0 and property_data.get('existing_building_size', 0) > 0:
            current_utilization = property_data['existing_building_size'] / property_data['plot_size']
        
        utilization_diff = utilization_score - current_utilization
        
        if utilization_diff > 0.2:  # Betydelig potensial for bedre utnyttelse
            plot_size = property_data.get('plot_size', 0)
            potential_additional_area = plot_size * utilization_diff
            
            recommendations.append({
                'category': 'utilization',
                'title': 'Betydelig utviklingspotensial',
                'description': f'Eiendommen har potensial for ca. {potential_additional_area:.1f} m² ytterligere bygningsmasse',
                'impact': 'high',
                'actions': [
                    'Vurder utvidelse av eksisterende bygningsmasse',
                    'Undersøk muligheter for tilbygg',
                    'Vurder oppdeling i flere enheter',
                    f'Øk utnyttelsesgraden fra {current_utilization:.2f} til nærmere {utilization_score:.2f}'
                ],
                'estimated_cost': self._estimate_development_cost(potential_additional_area, property_data),
                'potential_value_increase': self._estimate_value_increase(potential_additional_area, property_data)
            })
        elif utilization_diff > 0.05:  # Moderat potensial
            recommendations.append({
                'category': 'utilization',
                'title': 'Moderat utviklingspotensial',
                'description': 'Eiendommen har noe potensial for bedre utnyttelse',
                'impact': 'medium',
                'actions': [
                    'Optimaliser eksisterende areal',
                    'Vurder mindre utvidelser eller ombygginger'
                ]
            })
        
        # Romdistribusjon anbefalinger
        room_distribution = results.get('room_distribution', 0)
        
        if room_distribution > 0.7:
            recommendations.append({
                'category': 'layout',
                'title': 'Høyt potensiale for planløsningsoptimalisering',
                'description': 'Betydelig potensial for forbedret romfordeling',
                'impact': 'high',
                'actions': [
                    'Optimaliser rominndeling for mer effektiv bruk',
                    'Vurder åpen planløsning i fellesområder',
                    'Undersøk muligheter for flere bad/soverom',
                    'Vurder hvordan naturlige lysforhold kan forbedres'
                ],
                'estimated_cost': self._estimate_renovation_cost('major_layout', property_data),
                'potential_value_increase': self._estimate_layout_value_increase(property_data)
            })
        elif room_distribution > 0.5:
            recommendations.append({
                'category': 'layout',
                'title': 'Moderat planløsningspotensial',
                'description': 'Noe potensial for forbedret romfordeling',
                'impact': 'medium',
                'actions': [
                    'Vurder mindre endringer i rominndeling',
                    'Undersøk muligheter for å forbedre trafikkflyt',
                    'Moderniser kjøkken/bad for bedre funksjonalitet'
                ]
            })
        
        # Energieffektivitet anbefalinger
        energy_efficiency = results.get('energy_efficiency', 0)
        
        if energy_efficiency < 0.4:
            recommendations.append({
                'category': 'energy',
                'title': 'Betydelig potensial for energioptimalisering',
                'description': 'Eiendommen har stort potensial for energieffektiviseringer',
                'impact': 'high',
                'actions': [
                    'Gjennomfør omfattende etterisolering av vegger, tak og gulv',
                    'Oppgrader til energieffektive vinduer og dører',
                    'Vurder balansert ventilasjon med varmegjenvinning',
                    'Installer varmepumpe eller annen fornybar oppvarmingskilde',
                    'Vurder solceller for strømproduksjon'
                ],
                'estimated_cost': self._estimate_energy_upgrade_cost('comprehensive', property_data),
                'potential_savings': self._estimate_energy_savings('comprehensive', property_data),
                'subsidy_potential': 'høy',
                'roi_years': self._calculate_energy_roi('comprehensive', property_data)
            })
        elif energy_efficiency < 0.7:
            recommendations.append({
                'category': 'energy',
                'title': 'Moderat potensial for energioptimalisering',
                'description': 'Eiendommen har potensial for energieffektiviseringer',
                'impact': 'medium',
                'actions': [
                    'Vurder tilleggsisolasjon på loft og i kjeller',
                    'Oppgrader til bedre vinduer der det er behov',
                    'Installer varmepumpe',
                    'Oppgrader til smart varmestyring'
                ],
                'estimated_cost': self._estimate_energy_upgrade_cost('moderate', property_data),
                'potential_savings': self._estimate_energy_savings('moderate', property_data),
                'roi_years': self._calculate_energy_roi('moderate', property_data)
            })
        
        # Økonomiske potensial anbefalinger
        economic_potential = results.get('economic_potential', 0)
        
        if economic_potential > 0.7:
            recommendations.append({
                'category': 'economic',
                'title': 'Høyt økonomisk potensial',
                'description': 'Eiendommen har betydelig økonomisk utviklingspotensial',
                'impact': 'high',
                'actions': [
                    'Vurder strategisk utviklingsplan med flere faser',
                    'Undersøk muligheter for endring av reguleringsformål',
                    'Vurder om eiendommen kan deles opp i flere seksjoner',
                    'Utforsk kombinert bolig og næring hvis mulig'
                ],
                'financial_metrics': {
                    'estimated_roi': f"{economic_potential * 100:.1f}%",
                    'investment_attractiveness': 'høy',
                    'market_timing': 'gunstig' if property_data.get('price_trend', 0) > 0.05 else 'nøytral'
                }
            })
        
        # Renovering prioritering
        renovation_priority = results.get('renovation_priority', 0)
        
        if renovation_priority > 0.7:
            recommendations.append({
                'category': 'renovation',
                'title': 'Høy prioritet for renovering',
                'description': 'Renovering bør prioriteres for optimal verdiutvikling',
                'impact': 'high',
                'actions': [
                    'Gjennomfør teknisk tilstandsanalyse',
                    'Prioriter kritiske renoveringsbehov med høy ROI',
                    'Utarbeid langsiktig vedlikeholdsplan'
                ],
                'suggested_timeframe': 'Innen 1-2 år'
            })
        
        return recommendations
    
    def _generate_development_plan(self, results: Dict[str, float], property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genererer en faseinndelt utviklingsplan for eiendommen"""
        optimal_utilization = results.get('optimal_utilization', 0)
        development_complexity = results.get('development_complexity', 0.5)
        roi_timeline = results.get('roi_timeline', 10)
        
        # Beregn nøkkelvariabler for utviklingsplanen
        plot_size = property_data.get('plot_size', 0)
        current_building_size = property_data.get('existing_building_size', 0)
        current_utilization = current_building_size / plot_size if plot_size > 0 else 0
        
        target_building_size = plot_size * optimal_utilization
        additional_area = max(0, target_building_size - current_building_size)
        
        # Fastsett tidslinjen basert på kompleksitet
        timeline_factor = 1 + development_complexity
        phase_duration = [3, 6, 9]  # Måneder per fase
        
        # Juster fasenes varighet basert på kompleksitet
        adjusted_phases = [int(p * timeline_factor) for p in phase_duration]
        
        # Beregn totale kostnader
        cost_per_sqm = property_data.get('construction_cost_per_sqm', 30000)
        total_development_cost = additional_area * cost_per_sqm
        
        # Beregn forventet verdiøkning
        value_increase_factor = 1.3  # 30% økning over byggekostnad
        expected_value_increase = total_development_cost * value_increase_factor
        
        # Lag utviklingsplanen
        plan = {
            'current_state': {
                'building_size': float(current_building_size),
                'utilization_ratio': float(current_utilization),
                'estimated_value': float(property_data.get('market_value_per_sqm', 0) * current_building_size)
            },
            'target_state': {
                'building_size': float(target_building_size),
                'utilization_ratio': float(optimal_utilization),
                'additional_area': float(additional_area),
                'estimated_value_after_development': float(
                    property_data.get('market_value_per_sqm', 0) * target_building_size * 1.1
                )
            },
            'financial_summary': {
                'estimated_development_cost': float(total_development_cost),
                'expected_value_increase': float(expected_value_increase),
                'roi_percentage': float((expected_value_increase / total_development_cost - 1) * 100) if total_development_cost > 0 else 0,
                'estimated_roi_timeline_years': float(roi_timeline)
            },
            'phases': [
                {
                    'name': 'Fase 1: Planlegging og prosjektering',
                    'duration_months': adjusted_phases[0],
                    'key_activities': [
                        'Gjennomføre detaljert teknisk vurdering',
                        'Utarbeide arkitekttegninger',
                        'Søke byggetillatelse',
                        'Innhente tilbud fra entreprenører'
                    ],
                    'estimated_cost_percentage': 10,
                    'estimated_cost': float(total_development_cost * 0.1),
                    'risk_factors': ['Reguleringsutfordringer', 'Prosjektforsinkelser']
                },
                {
                    'name': 'Fase 2: Bygging og renovering',
                    'duration_months': adjusted_phases[1],
                    'key_activities': [
                        'Igangsette byggeprosess',
                        'Gjennomføre strukturelle endringer',
                        'Installere tekniske systemer',
                        'Oppgradering av eksisterende arealer'
                    ],
                    'estimated_cost_percentage': 75,
                    'estimated_cost': float(total_development_cost * 0.75),
                    'risk_factors': ['Budsjettoverskridelser', 'Uforutsette tekniske utfordringer']
                },
                {
                    'name': 'Fase 3: Ferdigstillelse og markedsintroduksjon',
                    'duration_months': adjusted_phases[2],
                    'key_activities': [
                        'Innredning og ferdigstillelse',
                        'Teknisk kontroll og sertifisering',
                        'Markedsføring av eiendommen',
                        'Salg eller utleie av enheter'
                    ],
                    'estimated_cost_percentage': 15,
                    'estimated_cost': float(total_development_cost * 0.15),
                    'risk_factors': ['Markedsendringer', 'Prispress i området']
                }
            ],
            'total_timeline_months': sum(adjusted_phases),
            'regulatory_considerations': self._get_regulatory_considerations(property_data),
            'sustainability_integration': self._get_sustainability_integration(results, property_data)
        }
        
        return plan
    
    def _get_regulatory_considerations(self, property_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifiserer regulatoriske hensyn for eiendomsutvikling"""
        considerations = []
        
        # Sjekk utnyttelsesgrad mot regulering
        if 'building_coverage_ratio' in property_data and 'floor_area_ratio' in property_data:
            considerations.append({
                'type': 'zoning',
                'description': 'Regulert tomteutnyttelse',
                'details': f"BYA: {property_data['building_coverage_ratio']*100:.0f}%, TU: {property_data['floor_area_ratio']:.1f}",
                'impact': 'high'
            })
        
        # Sjekk høydebegrensninger
        if 'max_height' in property_data:
            considerations.append({
                'type': 'height_restrictions',
                'description': 'Høydebegrensninger',
                'details': f"Maks byggehøyde: {property_data['max_height']:.1f}m",
                'impact': 'medium'
            })
        
        # Sjekk parkeringskrav
        if 'parking_requirements' in property_data:
            considerations.append({
                'type': 'parking',
                'description': 'Parkeringskrav',
                'details': f"Krav: {property_data['parking_requirements']} plasser per enhet",
                'impact': 'medium'
            })
        
        # Sjekk avstandskrav
        if 'min_distance_to_neighbor' in property_data:
            considerations.append({
                'type': 'distance_requirements',
                'description': 'Avstandskrav til nabogrense',
                'details': f"Minimum: {property_data['min_distance_to_neighbor']}m",
                'impact': 'medium'
            })
        
        return considerations
    
    def _get_sustainability_integration(self, results: Dict[str, float], property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genererer bærekraftig integrasjonsplan"""
        energy_efficiency = results.get('energy_efficiency', 0.5)
        
        # Velg tiltak basert på energieffektivitetsscore
        if energy_efficiency < 0.4:
            energy_level = 'high'
            energy_measures = [
                'Omfattende etterisolering',
                'Installasjon av varmepumpe',
                'Balansert ventilasjon med varmegjenvinning',
                'Trippelglass vinduer med lav U-verdi',
                'Solceller for strømproduksjon'
            ]
        elif energy_efficiency < 0.7:
            energy_level = 'medium'
            energy_measures = [
                'Tilleggsisolering av loft og kjeller',
                'Utskifting av vinduer med dårlig U-verdi',
                'Luft-til-luft varmepumpe',
                'Smart energistyring'
            ]
        else:
            energy_level = 'low'
            energy_measures = [
                'Mindre isolasjonstiltak',
                'Energieffektiv belysning',
                'Smarte termostater'
            ]
        
        # Vurder solpotensial
        solar_potential = property_data.get('solar_potential', 0.5)
        if solar_potential > 0.7:
            solar_measures = [
                'Solcelleanlegg på tak',
                'Solfangere for oppvarming av vann',
                'Batterisystem for energilagring'
            ]
        elif solar_potential > 0.4:
            solar_measures = [
                'Mindre solcelleanlegg for delvis strømforsyning',
                'Passiv solvarme gjennom strategisk plasserte vinduer'
            ]
        else:
            solar_measures = [
                'Begrenset potensial for aktiv solenergi',
                'Vurder andre fornybare energikilder'
            ]
        
        return {
            'energy_efficiency': {
                'priority_level': energy_level,
                'recommended_measures': energy_measures,
                'potential_energy_rating_improvement': 2 if energy_level == 'high' else (1 if energy_level == 'medium' else 0),
                'estimated_energy_savings_percentage': 50 if energy_level == 'high' else (30 if energy_level == 'medium' else 15)
            },
            'renewable_energy': {
                'solar_potential': float(solar_potential),
                'recommended_solutions': solar_measures
            },
            'materials': [
                'Bruk av miljøsertifiserte byggematerialer',
                'Gjenbruk av eksisterende materialer der mulig',
                'Velg materialer med lavt klimaavtrykk og god holdbarhet'
            ],
            'water_management': [
                'Vannbesparende armaturer',
                'System for oppsamling av regnvann',
                'Permeable overflater for håndtering av overvann'
            ],
            'potential_certifications': self._get_potential_certifications(results, property_data)
        }
    
    def _get_potential_certifications(self, results: Dict[str, float], property_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifiserer potensielle miljøsertifiseringer for prosjektet"""
        energy_efficiency = results.get('energy_efficiency', 0.5)
        certifications = []
        
        # BREEAM-NOR
        if energy_efficiency > 0.7:
            certifications.append({
                'name': 'BREEAM-NOR',
                'potential_level': 'Very Good / Excellent',
                'benefits': 'Internasjonalt anerkjent miljøsertifisering som kan øke eiendomsverdien',
                'estimated_cost_premium': '3-5% av byggebudsjettet'
            })
        elif energy_efficiency > 0.5:
            certifications.append({
                'name': 'BREEAM-NOR',
                'potential_level': 'Good',
                'benefits': 'Anerkjent miljøsertifisering som viser miljøfokus',
                'estimated_cost_premium': '2-3% av byggebudsjettet'
            })
        
        # Svanemerket
        if energy_efficiency > 0.8:
            certifications.append({
                'name': 'Svanemerket',
                'potential_level': 'Sertifisert',
                'benefits': 'Nordisk miljømerke med høy gjenkjennelse blant forbrukere',
                'estimated_cost_premium': '2-4% av byggebudsjettet'
            })
        
        # Passivhus/nZEB
        if energy_efficiency > 0.75:
            certifications.append({
                'name': 'Passivhus/nZEB',
                'potential_level': 'Sertifisert',
                'benefits': 'Svært energieffektiv bygning med lavt energibehov',
                'estimated_cost_premium': '5-8% av byggebudsjettet'
            })
        
        return certifications
    
    def _calculate_financial_projections(self, results: Dict[str, float], property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner finansielle framskrivninger for prosjektet"""
        economic_potential = results.get('economic_potential', 0.5)
        roi_timeline = results.get('roi_timeline', 10)
        
        # Beregn nøkkelvariabler
        plot_size = property_data.get('plot_size', 0)
        current_building_size = property_data.get('existing_building_size', 0)
        optimal_utilization = results.get('optimal_utilization', 0)
        target_building_size = plot_size * optimal_utilization
        additional_area = max(0, target_building_size - current_building_size)
        
        # Kostnader
        construction_cost_per_sqm = property_data.get('construction_cost_per_sqm', 30000)
        renovation_cost_per_sqm = construction_cost_per_sqm * 0.6  # Renovering er typisk billigere enn nybygg
        
        development_cost = additional_area * construction_cost_per_sqm
        renovation_cost = current_building_size * renovation_cost_per_sqm * (1 - results.get('energy_efficiency', 0.5))
        
        total_project_cost = development_cost + renovation_cost
        
        # Inntektspotensial
        market_value_per_sqm = property_data.get('market_value_per_sqm', 50000)
        current_value = current_building_size * market_value_per_sqm
        
        # Justert verdi basert på økonomisk potensial og tilstand
        value_increase_factor = 1.1 + (economic_potential * 0.3)
        future_value_per_sqm = market_value_per_sqm * value_increase_factor
        
        future_value = target_building_size * future_value_per_sqm
        value_increase = future_value - current_value
        
        # ROI beregning
        roi_percentage = (value_increase / total_project_cost * 100) if total_project_cost > 0 else 0
        annual_roi = roi_percentage / roi_timeline if roi_timeline > 0 else 0
        
        # Kontantstrømsprojeksjoner
        rental_potential = property_data.get('rental_potential', 0)
        monthly_rental_income = target_building_size * rental_potential
        annual_rental_income = monthly_rental_income * 12
        
        # Driftskostnader
        operating_expenses_percentage = 0.3  # 30% av leieinntekter
        annual_operating_expenses = annual_rental_income * operating_expenses_percentage
        net_annual_income = annual_rental_income - annual_operating_expenses
        
        # Avkastning ved utleie
        rental_yield = (net_annual_income / future_value * 100) if future_value > 0 else 0
        
        return {
            'investment_summary': {
                'development_cost': float(development_cost),
                'renovation_cost': float(renovation_cost),
                'total_project_cost': float(total_project_cost),
                'contingency_reserve': float(total_project_cost * 0.1)  # 10% buffer
            },
            'value_projection': {
                'current_value': float(current_value),
                'projected_future_value': float(future_value),
                'value_increase': float(value_increase),
                'value_increase_percentage': float((future_value / current_value - 1) * 100) if current_value > 0 else 0
            },
            'roi_analysis': {
                'project_roi_percentage': float(roi_percentage),
                'estimated_roi_timeline_years': float(roi_timeline),
                'annual_roi_percentage': float(annual_roi)
            },
            'rental_scenario': {
                'monthly_rental_income': float(monthly_rental_income),
                'annual_rental_income': float(annual_rental_income),
                'annual_operating_expenses': float(annual_operating_expenses),
                'net_annual_income': float(net_annual_income),
                'rental_yield_percentage': float(rental_yield)
            },
            'funding_options': self._suggest_funding_options(total_project_cost, property_data),
            'sensitivity_analysis': self._perform_sensitivity_analysis(
                total_project_cost, 
                future_value, 
                net_annual_income,
                roi_timeline
            )
        }
    
    def _suggest_funding_options(self, total_cost: float, property_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Foreslår finansieringsalternativer for prosjektet"""
        current_value = property_data.get('existing_building_size', 0) * property_data.get('market_value_per_sqm', 0)
        
        funding_options = []
        
        # Tradisjonelt byggelån
        funding_options.append({
            'type': 'construction_loan',
            'name': 'Byggelån',
            'description': 'Tradisjonelt byggelån fra bank',
            'typical_terms': '60-75% av prosjektkostnad, 3-5% rente',
            'pros': ['Velkjent finansieringsform', 'Forutsigbare vilkår'],
            'cons': ['Krever egenkapital', 'Personlig garanti kan være nødvendig']
        })
        
        # Refinansiering av eksisterende eiendom
        if current_value > 0:
            funding_options.append({
                'type': 'refinancing',
                'name': 'Refinansiering av eksisterende eiendom',
                'description': 'Frigjøre egenkapital fra eksisterende eiendom',
                'typical_terms': 'Opptil 75% belåningsgrad, 2.5-4% rente',
                'pros': ['Utnytter eksisterende egenkapital', 'Ofte lavere rente enn byggelån'],
                'cons': ['Øker belåningsgrad på eksisterende eiendom', 'Begrenset til eksisterende verdi']
            })
        
        # Investorkapital
        if total_cost > 10000000:  # For større prosjekter
            funding_options.append({
                'type': 'investor_equity',
                'name': 'Investorkapital',
                'description': 'Hente inn eksterne investorer for delt eierskap',
                'typical_terms': '20-40% av prosjektkostnad mot eierandel',
                'pros': ['Reduserer egen kapitalrisiko', 'Potensielt tilgang til investornettverk'],
                'cons': ['Gir fra seg eierandel', 'Krav til avkastning fra investorer']
            })
        
        # Grønn finansiering
        energy_efficiency = property_data.get('energy_rating', 0)
        if energy_efficiency > 0.7:
            funding_options.append({
                'type': 'green_financing',
                'name': 'Grønn finansiering',
                'description': 'Spesialfinansiering for miljøvennlige byggeprosjekter',
                'typical_terms': 'Opptil 80% av kostnad, 0.2-0.5% rabatt på rente',
                'pros': ['Bedre vilkår enn tradisjonelle lån', 'I tråd med bærekraftig utvikling'],
                'cons': ['Krav til miljøsertifisering', 'Begrenset tilgjengelighet']
            })
        
        return funding_options
    
    def _perform_sensitivity_analysis(self, 
                                     total_cost: float, 
                                     future_value: float, 
                                     annual_income: float,
                                     roi_timeline: float) -> Dict[str, Any]:
        """Utfører sensitivitetsanalyse for prosjektøkonomien"""
        # Definer scenarioer
        scenarios = {
            'base_case': {'cost_factor': 1.0, 'value_factor': 1.0, 'income_factor': 1.0},
            'best_case': {'cost_factor': 0.9, 'value_factor': 1.1, 'income_factor': 1.15},
            'worst_case': {'cost_factor': 1.2, 'value_factor': 0.9, 'income_factor': 0.85}
        }
        
        scenario_results = {}
        
        for name, factors in scenarios.items():
            adj_cost = total_cost * factors['cost_factor']
            adj_value = future_value * factors['value_factor']
            adj_income = annual_income * factors['income_factor']
            
            value_increase = adj_value - (future_value / factors['value_factor'])
            roi_percentage = (value_increase / adj_cost * 100) if adj_cost > 0 else 0
            annual_roi = roi_percentage / roi_timeline if roi_timeline > 0 else 0
            rental_yield = (adj_income / adj_value * 100) if adj_value > 0 else 0
            
            scenario_results[name] = {
                'total_cost': float(adj_cost),
                'future_value': float(adj_value),
                'value_increase': float(value_increase),
                'roi_percentage': float(roi_percentage),
                'annual_roi': float(annual_roi),
                'annual_income': float(adj_income),
                'rental_yield': float(rental_yield)
            }
        
        # Risikofaktorer som påvirker scenarioene
        risk_factors = [
            {
                'name': 'Byggekostnadsøkning',
                'impact': 'Høyere byggekostnader reduserer ROI',
                'mitigation': 'Fastpriskontrakter, tidlig innkjøp av materialer'
            },
            {
                'name': 'Markedsverdifall',
                'impact': 'Lavere salgsverdi reduserer prosjektets lønnsomhet',
                'mitigation': 'Faset utvikling, fleksibel utleie/salgsstrategi'
            },
            {
                'name': 'Lavere leieinntekter',
                'impact': 'Redusert kontantstrøm ved utleie',
                'mitigation': 'Langsiktige leiekontrakter, kvalitetsfokus for å tiltrekke stabile leietakere'
            },
            {
                'name': 'Forsinket ferdigstillelse',
                'impact': 'Økte finansieringskostnader, utsatt inntekt',
                'mitigation': 'Realistisk tidsplan, klare milepæler, erfarne entreprenører'
            }
        ]
        
        return {
            'scenarios': scenario_results,
            'risk_factors': risk_factors,
            'break_even_analysis': {
                'value_reduction_tolerance': float(
                    ((future_value - total_cost) / future_value) * 100
                ) if future_value > 0 else 0,
                'cost_increase_tolerance': float(
                    ((future_value - total_cost) / total_cost) * 100
                ) if total_cost > 0 else 0
            }
        }
    
    def _analyze_risks(self, results: Dict[str, float], property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyserer risiko og usikkerhetsfaktorer"""
        development_complexity = results.get('development_complexity', 0.5)
        
        # Identifiser risikokategorier basert på kompleksitet
        risk_categories = ['regulatory', 'financial', 'technical', 'market']
        
        # Tilordne risikofaktorer basert på prosjektets egenskaper
        risk_factors = []
        
        # Regulatoriske risikoer
        if 'max_height' in property_data or 'building_coverage_ratio' in property_data:
            risk_factors.append({
                'category': 'regulatory',
                'name': 'Reguleringsbegrensninger',
                'description': 'Risiko for at reguleringsplan begrenser utviklingspotensialet',
                'probability': 'medium' if development_complexity > 0.6 else 'low',
                'impact': 'high',
                'mitigation': 'Tidlig dialog med planmyndigheter, grundig gjennomgang av reguleringsbestemmelser'
            })
        
        # Finansielle risikoer
        construction_cost = property_data.get('construction_cost_per_sqm', 30000)
        if construction_cost > 35000:  # Høye byggekostnader
            risk_factors.append({
                'category': 'financial',
                'name': 'Kostnadsoverskridelser',
                'description': 'Risiko for at byggekostnader overskrider budsjett',
                'probability': 'high' if development_complexity > 0.7 else 'medium',
                'impact': 'high',
                'mitigation': 'Detaljert kostnadsestimat, fastpriskontrakter, kontinuerlig budsjettoppfølging'
            })
        
        risk_factors.append({
            'category': 'financial',
            'name': 'Renteøkning',
            'description': 'Risiko for økte finansieringskostnader ved renteøkning',
            'probability': 'medium',
            'impact': 'medium',
            'mitigation': 'Vurdere rentesikring, faset utvikling for å redusere lånebeløp'
        })
        
        # Tekniske risikoer
        building_age = property_data.get('building_age', 0)
        if building_age > 30:  # Eldre bygg
            risk_factors.append({
                'category': 'technical',
                'name': 'Skjulte bygningsfeil',
                'description': 'Risiko for å avdekke skjulte feil i eksisterende bygningsmasse',
                'probability': 'high',
                'impact': 'medium',
                'mitigation': 'Grundig teknisk due diligence, sette av buffer i budsjettet'
            })
        
        # Markedsrisikoer
        market_trend = property_data.get('price_trend', 0)
        if market_trend < 0.02:  # Svak eller negativ prisutvikling
            risk_factors.append({
                'category': 'market',
                'name': 'Markedsavkjøling',
                'description': 'Risiko for redusert etterspørsel eller prisfall i markedet',
                'probability': 'medium',
                'impact': 'high',
                'mitigation': 'Fleksibel strategi for salg/utleie, fokus på kvalitet for å skille seg ut i markedet'
            })
        
        # Beregn risikoscore
        risk_scores = {category: 0 for category in risk_categories}
        
        for risk in risk_factors:
            category = risk['category']
            probability_score = {'low': 1, 'medium': 2, 'high': 3}.get(risk['probability'], 0)
            impact_score = {'low': 1, 'medium': 2, 'high': 3}.get(risk['impact'], 0)
            risk_scores[category] += probability_score * impact_score
        
        # Normaliser risikoscorer til 0-1 skala
        max_possible_score = 9 * 3  # Anta maks 3 høy-høy risikoer per kategori
        for category in risk_scores:
            risk_scores[category] = min(1, risk_scores[category] / max_possible_score)
        
        # Beregn total risikoscore
        total_risk_score = sum(risk_scores.values()) / len(risk_scores)
        
        return {
            'overall_risk_level': {
                'score': float(total_risk_score),
                'category': 'high' if total_risk_score > 0.7 else ('medium' if total_risk_score > 0.4 else 'low')
            },
            'risk_by_category': {k: float(v) for k, v in risk_scores.items()},
            'risk_factors': risk_factors,
            'major_risk_areas': [category for category, score in risk_scores.items() if score > 0.6],
            'recommended_contingency': f"{(10 + total_risk_score * 15):.1f}%"  # 10-25% avhengig av risiko
        }
    
    def _estimate_development_cost(self, area: float, property_data: Dict[str, Any]) -> float:
        """Estimerer utviklingskostnad basert på areal og eiendomsdata"""
        base_cost_per_sqm = property_data.get('construction_cost_per_sqm', 30000)
        
        # Juster basert på terreng og grunnforhold
        ground_quality = property_data.get('ground_quality', 0.5)
        ground_factor = 1 + (0.5 - ground_quality) * 0.4  # Dårligere grunnforhold gir høyere kostnader
        
        # Juster basert på helning
        slope_angle = property_data.get('slope_angle', 0)
        slope_factor = 1 + (slope_angle / 45) * 0.3  # Brattere terreng gir høyere kostnader
        
        adjusted_cost_per_sqm = base_cost_per_sqm * ground_factor * slope_factor
        
        return area * adjusted_cost_per_sqm
    
    def _estimate_value_increase(self, additional_area: float, property_data: Dict[str, Any]) -> float:
        """Estimerer verdiøkning basert på nye arealer"""
        market_value_per_sqm = property_data.get('market_value_per_sqm', 50000)
        
        # Juster basert på markedstrender
        price_trend = property_data.get('price_trend', 0)
        trend_factor = 1 + price_trend
        
        # Juster basert på nabolagskvalitet
        neighborhood_rating = property_data.get('neighborhood_rating', 0.5)
        neighborhood_factor = 0.8 + neighborhood_rating * 0.4  # 0.8-1.2 faktor
        
        # Premium for nybygg
        new_build_premium = 1.15
        
        adjusted_value_per_sqm = market_value_per_sqm * trend_factor * neighborhood_factor * new_build_premium
        
        return additional_area * adjusted_value_per_sqm
    
    def _estimate_renovation_cost(self, renovation_type: str, property_data: Dict[str, Any]) -> float:
        """Estimerer renoveringskostnader basert på type og eiendomsdata"""
        area = property_data.get('existing_building_size', 100)
        
        # Basispriser per kvadratmeter for ulike renoveringstyper
        cost_per_sqm = {
            'minor_layout': 5000,      # Mindre planløsningsendringer
            'major_layout': 10000,     # Større planløsningsendringer (åpen løsning, etc.)
            'complete': 15000          # Komplett renovering
        }
        
        base_cost = area * cost_per_sqm.get(renovation_type, 7500)
        
        # Juster basert på bygningsalder
        building_age = property_data.get('building_age', 0)
        age_factor = 1 + min(1, building_age / 100) * 0.5  # Eldre bygg er dyrere å renovere
        
        # Juster basert på teknisk tilstand
        technical_condition = property_data.get('technical_condition', 0.5)
        condition_factor = 1 + (1 - technical_condition) * 0.7  # Dårligere tilstand gir høyere kostnader
        
        adjusted_cost = base_cost * age_factor * condition_factor
        
        return adjusted_cost
    
    def _estimate_layout_value_increase(self, property_data: Dict[str, Any]) -> float:
        """Estimerer verdiøkning ved forbedret planløsning"""
        area = property_data.get('existing_building_size', 100)
        market_value_per_sqm = property_data.get('market_value_per_sqm', 50000)
        current_value = area * market_value_per_sqm
        
        # Generelt kan en god planløsningsoptimalisering øke verdien med 5-15%
        value_increase_percentage = 0.1  # 10% standard
        
        # Juster basert på alder (eldre bygg har ofte større forbedringspotensial)
        building_age = property_data.get('building_age', 0)
        if building_age > 30:
            value_increase_percentage += 0.05
        
        # Juster basert på område (dyrere områder gir større absolutt avkastning på forbedringer)
        if market_value_per_sqm > 60000:
            value_increase_percentage += 0.03
        
        return current_value * value_increase_percentage
    
    def _estimate_energy_upgrade_cost(self, scope: str, property_data: Dict[str, Any]) -> float:
        """Estimerer kostnad for energioppgradering"""
        area = property_data.get('existing_building_size', 100)
        
        # Basiskostnader per kvadratmeter for ulike oppgraderingsomfang
        cost_per_sqm = {
            'minor': 1000,         # Mindre tiltak (utskifting av vinduer, litt isolasjon)
            'moderate': 2500,      # Moderate tiltak (vinduer, etterisolering av tak/loft)
            'comprehensive': 5000  # Omfattende tiltak (full etterisolering, nye tekniske systemer)
        }
        
        base_cost = area * cost_per_sqm.get(scope, 2500)
        
        # Juster basert på eksisterende energirating (dårligere rating = mer arbeid nødvendig)
        energy_rating = property_data.get('energy_rating', 0.5)
        energy_factor = 1 + (1 - energy_rating) * 0.5
        
        # Juster basert på bygningsalder
        building_age = property_data.get('building_age', 0)
        age_factor = 1 + min(1, building_age / 80) * 0.6
        
        adjusted_cost = base_cost * energy_factor * age_factor
        
        return adjusted_cost
    
    def _estimate_energy_savings(self, scope: str, property_data: Dict[str, Any]) -> Dict[str, float]:
        """Estimerer energibesparelser ved oppgradering"""
        area = property_data.get('existing_building_size', 100)
        energy_rating = property_data.get('energy_rating', 0.5)
        
        # Estimer nåværende energiforbruk (kWh/m²/år)
        # Dårligere energirating gir høyere forbruk
        current_consumption = 200 + (1 - energy_rating) * 200  # 200-400 kWh/m²/år
        
        # Forbedringspotensialer for ulike omfang
        improvement_percentage = {
            'minor': 0.15,             # 15% reduksjon
            'moderate': 0.35,          # 35% reduksjon
            'comprehensive': 0.6       # 60% reduksjon
        }
        
        # Beregn besparelser
        percentage = improvement_percentage.get(scope, 0.3)
        saved_kwh_per_year = current_consumption * area * percentage
        
        # Beregn økonomiske besparelser
        electricity_price = 1.5  # NOK per kWh
        annual_savings = saved_kwh_per_year * electricity_price
        
        return {
            'current_consumption_kwh': float(current_consumption * area),
            'consumption_reduction_percentage': float(percentage * 100),
            'saved_kwh_per_year': float(saved_kwh_per_year),
            'annual_cost_savings': float(annual_savings),
            'co2_reduction_kg': float(saved_kwh_per_year * 0.017)  # Norsk strømmiks
        }
    
    def _calculate_energy_roi(self, scope: str, property_data: Dict[str, Any]) -> float:
        """Beregner ROI for energitiltak i år"""
        upgrade_cost = self._estimate_energy_upgrade_cost(scope, property_data)
        savings = self._estimate_energy_savings(scope, property_data)
        annual_savings = savings.get('annual_cost_savings', 0)
        
        if annual_savings <= 0:
            return float('inf')
        
        # Beregn enkelt tilbakebetalingstid
        payback_years = upgrade_cost / annual_savings
        
        # Juster for Enova-subsidier
        if scope == 'comprehensive':
            subsidy_percentage = 0.25  # Typisk Enova-støtte for større energitiltak
            payback_years *= (1 - subsidy_percentage)
        
        return payback_years
    
    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """Beregner konfidensgrad for prediksjonene"""
        # For neural network outputs, confidence can be estimated 
        # based on how close the values are to the extremes
        
        # Normalize predictions to 0-1 range if not already
        normalized_predictions = np.clip(predictions, 0, 1)
        
        # Calculate distance from the middle (0.5)
        distances = np.abs(normalized_predictions - 0.5)
        
        # Average distance, normalized to 0-1 range
        avg_distance = np.mean(distances) * 2  # *2 to scale from 0-0.5 to 0-1
        
        # Apply a scaling factor to map to a reasonable confidence range
        confidence = 0.6 + avg_distance * 0.3  # Range: 0.6-0.9
        
        return float(confidence)
    
    def train(self, train_data: np.ndarray, train_targets: np.ndarray, 
              val_data: Optional[np.ndarray] = None, val_targets: Optional[np.ndarray] = None,
              test_data: Optional[np.ndarray] = None, test_targets: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Trener modellen på gitt datasett"""
        # Preprocess data
        self.scaler.fit(train_data)
        scaled_train_data = self.scaler.transform(train_data)
        
        if val_data is not None and val_targets is not None:
            scaled_val_data = self.scaler.transform(val_data)
            validation_set = (scaled_val_data, val_targets)
        else:
            # Split train data to create validation set
            split_idx = int(len(scaled_train_data) * 0.8)
            scaled_val_data = scaled_train_data[split_idx:]
            val_targets = train_targets[split_idx:]
            scaled_train_data = scaled_train_data[:split_idx]
            train_targets = train_targets[:split_idx]
            validation_set = (scaled_val_data, val_targets)
        
        # Create datasets
        train_dataset = PropertyDataset(scaled_train_data, train_targets)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        val_dataset = PropertyDataset(*validation_set)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        # Create optimizer
        optimizer = optim.Adam(
            self.nn_model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        if self.config.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                verbose=True
            )
        
        # Mixed precision
        if self.config.use_mixed_precision and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"Starting training on {len(train_dataset)} samples")
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.nn_model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        if self.config.use_feature_extraction:
                            output, _ = self.nn_model(data)
                        else:
                            output = self.nn_model(data)
                        loss = criterion(output, target)
                    
                    # Backward and optimize with scaler
                    scaler.scale(loss).backward()
                    
                    # Add L1 regularization if enabled
                    if self.config.use_l1_reg:
                        l1_penalty = torch.tensor(0.0).to(self.device)
                        for param in self.nn_model.parameters():
                            l1_penalty += torch.norm(param, 1)
                        
                        scaled_penalty = scaler.scale(l1_penalty * self.config.l1_lambda)
                        scaled_penalty.backward()
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    if self.config.use_feature_extraction:
                        output, _ = self.nn_model(data)
                    else:
                        output = self.nn_model(data)
                    loss = criterion(output, target)
                    
                    loss.backward()
                    
                    # Add L1 regularization if enabled
                    if self.config.use_l1_reg:
                        l1_penalty = torch.tensor(0.0).to(self.device)
                        for param in self.nn_model.parameters():
                            l1_penalty += torch.norm(param, 1)
                        
                        (l1_penalty * self.config.l1_lambda).backward()
                    
                    optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            training_history['train_loss'].append(train_loss)
            
            # Validation phase
            self.nn_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.config.use_feature_extraction:
                        output, _ = self.nn_model(data)
                    else:
                        output = self.nn_model(data)
                    
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            training_history['val_loss'].append(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Learning rate scheduler step
            if self.config.use_lr_scheduler:
                scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_model_state = self.nn_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.nn_model.load_state_dict(best_model_state)
        
        # Train ensemble models if enabled
        ensemble_metrics = {}
        if self.config.use_ensemble and self.ensemble_models:
            for model_name, model in self.ensemble_models.items():
                try:
                    logger.info(f"Training {model_name} model...")
                    model.fit(scaled_train_data, train_targets)
                    
                    # Evaluate on validation set
                    val_preds = model.predict(scaled_val_data)
                    val_mse = np.mean((val_preds - val_targets) ** 2)
                    ensemble_metrics[model_name] = {'val_mse': float(val_mse)}
                    
                    logger.info(f"{model_name} validation MSE: {val_mse:.6f}")
                except Exception as e:
                    logger.error(f"Error training {model_name} model: {str(e)}")
        
        # Evaluate on test set if provided
        test_metrics = {}
        if test_data is not None and test_targets is not None:
            scaled_test_data = self.scaler.transform(test_data)
            test_dataset = PropertyDataset(scaled_test_data, test_targets)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Neural network evaluation
            self.nn_model.eval()
            test_loss = 0.0
            all_outputs = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.config.use_feature_extraction:
                        output, _ = self.nn_model(data)
                    else:
                        output = self.nn_model(data)
                    
                    loss = criterion(output, target)
                    test_loss += loss.item()
                    all_outputs.append(output.cpu().numpy())
            
            test_loss /= len(test_loader)
            test_metrics['nn'] = {'test_mse': float(test_loss)}
            
            # Ensemble models evaluation
            if self.config.use_ensemble and self.ensemble_models:
                for model_name, model in self.ensemble_models.items():
                    try:
                        test_preds = model.predict(scaled_test_data)
                        test_mse = np.mean((test_preds - test_targets) ** 2)
                        test_metrics[model_name] = {'test_mse': float(test_mse)}
                    except Exception as e:
                        logger.error(f"Error evaluating {model_name} model: {str(e)}")
            
            # Combined ensemble evaluation
            all_preds = np.zeros_like(test_targets)
            total_weight = 0
            
            # Neural network predictions
            nn_preds = np.concatenate(all_outputs, axis=0)
            nn_weight = self.config.ensemble_weights.get('nn', 0.5)
            all_preds += nn_preds * nn_weight
            total_weight += nn_weight
            
            # Other ensemble models
            for model_name, model in self.ensemble_models.items():
                try:
                    model_preds = model.predict(scaled_test_data)
                    model_weight = self.config.ensemble_weights.get(model_name, 0.2)
                    all_preds += model_preds * model_weight
                    total_weight += model_weight
                except Exception as e:
                    logger.error(f"Error in ensemble prediction for {model_name}: {str(e)}")
            
            if total_weight > 0:
                all_preds /= total_weight
            
            ensemble_test_mse = np.mean((all_preds - test_targets) ** 2)
            test_metrics['ensemble'] = {'test_mse': float(ensemble_test_mse)}
            
            logger.info(f"Test MSE - NN: {test_metrics['nn']['test_mse']:.6f}, Ensemble: {ensemble_test_mse:.6f}")
        
        return {
            'training_history': training_history,
            'ensemble_metrics': ensemble_metrics,
            'test_metrics': test_metrics,
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch + 1
        }
    
    def evaluate_feature_importance(self, data: np.ndarray) -> Dict[str, float]:
        """Evaluerer feature-viktighet ved å bruke attention-mekanismen"""
        if not self.config.use_feature_extraction:
            logger.warning("Feature extraction is disabled. Cannot evaluate feature importance.")
            return {}
        
        # Preprocess data
        scaled_data = self.scaler.transform(data)
        input_tensor = torch.FloatTensor(scaled_data).to(self.device)
        
        # Extract feature importance
        self.nn_model.eval()
        with torch.no_grad():
            _, attention_weights = self.nn_model(input_tensor)
        
        attention_weights = attention_weights.cpu().numpy()
        
        # Average across all samples
        avg_attention = np.mean(attention_weights, axis=0)
        
        # Map to feature names
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names[:len(avg_attention)]):
            feature_importance[feature_name] = float(avg_attention[i])
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        
        return feature_importance
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], top_n: int = 10) -> plt.Figure:
        """Plotter feature-viktighet"""
        # Sort and get top N features
        sorted_features = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:top_n])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar plot
        bars = ax.barh(
            list(sorted_features.keys()),
            list(sorted_features.values()),
            color='steelblue'
        )
        
        # Annotate bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                va='center'
            )
        
        # Set labels and title
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def save_model(self, path: str):
        """Lagrer modell, scaler og konfigurasjon"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save neural network model
        torch.save(self.nn_model.state_dict(), f"{path}_nn_model.pth")
        
        # Save scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Save configuration
        self.config.to_json(f"{path}_config.json")
        
        # Save ensemble models if available
        if self.config.use_ensemble and self.ensemble_models:
            for model_name, model in self.ensemble_models.items():
                try:
                    joblib.dump(model, f"{path}_{model_name}_model.joblib")
                except Exception as e:
                    logger.error(f"Error saving {model_name} model: {str(e)}")
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Laster modell, scaler og konfigurasjon"""
        # Load configuration
        config_path = f"{path}_config.json"
        if os.path.exists(config_path):
            self.config = PropertyOptimizationConfig.from_json(config_path)
            
            # Recreate neural network with loaded config
            self.nn_model = PropertyOptimizationModel(self.config).to(self.device)
        
        # Load neural network model
        self.nn_model.load_state_dict(torch.load(f"{path}_nn_model.pth", map_location=self.device))
        self.nn_model.eval()
        
        # Load scaler
        self.scaler = joblib.load(f"{path}_scaler.joblib")
        
        # Load ensemble models if enabled
        if self.config.use_ensemble:
            for model_name in self.config.ensemble_models:
                model_path = f"{path}_{model_name}_model.joblib"
                if os.path.exists(model_path):
                    try:
                        self.ensemble_models[model_name] = joblib.load(model_path)
                    except Exception as e:
                        logger.error(f"Error loading {model_name} model: {str(e)}")
        
        logger.info(f"Model loaded from {path}")
    
    def generate_optimization_report(self, property_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Genererer en komplett optimaliseringsrapport"""
        # Extract key results
        optimization_results = analysis_result.get('optimization_results', {})
        recommendations = analysis_result.get('recommendations', [])
        feature_importance = analysis_result.get('feature_importance', {})
        development_plan = analysis_result.get('development_plan', {})
        financial_projections = analysis_result.get('financial_projections', {})
        risk_analysis = analysis_result.get('risk_analysis', {})
        
        # Prepare property summary
        property_summary = {
            'address': property_data.get('address', 'Ikke spesifisert'),
            'plot_size': property_data.get('plot_size', 0),
            'existing_building_size': property_data.get('existing_building_size', 0),
            'current_utilization': property_data.get('existing_building_size', 0) / property_data.get('plot_size', 1) if property_data.get('plot_size', 0) > 0 else 0,
            'estimated_market_value': property_data.get('existing_building_size', 0) * property_data.get('market_value_per_sqm', 0),
            'energy_rating': property_data.get('energy_rating', 0),
            'technical_condition': property_data.get('technical_condition', 0)
        }
        
        # Create executive summary
        executive_summary = self._create_executive_summary(
            property_data, 
            optimization_results, 
            recommendations, 
            financial_projections
        )
        
        # Prepare visualization data (for frontend use)
        visualization_data = {
            'feature_importance': feature_importance,
            'financial_projections': {
                'labels': ['Nåverdi', 'Fremtidig verdi'],
                'values': [
                    property_summary['estimated_market_value'],
                    financial_projections.get('value_projection', {}).get('projected_future_value', 0)
                ]
            },
            'risk_analysis': {
                'categories': list(risk_analysis.get('risk_by_category', {}).keys()),
                'values': list(risk_analysis.get('risk_by_category', {}).values())
            }
        }
        
        # Complete report
        report = {
            'report_id': f"OPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'report_date': datetime.now().isoformat(),
            'property_summary': property_summary,
            'executive_summary': executive_summary,
            'optimization_results': optimization_results,
            'top_recommendations': recommendations[:3] if len(recommendations) > 3 else recommendations,
            'development_plan': development_plan,
            'financial_analysis': {
                'projected_value': financial_projections.get('value_projection', {}),
                'roi_analysis': financial_projections.get('roi_analysis', {}),
                'recommended_funding': financial_projections.get('funding_options', [])[0] if financial_projections.get('funding_options', []) else {}
            },
            'risk_assessment': {
                'overall_risk': risk_analysis.get('overall_risk_level', {}),
                'major_risk_areas': risk_analysis.get('major_risk_areas', []),
                'recommended_contingency': risk_analysis.get('recommended_contingency', '')
            },
            'visualization_data': visualization_data,
            'confidence_score': analysis_result.get('confidence_score', 0)
        }
        
        return report
    
    def _create_executive_summary(
        self,
        property_data: Dict[str, Any],
        optimization_results: Dict[str, float],
        recommendations: List[Dict[str, Any]],
        financial_projections: Dict[str, Any]
    ) -> str:
        """Lager en kort oppsummering av analysens hovedfunn"""
        # Extract key metrics
        optimal_utilization = optimization_results.get('optimal_utilization', 0)
        economic_potential = optimization_results.get('economic_potential', 0)
        roi_timeline = optimization_results.get('roi_timeline', 0)
        
        current_value = property_data.get('existing_building_size', 0) * property_data.get('market_value_per_sqm', 0)
        future_value = financial_projections.get('value_projection', {}).get('projected_future_value', 0)
        value_increase = future_value - current_value
        
        # Create summary text
        if economic_potential > 0.7:
            potential_desc = "betydelig"
        elif economic_potential > 0.4:
            potential_desc = "moderat"
        else:
            potential_desc = "begrenset"
            
        if optimal_utilization > 0.8:
            utilization_desc = "svært høy"
        elif optimal_utilization > 0.6:
            utilization_desc = "høy"
        elif optimal_utilization > 0.4:
            utilization_desc = "moderat"
        else:
            utilization_desc = "lav"
            
        # Get top recommendation
        top_rec = recommendations[0] if recommendations else {}
        
        summary = (
            f"Eiendommen har {potential_desc} utviklingspotensial med en {utilization_desc} optimal "
            f"utnyttelsesgrad på {optimal_utilization:.2f}. Den estimerte verdiøkningen er {value_increase:,.0f} NOK "
            f"med en estimert ROI-tidshorisont på {roi_timeline:.1f} år. "
            f"Den mest anbefalte handlingen er: {top_rec.get('title', 'Ingen spesifikk anbefaling')}."
        )
        
        return summary
