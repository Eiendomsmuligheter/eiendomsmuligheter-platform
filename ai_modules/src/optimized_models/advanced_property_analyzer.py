import torch
import torch.nn as nn
from transformers import LayoutLMv2Model, AutoFeatureExtractor
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
from dataclasses import dataclass
import tensorflow as tf
import asyncio
import logging
from scipy.optimize import minimize
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Konfigurasjon for analysemodellen"""
    image_size: tuple = (1024, 1024)
    batch_size: int = 8
    num_channels: int = 3
    num_classes: int = 12  # Utvidet til flere klasser
    learning_rate: float = 5e-5  # Optimalisert læringsrate
    weight_decay: float = 2e-6
    dropout_rate: float = 0.3  # Økt dropout for bedre generalisering
    use_mixed_precision: bool = True  # Aktiver mixed precision for raskere beregning
    enable_augmentation: bool = True  # Aktiver data-augmentering
    model_version: str = "2.5.0"  # For versjonskontroll
    use_ensemble: bool = True  # Bruk ensemble-læring for bedre resultater
    cache_features: bool = True  # Mellomlagre features for raskere gjentatte analyser
    multitask_learning: bool = True  # Aktiver multitask-læring
    market_data_integration: bool = True  # Integrer markedsdata i analysen

class PropertyFeatureExtractor(nn.Module):
    def __init__(self, config: AnalysisConfig):
        super().__init__()
        self.config = config
        
        # Oppgradert backbone til mer avansert arkitektur
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b7', pretrained=True)
        
        # Adaptivt feature extraction
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Utvidet feature pyramid network
        self.feature_pyramid = nn.ModuleList([
            nn.Conv2d(2560, 512, kernel_size=1),  # EfficientNetB7 har 2560 kanaler
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.Conv2d(640, 512, kernel_size=1),
            nn.Conv2d(320, 512, kernel_size=1),
            nn.Conv2d(160, 512, kernel_size=1)
        ])
        
        # Forbedret attention mekanisme
        self.attention = nn.MultiheadAttention(512, num_heads=16, dropout=config.dropout_rate)
        
        # Mer sofistikert feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 * 5, 1024),
            nn.SiLU(),  # Nyere aktivering (Sigmoid Linear Unit)
            nn.Dropout(config.dropout_rate),
            nn.Linear(1024, 768),
            nn.SiLU(),
            nn.Dropout(config.dropout_rate * 0.8),
            nn.Linear(768, 512)
        )
        
        # Nye moduler for spesifikke egenskaper
        self.spatial_understanding = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=config.dropout_rate),
            num_layers=4
        )
        
        # Feature normalisering
        self.layer_norm = nn.LayerNorm(512)
        
        # Adaptiv pooling for variable størrelse på input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def extract_backbone_features(self, x):
        """Ekstrakt features fra backbone-nettverket"""
        # Få tilgang til intermediate features fra EfficientNet
        features = []
        
        # For EfficientNet struktur
        x = self.backbone.features[0](x)
        features.append(x)
        
        x = self.backbone.features[1](x)
        features.append(x)
        
        x = self.backbone.features[2](x)
        features.append(x)
        
        x = self.backbone.features[3](x)
        features.append(x)
        
        x = self.backbone.features[4](x)
        features.append(x)
        
        return features

    def forward(self, x):
        # Ekstraher features på ulike nivåer
        features = self.extract_backbone_features(x)
        
        # Anvend feature pyramid
        pyramid_features = [
            self.feature_pyramid[i](features[i])
            for i in range(len(self.feature_pyramid))
        ]
        
        # Global adaptive pooling på hver feature map
        pooled_features = [
            self.adaptive_pool(f).view(f.size(0), -1) for f in pyramid_features
        ]
        
        # Stablende features for attention
        stacked_features = torch.stack(pooled_features, dim=0)
        
        # Self-attention med residual connection
        attended_features, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        attended_features = attended_features + stacked_features  # Residual connection
        
        # Transformer encoder for spatial understanding
        spatial_features = self.spatial_understanding(attended_features)
        
        # Fusion
        features = spatial_features.transpose(0, 1).reshape(-1, 512 * 5)
        features = self.fusion(features)
        
        # Normaliser output features
        features = self.layer_norm(features)
        
        return features

class AdvancedPropertyAnalyzer:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialiserer den avanserte eiendomsanalysatoren med konfigurasjon
        
        Args:
            config_path: Valgfri sti til konfigurasjonsfil
        """
        # Last tilpasset konfigurasjon hvis tilgjengelig
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        else:
            self.config = AnalysisConfig()
        
        logger.info(f"Initialiserer AdvancedPropertyAnalyzer v{self.config.model_version}")
        
        # Initialiserer feature extractor med forbedret arkitektur
        self.feature_extractor = PropertyFeatureExtractor(self.config)
        
        # Oppgrader til større LayoutLM-modell for bedre dokumentforståelse
        self.layout_model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-large-uncased")
        self.feature_processor = AutoFeatureExtractor.from_pretrained("microsoft/layoutlmv2-large-uncased")
        
        # Spesialiserte analysemoduler med ensemble-læring
        self.room_detector = self._load_room_detector()
        self.dimension_analyzer = self._load_dimension_analyzer()
        self.material_analyzer = self._load_material_analyzer()
        self.structure_analyzer = self._load_structure_analyzer()
        self.energy_analyzer = self._load_energy_analyzer()
        self.value_estimator = self._load_value_estimator()
        self.rental_potential_analyzer = self._load_rental_potential_analyzer()
        self.renovation_cost_estimator = self._load_renovation_cost_estimator()
        
        # Nye analysemoduler for enda mer verdi
        self.market_trends_analyzer = self._load_market_trends_analyzer()
        self.neighborhood_analyzer = self._load_neighborhood_analyzer()
        self.regulatory_compliance_checker = self._load_regulatory_compliance_checker()
        self.sustainability_analyzer = self._load_sustainability_analyzer()
        
        # Cache for mellomlagring av resultater
        self.analysis_cache = {}
        
        # GPU-akselerasjon hvis tilgjengelig
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Bruker {self.device} for beregninger")
        
        # Mixed precision for raskere beregning
        if self.config.use_mixed_precision and self.device == torch.device("cuda"):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # Flytt modeller til GPU
        self.feature_extractor.to(self.device)
        self.layout_model.to(self.device)
        
        # Thread pool for parallell prosessering
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        logger.info("AdvancedPropertyAnalyzer initalisering fullført")

    def _load_config(self, config_path: str) -> AnalysisConfig:
        """Laster konfigurasjon fra fil"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return AnalysisConfig(**config_dict)

    def _load_room_detector(self):
        """Laster spesialisert romdeteksjonsmodell med ensemble-arkitektur"""
        if self.config.use_ensemble:
            # Ensemble av modeller for romdeteksjon
            models = []
            
            # Basismodell
            models.append(self._create_room_detector_model())
            
            # Modell spesialisert for åpen planløsning
            models.append(self._create_specialized_room_detector("open_plan"))
            
            # Modell spesialisert for tradisjonell planløsning
            models.append(self._create_specialized_room_detector("traditional"))
            
            return models
        else:
            return self._create_room_detector_model()

    def _create_room_detector_model(self):
        """Oppretter en avansert romdeteksjonsmodell"""
        # Bruk EfficientNetV2 for bedre ytelse
        base_model = tf.keras.applications.EfficientNetV2L(
            include_top=False,
            weights='imagenet',
            input_shape=(self.config.image_size[0], self.config.image_size[1], self.config.num_channels)
        )
        
        # Legg til custom layers for romdeteksjon
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Legg til kontekstbevisst feature extractor
        context_branch = tf.keras.layers.Dense(1024, activation='swish')(x)
        context_branch = tf.keras.layers.Dropout(self.config.dropout_rate)(context_branch)
        
        # Spatial branch
        spatial_branch = tf.keras.layers.Dense(1024, activation='swish')(x)
        spatial_branch = tf.keras.layers.Dropout(self.config.dropout_rate)(spatial_branch)
        
        # Kombinere branches med attention
        attention_weights = tf.keras.layers.Dense(2, activation='softmax')(x)
        combined = (attention_weights[:, 0:1] * context_branch) + (attention_weights[:, 1:2] * spatial_branch)
        
        x = tf.keras.layers.Dense(1536, activation='swish')(combined)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        x = tf.keras.layers.Dense(768, activation='swish')(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate * 0.8)(x)
        
        # Output layers for ulike aspekter
        room_type = tf.keras.layers.Dense(self.config.num_classes, activation='softmax', name='room_type')(x)
        room_size = tf.keras.layers.Dense(3, activation='linear', name='room_size')(x)
        room_features = tf.keras.layers.Dense(25, activation='sigmoid', name='room_features')(x)
        room_quality = tf.keras.layers.Dense(5, activation='softmax', name='room_quality')(x)
        room_condition = tf.keras.layers.Dense(5, activation='softmax', name='room_condition')(x)
        
        model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[room_type, room_size, room_features, room_quality, room_condition]
        )
        
        # Kompiler modellen med custom loss functions
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss={
                'room_type': 'categorical_crossentropy',
                'room_size': 'mean_squared_error',
                'room_features': 'binary_crossentropy',
                'room_quality': 'categorical_crossentropy',
                'room_condition': 'categorical_crossentropy'
            },
            metrics={
                'room_type': 'accuracy',
                'room_size': 'mae',
                'room_features': 'accuracy',
                'room_quality': 'accuracy',
                'room_condition': 'accuracy'
            }
        )
        
        return model

    def _create_specialized_room_detector(self, specialization: str):
        """Oppretter en spesialisert romdeteksjonsmodell for spesifikke typer planløsninger"""
        # Start med basismodell og juster parametre
        model = self._create_room_detector_model()
        
        # Juster vekter basert på spesialisering (i praksis ville disse være forhåndstrent)
        if specialization == "open_plan":
            # Hypotetisk justering for åpne planløsninger
            logger.info("Laster spesialisert modell for åpne planløsninger")
        elif specialization == "traditional":
            # Hypotetisk justering for tradisjonelle planløsninger
            logger.info("Laster spesialisert modell for tradisjonelle planløsninger")
        
        return model

    def _load_dimension_analyzer(self):
        """Laster oppgradert modell for dimensjonsanalyse"""
        # Bruk YOLOv7 istedenfor YOLOv5 for bedre presisjon
        try:
            model = torch.hub.load('WongKinYiu/yolov7', 'custom', path='models/dimension_analyzer_v2.pt')
        except Exception as e:
            logger.warning(f"Kunne ikke laste YOLOv7 modell: {e}. Faller tilbake til YOLOv5.")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/dimension_analyzer.pt')
        
        model.conf = 0.6  # Økt terskel for bedre presisjon
        model.iou = 0.5   # Justert NMS IoU terskel
        
        # Konfigurer for å inkludere presisjonsmålinger
        model.agnostic = True  # Klasseagnostisk NMS
        model.max_det = 300    # Maksimalt antall deteksjoner
        
        return model

    def _load_material_analyzer(self):
        """Laster oppgradert modell for materialanalyse"""
        try:
            # Prøv å laste nyere modell først
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/material_analyzer_v2.pt')
        except Exception as e:
            logger.warning(f"Kunne ikke laste nyere materialanalysemodell: {e}. Faller tilbake til original.")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/material_analyzer.pt')
        
        model.conf = 0.5  # Justert terskel for materialdeteksjon
        model.iou = 0.5
        
        return model

    def _load_structure_analyzer(self):
        """Laster oppgradert modell for strukturanalyse"""
        try:
            # Prøv å laste nyere modell først
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/structure_analyzer_v3.pt')
        except Exception as e:
            logger.warning(f"Kunne ikke laste nyere strukturanalysemodell: {e}. Faller tilbake til original.")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/structure_analyzer.pt')
        
        model.conf = 0.65  # Høyere terskel for strukturelle elementer for bedre presisjon
        model.iou = 0.45
        
        return model

    def _load_energy_analyzer(self):
        """Laster spesialisert modell for energianalyse"""
        # Gradient Boosting Regressor for energiberegninger
        model = GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        
        # Last forhåndstrente vekter hvis de eksisterer
        try:
            model = self._load_model('models/energy_analyzer.pkl')
            logger.info("Energianalysemodell lastet vellykket")
        except FileNotFoundError:
            logger.warning("Kunne ikke finne forhåndstrent energianalysemodell. Bruker utrenet modell.")
        
        return model

    def _load_value_estimator(self):
        """Laster modell for verdiestimering"""
        # Ensemble av Random Forest og Gradient Boosting for bedre prediksjoner
        rf_model = RandomForestRegressor(
            n_estimators=150, 
            max_depth=10, 
            random_state=42
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=150, 
            learning_rate=0.1, 
            max_depth=6, 
            random_state=42
        )
        
        # Last forhåndstrente vekter hvis de eksisterer
        try:
            rf_model = self._load_model('models/value_estimator_rf.pkl')
            gb_model = self._load_model('models/value_estimator_gb.pkl')
            logger.info("Verdiestimatormodeller lastet vellykket")
        except FileNotFoundError:
            logger.warning("Kunne ikke finne forhåndstrente verdiestimatormodeller. Bruker utrenede modeller.")
        
        return {'rf': rf_model, 'gb': gb_model}

    def _load_rental_potential_analyzer(self):
        """Laster modell for analyse av leiepotensial"""
        # Gradient Boosting Regressor for leieprisprediksjoner
        model = GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42
        )
        
        # Last forhåndstrente vekter hvis de eksisterer
        try:
            model = self._load_model('models/rental_potential_analyzer.pkl')
            logger.info("Leieanalytiker-modell lastet vellykket")
        except FileNotFoundError:
            logger.warning("Kunne ikke finne forhåndstrent leieanalytiker-modell. Bruker utrenet modell.")
        
        return model

    def _load_renovation_cost_estimator(self):
        """Laster modell for estimering av renoveringskostnader"""
        # Gradient Boosting Regressor for kostnadsestimering
        model = GradientBoostingRegressor(
            n_estimators=250, 
            learning_rate=0.03, 
            max_depth=7, 
            random_state=42
        )
        
        # Last forhåndstrente vekter hvis de eksisterer
        try:
            model = self._load_model('models/renovation_cost_estimator.pkl')
            logger.info("Kostnadsestimator-modell lastet vellykket")
        except FileNotFoundError:
            logger.warning("Kunne ikke finne forhåndstrent kostnadsestimator-modell. Bruker utrenet modell.")
        
        return model

    def _load_market_trends_analyzer(self):
        """Laster markedstrendsanalysator"""
        # Last inn markedsdata hvis tilgjengelig
        try:
            market_data = pd.read_csv('data/market_trends.csv')
            logger.info("Markedsdata lastet vellykket")
        except FileNotFoundError:
            market_data = None
            logger.warning("Kunne ikke finne markedsdata. Markedstrendsanalyse vil være begrenset.")
        
        return market_data

    def _load_neighborhood_analyzer(self):
        """Laster nabolagsanalysator"""
        # Last nabolagsdata hvis tilgjengelig
        try:
            neighborhood_data = pd.read_csv('data/neighborhood_data.csv')
            logger.info("Nabolagsdata lastet vellykket")
        except FileNotFoundError:
            neighborhood_data = None
            logger.warning("Kunne ikke finne nabolagsdata. Nabolagsanalyse vil være begrenset.")
        
        return neighborhood_data

    def _load_regulatory_compliance_checker(self):
        """Laster regulatorisk samsvarskontroll"""
        # Last regelverksdata hvis tilgjengelig
        try:
            regulatory_data = json.load(open('data/regulatory_requirements.json', 'r'))
            logger.info("Regulatoriske data lastet vellykket")
        except FileNotFoundError:
            regulatory_data = {}
            logger.warning("Kunne ikke finne regulatoriske data. Samsvarskontroll vil være begrenset.")
        
        return regulatory_data

    def _load_sustainability_analyzer(self):
        """Laster bærekraftsanalysator"""
        # Last bærekraftsdata hvis tilgjengelig
        try:
            sustainability_data = json.load(open('data/sustainability_metrics.json', 'r'))
            logger.info("Bærekraftsdata lastet vellykket")
        except FileNotFoundError:
            sustainability_data = {}
            logger.warning("Kunne ikke finne bærekraftsdata. Bærekraftsanalyse vil være begrenset.")
        
        return sustainability_data

    def _load_model(self, path: str):
        """Hjelpefunksjon for å laste modeller fra disk"""
        import pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def _load_and_preprocess_image(self, image_path: str):
        """Laster og preprosesserer bilde for analyse"""
        # Sjekk om bildet eksisterer
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Bildefil ikke funnet: {image_path}")
        
        # Last bilde
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Kunne ikke lese bildefil: {image_path}")
        
        # Konverter fra BGR til RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize til konfigurasjonsstørrelse
        image = cv2.resize(image, self.config.image_size)
        
        # Normaliser bildet
        image = image.astype(np.float32) / 255.0
        
        # Data augmentering hvis aktivert
        if self.config.enable_augmentation and np.random.random() < 0.5:
            image = self._apply_augmentation(image)
        
        # Konverter til tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        
        return image_tensor.to(self.device)

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Anvender data-augmentering på bildet"""
        # Implementer tilfeldige transformasjoner
        # Dette er en enkel implementasjon - kan utvides
        
        # Tilfeldig justering av lysstyrke
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 1)
        
        # Tilfeldig justering av kontrast
        contrast_factor = np.random.uniform(0.8, 1.2)
        image = np.clip((image - 0.5) * contrast_factor + 0.5, 0, 1)
        
        return image

    async def analyze_property(self, image_path: str, property_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Utfører komplett analyse av eiendom
        
        Args:
            image_path: Sti til bildet som skal analyseres
            property_data: Valgfri ekstra informasjon om eiendommen
            
        Returns:
            Dict med analyseresultater
        """
        # Sjekk cache først hvis aktivert
        cache_key = f"{image_path}_{hash(str(property_data))}"
        if self.config.cache_features and cache_key in self.analysis_cache:
            logger.info(f"Bruker mellomlagret analyse for {image_path}")
            return self.analysis_cache[cache_key]
        
        # Start tidtaking
        start_time = datetime.now()
        logger.info(f"Starter analyse av eiendom: {image_path}")
        
        # Last og preprocess bilde
        try:
            image = self._load_and_preprocess_image(image_path)
        except Exception as e:
            logger.error(f"Feil ved lasting av bilde: {e}")
            return {'error': f"Kunne ikke laste bilde: {str(e)}"}
        
        # Parallell prosessering av ulike analyser
        try:
            results = await asyncio.gather(
                self._analyze_rooms(image),
                self._analyze_dimensions(image),
                self._analyze_materials(image),
                self._analyze_structure(image),
                self._analyze_layout(image)
            )
            
            room_analysis, dimension_analysis, material_analysis, structure_analysis, layout_analysis = results
            
            # Beregn energianalyse
            energy_analysis = self._analyze_energy_efficiency(
                material_analysis,
                structure_analysis
            )
            
            # Beregn utviklingspotensial
            development_potential = self._calculate_development_potential(
                room_analysis,
                dimension_analysis,
                structure_analysis,
                layout_analysis
            )
            
            # Beregn verdianalyse
            value_analysis = self._analyze_property_value(
                room_analysis,
                dimension_analysis,
                material_analysis,
                energy_analysis,
                property_data
            )
            
            # Beregn leiepotensial
            rental_analysis = self._analyze_rental_potential(
                room_analysis,
                dimension_analysis,
                location_data=property_data.get('location') if property_data else None
            )
            
            # Beregn renoveringspotensial og ROI
            renovation_analysis = self._analyze_renovation_potential(
                room_analysis,
                material_analysis,
                structure_analysis,
                energy_analysis,
                value_analysis
            )
            
            # Generer anbefalinger
            recommendations = self._generate_recommendations(
                room_analysis,
                material_analysis,
                structure_analysis,
                energy_analysis,
                value_analysis,
                rental_analysis,
                renovation_analysis,
                development_potential
            )
            
            # Markedsanalyse hvis data er tilgjengelig
            market_analysis = self._analyze_market_trends(
                property_data.get('location') if property_data else None
            )
            
            # Nabolagsanalyse hvis data er tilgjengelig
            neighborhood_analysis = self._analyze_neighborhood(
                property_data.get('location') if property_data else None
            )
            
            # Regulatorisk samsvarskontroll
            compliance_analysis = self._check_regulatory_compliance(
                room_analysis,
                dimension_analysis,
                property_data
            )
            
            # Bærekraftsanalyse
            sustainability_analysis = self._analyze_sustainability(
                material_analysis,
                energy_analysis,
                property_data
            )
        
        except Exception as e:
            logger.error(f"Feil under analyse: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f"Analysefeil: {str(e)}"}
        
        # Samle alt i én komplett analyse
        complete_analysis = {
            'property_details': {
                'rooms': room_analysis,
                'dimensions': dimension_analysis,
                'materials': material_analysis,
                'structure': structure_analysis,
                'layout': layout_analysis
            },
            'performance_analysis': {
                'energy': energy_analysis,
                'value': value_analysis,
                'rental': rental_analysis,
                'renovation': renovation_analysis,
                'development': development_potential,
                'sustainability': sustainability_analysis
            },
            'market_analysis': {
                'trends': market_analysis,
                'neighborhood': neighborhood_analysis
            },
            'recommendations': {
                'prioritized': recommendations.get('prioritized', []),
                'by_category': recommendations.get('by_category', {}),
                'roi_optimized': recommendations.get('roi_optimized', [])
            },
            'compliance': {
                'regulatory': compliance_analysis,
                'building_code': self._check_building_code_compliance(
                    room_analysis,
                    dimension_analysis
                ),
                'accessibility': self._analyze_accessibility(
                    room_analysis,
                    dimension_analysis
                )
            },
            'visualization_data': {
                '3d_model_parameters': self._generate_3d_model_parameters(
                    room_analysis,
                    dimension_analysis
                ),
                'material_mappings': self._generate_material_mappings(
                    material_analysis
                ),
                'lighting_data': self._analyze_lighting(
                    room_analysis,
                    structure_analysis
                ),
                'renovation_visualization': self._generate_renovation_visualization(
                    recommendations,
                    room_analysis,
                    dimension_analysis
                )
            },
            'meta': {
                'analysis_version': self.config.model_version,
                'analysis_date': datetime.now().isoformat(),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'confidence_score': self._calculate_confidence_score(
                    room_analysis,
                    dimension_analysis,
                    material_analysis,
                    structure_analysis
                )
            }
        }
        
        # Mellomlagre analysen hvis caching er aktivert
        if self.config.cache_features:
            self.analysis_cache[cache_key] = complete_analysis
        
        logger.info(f"Analyse fullført på {(datetime.now() - start_time).total_seconds():.2f} sekunder")
        
        return complete_analysis

    async def _analyze_rooms(self, image: torch.Tensor) -> Dict[str, Any]:
        """Analyserer rom i bildet"""
        # Konverter til format for romdeteksjonsmodell
        if isinstance(image, torch.Tensor):
            np_image = image.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = image
        
        # Bruk ensemble hvis konfigurert
        if self.config.use_ensemble and isinstance(self.room_detector, list):
            # Ensemble av modeller
            all_predictions = []
            
            for model in self.room_detector:
                predictions = model.predict(np.expand_dims(np_image, axis=0))
                all_predictions.append(predictions)
            
            # Slå sammen prediksjoner med vekting
            # For romtype: gjennomsnittlig softmax-sannsynlighet
            room_type_probas = np.mean([pred[0] for pred in all_predictions], axis=0)
            room_types = np.argmax(room_type_probas, axis=1)
            
            # For romstørrelse: gjennomsnittlig prediksjon
            room_sizes = np.mean([pred[1] for pred in all_predictions], axis=0)
            
            # For romegenskaper: gjennomsnittlig prediksjon med terskel
            room_features_probas = np.mean([pred[2] for pred in all_predictions], axis=0)
            room_features = (room_features_probas > 0.5).astype(np.int32)
            
            # For kvalitet og tilstand: gjennomsnittlig softmax-sannsynlighet
            room_quality_probas = np.mean([pred[3] for pred in all_predictions], axis=0)
            room_qualities = np.argmax(room_quality_probas, axis=1)
            
            room_condition_probas = np.mean([pred[4] for pred in all_predictions], axis=0)
            room_conditions = np.argmax(room_condition_probas, axis=1)
            
        else:
            # Enkeltmodell
            if isinstance(self.room_detector, list):
                model = self.room_detector[0]  # Bruk første modell hvis liste
            else:
                model = self.room_detector
                
            predictions = model.predict(np.expand_dims(np_image, axis=0))
            room_types = np.argmax(predictions[0], axis=1)
            room_sizes = predictions[1]
            room_features = (predictions[2] > 0.5).astype(np.int32)
            room_qualities = np.argmax(predictions[3], axis=1)
            room_conditions = np.argmax(predictions[4], axis=1)
        
        # Mapping av prediksjoner til faktiske verdier
        room_type_mapping = {
            0: 'kitchen',
            1: 'living_room',
            2: 'bedroom',
            3: 'bathroom',
            4: 'hallway',
            5: 'dining_room',
            6: 'office',
            7: 'basement',
            8: 'attic',
            9: 'garage',
            10: 'balcony',
            11: 'other'
        }
        
        room_feature_mapping = [
            'windows', 'natural_light', 'built_in_storage', 'hardwood_floor', 'carpet',
            'tile', 'high_ceiling', 'open_layout', 'fireplace', 'modern_fixtures',
            'outdated_fixtures', 'good_insulation', 'poor_insulation', 'heating_system',
            'air_conditioning', 'moisture_issues', 'water_damage', 'structural_issues',
            'renovation_potential', 'energy_efficient', 'smart_home_capability',
            'accessibility_features', 'noise_insulation', 'safety_features', 'eco_friendly_materials'
        ]
        
        quality_mapping = {
            0: 'poor',
            1: 'below_average',
            2: 'average',
            3: 'good',
            4: 'excellent'
        }
        
        condition_mapping = {
            0: 'needs_major_renovation',
            1: 'needs_minor_renovation',
            2: 'average',
            3: 'good',
            4: 'excellent'
        }
        
        # Konverter numeriske prediksjoner til meningsfulle verdier
        room_info = {
            'type': room_type_mapping[room_types[0]],
            'size': {
                'area': float(room_sizes[0][0]),
                'width': float(room_sizes[0][1]),
                'length': float(room_sizes[0][2])
            },
            'features': [room_feature_mapping[i] for i in range(len(room_features[0])) if room_features[0][i] == 1],
            'quality': quality_mapping[room_qualities[0]],
            'condition': condition_mapping[room_conditions[0]],
            'confidence': float(np.max(room_type_probas[0]) if self.config.use_ensemble else np.max(predictions[0][0]))
        }
        
        # Legg til estimert takhøyde
        room_info['ceiling_height'] = self._estimate_ceiling_height(np_image)
        
        # Beregn potensiell bruk basert på romtype og egenskaper
        room_info['potential_uses'] = self._identify_potential_room_uses(room_info)
        
        return room_info

    def _estimate_ceiling_height(self, image: np.ndarray) -> float:
        """Estimerer takhøyde fra bilde (forenklet implementasjon)"""
        # Dette er en placeholder-implementasjon
        # I praksis ville dette involvere mer avansert bildeanalyse
        
        # Basert på romstørrelse, proporsjoner, dørhøyde, osv.
        # Her returnerer vi bare en standard verdi
        return 240.0  # cm

    def _identify_potential_room_uses(self, room_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifiserer potensielle bruksområder for rommet"""
        potential_uses = []
        
        room_type = room_info['type']
        features = room_info['features']
        area = room_info['size']['area']
        condition = room_info['condition']
        
        # Vurder potensiale for hjemmekontor
        if area >= 8 and 'natural_light' in features:
            potential_uses.append({
                'use_type': 'home_office',
                'suitability': 'high' if area >= 12 else 'medium',
                'requirements': ['desk', 'chair', 'internet'] if area < 12 else []
            })
        
        # Vurder potensiale for utleie (avhengig av romtype)
        if room_type in ['bedroom', 'basement', 'attic'] and area >= 10:
            potential_uses.append({
                'use_type': 'rental',
                'suitability': 'high' if area >= 15 and condition in ['good', 'excellent'] else 'medium',
                'requirements': ['separate_entrance', 'privacy'] if room_type != 'bedroom' else ['privacy']
            })
        
        # Vurder potensiale for treningsrom
        if area >= 8:
            potential_uses.append({
                'use_type': 'gym',
                'suitability': 'high' if area >= 15 else 'medium',
                'requirements': ['ventilation', 'noise_insulation'] if 'noise_insulation' not in features else ['ventilation']
            })
        
        # Vurder potensiale for endring til annen romtype
        if room_type == 'bedroom' and area >= 20:
            potential_uses.append({
                'use_type': 'bedroom_with_ensuite',
                'suitability': 'medium',
                'requirements': ['plumbing', 'ventilation']
            })
        
        elif room_type == 'living_room' and area >= 30:
            potential_uses.append({
                'use_type': 'open_plan_living_dining',
                'suitability': 'high',
                'requirements': ['wall_removal'] if 'open_layout' not in features else []
            })
        
        return potential_uses

    async def _analyze_dimensions(self, image: torch.Tensor) -> Dict[str, Any]:
        """Analyserer dimensjoner i bildet"""
        # Konverter tensor til format for dimensjonsmodellen
        if isinstance(image, torch.Tensor):
            np_image = image.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = image
        
        # Kjør dimensjonsanalyse
        results = self.dimension_analyzer(np_image)
        
        # Prosesser resultatene
        dimensions = {
            'detected_objects': [],
            'room_measurements': {},
            'scale_factor': 1.0,  # Pixeler til cm
        }
        
        # Tolk deteksjoner
        detections = results.xyxy[0].cpu().numpy()
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            obj_class = int(cls)
            
            # Eksempel på klasser: 0=dør, 1=vindu, 2=vegg, 3=gulv, osv.
            class_mapping = {
                0: 'door',
                1: 'window',
                2: 'wall',
                3: 'floor',
                4: 'ceiling',
                5: 'counter',
                6: 'cabinet',
                7: 'staircase',
                8: 'column',
                9: 'furniture'
            }
            
            obj_type = class_mapping.get(obj_class, 'unknown')
            
            # Beregn dimensjoner basert på bounding box
            width = x2 - x1
            height = y2 - y1
            
            # Kalibrer med kjente objektstørrelser (f.eks. standard dørhøyde)
            if obj_type == 'door' and conf > 0.8:
                # Standard dørhøyde er ca 200 cm
                dimensions['scale_factor'] = 200.0 / height
            
            # Legg til detektert objekt
            dimensions['detected_objects'].append({
                'type': obj_type,
                'confidence': float(conf),
                'bounding_box': [float(x1), float(y1), float(x2), float(y2)],
                'estimated_dimensions': {
                    'width': float(width * dimensions['scale_factor']),
                    'height': float(height * dimensions['scale_factor'])
                }
            })
        
        # Beregn romstørrelse basert på gulvdeteksjon
        floor_objects = [obj for obj in dimensions['detected_objects'] if obj['type'] == 'floor']
        if floor_objects:
            floor = max(floor_objects, key=lambda x: (x['bounding_box'][2] - x['bounding_box'][0]) * (x['bounding_box'][3] - x['bounding_box'][1]))
            floor_width = (floor['bounding_box'][2] - floor['bounding_box'][0]) * dimensions['scale_factor']
            floor_length = (floor['bounding_box'][3] - floor['bounding_box'][1]) * dimensions['scale_factor']
            
            dimensions['room_measurements'] = {
                'floor_area': floor_width * floor_length / 10000,  # konverter til kvadratmeter
                'width': floor_width / 100,  # konverter til meter
                'length': floor_length / 100,  # konverter til meter
            }
        
        # Beregn takhøyde basert på vegg- og takdeteksjoner
        wall_objects = [obj for obj in dimensions['detected_objects'] if obj['type'] == 'wall']
        if wall_objects:
            avg_wall_height = np.mean([obj['estimated_dimensions']['height'] for obj in wall_objects])
            dimensions['room_measurements']['ceiling_height'] = avg_wall_height / 100  # konverter til meter
        
        return dimensions

    async def _analyze_materials(self, image: torch.Tensor) -> Dict[str, Any]:
        """Analyserer materialer i bildet"""
        # Konverter tensor til format for materialmodellen
        if isinstance(image, torch.Tensor):
            np_image = image.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = image
        
        # Kjør materialanalyse
        results = self.material_analyzer(np_image)
        
        # Prosesser resultatene
        materials = {
            'detected_materials': [],
            'surface_coverage': {},
            'quality_assessment': {},
            'sustainability_score': 0.0
        }
        
        # Tolk deteksjoner
        detections = results.xyxy[0].cpu().numpy()
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            material_class = int(cls)
            
            # Eksempel på materialkategorier
            material_mapping = {
                0: 'hardwood',
                1: 'laminate',
                2: 'tile',
                3: 'carpet',
                4: 'concrete',
                5: 'stone',
                6: 'marble',
                7: 'vinyl',
                8: 'drywall',
                9: 'brick',
                10: 'glass',
                11: 'metal',
                12: 'plastic',
                13: 'ceramic',
                14: 'granite',
                15: 'formica'
            }
            
            material_type = material_mapping.get(material_class, 'unknown')
            
            # Beregn areal av materialet
            area = (x2 - x1) * (y2 - y1)
            
            # Evaluer materialkvalitet basert på utseende
            quality = self._evaluate_material_quality(np_image[int(y1):int(y2), int(x1):int(x2)], material_type)
            
            # Beregn bærekraftsscore for materialet
            sustainability = self._calculate_material_sustainability(material_type, quality)
            
            # Legg til detektert materiale
            materials['detected_materials'].append({
                'type': material_type,
                'confidence': float(conf),
                'bounding_box': [float(x1), float(y1), float(x2), float(y2)],
                'area_pixels': float(area),
                'quality': quality,
                'sustainability': sustainability
            })
        
        # Beregn overflatedekning for hvert materiale
        total_area = np_image.shape[0] * np_image.shape[1]
        for material in set([m['type'] for m in materials['detected_materials']]):
            material_areas = [m['area_pixels'] for m in materials['detected_materials'] if m['type'] == material]
            materials['surface_coverage'][material] = sum(material_areas) / total_area
        
        # Beregn gjennomsnittlig kvalitetsvurdering
        for material in set([m['type'] for m in materials['detected_materials']]):
            material_qualities = [m['quality']['score'] for m in materials['detected_materials'] if m['type'] == material]
            materials['quality_assessment'][material] = np.mean(material_qualities)
        
        # Beregn total bærekraftsscore
        sustainability_scores = [m['sustainability']['score'] * m['area_pixels'] for m in materials['detected_materials']]
        total_material_area = sum([m['area_pixels'] for m in materials['detected_materials']])
        if total_material_area > 0:
            materials['sustainability_score'] = sum(sustainability_scores) / total_material_area
        
        # Legg til anbefalinger for materialer som kan oppgraderes
        materials['upgrade_recommendations'] = self._generate_material_upgrade_recommendations(materials)
        
        return materials

    def _evaluate_material_quality(self, material_image: np.ndarray, material_type: str) -> Dict[str, Any]:
        """Evaluerer kvaliteten på materialet basert på bildeanalyse"""
        # Placeholder for avansert materialanalyse
        # I praksis ville dette involvere teksturanalyse, fargekonsistens, etc.
        
        # Eksempel på en enkel kvalitetsberegning
        # Basert på variasjoner i tekstur og farger
        
        # Konverter til gråtone
        gray = cv2.cvtColor(material_image, cv2.COLOR_RGB2GRAY)
        
        # Beregn teksturvariasjon (standardavvik)
        texture_variation = np.std(gray)
        
        # Beregn fargevariasjon
        color_variation = np.std(material_image, axis=(0, 1)).mean()
        
        # Beregn kvalitetsscore basert på tekstur og farge
        # Forskjellige materialer har forskjellige forventede verdier
        
        quality_score = 0.5  # Standard middelverdi
        quality_issues = []
        
        if material_type in ['hardwood', 'laminate', 'tile', 'marble', 'granite']:
            # For disse materialene er konsistens ofte et kvalitetstegn
            if texture_variation > 40:
                quality_issues.append('inconsistent_texture')
                quality_score -= 0.1
            
            if color_variation > 30:
                quality_issues.append('color_inconsistency')
                quality_score -= 0.1
        
        elif material_type in ['carpet']:
            # For tepper kan for lav variasjon tyde på slitasje
            if texture_variation < 15:
                quality_issues.append('worn_out')
                quality_score -= 0.2
        
        # Juster kvalitetsscore basert på materialet
        if material_type in ['marble', 'granite', 'hardwood']:
            quality_score += 0.2  # Premium materialer
        elif material_type in ['laminate', 'vinyl']:
            quality_score -= 0.1  # Rimeligere materialer
        
        # Begrens score til området [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Tildel kvalitetskategori
        if quality_score >= 0.8:
            quality_category = 'excellent'
        elif quality_score >= 0.6:
            quality_category = 'good'
        elif quality_score >= 0.4:
            quality_category = 'average'
        elif quality_score >= 0.2:
            quality_category = 'below_average'
        else:
            quality_category = 'poor'
        
        return {
            'score': float(quality_score),
            'category': quality_category,
            'issues': quality_issues
        }

    def _calculate_material_sustainability(self, material_type: str, quality: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner bærekraftsscore for materialet"""
        # Basisskåre basert på materialtype
        base_scores = {
            'hardwood': 0.7,  # Kan være bærekraftig hvis fra ansvarlig skogbruk
            'laminate': 0.5,  # Syntetisk, men holdbart
            'tile': 0.6,
            'carpet': 0.4,
            'concrete': 0.5,
            'stone': 0.8,  # Naturlig og holdbart
            'marble': 0.7,
            'vinyl': 0.3,  # Plastbasert
            'drywall': 0.5,
            'brick': 0.7,  # Holdbart og naturlig
            'glass': 0.6,
            'metal': 0.6,
            'plastic': 0.2,  # Lav bærekraft
            'ceramic': 0.6,
            'granite': 0.7,
            'formica': 0.4
        }
        
        # Standardverdi for ukjente materialer
        base_score = base_scores.get(material_type, 0.5)
        
        # Juster basert på kvalitet (høyere kvalitet = mer holdbart = mer bærekraftig)
        quality_adjustment = (quality['score'] - 0.5) * 0.2
        
        # Beregn total score
        sustainability_score = base_score + quality_adjustment
        
        # Begrens score til området [0, 1]
        sustainability_score = max(0.0, min(1.0, sustainability_score))
        
        # Kategoriser bærekraftsnivå
        if sustainability_score >= 0.8:
            category = 'excellent'
        elif sustainability_score >= 0.6:
            category = 'good'
        elif sustainability_score >= 0.4:
            category = 'average'
        elif sustainability_score >= 0.2:
            category = 'below_average'
        else:
            category = 'poor'
        
        # Finn forbedringsmuligheter
        improvement_options = []
        if material_type in ['vinyl', 'plastic', 'carpet']:
            improvement_options.append({
                'option': 'replace_with_natural',
                'description': f'Erstatt {material_type} med mer naturlige alternativer som tre eller stein',
                'sustainability_gain': 0.3
            })
        
        if material_type in ['hardwood'] and quality['score'] < 0.6:
            improvement_options.append({
                'option': 'refinish',
                'description': 'Refinishing av tregulvet i stedet for utskifting',
                'sustainability_gain': 0.2
            })
        
        return {
            'score': float(sustainability_score),
            'category': category,
            'improvement_options': improvement_options
        }

    def _generate_material_upgrade_recommendations(self, materials: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genererer anbefalinger for materialoppgraderinger"""
        recommendations = []
        
        # Finn materialer med lav kvalitet
        for material, quality_score in materials['quality_assessment'].items():
            if quality_score < 0.5:
                # Finn beste erstatning for materialet
                replacement = self._find_best_material_replacement(material)
                
                recommendations.append({
                    'original_material': material,
                    'recommended_replacement': replacement['material'],
                    'reason': replacement['reason'],
                    'estimated_improvement': replacement['improvement'],
                    'sustainability_impact': replacement['sustainability_impact'],
                    'priority': 'high' if quality_score < 0.3 else 'medium'
                })
        
        return recommendations

    def _find_best_material_replacement(self, material: str) -> Dict[str, Any]:
        """Finner beste erstatningsmateriale basert på type"""
        # Dette er en forenklet implementasjon
        replacements = {
            'carpet': {
                'material': 'hardwood',
                'reason': 'Bedre holdbarhet og luftkvalitet',
                'improvement': 0.4,
                'sustainability_impact': 'positive'
            },
            'vinyl': {
                'material': 'tile',
                'reason': 'Mer holdbart og miljøvennlig',
                'improvement': 0.3,
                'sustainability_impact': 'positive'
            },
            'laminate': {
                'material': 'engineered_hardwood',
                'reason': 'Bedre kvalitet med lignende prisnivå',
                'improvement': 0.3,
                'sustainability_impact': 'positive'
            },
            'formica': {
                'material': 'quartz',
                'reason': 'Mer holdbart og moderne utseende',
                'improvement': 0.4,
                'sustainability_impact': 'neutral'
            },
            'drywall': {
                'material': 'drywall_with_insulation',
                'reason': 'Bedre lydisolasjon og energieffektivitet',
                'improvement': 0.2,
                'sustainability_impact': 'positive'
            }
        }
        
        # Standard erstatning hvis ingen spesifikk erstatning er definert
        default_replacement = {
            'material': 'higher_quality_' + material,
            'reason': 'Oppgrader til høyere kvalitet av samme material',
            'improvement': 0.2,
            'sustainability_impact': 'neutral'
        }
        
        return replacements.get(material, default_replacement)

    async def _analyze_structure(self, image: torch.Tensor) -> Dict[str, Any]:
        """Analyserer strukturelle elementer i bildet"""
        # Konverter tensor til format for strukturmodellen
        if isinstance(image, torch.Tensor):
            np_image = image.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = image
        
        # Kjør strukturanalyse
        results = self.structure_analyzer(np_image)
        
        # Prosesser resultatene
        structure = {
            'structural_elements': [],
            'load_bearing_walls': [],
            'non_load_bearing_walls': [],
            'modification_potential': {},
            'structural_issues': []
        }
        
        # Tolk deteksjoner
        detections = results.xyxy[0].cpu().numpy()
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            element_class = int(cls)
            
            # Eksempel på strukturelle elementkategorier
            element_mapping = {
                0: 'load_bearing_wall',
                1: 'non_load_bearing_wall',
                2: 'column',
                3: 'beam',
                4: 'joist',
                5: 'foundation',
                6: 'roof_truss',
                7: 'staircase',
                8: 'window_opening',
                9: 'door_opening'
            }
            
            element_type = element_mapping.get(element_class, 'unknown')
            
            # Legg til detektert strukturelt element
            element = {
                'type': element_type,
                'confidence': float(conf),
                'bounding_box': [float(x1), float(y1), float(x2), float(y2)],
                'removable': element_type not in ['load_bearing_wall', 'column', 'beam', 'foundation']
            }
            
            structure['structural_elements'].append(element)
            
            # Kategoriser vegger
            if element_type == 'load_bearing_wall':
                structure['load_bearing_walls'].append(element)
            elif element_type == 'non_load_bearing_wall':
                structure['non_load_bearing_walls'].append(element)
        
        # Vurder modifikasjonspotensiale
        structure['modification_potential'] = self._evaluate_modification_potential(structure)
        
        # Identifiser strukturelle problemer
        structure['structural_issues'] = self._identify_structural_issues(structure, np_image)
        
        return structure

    def _evaluate_modification_potential(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluerer potensiale for strukturelle modifikasjoner"""
        non_load_bearing_walls = len(structure['non_load_bearing_walls'])
        load_bearing_walls = len(structure['load_bearing_walls'])
        
        # Beregn åpen planpotensial
        open_plan_score = min(1.0, non_load_bearing_walls / max(1, non_load_bearing_walls + load_bearing_walls))
        
        # Identifiser veggpartier som kan fjernes
        removable_sections = []
        for wall in structure['non_load_bearing_walls']:
            removable_sections.append({
                'element_index': structure['structural_elements'].index(wall),
                'bounding_box': wall['bounding_box'],
                'removal_complexity': 'low'
            })
        
        # Identifiser veggpartier som kan modifiseres (f.eks. åpning for dør/vindu)
        modifiable_sections = []
        for wall in structure['load_bearing_walls']:
            # Forenklet: anta at midtseksjoner av bærende vegger kan modifiseres
            x1, y1, x2, y2 = wall['bounding_box']
            center_x = (x1 + x2) / 2
            center_section = [center_x - (x2 - x1) * 0.2, y1, center_x + (x2 - x1) * 0.2, y2]
            
            modifiable_sections.append({
                'element_index': structure['structural_elements'].index(wall),
                'bounding_box': center_section,
                'modification_type': 'opening',
                'complexity': 'high',
                'requires_structural_support': True
            })
        
        return {
            'open_plan_potential': {
                'score': float(open_plan_score),
                'category': 'high' if open_plan_score > 0.7 else ('medium' if open_plan_score > 0.3 else 'low')
            },
            'removable_sections': removable_sections,
            'modifiable_sections': modifiable_sections,
            'extension_potential': self._evaluate_extension_potential(structure)
        }

    def _evaluate_extension_potential(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluerer potensial for utvidelse/påbygg basert på strukturanalyse"""
        # Dette er en forenklet implementasjon
        # I praksis ville dette kreve mer informasjon om tomten, bygningens fotavtrykk, etc.
        
        # Teller strukturelle elementer som kunne muliggjøre utvidelse
        load_bearing_elements = sum(1 for e in structure['structural_elements'] 
                                    if e['type'] in ['load_bearing_wall', 'column', 'beam'])
        
        # Forenklet scoring basert på antall lastbærende elementer
        # Flere lastbærende elementer indikerer en sterkere struktur som kan utvides
        extension_score = min(1.0, load_bearing_elements / 10.0)
        
        return {
            'score': float(extension_score),
            'category': 'high' if extension_score > 0.7 else ('medium' if extension_score > 0.4 else 'low'),
            'possible_directions': ['vertical', 'horizontal'],
            'structural_considerations': [
                'Eksisterende fundament må vurderes for horisontal utvidelse',
                'Takkonstruksjon må vurderes for vertikal utvidelse'
            ]
        }

    def _identify_structural_issues(self, structure: Dict[str, Any], image: np.ndarray) -> List[Dict[str, Any]]:
        """Identifiserer potensielle strukturelle problemer"""
        issues = []
        
        # Sjekk etter tegn på setningsskader (forenklet implementasjon)
        # I praksis ville dette involvert mer avansert analyse
        
        # Sjekk etter skjeve linjer i vegger og søyler
        for element in structure['structural_elements']:
            if element['type'] in ['load_bearing_wall', 'column']:
                x1, y1, x2, y2 = element['bounding_box']
                
                # Beregn evt. helning på vertikale elementer
                if element['type'] == 'column' or abs(x2 - x1) < abs(y2 - y1):
                    angle = np.arctan2(abs(x2 - x1), abs(y2 - y1)) * 180 / np.pi
                    
                    if angle > 2.0:  # Mer enn 2 grader helning
                        issues.append({
                            'type': 'structural_tilt',
                            'element_type': element['type'],
                            'severity': 'high' if angle > 5 else 'medium',
                            'description': f'Helning på {angle:.1f} grader detektert i {element["type"]}',
                            'location': element['bounding_box']
                        })
        
        # Sjekk for manglende strukturelle elementer
        if len(structure['load_bearing_walls']) < 2:
            issues.append({
                'type': 'insufficient_support',
                'severity': 'high',
                'description': 'Mulig utilstrekkelig strukturell støtte detektert',
                'recommendation': 'Grundig strukturell vurdering anbefales'
            })
        
        return issues

    async def _analyze_layout(self, image: torch.Tensor) -> Dict[str, Any]:
        """Analyserer romoppsett og planløsning"""
        # Konverter tensor til layout-format
        if isinstance(image, torch.Tensor):
            np_image = image.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = image
        
        # Forenklet layout-analyse
        # I praksis ville dette kravet mer avanserte algoritmer
        
        # Identifiser hovedområder i bildet ved hjelp av segmentering
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Identifiser hovedområder
        layout_areas = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:  # Ignorer for små områder
                x, y, w, h = cv2.boundingRect(contour)
                layout_areas.append({
                    'id': i,
                    'area_pixels': float(area),
                    'bounding_box': [float(x), float(y), float(x + w), float(y + h)],
                    'aspect_ratio': float(w / h)
                })
        
        # Vurder rom-flow og funksjonalitet
        layout_assessment = self._assess_layout_functionality(layout_areas, np_image.shape[:2])
        
        # Vurder potensial for layoutforbedringer
        improvement_potential = self._evaluate_layout_improvement_potential(layout_areas, layout_assessment)
        
        return {
            'areas': layout_areas,
            'functionality_assessment': layout_assessment,
            'improvement_potential': improvement_potential,
            'open_plan_score': self._calculate_open_plan_score(layout_areas, np_image.shape[:2])
        }

    def _assess_layout_functionality(self, areas: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Vurderer layoutets funksjonalitet"""
        total_image_area = image_shape[0] * image_shape[1]
        total_identified_area = sum(area['area_pixels'] for area in areas)
        coverage_ratio = total_identified_area / total_image_area
        
        # Beregn gjennomsnittlig og median aspektforhold
        aspect_ratios = [area['aspect_ratio'] for area in areas]
        avg_aspect_ratio = np.mean(aspect_ratios)
        
        # Vurder balanse i romstørrelser
        area_sizes = [area['area_pixels'] for area in areas]
        area_variance = np.var(area_sizes) / (np.mean(area_sizes) ** 2) if areas else 0
        
        # Høy varians indikerer ubalanserte romstørrelser
        balance_score = max(0, 1 - area_variance)
        
        # Vurder rommenes form (forholdet mellom bredde og lengde)
        shape_score = 0.5
        if aspect_ratios:
            # Ideelt aspektforhold er ofte mellom 1:1 og 1:2
            ideal_ratios = [abs(1 - min(r, 1/r)) for r in aspect_ratios]
            shape_score = 1 - min(1, np.mean(ideal_ratios))
        
        # Samlet funksjonalitetsscore
        functionality_score = (balance_score * 0.4 + shape_score * 0.6) * coverage_ratio
        
        return {
            'coverage_ratio': float(coverage_ratio),
            'balance_score': float(balance_score),
            'shape_score': float(shape_score),
            'functionality_score': float(functionality_score),
            'functionality_category': 'high' if functionality_score > 0.7 else 
                                      ('medium' if functionality_score > 0.4 else 'low'),
            'issues': self._identify_layout_issues(areas, balance_score, shape_score)
        }

    def _identify_layout_issues(self, areas: List[Dict[str, Any]], 
                              balance_score: float, shape_score: float) -> List[Dict[str, Any]]:
        """Identifiserer layoutproblemer"""
        issues = []
        
        if balance_score < 0.4:
            issues.append({
                'type': 'unbalanced_room_sizes',
                'severity': 'medium',
                'description': 'Store variasjoner i romstørrelser kan redusere funksjonalitet'
            })
        
        if shape_score < 0.4:
            issues.append({
                'type': 'poor_room_proportions',
                'severity': 'medium',
                'description': 'Flere rom har ugunstige proporsjoner (for smale eller irregulære)'
            })
        
        # Sjekk for for små rom
        small_areas = [area for area in areas if area['area_pixels'] < 10000]
        if small_areas and len(small_areas) / len(areas) > 0.3:
            issues.append({
                'type': 'excessive_small_areas',
                'severity': 'medium',
                'description': 'For mange små rom/områder kan redusere brukbarheten'
            })
        
        return issues

    def _evaluate_layout_improvement_potential(self, areas: List[Dict[str, Any]], 
                                              assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluerer potensial for layoutforbedringer"""
        # Vurder forbedringspotensial basert på funksjonalitetsvurdering
        functionality_score = assessment.get('functionality_score', 0.5)
        improvement_potential = 1 - functionality_score
        
        # Generer spesifikke forbedringsforslag
        suggestions = []
        
        if assessment.get('balance_score', 0.5) < 0.5:
            suggestions.append({
                'type': 'rebalance_areas',
                'description': 'Omfordel romstørrelser for bedre balanse',
                'complexity': 'high',
                'potential_impact': 'medium'
            })
        
        if assessment.get('shape_score', 0.5) < 0.5:
            suggestions.append({
                'type': 'improve_proportions',
                'description': 'Endre romformer for bedre proporsjoner og brukbarhet',
                'complexity': 'medium',
                'potential_impact': 'high'
            })
        
        # Se etter potensial for åpen planløsning
        if len(areas) > 3:
            suggestions.append({
                'type': 'open_plan',
                'description': 'Vurder åpen planløsning for å forbedre romfølelsen og lysforhold',
                'complexity': 'medium',
                'potential_impact': 'high'
            })
        
        return {
            'improvement_score': float(improvement_potential),
            'category': 'high' if improvement_potential > 0.6 else 
                        ('medium' if improvement_potential > 0.3 else 'low'),
            'suggestions': suggestions
        }

    def _calculate_open_plan_score(self, areas: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> float:
        """Beregner en score for åpen planløsning basert på layout"""
        if not areas:
            return 0.5  # Standard midtverdi hvis ingen områder er identifisert
        
        total_image_area = image_shape[0] * image_shape[1]
        
        # Beregn gjennomsnittlig områdestørrelse
        avg_area_size = np.mean([area['area_pixels'] for area in areas])
        
        # Større gjennomsnittlig områdestørrelse indikerer mer åpen planløsning
        size_score = min(1.0, avg_area_size / (total_image_area * 0.3))
        
        # Færre områder indikerer mer åpen planløsning
        count_score = max(0.0, 1.0 - (len(areas) / 10.0))
        
        # Kombiner scores
        open_plan_score = size_score * 0.7 + count_score * 0.3
        
        return float(open_plan_score)

    def _analyze_energy_efficiency(
        self,
        material_analysis: Dict[str, Any],
        structure_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyserer energieffektivitet"""
        # Beregn U-verdier for ulike bygningsdeler basert på detekterte materialer
        u_values = self._calculate_u_values(material_analysis)
        
        # Beregn varmetap
        heat_loss = self._calculate_heat_loss(u_values, structure_analysis)
        
        # Analyser oppvarmingsbehov
        heating_need = self._calculate_heating_need(heat_loss, structure_analysis)
        
        # Identifiser energiforbedringer
        improvements = self._identify_energy_improvements(u_values, heat_loss)
        
        # Beregn energirating
        energy_rating = self._calculate_energy_rating(heating_need)
        
        # Beregn kostnadsbesparelser ved forbedringer
        savings = self._calculate_energy_improvement_savings(improvements, heating_need)
        
        # Sjekk eligibility for subsidier (Enova etc.)
        subsidies = self._check_energy_subsidy_eligibility(improvements, energy_rating)
        
        return {
            'u_values': u_values,
            'heat_loss': heat_loss,
            'heating_need': heating_need,
            'energy_rating': energy_rating,
            'improvement_potential': improvements,
            'annual_savings': savings,
            'subsidies': subsidies,
            'payback_period': self._calculate_energy_payback_period(improvements, savings, subsidies),
            'co2_reduction': self._calculate_co2_reduction(improvements, heating_need)
        }

    def _calculate_u_values(self, material_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Beregner U-verdier for bygningsdeler basert på materialanalyse"""
        u_values = {}
        
        # Typiske U-verdier for forskjellige materialer og konstruksjoner
        material_u_values = {
            'walls': {
                'brick': 2.0,
                'concrete': 2.5,
                'drywall': 3.0,
                'stone': 2.2,
                'wood': 1.5
            },
            'roof': {
                'tile': 2.3,
                'metal': 3.0,
                'asphalt': 2.5
            },
            'windows': {
                'single_pane': 5.0,
                'double_pane': 2.8,
                'triple_pane': 1.0,
                'glass': 3.5
            },
            'floor': {
                'concrete': 2.0,
                'wood': 1.5,
                'carpet': 1.7,
                'tile': 1.8,
                'laminate': 1.7
            }
        }
        
        # Identifiser materialer for hver bygningsdel
        for material in material_analysis.get('detected_materials', []):
            material_type = material.get('type', 'unknown')
            
            # Kategoriser materiale
            if material_type in ['brick', 'concrete', 'drywall', 'stone']:
                # Anta at materialet er i veggen
                u_values['walls'] = material_u_values['walls'].get(material_type, 2.0)
            elif material_type in ['tile', 'metal', 'asphalt']:
                # Anta at materialet er i taket
                u_values['roof'] = material_u_values['roof'].get(material_type, 2.5)
            elif material_type in ['glass']:
                # Anta at materialet er i vinduet
                # Simplifisert: vi gjetter på vindustype basert på detekteringskvalitet
                quality = material.get('quality', {}).get('score', 0.5)
                if quality > 0.8:
                    u_values['windows'] = material_u_values['windows']['triple_pane']
                elif quality > 0.5:
                    u_values['windows'] = material_u_values['windows']['double_pane']
                else:
                    u_values['windows'] = material_u_values['windows']['single_pane']
            elif material_type in ['concrete', 'wood', 'carpet', 'tile', 'laminate']:
                # Anta at materialet er i gulvet
                u_values['floor'] = material_u_values['floor'].get(material_type, 1.8)
        
        # Standardverdier for bygningsdeler som ikke er detektert
        default_u_values = {
            'walls': 2.0,
            'roof': 2.5,
            'windows': 2.8,
            'floor': 1.8
        }
        
        # Fyll inn manglende verdier med standardverdier
        for part, default_value in default_u_values.items():
            if part not in u_values:
                u_values[part] = default_value
        
        return u_values

    def _calculate_heat_loss(self, u_values: Dict[str, float], 
                            structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner varmetap basert på U-verdier og strukturanalyse"""
        # Forenklet beregning - i praksis ville dette kreve mer informasjon
        
        # Estimer areal for hver bygningsdel basert på strukturanalyse
        # Dette er en veldig forenklet tilnærming
        areas = {
            'walls': 100.0,  # m²
            'roof': 60.0,    # m²
            'windows': 15.0, # m²
            'floor': 60.0    # m²
        }
        
        # Juster områder basert på strukturelle elementer hvis tilgjengelig
        wall_elements = [e for e in structure_analysis.get('structural_elements', []) 
                         if e['type'] in ['load_bearing_wall', 'non_load_bearing_wall']]
        
        if wall_elements:
            # Estimer veggareal basert på detekterte vegger
            wall_area = sum((e['bounding_box'][2] - e['bounding_box'][0]) * 
                            (e['bounding_box'][3] - e['bounding_box'][1]) 
                            for e in wall_elements) / 10000  # konverter til m²
            if wall_area > 0:
                areas['walls'] = wall_area
        
        # Beregn varmetap for hver bygningsdel
        heat_loss_components = {}
        total_heat_loss = 0
        
        for part, area in areas.items():
            u_value = u_values.get(part, 2.0)
            heat_loss = u_value * area
            heat_loss_components[part] = {
                'area': float(area),
                'u_value': float(u_value),
                'heat_loss': float(heat_loss)
            }
            total_heat_loss += heat_loss
        
        return {
            'components': heat_loss_components,
            'total_heat_loss': float(total_heat_loss),
            'heat_loss_per_sqm': float(total_heat_loss / sum(areas.values()))
        }

    def _calculate_heating_need(self, heat_loss: Dict[str, Any], 
                               structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner oppvarmingsbehov basert på varmetap"""
        # Forenklet beregning - i praksis ville dette krevd mer data
        
        # Standardverdier for beregning
        heating_degree_days = 4000  # Graddager (avhenger av klima)
        internal_gains = 5.0        # W/m²
        
        # Estimer totalt gulvareal
        floor_area = 60.0  # m²
        
        # Beregn årlig oppvarmingsbehov
        heat_loss_coefficient = heat_loss.get('total_heat_loss', 200.0)
        annual_heating_need = heat_loss_coefficient * heating_degree_days * 24 / 1000  # kWh
        
        # Trekk fra interne varmegevinster
        internal_gain_reduction = internal_gains * floor_area * 365 * 24 / 1000  # kWh
        net_heating_need = max(0, annual_heating_need - internal_gain_reduction)
        
        # Beregn oppvarmingsbehov per kvadratmeter
        heating_need_per_sqm = net_heating_need / floor_area if floor_area > 0 else 0
        
        return {
            'annual_heating_need': float(net_heating_need),
            'heating_need_per_sqm': float(heating_need_per_sqm),
            'energy_class_threshold': self._get_energy_class_threshold(heating_need_per_sqm)
        }

    def _get_energy_class_threshold(self, heating_need_per_sqm: float) -> str:
        """Returnerer energiklasseterskel basert på oppvarmingsbehov per kvm"""
        # Norske energimerketerskel (forenklet)
        if heating_need_per_sqm <= 95:
            return 'A'
        elif heating_need_per_sqm <= 120:
            return 'B'
        elif heating_need_per_sqm <= 145:
            return 'C'
        elif heating_need_per_sqm <= 175:
            return 'D'
        elif heating_need_per_sqm <= 205:
            return 'E'
        elif heating_need_per_sqm <= 250:
            return 'F'
        else:
            return 'G'

    def _identify_energy_improvements(self, u_values: Dict[str, float], 
                                    heat_loss: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifiserer potensielle energiforbedringer"""
        improvements = []
        
        # Sjekk hver bygningsdel og foreslå forbedringer basert på U-verdier
        components = heat_loss.get('components', {})
        
        # Veggisolasjon
        if u_values.get('walls', 0) > 0.8:
            wall_heat_loss = components.get('walls', {}).get('heat_loss', 0)
            wall_area = components.get('walls', {}).get('area', 0)
            
            improvements.append({
                'type': 'wall_insulation',
                'description': 'Etterisolering av vegger',
                'current_u_value': float(u_values.get('walls', 0)),
                'target_u_value': 0.18,
                'area': float(wall_area),
                'heat_loss_reduction': float(wall_heat_loss * (1 - 0.18 / u_values.get('walls', 1))),
                'estimated_cost': float(wall_area * 1500),  # NOK per m²
                'priority': 'high' if u_values.get('walls', 0) > 1.2 else 'medium'
            })
        
        # Takisolasjon
        if u_values.get('roof', 0) > 0.6:
            roof_heat_loss = components.get('roof', {}).get('heat_loss', 0)
            roof_area = components.get('roof', {}).get('area', 0)
            
            improvements.append({
                'type': 'roof_insulation',
                'description': 'Etterisolering av tak/loft',
                'current_u_value': float(u_values.get('roof', 0)),
                'target_u_value': 0.13,
                'area': float(roof_area),
                'heat_loss_reduction': float(roof_heat_loss * (1 - 0.13 / u_values.get('roof', 1))),
                'estimated_cost': float(roof_area * 1000),  # NOK per m²
                'priority': 'high' if u_values.get('roof', 0) > 1.0 else 'medium'
            })
        
        # Vindusutskifting
        if u_values.get('windows', 0) > 1.6:
            window_heat_loss = components.get('windows', {}).get('heat_loss', 0)
            window_area = components.get('windows', {}).get('area', 0)
            
            improvements.append({
                'type': 'window_replacement',
                'description': 'Utskifting til energieffektive vinduer',
                'current_u_value': float(u_values.get('windows', 0)),
                'target_u_value': 0.8,
                'area': float(window_area),
                'heat_loss_reduction': float(window_heat_loss * (1 - 0.8 / u_values.get('windows', 1))),
                'estimated_cost': float(window_area * 5000),  # NOK per m²
                'priority': 'high' if u_values.get('windows', 0) > 2.5 else 'medium'
            })
        
        # Gulvisolasjon
        if u_values.get('floor', 0) > 0.6:
            floor_heat_loss = components.get('floor', {}).get('heat_loss', 0)
            floor_area = components.get('floor', {}).get('area', 0)
            
            improvements.append({
                'type': 'floor_insulation',
                'description': 'Etterisolering av gulv/kjeller',
                'current_u_value': float(u_values.get('floor', 0)),
                'target_u_value': 0.15,
                'area': float(floor_area),
                'heat_loss_reduction': float(floor_heat_loss * (1 - 0.15 / u_values.get('floor', 1))),
                'estimated_cost': float(floor_area * 1200),  # NOK per m²
                'priority': 'medium'
            })
        
        # Varmepumpe
        improvements.append({
            'type': 'heat_pump',
            'description': 'Installasjon av luft-til-luft varmepumpe',
            'efficiency_gain': 0.3,  # 30% reduksjon i oppvarmingsbehov
            'estimated_cost': 25000,  # NOK
            'priority': 'high'
        })
        
        # Balansert ventilasjon med varmegjenvinning
        improvements.append({
            'type': 'balanced_ventilation',
            'description': 'Balansert ventilasjon med varmegjenvinning',
            'efficiency_gain': 0.2,  # 20% reduksjon i oppvarmingsbehov
            'estimated_cost': 80000,  # NOK
            'priority': 'medium'
        })
        
        return improvements

    def _calculate_energy_rating(self, heating_need: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner energimerking basert på oppvarmingsbehov"""
        heating_need_per_sqm = heating_need.get('heating_need_per_sqm', 200)
        energy_class = self._get_energy_class_threshold(heating_need_per_sqm)
        
        # Beregn omtrentlig oppvarmingskostnad
        electricity_price = 1.5  # NOK per kWh
        annual_cost = heating_need.get('annual_heating_need', 0) * electricity_price
        
        return {
            'rating': energy_class,
            'heating_need_per_sqm': float(heating_need_per_sqm),
            'annual_cost': float(annual_cost),
            'rating_description': self._get_energy_rating_description(energy_class)
        }

    def _get_energy_rating_description(self, energy_class: str) -> str:
        """Returnerer beskrivelse av energiklasse"""
        descriptions = {
            'A': 'Svært energieffektiv bygning, lavt energibehov',
            'B': 'Meget energieffektiv bygning',
            'C': 'Energieffektiv bygning',
            'D': 'Moderat energieffektiv bygning',
            'E': 'Bygning med lavere energieffektivitet enn moderne standard',
            'F': 'Bygning med dårlig energieffektivitet',
            'G': 'Bygning med svært dårlig energieffektivitet, høyt energibehov'
        }
        
        return descriptions.get(energy_class, 'Ukjent energimerke')

    def _calculate_energy_improvement_savings(self, improvements: List[Dict[str, Any]], 
                                            heating_need: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner kostnadsbesparelser ved energiforbedringer"""
        annual_heating_need = heating_need.get('annual_heating_need', 0)
        electricity_price = 1.5  # NOK per kWh
        
        # Beregn besparelser for hver forbedring
        savings_per_improvement = {}
        total_annual_savings = 0
        
        for improvement in improvements:
            imp_type = improvement.get('type', '')
            
            if 'heat_loss_reduction' in improvement:
                # For isolasjonstiltak
                energy_saving = improvement.get('heat_loss_reduction', 0) * 24 * 4000 / 1000  # kWh
                cost_saving = energy_saving * electricity_price
            elif 'efficiency_gain' in improvement:
                # For systemer som varmepumpe eller balansert ventilasjon
                energy_saving = annual_heating_need * improvement.get('efficiency_gain', 0)
                cost_saving = energy_saving * electricity_price
            else:
                energy_saving = 0
                cost_saving = 0
            
            savings_per_improvement[imp_type] = {
                'annual_energy_saving_kwh': float(energy_saving),
                'annual_cost_saving': float(cost_saving)
            }
            
            total_annual_savings += cost_saving
        
        # Beregn totalt
        return {
            'per_improvement': savings_per_improvement,
            'total_annual_savings': float(total_annual_savings),
            'payback_years': self._calculate_simple_payback(improvements, total_annual_savings)
        }

    def _calculate_simple_payback(self, improvements: List[Dict[str, Any]], 
                                annual_savings: float) -> float:
        """Beregner enkel tilbakebetalingstid"""
        total_cost = sum(improvement.get('estimated_cost', 0) for improvement in improvements)
        
        if annual_savings <= 0:
            return float('inf')
        
        return total_cost / annual_savings

    def _check_energy_subsidy_eligibility(self, improvements: List[Dict[str, Any]], 
                                        energy_rating: Dict[str, Any]) -> Dict[str, Any]:
        """Sjekker eligibility for energisubsidier (f.eks. Enova-støtte)"""
        # Forenklet implementasjon av norske Enova-regler
        
        eligible_improvements = []
        total_subsidy = 0
        
        # Regler for ulike typer tiltak (forenklet)
        for improvement in improvements:
            imp_type = improvement.get('type', '')
            estimated_cost = improvement.get('estimated_cost', 0)
            
            if imp_type == 'wall_insulation':
                if improvement.get('current_u_value', 0) > 0.8 and improvement.get('target_u_value', 0) <= 0.22:
                    subsidy_amount = min(improvement.get('area', 0) * 500, estimated_cost * 0.25)
                    eligible_improvements.append({
                        'type': imp_type,
                        'subsidy_amount': float(subsidy_amount),
                        'requirements': 'U-verdi ≤ 0.22 W/m²K etter tiltak'
                    })
                    total_subsidy += subsidy_amount
            
            elif imp_type == 'roof_insulation':
                if improvement.get('current_u_value', 0) > 0.6 and improvement.get('target_u_value', 0) <= 0.18:
                    subsidy_amount = min(improvement.get('area', 0) * 400, estimated_cost * 0.25)
                    eligible_improvements.append({
                        'type': imp_type,
                        'subsidy_amount': float(subsidy_amount),
                        'requirements': 'U-verdi ≤ 0.18 W/m²K etter tiltak'
                    })
                    total_subsidy += subsidy_amount
            
            elif imp_type == 'window_replacement':
                if improvement.get('current_u_value', 0) > 1.6 and improvement.get('target_u_value', 0) <= 0.8:
                    subsidy_amount = min(improvement.get('area', 0) * 1000, estimated_cost * 0.25)
                    eligible_improvements.append({
                        'type': imp_type,
                        'subsidy_amount': float(subsidy_amount),
                        'requirements': 'U-verdi ≤ 0.8 W/m²K for nye vinduer'
                    })
                    total_subsidy += subsidy_amount
            
            elif imp_type == 'heat_pump':
                subsidy_amount = min(10000, estimated_cost * 0.25)
                eligible_improvements.append({
                    'type': imp_type,
                    'subsidy_amount': float(subsidy_amount),
                    'requirements': 'SCOP ≥ 3.5'
                })
                total_subsidy += subsidy_amount
            
            elif imp_type == 'balanced_ventilation':
                subsidy_amount = min(20000, estimated_cost * 0.25)
                eligible_improvements.append({
                    'type': imp_type,
                    'subsidy_amount': float(subsidy_amount),
                    'requirements': 'Varmegjenvinningsgrad ≥ 80%'
                })
                total_subsidy += subsidy_amount
        
        return {
            'eligible_improvements': eligible_improvements,
            'total_subsidy': float(total_subsidy),
            'subsidy_percentage': float(total_subsidy / sum(improvement.get('estimated_cost', 0) for improvement in improvements) * 100 if improvements else 0),
            'program': 'Enova',
            'additional_requirements': 'Se enova.no for komplett regelverk og søknadsprosess'
        }

    def _calculate_energy_payback_period(self, improvements: List[Dict[str, Any]], 
                                       savings: Dict[str, Any], 
                                       subsidies: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner tilbakebetalingstid med subsidier"""
        total_cost = sum(improvement.get('estimated_cost', 0) for improvement in improvements)
        total_subsidy = subsidies.get('total_subsidy', 0)
        net_cost = total_cost - total_subsidy
        
        annual_savings = savings.get('total_annual_savings', 0)
        
        if annual_savings <= 0:
            payback_years = float('inf')
        else:
            payback_years = net_cost / annual_savings
        
        return {
            'gross_cost': float(total_cost),
            'subsidies': float(total_subsidy),
            'net_cost': float(net_cost),
            'annual_savings': float(annual_savings),
            'payback_years': float(payback_years),
            'payback_category': 'excellent' if payback_years <= 5 else 
                               ('good' if payback_years <= 10 else 
                               ('acceptable' if payback_years <= 15 else 'poor'))
        }

    def _calculate_co2_reduction(self, improvements: List[Dict[str, Any]], 
                               heating_need: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner CO2-reduksjon ved energiforbedringer"""
        annual_heating_need = heating_need.get('annual_heating_need', 0)
        
        # CO2-faktor for elektrisitet i Norge (veldig lav pga. vannkraft)
        co2_factor = 0.017  # kg CO2 per kWh
        
        # Beregn reduksjon for hver forbedring
        co2_reduction_per_improvement = {}
        total_energy_savings = 0
        
        for improvement in improvements:
            imp_type = improvement.get('type', '')
            
            if 'heat_loss_reduction' in improvement:
                # For isolasjonstiltak
                energy_saving = improvement.get('heat_loss_reduction', 0) * 24 * 4000 / 1000  # kWh
            elif 'efficiency_gain' in improvement:
                # For systemer som varmepumpe eller balansert ventilasjon
                energy_saving = annual_heating_need * improvement.get('efficiency_gain', 0)
            else:
                energy_saving = 0
            
            co2_saving = energy_saving * co2_factor
            
            co2_reduction_per_improvement[imp_type] = {
                'annual_energy_saving_kwh': float(energy_saving),
                'annual_co2_reduction_kg': float(co2_saving)
            }
            
            total_energy_savings += energy_saving
        
        total_co2_reduction = total_energy_savings * co2_factor
        
        return {
            'per_improvement': co2_reduction_per_improvement,
            'total_annual_energy_saving_kwh': float(total_energy_savings),
            'total_annual_co2_reduction_kg': float(total_co2_reduction),
            'equivalent_trees': float(total_co2_reduction / 21)  # Et tre absorberer ca. 21 kg CO2 per år
        }

    def _analyze_property_value(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        property_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyserer eiendomsverdi basert på alle faktorer"""
        # Samle data for verdiestimering
        features = self._prepare_value_estimation_features(
            room_analysis, 
            dimension_analysis, 
            material_analysis, 
            energy_analysis, 
            property_data
        )
        
        # Bruk ensemble-modeller for prediksjon
        value_predictions = {}
        
        if isinstance(self.value_estimator, dict):
            # RF-modell prediksjon
            if 'rf' in self.value_estimator:
                try:
                    rf_prediction = self.value_estimator['rf'].predict([features])[0]
                    value_predictions['random_forest'] = float(rf_prediction)
                except Exception as e:
                    logger.warning(f"Feil i RandomForest verdiestimering: {e}")
                    value_predictions['random_forest'] = None
            
            # GB-modell prediksjon
            if 'gb' in self.value_estimator:
                try:
                    gb_prediction = self.value_estimator['gb'].predict([features])[0]
                    value_predictions['gradient_boosting'] = float(gb_prediction)
                except Exception as e:
                    logger.warning(f"Feil i GradientBoosting verdiestimering: {e}")
                    value_predictions['gradient_boosting'] = None
        else:
            # Fallback til enkel heuristisk beregning
            value_predictions['heuristic'] = self._heuristic_value_estimation(
                room_analysis, 
                dimension_analysis, 
                material_analysis, 
                energy_analysis, 
                property_data
            )
        
        # Kombiner prediksjoner for endelig estimat
        valid_predictions = [pred for pred in value_predictions.values() if pred is not None]
        if valid_predictions:
            estimated_value = sum(valid_predictions) / len(valid_predictions)
        else:
            # Absolut fallback
            estimated_value = 25000 * (property_data.get('area', 100) if property_data else 100)
        
        # Beregn prisintervall (± 10%)
        value_range = {
            'low': float(estimated_value * 0.9),
            'estimated': float(estimated_value),
            'high': float(estimated_value * 1.1)
        }
        
        # Identifiser verdidrivere
        value_drivers = self._identify_value_drivers(
            room_analysis, 
            dimension_analysis, 
            material_analysis, 
            energy_analysis, 
            property_data
        )
        
        return {
            'estimated_value': float(estimated_value),
            'value_range': value_range,
            'predictions': value_predictions,
            'value_drivers': value_drivers,
            'value_detractors': self._identify_value_detractors(
                room_analysis, 
                material_analysis, 
                energy_analysis
            ),
            'potential_value_increase': self._estimate_value_increase_potential(
                room_analysis, 
                dimension_analysis, 
                material_analysis, 
                energy_analysis, 
                property_data,
                estimated_value
            )
        }

    def _prepare_value_estimation_features(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        property_data: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Forbereder features for verdiestimering"""
        features = []
        
        # Areal
        area = property_data.get('area', 0) if property_data else dimension_analysis.get('room_measurements', {}).get('floor_area', 0) * 10
        features.append(area)
        
        # Rom
        room_type = room_analysis.get('type', 'unknown')
        room_quality_score = {'poor': 0, 'below_average': 0.25, 'average': 0.5, 'good': 0.75, 'excellent': 1.0}
        room_quality = room_quality_score.get(room_analysis.get('quality', 'average'), 0.5)
        features.append(room_quality)
        
        # Materialkvalitet
        material_quality = 0.5  # Standardverdi
        if 'quality_assessment' in material_analysis:
            material_qualities = list(material_analysis['quality_assessment'].values())
            if material_qualities:
                material_quality = sum(material_qualities) / len(material_qualities)
        features.append(material_quality)
        
        # Energieffektivitet
        energy_rating_map = {'A': 1.0, 'B': 0.85, 'C': 0.7, 'D': 0.55, 'E': 0.4, 'F': 0.25, 'G': 0.1}
        energy_rating = energy_rating_map.get(energy_analysis.get('energy_rating', {}).get('rating', 'E'), 0.4)
        features.append(energy_rating)
        
        # Beliggenhet (hvis tilgjengelig)
        location_score = 0.5  # Standard
        if property_data and 'location' in property_data:
            location = property_data['location']
            # Her ville det normalt vært en mer kompleks beregning basert på markedsdata
            if isinstance(location, dict) and 'postal_code' in location:
                # Eksempel: Postal kode kan være en proxy for området
                postal_score = min(1.0, int(location['postal_code'][:2]) / 100 + 0.3)
                location_score = postal_score
        features.append(location_score)
        
        return features

    def _heuristic_value_estimation(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        property_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Heuristisk verdiestimering når ML-modeller ikke er tilgjengelige"""
        # Basis kvadratmeterpris (kan varieres basert på område)
        base_price_per_sqm = 50000  # NOK per m²
        
        # Juster basert på område hvis tilgjengelig
        if property_data and 'location' in property_data:
            location = property_data['location']
            if isinstance(location, dict):
                if 'city' in location:
                    city = location['city'].lower()
                    if 'oslo' in city:
                        base_price_per_sqm = 90000
                    elif 'bergen' in city or 'trondheim' in city or 'stavanger' in city:
                        base_price_per_sqm = 70000
                    elif 'tromsø' in city or 'kristiansand' in city:
                        base_price_per_sqm = 60000
        
        # Hent areal
        area = property_data.get('area', 0) if property_data else dimension_analysis.get('room_measurements', {}).get('floor_area', 0) * 10
        
        if area <= 0:
            area = 100  # Standardverdi hvis areal mangler
        
        # Basispris
        base_value = area * base_price_per_sqm
        
        # Justeringsfaktorer
        adjustment_factors = 1.0
        
        # Juster for romkvalitet
        room_quality_map = {'poor': 0.85, 'below_average': 0.92, 'average': 1.0, 'good': 1.08, 'excellent': 1.15}
        adjustment_factors *= room_quality_map.get(room_analysis.get('quality', 'average'), 1.0)
        
        # Juster for materialkvalitet
        if 'quality_assessment' in material_analysis:
            material_qualities = list(material_analysis['quality_assessment'].values())
            if material_qualities:
                avg_material_quality = sum(material_qualities) / len(material_qualities)
                material_adjustment = 0.85 + avg_material_quality * 0.3  # 0.85 - 1.15
                adjustment_factors *= material_adjustment
        
        # Juster for energirating
        energy_rating_map = {'A': 1.1, 'B': 1.05, 'C': 1.02, 'D': 1.0, 'E': 0.98, 'F': 0.95, 'G': 0.9}
        adjustment_factors *= energy_rating_map.get(energy_analysis.get('energy_rating', {}).get('rating', 'D'), 1.0)
        
        return base_value * adjustment_factors

    def _identify_value_drivers(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        property_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Identifiserer faktorer som driver eiendomsverdien opp"""
        value_drivers = []
        
        # Romkvalitet
        if room_analysis.get('quality', 'average') in ['good', 'excellent']:
            value_drivers.append({
                'type': 'room_quality',
                'description': f"Høy romkvalitet ({room_analysis.get('quality', 'ukjent')})",
                'impact': 'medium'
            })
        
        # Romegenskaper
        for feature in room_analysis.get('features', []):
            if feature in ['natural_light', 'high_ceiling', 'open_layout', 'hardwood_floor', 'fireplace']:
                value_drivers.append({
                    'type': 'feature',
                    'description': f"Verdiøkende egenskap: {feature.replace('_', ' ')}",
                    'impact': 'medium' if feature in ['fireplace', 'high_ceiling'] else 'low'
                })
        
        # Energieffektivitet
        energy_rating = energy_analysis.get('energy_rating', {}).get('rating', 'E')
        if energy_rating in ['A', 'B', 'C']:
            value_drivers.append({
                'type': 'energy_rating',
                'description': f"God energimerking ({energy_rating})",
                'impact': 'high' if energy_rating == 'A' else 'medium'
            })
        
        # Materialkvalitet
        high_quality_materials = []
        for material in material_analysis.get('detected_materials', []):
            if material.get('type') in ['hardwood', 'marble', 'granite', 'stone'] and \
               material.get('quality', {}).get('category', '') in ['good', 'excellent']:
                high_quality_materials.append(material.get('type'))
        
        if high_quality_materials:
            value_drivers.append({
                'type': 'premium_materials',
                'description': f"Premium materialer: {', '.join(high_quality_materials)}",
                'impact': 'medium'
            })
        
        # Beliggenhet (hvis tilgjengelig)
        if property_data and 'location' in property_data and isinstance(property_data['location'], dict):
            location = property_data['location']
            if 'amenities' in location and isinstance(location['amenities'], list):
                valuable_amenities = [a for a in location['amenities'] 
                                    if a in ['sea_view', 'park_nearby', 'good_schools', 'public_transport']]
                
                if valuable_amenities:
                    value_drivers.append({
                        'type': 'location_amenities',
                        'description': f"Verdifulle fasiliteter i nærheten: {', '.join(valuable_amenities)}",
                        'impact': 'high'
                    })
        
        return value_drivers

    def _identify_value_detractors(
        self,
        room_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identifiserer faktorer som trekker eiendomsverdien ned"""
        value_detractors = []
        
        # Romkvalitet og tilstand
        if room_analysis.get('quality', 'average') in ['poor', 'below_average']:
            value_detractors.append({
                'type': 'poor_room_quality',
                'description': f"Lav romkvalitet ({room_analysis.get('quality', 'ukjent')})",
                'impact': 'high' if room_analysis.get('quality') == 'poor' else 'medium',
                'improvement_potential': 'high'
            })
        
        if room_analysis.get('condition', 'average') in ['needs_major_renovation', 'needs_minor_renovation']:
            value_detractors.append({
                'type': 'poor_condition',
                'description': f"Behov for renovering ({room_analysis.get('condition', 'ukjent').replace('_', ' ')})",
                'impact': 'high' if room_analysis.get('condition') == 'needs_major_renovation' else 'medium',
                'improvement_potential': 'high'
            })
        
        # Problematiske romegenskaper
        for feature in room_analysis.get('features', []):
            if feature in ['poor_insulation', 'moisture_issues', 'water_damage', 'structural_issues']:
                value_detractors.append({
                    'type': 'negative_feature',
                    'description': f"Verdireduserende egenskap: {feature.replace('_', ' ')}",
                    'impact': 'high' if feature in ['water_damage', 'structural_issues'] else 'medium',
                    'improvement_potential': 'medium'
                })
        
        # Energieffektivitet
        energy_rating = energy_analysis.get('energy_rating', {}).get('rating', 'E')
        if energy_rating in ['F', 'G']:
            value_detractors.append({
                'type': 'poor_energy_rating',
                'description': f"Dårlig energimerking ({energy_rating})",
                'impact': 'medium',
                'improvement_potential': 'high'
            })
        
        # Materialkvalitet
        low_quality_materials = []
        for material in material_analysis.get('detected_materials', []):
            if material.get('quality', {}).get('category', '') in ['poor', 'below_average']:
                low_quality_materials.append(material.get('type'))
        
        if low_quality_materials:
            value_detractors.append({
                'type': 'low_quality_materials',
                'description': f"Lav materialkvalitet: {', '.join(low_quality_materials)}",
                'impact': 'medium',
                'improvement_potential': 'high'
            })
        
        return value_detractors

    def _estimate_value_increase_potential(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        property_data: Optional[Dict[str, Any]] = None,
        current_value: float = 0
    ) -> Dict[str, Any]:
        """Estimerer potensial for verdiøkning gjennom forbedringer"""
        improvement_opportunities = []
        total_potential_increase = 0
        total_estimated_cost = 0
        
        # Vurder romforbedringer
        if room_analysis.get('quality', 'average') in ['poor', 'below_average', 'average']:
            cost_per_sqm = 5000  # NOK per m² for generell oppussing
            area = property_data.get('area', 0) if property_data else dimension_analysis.get('room_measurements', {}).get('floor_area', 0) * 10
            
            if area <= 0:
                area = 20  # Antatt størrelse for et enkelt rom
                
            renovation_cost = area * cost_per_sqm
            value_increase = renovation_cost * 1.5  # 50% ROI på generell oppussing
            
            improvement_opportunities.append({
                'type': 'room_renovation',
                'description': f"Generell oppussing for å øke romkvalitet fra {room_analysis.get('quality', 'average')} til good",
                'estimated_cost': float(renovation_cost),
                'potential_value_increase': float(value_increase),
                'roi_percentage': 50.0,
                'priority': 'high' if room_analysis.get('quality') == 'poor' else 'medium'
            })
            
            total_potential_increase += value_increase
            total_estimated_cost += renovation_cost
        
        # Vurder energiforbedringer
        energy_improvements = energy_analysis.get('improvement_potential', [])
        energy_payback = energy_analysis.get('payback_period', {})
        
        for improvement in energy_improvements:
            if improvement.get('type') in ['wall_insulation', 'roof_insulation', 'window_replacement']:
                cost = improvement.get('estimated_cost', 0)
                
                # Beregn verdiøkning basert på energirating
                current_rating = energy_analysis.get('energy_rating', {}).get('rating', 'E')
                rating_value_map = {'G': 0, 'F': 1, 'E': 2, 'D': 3, 'C': 4, 'B': 5, 'A': 6}
                current_rating_value = rating_value_map.get(current_rating, 2)
                
                # Anta at dette tiltaket kan forbedre rating med 1 nivå
                new_rating_value = min(6, current_rating_value + 1)
                
                # Verdiøkning per forbedring i rating (ca. 3-5% per nivå)
                if current_value > 0:
                    value_increase = current_value * (new_rating_value - current_rating_value) * 0.04
                else:
                    # Fallback hvis current_value ikke er tilgjengelig
                    value_increase = cost * 1.2
                
                improvement_opportunities.append({
                    'type': improvement.get('type'),
                    'description': improvement.get('description', ''),
                    'estimated_cost': float(cost),
                    'potential_value_increase': float(value_increase),
                    'roi_percentage': float(value_increase / cost * 100 if cost > 0 else 0),
                    'energy_saving': improvement.get('heat_loss_reduction', 0),
                    'priority': improvement.get('priority', 'medium')
                })
                
                total_potential_increase += value_increase
                total_estimated_cost += cost
        
        # Vurder materialoppgraderinger
        for recommendation in material_analysis.get('upgrade_recommendations', []):
            material_type = recommendation.get('original_material', '')
            replacement = recommendation.get('recommended_replacement', '')
            
            if material_type and replacement:
                # Grov estimering av kostnad og verdiøkning
                area = 20  # m² (antatt)
                cost_per_sqm = {
                    'hardwood': 1500,
                    'tile': 1200,
                    'engineered_hardwood': 1000,
                    'quartz': 3000,
                    'carpet': 700,
                    'laminate': 600,
                    'vinyl': 500,
                    'drywall_with_insulation': 800
                }
                
                estimated_cost = area * cost_per_sqm.get(replacement, 1000)
                value_increase = estimated_cost * (1 + recommendation.get('estimated_improvement', 0.2))
                
                improvement_opportunities.append({
                    'type': 'material_upgrade',
                    'description': f"Oppgrader {material_type} til {replacement}",
                    'estimated_cost': float(estimated_cost),
                    'potential_value_increase': float(value_increase),
                    'roi_percentage': float(value_increase / estimated_cost * 100 if estimated_cost > 0 else 0),
                    'priority': recommendation.get('priority', 'medium')
                })
                
                total_potential_increase += value_increase
                total_estimated_cost += estimated_cost
        
        return {
            'improvement_opportunities': improvement_opportunities,
            'total_potential_increase': float(total_potential_increase),
            'total_estimated_cost': float(total_estimated_cost),
            'overall_roi_percentage': float(total_potential_increase / total_estimated_cost * 100 if total_estimated_cost > 0 else 0),
            'value_increase_percentage': float(total_potential_increase / current_value * 100 if current_value > 0 else 0)
        }

    def _analyze_rental_potential(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        location_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyserer leiepotensial for eiendommen"""
        # Basisberegning av leiepotensial
        area = dimension_analysis.get('room_measurements', {}).get('floor_area', 0) * 10
        if area <= 0 and location_data and 'area' in location_data:
            area = location_data.get('area', 0)
        
        if area <= 0:
            area = 60  # Standard leilighetsstørrelse
        
        # Basis leiepris per kvadratmeter
        base_rent_per_sqm = 200  # NOK per måned per m²
        
        # Juster basert på beliggenhet
        location_factor = 1.0
        if location_data:
            city = location_data.get('city', '').lower() if isinstance(location_data.get('city'), str) else ''
            
            if 'oslo' in city:
                location_factor = 1.5
            elif 'bergen' in city or 'trondheim' in city or 'stavanger' in city:
                location_factor = 1.3
            elif 'tromsø' in city or 'kristiansand' in city:
                location_factor = 1.2
        
        # Juster basert på romkvalitet
        quality_factor = {
            'excellent': 1.2,
            'good': 1.1,
            'average': 1.0,
            'below_average': 0.9,
            'poor': 0.8
        }.get(room_analysis.get('quality', 'average'), 1.0)
        
        # Beregn månedlig leiepris
        monthly_rent = area * base_rent_per_sqm * location_factor * quality_factor
        
        # Identifiser leieforbedringspotensial
        improvement_potential = self._identify_rental_improvement_potential(
            room_analysis, 
            monthly_rent, 
            quality_factor
        )
        
        # Vurder utleiealternativer
        rental_options = self._evaluate_rental_options(
            room_analysis, 
            dimension_analysis, 
            monthly_rent, 
            location_data
        )
        
        return {
            'monthly_rent': float(monthly_rent),
            'annual_rent': float(monthly_rent * 12),
            'rent_per_sqm': float(base_rent_per_sqm * location_factor * quality_factor),
            'occupancy_rate': 0.95,  # Antatt utleiegrad
            'net_yield': float((monthly_rent * 12 * 0.95 * 0.85) / (area * 50000) * 100),  # Antatt driftskostnader på 15%
            'improvement_potential': improvement_potential,
            'rental_options': rental_options
        }

    def _identify_rental_improvement_potential(
        self,
        room_analysis: Dict[str, Any],
        current_rent: float,
        quality_factor: float
    ) -> Dict[str, Any]:
        """Identifiserer potensial for økt leieinntekt gjennom forbedringer"""
        improvements = []
        
        # Hvis romkvaliteten er under 'god', foreslå oppussing
        if quality_factor < 1.1:
            current_quality = room_analysis.get('quality', 'average')
            target_quality = 'good' if current_quality != 'good' else 'excellent'
            
            target_factor = 1.1 if target_quality == 'good' else 1.2
            rent_increase = current_rent * (target_factor / quality_factor - 1)
            
            # Grov estimering av oppussingskostnad
            renovation_cost = current_rent * 10  # Rundt 10 måneders leie
            
            improvements.append({
                'type': 'renovation',
                'description': f"Oppussing fra {current_quality} til {target_quality} standard",
                'estimated_cost': float(renovation_cost),
                'monthly_rent_increase': float(rent_increase),
                'annual_rent_increase': float(rent_increase * 12),
                'roi_percentage': float(rent_increase * 12 / renovation_cost * 100),
                'payback_months': float(renovation_cost / rent_increase)
            })
        
        # Sjekk for spesifikke fasiliteter som kan øke leien
        missing_features = []
        for feature in ['dishwasher', 'washing_machine', 'air_conditioning']:
            if feature not in room_analysis.get('features', []):
                missing_features.append(feature)
        
        if missing_features:
            feature_costs = {
                'dishwasher': 8000,
                'washing_machine': 7000,
                'air_conditioning': 15000
            }
            
            feature_rent_increases = {
                'dishwasher': 300,
                'washing_machine': 250,
                'air_conditioning': 400
            }
            
            for feature in missing_features:
                cost = feature_costs.get(feature, 10000)
                rent_increase = feature_rent_increases.get(feature, 300)
                
                improvements.append({
                    'type': 'add_feature',
                    'feature': feature,
                    'description': f"Installasjon av {feature.replace('_', ' ')}",
                    'estimated_cost': float(cost),
                    'monthly_rent_increase': float(rent_increase),
                    'annual_rent_increase': float(rent_increase * 12),
                    'roi_percentage': float(rent_increase * 12 / cost * 100),
                    'payback_months': float(cost / rent_increase)
                })
        
        # Beregn total forbedringspotensial
        total_cost = sum(improvement.get('estimated_cost', 0) for improvement in improvements)
        total_monthly_increase = sum(improvement.get('monthly_rent_increase', 0) for improvement in improvements)
        
        return {
            'improvements': improvements,
            'total_cost': float(total_cost),
            'total_monthly_increase': float(total_monthly_increase),
            'total_annual_increase': float(total_monthly_increase * 12),
            'combined_roi': float(total_monthly_increase * 12 / total_cost * 100 if total_cost > 0 else 0)
        }

    def _evaluate_rental_options(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        base_monthly_rent: float,
        location_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Evaluerer ulike utleiealternativer"""
        options = []
        
        # Standard langtidsleie
        options.append({
            'type': 'long_term',
            'description': 'Standard langtidsleie (12+ måneder)',
            'monthly_income': float(base_monthly_rent),
            'annual_income': float(base_monthly_rent * 12),
            'occupancy_rate': 0.95,
            'management_effort': 'low',
            'risk_level': 'low',
            'pros': ['Stabil inntekt', 'Mindre administrasjon', 'Lavere slitasje'],
            'cons': ['Lavere leieinntekt enn korttidsutleie', 'Risiko for dårlige leietakere']
        })
        
        # Korttidsleie (Airbnb)
        # Leieprisen er typisk høyere, men også lavere belegg og mer administrasjon
        airbnb_nightly_rate = base_monthly_rent / 30 * 2.5  # ca. 2.5x daglig rate
        
        # Juster belegg basert på beliggenhet
        occupancy_rate = 0.65  # Standard
        if location_data:
            city = location_data.get('city', '').lower() if isinstance(location_data.get('city'), str) else ''
            
            if 'oslo' in city or 'bergen' in city:
                occupancy_rate = 0.75  # Høyere i store turistbyer
        
        airbnb_monthly = airbnb_nightly_rate * 30 * occupancy_rate
        
        options.append({
            'type': 'short_term',
            'description': 'Korttidsutleie (f.eks. Airbnb)',
            'nightly_rate': float(airbnb_nightly_rate),
            'monthly_income': float(airbnb_monthly),
            'annual_income': float(airbnb_monthly * 12),
            'occupancy_rate': float(occupancy_rate),
            'management_effort': 'high',
            'risk_level': 'medium',
            'pros': ['Høyere potensielle inntekter', 'Fleksibilitet', 'Kan bruke boligen selv innimellom'],
            'cons': ['Mer administrasjon', 'Variabel belegg', 'Høyere slitasje', 'Regulatoriske begrensninger']
        })
        
        # Hybel/rom-utleie hvis eiendommen er stor nok
        area = dimension_analysis.get('room_measurements', {}).get('floor_area', 0) * 10
        if area > 80:
            room_rent = base_monthly_rent * 0.4  # ca. 40% av full leiepris for ett rom
            
            options.append({
                'type': 'room_rental',
                'description': 'Utleie av rom/hybel (del av bolig)',
                'monthly_income': float(room_rent),
                'annual_income': float(room_rent * 12),
                'occupancy_rate': 0.9,
                'management_effort': 'medium',
                'risk_level': 'low',
                'pros': ['Kan bo i boligen samtidig', 'Mindre investeringsbehov', 'Flexibilitet'],
                'cons': ['Deler bolig med leietaker', 'Begrenset privatliv', 'Lavere total inntekt']
            })
        
        # Identifiser beste alternativ basert på ROI
        for option in options:
            option['net_annual_income'] = option['annual_income'] * option['occupancy_rate'] * 0.85  # Anta 15% driftskostnader
        
        options.sort(key=lambda x: x['net_annual_income'], reverse=True)
        if options:
            options[0]['recommended'] = True
        
        return options

    def _analyze_renovation_potential(
        self,
        room_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        structure_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        value_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyserer renoveringspotensial og ROI"""
        renovation_options = []
        
        # Samle ulike renoveringsalternativer
        # 1. Fra verdianalyse
        if 'potential_value_increase' in value_analysis:
            for opportunity in value_analysis['potential_value_increase'].get('improvement_opportunities', []):
                renovation_options.append({
                    'type': opportunity.get('type', 'unknown'),
                    'description': opportunity.get('description', ''),
                    'estimated_cost': opportunity.get('estimated_cost', 0),
                    'value_increase': opportunity.get('potential_value_increase', 0),
                    'roi_percentage': opportunity.get('roi_percentage', 0),
                    'priority': opportunity.get('priority', 'medium'),
                    'category': 'value'
                })
        
        # 2. Fra energianalyse
        for improvement in energy_analysis.get('improvement_potential', []):
            energy_saving = 0
            if 'heat_loss_reduction' in improvement:
                energy_saving = improvement['heat_loss_reduction'] * 24 * 4000 / 1000  # kWh/år
            
            # Beregn kostnadsbesparelse
            cost_saving = energy_saving * 1.5  # NOK/kWh
            
            renovation_options.append({
                'type': improvement.get('type', 'unknown'),
                'description': improvement.get('description', ''),
                'estimated_cost': improvement.get('estimated_cost', 0),
                'energy_saving_kwh': float(energy_saving),
                'annual_cost_saving': float(cost_saving),
                'value_increase': improvement.get('estimated_cost', 0) * 0.8,  # Antatt 80% av kostnad
                'roi_percentage': float(cost_saving / improvement.get('estimated_cost', 1) * 100),
                'priority': improvement.get('priority', 'medium'),
                'category': 'energy'
            })
        
        # 3. Fra materialanalyse
        for upgrade in material_analysis.get('upgrade_recommendations', []):
            renovation_options.append({
                'type': 'material_upgrade',
                'description': f"Oppgradere {upgrade.get('original_material', '')} til {upgrade.get('recommended_replacement', '')}",
                'estimated_cost': 20000,  # Placeholder - ville vært beregnet basert på område
                'value_increase': 25000,  # Placeholder - ville vært beregnet basert på område og materialkvalitet
                'roi_percentage': 25,
                'priority': upgrade.get('priority', 'medium'),
                'category': 'material'
            })
        
        # 4. Fra strukturanalyse (hvis modifikasjonspotensial er høyt)
        modification_potential = structure_analysis.get('modification_potential', {})
        open_plan_potential = modification_potential.get('open_plan_potential', {})
        
        if open_plan_potential.get('category', 'low') in ['medium', 'high']:
            renovation_options.append({
                'type': 'open_plan_conversion',
                'description': 'Konvertere til åpen planløsning',
                'estimated_cost': 100000,  # Placeholder - ville vært beregnet basert på vegger som må fjernes
                'value_increase': 150000,  # Placeholder - ville vært beregnet basert på markedsverdi
                'roi_percentage': 50,
                'priority': 'high' if open_plan_potential.get('category') == 'high' else 'medium',
                'category': 'structural'
            })
        
        # Beregn optimal renoveringspakke basert på budsjettbegrensninger
        budget_options = self._calculate_optimal_renovation_packages(renovation_options)
        
        return {
            'renovation_options': renovation_options,
            'budget_options': budget_options,
            'best_roi_options': sorted(renovation_options, key=lambda x: x.get('roi_percentage', 0), reverse=True)[:3],
            'recommendations': self._generate_renovation_recommendations(renovation_options)
        }

    def _calculate_optimal_renovation_packages(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Beregner optimale renoveringspakker for ulike budsjetter"""
        budget_levels = [100000, 250000, 500000, 1000000]  # NOK
        packages = []
        
        # Sorter alternativer etter ROI
        sorted_options = sorted(options, key=lambda x: x.get('roi_percentage', 0), reverse=True)
        
        for budget in budget_levels:
            package = {
                'budget': float(budget),
                'selected_options': [],
                'total_cost': 0.0,
                'total_value_increase': 0.0,
                'roi_percentage': 0.0
            }
            
            # Grådig algoritme - velg høyest ROI først innenfor budsjett
            for option in sorted_options:
                cost = option.get('estimated_cost', 0)
                if package['total_cost'] + cost <= budget:
                    package['selected_options'].append(option)
                    package['total_cost'] += cost
                    package['total_value_increase'] += option.get('value_increase', 0)
            
            # Beregn ROI for pakken
            if package['total_cost'] > 0:
                package['roi_percentage'] = package['total_value_increase'] / package['total_cost'] * 100
            
            packages.append(package)
        
        return packages

    def _generate_renovation_recommendations(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genererer anbefalinger for renovering basert på ulike mål"""
        recommendations = {
            'best_value': [],
            'best_energy': [],
            'best_quick_wins': [],
            'comprehensive': []
        }
        
        # Sorter etter ulike kriterier
        value_options = sorted([o for o in options if o.get('value_increase', 0) > 0], 
                              key=lambda x: x.get('value_increase', 0), reverse=True)
        
        energy_options = sorted([o for o in options if o.get('category') == 'energy'], 
                               key=lambda x: x.get('energy_saving_kwh', 0), reverse=True)
        
        quick_win_options = sorted([o for o in options if o.get('estimated_cost', 0) < 50000], 
                                  key=lambda x: x.get('roi_percentage', 0), reverse=True)
        
        # Velg topp alternativer for hver kategori
        recommendations['best_value'] = value_options[:3]
        recommendations['best_energy'] = energy_options[:3]
        recommendations['best_quick_wins'] = quick_win_options[:3]
        
        # Lag en omfattende pakke som balanserer verdi, energi og materialer
        comprehensive = []
        categories_added = set()
        
        # Først legg til topp alternativ fra hver kategori
        for option in sorted(options, key=lambda x: x.get('roi_percentage', 0), reverse=True):
            category = option.get('category', 'unknown')
            if category not in categories_added:
                comprehensive.append(option)
                categories_added.add(category)
                
                if len(categories_added) >= 4:  # Anta 4 hovedkategorier (value, energy, material, structural)
                    break
        
        # Deretter legg til flere høy-ROI alternativer
        for option in sorted(options, key=lambda x: x.get('roi_percentage', 0), reverse=True):
            if option not in comprehensive and len(comprehensive) < 6:
                comprehensive.append(option)
        
        recommendations['comprehensive'] = comprehensive
        
        return recommendations

    def _analyze_market_trends(self, location_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyserer markedstrender for området"""
        if not self.market_trends_analyzer or not location_data:
            return {
                'available': False,
                'reason': 'Manglende markedsdata eller lokasjonsinformasjon'
            }
        
        # I en reell implementasjon ville dette hentet data fra en database
        # Her returnerer vi bare noen plausible verdier
        
        city = location_data.get('city', '').lower() if isinstance(location_data.get('city'), str) else ''
        
        # Dummy data for demonstrasjon
        if 'oslo' in city:
            return {
                'available': True,
                'price_trends': {
                    'yearly_growth': 5.2,
                    'quarterly_growth': 1.3,
                    'forecast_next_year': 4.0
                },
                'market_liquidity': 'high',
                'average_days_on_market': 30,
                'supply_demand_balance': 'seller_market',
                'comparable_properties': [
                    {
                        'address': 'Eksempelveien 1',
                        'area': 85,
                        'sale_price': 5850000,
                        'price_per_sqm': 68824,
                        'days_on_market': 21
                    },
                    {
                        'address': 'Eksempelveien 2',
                        'area': 92,
                        'sale_price': 6100000,
                        'price_per_sqm': 66304,
                        'days_on_market': 35
                    }
                ]
            }
        else:
            # Generiske data for andre områder
            return {
                'available': True,
                'price_trends': {
                    'yearly_growth': 3.8,
                    'quarterly_growth': 0.9,
                    'forecast_next_year': 3.0
                },
                'market_liquidity': 'medium',
                'average_days_on_market': 45,
                'supply_demand_balance': 'balanced',
                'comparable_properties': [
                    {
                        'address': 'Eksempelveien 10',
                        'area': 90,
                        'sale_price': 3950000,
                        'price_per_sqm': 43889,
                        'days_on_market': 40
                    }
                ]
            }

    def _analyze_neighborhood(self, location_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyserer nabolaget rundt eiendommen"""
        if not self.neighborhood_analyzer or not location_data:
            return {
                'available': False,
                'reason': 'Manglende nabolagsdata eller lokasjonsinformasjon'
            }
        
        # I en reell implementasjon ville dette hentet data fra en database
        # Her returnerer vi bare noen plausible verdier
        
        # Dummy data for demonstrasjon
        return {
            'available': True,
            'amenities': {
                'schools': {
                    'proximity': 'good',
                    'quality': 'high',
                    'nearest': 'Eksempelskolen (500m)'
                },
                'public_transport': {
                    'proximity': 'excellent',
                    'options': ['bus', 'subway'],
                    'nearest': 'Eksempel holdeplass (300m)'
                },
                'shopping': {
                    'proximity': 'good',
                    'options': ['grocery', 'mall'],
                    'nearest': 'Eksempel Senter (800m)'
                },
                'healthcare': {
                    'proximity': 'medium',
                    'options': ['doctor', 'pharmacy'],
                    'nearest': 'Eksempel Legesenter (1.2km)'
                },
                'parks': {
                    'proximity': 'excellent',
                    'nearest': 'Eksempelparken (200m)'
                }
            },
            'demographics': {
                'age_distribution': {'0-18': 22, '19-34': 28, '35-64': 35, '65+': 15},
                'income_level': 'above_average',
                'education_level': 'high'
            },
            'safety': {
                'crime_rate': 'low',
                'relative_to_city': 'better'
            },
            'development_plans': {
                'upcoming_projects': 'Ny infrastruktur planlagt for 2026',
                'impact_on_property': 'positive'
            }
        }

    def _check_regulatory_compliance(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        property_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sjekker regulatorisk samsvar"""
        if not self.regulatory_compliance_checker:
            return {
                'available': False,
                'reason': 'Manglende regulatoriske data'
            }
        
        # I en reell implementasjon ville dette sjekket mot faktiske regulatoriske krav
        # Her gjør vi en forenklet vurdering
        
        compliance_issues = []
        
        # Sjekk krav til romhøyde
        ceiling_height = room_analysis.get('ceiling_height', 240)
        if ceiling_height < 240:
            compliance_issues.append({
                'type': 'ceiling_height',
                'description': f'Takhøyde ({ceiling_height}cm) er under forskriftskrav (240cm)',
                'severity': 'high',
                'remediation': 'Krever dispensasjon eller ombygging'
            })
        
        # Sjekk vinduskrav (forenklet - antar at det burde være vindu i de fleste rom)
        if 'natural_light' not in room_analysis.get('features', []) and room_analysis.get('type') in ['bedroom', 'living_room', 'kitchen']:
            compliance_issues.append({
                'type': 'natural_light',
                'description': f'Mangel på naturlig lys i {room_analysis.get("type")}',
                'severity': 'medium',
                'remediation': 'Installasjon av vindu eller takvinduer'
            })
        
        # Sjekk krav til universell utforming (forenklet)
        if 'accessibility_features' not in room_analysis.get('features', []) and room_analysis.get('type') in ['bathroom']:
            compliance_issues.append({
                'type': 'accessibility',
                'description': 'Manglende tilgjengelighet i bad',
                'severity': 'low',
                'remediation': 'Installasjon av tilgjengelighetsfunksjoner ved renovering'
            })
        
        # Vurder samlet complianc
        compliance_score = 1.0 - (len(compliance_issues) * 0.2)
        compliance_score = max(0.0, min(1.0, compliance_score))
        
        return {
            'available': True,
            'compliance_score': float(compliance_score),
            'compliance_category': 'high' if compliance_score > 0.8 else ('medium' if compliance_score > 0.5 else 'low'),
            'compliance_issues': compliance_issues,
            'regulatory_requirements': [
                'Takhøyde: Minimum 240cm i oppholdsrom',
                'Naturlig lys: Vindu i alle oppholdsrom',
                'Universell utforming: Tilgjengelighetskrav for nye bygg/større renoveringer'
            ]
        }

    def _check_building_code_compliance(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sjekker samsvar med byggtekniske forskrifter"""
        # Dette er en forenklet implementasjon
        compliance_issues = []
        
        # Sjekk romstørrelse
        room_type = room_analysis.get('type', 'unknown')
        area = room_analysis.get('size', {}).get('area', 0)
        
        if room_type == 'bedroom' and area < 7:
            compliance_issues.append({
                'type': 'room_size',
                'description': f'Soverom ({area}m²) er under minimumskrav (7m²)',
                'severity': 'medium'
            })
        
        # Sjekk rømningsvei
        if 'safety_features' not in room_analysis.get('features', []) and room_type in ['bedroom']:
            compliance_issues.append({
                'type': 'emergency_exit',
                'description': 'Mangler tydelig rømningsvei fra soverom',
                'severity': 'high'
            })
        
        # Vurder samlet compliance
        compliance_score = 1.0 - (len(compliance_issues) * 0.3)
        compliance_score = max(0.0, min(1.0, compliance_score))
        
        return {
            'compliance_score': float(compliance_score),
            'compliance_category': 'high' if compliance_score > 0.8 else ('medium' if compliance_score > 0.5 else 'low'),
            'compliance_issues': compliance_issues
        }

    def _analyze_accessibility(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyserer tilgjengelighet og universell utforming"""
        # Dette er en forenklet implementasjon
        
        accessibility_issues = []
        accessibility_features = []
        
        # Sjekk for eksisterende tilgjengelighetsfunksjoner
        if 'accessibility_features' in room_analysis.get('features', []):
            accessibility_features.append('general_accessibility_features')
        
        # Sjekk terskler (antar at de eksisterer hvis ikke spesifisert)
        accessibility_issues.append({
            'type': 'thresholds',
            'description': 'Terskler kan være en hindring for rullestolbrukere',
            'severity': 'medium',
            'remediation': 'Installere terskeleliminator eller fjerne terskler'
        })
        
        # Sjekk dørbredde (antar standard dørbredde hvis ikke spesifisert)
        detected_doors = [obj for obj in dimension_analysis.get('detected_objects', []) if obj.get('type') == 'door']
        
        for door in detected_doors:
            door_width = door.get('estimated_dimensions', {}).get('width', 0)
            if door_width < 80:
                accessibility_issues.append({
                    'type': 'door_width',
                    'description': f'Dørbredde ({door_width}cm) er under anbefalt minimum (80cm) for rullestoltilgang',
                    'severity': 'high',
                    'remediation': 'Utvide døråpning ved renovering'
                })
        
        # Beregn tilgjengelighetsscore
        base_score = 0.5  # Start med middels score
        
        # Trekk fra for hvert problem
        issue_penalty = 0.1
        base_score -= len(accessibility_issues) * issue_penalty
        
        # Legg til for hver tilgjengelighetsfunksjon
        feature_bonus = 0.1
        base_score += len(accessibility_features) * feature_bonus
        
        # Begrens score til området [0, 1]
        accessibility_score = max(0.0, min(1.0, base_score))
        
        return {
            'accessibility_score': float(accessibility_score),
            'accessibility_category': 'high' if accessibility_score > 0.7 else ('medium' if accessibility_score > 0.4 else 'low'),
            'accessibility_issues': accessibility_issues,
            'accessibility_features': accessibility_features,
            'wheelchair_accessible': accessibility_score > 0.7,
            'improvement_recommendations': self._generate_accessibility_recommendations(accessibility_issues)
        }

    def _generate_accessibility_recommendations(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Genererer anbefalinger for tilgjengelighetsforbedringer"""
        recommendations = []
        
        for issue in issues:
            recommendations.append({
                'type': issue.get('type', 'unknown'),
                'description': issue.get('remediation', 'Utbedre problemer'),
                'priority': 'high' if issue.get('severity') == 'high' else ('medium' if issue.get('severity') == 'medium' else 'low'),
                'estimated_cost': 5000 if issue.get('type') == 'thresholds' else 15000  # Forenklede kostnadsestimater
            })
        
        # Legg til generelle anbefalinger hvis det er problemer
        if issues:
            recommendations.append({
                'type': 'general_accessibility',
                'description': 'Vurdere universell utforming ved neste renovering',
                'priority': 'medium',
                'estimated_cost': 50000
            })
        
        return recommendations

    def _analyze_sustainability(
        self,
        material_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        property_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyserer bærekraftaspekter ved eiendommen"""
        if not self.sustainability_analyzer:
            return {
                'available': False,
                'reason': 'Manglende bærekraftsdata'
            }
        
        # Beregn materialbærekraft
        material_sustainability = material_analysis.get('sustainability_score', 0.5)
        
        # Beregn energibærekraft basert på energimerking
        energy_rating = energy_analysis.get('energy_rating', {}).get('rating', 'E')
        energy_sustainability = {
            'A': 1.0, 'B': 0.85, 'C': 0.7, 'D': 0.55, 'E': 0.4, 'F': 0.25, 'G': 0.1
        }.get(energy_rating, 0.4)
        
        # Beregn samlet bærekraftsscore
        sustainability_score = material_sustainability * 0.4 + energy_sustainability * 0.6
        
        # Identifiser forbedringsmuligheter
        improvement_opportunities = []
        
        # Fra energiforbedringer
        for improvement in energy_analysis.get('improvement_potential', []):
            if 'heat_loss_reduction' in improvement:
                energy_saving = improvement['heat_loss_reduction'] * 24 * 4000 / 1000  # kWh/år
                co2_reduction = energy_saving * 0.017  # kg CO2 (Norsk strømmiks)
                
                improvement_opportunities.append({
                    'type': improvement.get('type', 'unknown'),
                    'description': improvement.get('description', ''),
                    'energy_saving_kwh': float(energy_saving),
                    'co2_reduction_kg': float(co2_reduction),
                    'estimated_cost': improvement.get('estimated_cost', 0),
                    'category': 'energy'
                })
        
        # Fra materialoppgraderinger
        for material in material_analysis.get('detected_materials', []):
            for option in material.get('sustainability', {}).get('improvement_options', []):
                improvement_opportunities.append({
                    'type': option.get('option', 'unknown'),
                    'description': option.get('description', ''),
                    'sustainability_gain': option.get('sustainability_gain', 0),
                    'estimated_cost': 20000,  # Placeholder - ville bli beregnet
                    'category': 'material'
                })
        
        # Beregn samlet CO2-reduksjon
        total_co2_reduction = sum(opt.get('co2_reduction_kg', 0) for opt in improvement_opportunities 
                                 if 'co2_reduction_kg' in opt)
        
        # Generer bærekraftsrapport
        return {
            'available': True,
            'sustainability_score': float(sustainability_score),
            'sustainability_category': 'high' if sustainability_score > 0.7 else 
                                       ('medium' if sustainability_score > 0.4 else 'low'),
            'components': {
                'material_sustainability': float(material_sustainability),
                'energy_sustainability': float(energy_sustainability)
            },
            'improvement_opportunities': improvement_opportunities,
            'potential_improvements': {
                'score_improvement': min(1.0, sustainability_score + 0.3),
                'total_co2_reduction_kg': float(total_co2_reduction),
                'equivalent_trees': float(total_co2_reduction / 21)  # Et tre absorberer ca. 21 kg CO2/år
            },
            'sustainability_certifications': self._check_sustainability_certifications(
                sustainability_score, 
                energy_analysis
            )
        }

    def _check_sustainability_certifications(
        self,
        sustainability_score: float,
        energy_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Sjekker potensielle bærekraftssertifiseringer"""
        certifications = []
        
        # BREEAM-NOR
        energy_rating = energy_analysis.get('energy_rating', {}).get('rating', 'E')
        if sustainability_score > 0.7 and energy_rating in ['A', 'B']:
            certifications.append({
                'name': 'BREEAM-NOR',
                'level': 'Very Good',
                'eligible': True,
                'requirements': 'Krever formell vurdering av sertifisert assessor'
            })
        elif sustainability_score > 0.5 and energy_rating in ['A', 'B', 'C']:
            certifications.append({
                'name': 'BREEAM-NOR',
                'level': 'Good',
                'eligible': True,
                'requirements': 'Krever formell vurdering av sertifisert assessor'
            })
        else:
            certifications.append({
                'name': 'BREEAM-NOR',
                'level': 'Ikke kvalifisert',
                'eligible': False,
                'requirements': 'Forbedre energieffektivitet og bærekraftige materialer'
            })
        
        # Svanemerket
        if sustainability_score > 0.8 and energy_rating in ['A']:
            certifications.append({
                'name': 'Svanemerket',
                'level': 'Sertifisert',
                'eligible': True,
                'requirements': 'Krever formell søknad og vurdering'
            })
        else:
            certifications.append({
                'name': 'Svanemerket',
                'level': 'Ikke kvalifisert',
                'eligible': False,
                'requirements': 'Krever omfattende bærekraftstiltak og energieffektivitet'
            })
        
        return certifications

    def _generate_recommendations(
        self,
        room_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        structure_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        value_analysis: Dict[str, Any],
        rental_analysis: Dict[str, Any],
        renovation_analysis: Dict[str, Any],
        development_potential: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genererer samlet anbefalinger basert på alle analyser"""
        all_recommendations = []
        
        # Samle anbefalinger fra ulike analyser
        # 1. Fra verdianalyse
        if 'potential_value_increase' in value_analysis:
            for opportunity in value_analysis['potential_value_increase'].get('improvement_opportunities', []):
                all_recommendations.append({
                    'type': opportunity.get('type', 'unknown'),
                    'description': opportunity.get('description', ''),
                    'estimated_cost': opportunity.get('estimated_cost', 0),
                    'benefit': {
                        'type': 'value_increase',
                        'amount': opportunity.get('potential_value_increase', 0)
                    },
                    'roi_percentage': opportunity.get('roi_percentage', 0),
                    'priority': opportunity.get('priority', 'medium'),
                    'category': 'value',
                    'timeline_weeks': 4 if 'renovation' in opportunity.get('type', '') else 2
                })
        
        # 2. Fra energianalyse
        for improvement in energy_analysis.get('improvement_potential', []):
            energy_saving = 0
            if 'heat_loss_reduction' in improvement:
                energy_saving = improvement['heat_loss_reduction'] * 24 * 4000 / 1000  # kWh/år
            
            # Beregn kostnadsbesparelse
            cost_saving = energy_saving * 1.5  # NOK/kWh
            
            all_recommendations.append({
                'type': improvement.get('type', 'unknown'),
                'description': improvement.get('description', ''),
                'estimated_cost': improvement.get('estimated_cost', 0),
                'benefit': {
                    'type': 'cost_saving',
                    'amount': float(cost_saving),
                    'recurring': True
                },
                'roi_percentage': float(cost_saving / improvement.get('estimated_cost', 1) * 100),
                'priority': improvement.get('priority', 'medium'),
                'category': 'energy',
                'timeline_weeks': 3 if 'insulation' in improvement.get('type', '') else 1
            })
        
        # 3. Fra utleieanalyse
        if 'improvement_potential' in rental_analysis:
            for improvement in rental_analysis['improvement_potential'].get('improvements', []):
                all_recommendations.append({
                    'type': improvement.get('type', 'unknown'),
                    'description': improvement.get('description', ''),
                    'estimated_cost': improvement.get('estimated_cost', 0),
                    'benefit': {
                        'type': 'rental_income',
                        'amount': improvement.get('annual_rent_increase', 0),
                        'recurring': True
                    },
                    'roi_percentage': improvement.get('roi_percentage', 0),
                    'priority': 'high' if improvement.get('roi_percentage', 0) > 25 else 'medium',
                    'category': 'rental',
                    'timeline_weeks': 2
                })
        
        # 4. Fra materialanalyse
        for upgrade in material_analysis.get('upgrade_recommendations', []):
            estimated_cost = 20000  # Placeholder
            value_increase = 25000  # Placeholder
            
            all_recommendations.append({
                'type': 'material_upgrade',
                'description': f"Oppgradere {upgrade.get('original_material', '')} til {upgrade.get('recommended_replacement', '')}",
                'estimated_cost': float(estimated_cost),
                'benefit': {
                    'type': 'value_increase',
                    'amount': float(value_increase)
                },
                'roi_percentage': float(value_increase / estimated_cost * 100),
                'priority': upgrade.get('priority', 'medium'),
                'category': 'material',
                'timeline_weeks': 2
            })
        
        # 5. Fra strukturanalyse
        modification_potential = structure_analysis.get('modification_potential', {})
        if modification_potential.get('open_plan_potential', {}).get('category', 'low') in ['medium', 'high']:
            estimated_cost = 100000  # Placeholder
            value_increase = 150000  # Placeholder
            
            all_recommendations.append({
                'type': 'open_plan_conversion',
                'description': 'Konvertere til åpen planløsning',
                'estimated_cost': float(estimated_cost),
                'benefit': {
                    'type': 'value_increase',
                    'amount': float(value_increase)
                },
                'roi_percentage': float(value_increase / estimated_cost * 100),
                'priority': 'high' if modification_potential.get('open_plan_potential', {}).get('category') == 'high' else 'medium',
                'category': 'structural',
                'timeline_weeks': 6
            })
        
        # Sorter alle anbefalinger etter ROI
        all_recommendations.sort(key=lambda x: x.get('roi_percentage', 0), reverse=True)
        
        # Klassifiser anbefalinger etter kategori
        recommendations_by_category = {}
        for rec in all_recommendations:
            category = rec.get('category', 'other')
            if category not in recommendations_by_category:
                recommendations_by_category[category] = []
            recommendations_by_category[category].append(rec)
        
        # Finn de beste anbefalingene for ulike budsjetter
        budget_levels = [50000, 100000, 250000, 500000]  # NOK
        budget_recommendations = []
        
        for budget in budget_levels:
            selected_recs = []
            remaining_budget = budget
            
            # Grådig algoritme basert på ROI
            for rec in all_recommendations:
                cost = rec.get('estimated_cost', 0)
                if cost <= remaining_budget:
                    selected_recs.append(rec)
                    remaining_budget -= cost
            
            total_benefit = 0
            for rec in selected_recs:
                benefit = rec.get('benefit', {})
                if benefit.get('type') == 'value_increase':
                    total_benefit += benefit.get('amount', 0)
                elif benefit.get('type') == 'cost_saving' and benefit.get('recurring', False):
                    total_benefit += benefit.get('amount', 0) * 5  # Antatt 5 års verdi
                elif benefit.get('type') == 'rental_income' and benefit.get('recurring', False):
                    total_benefit += benefit.get('amount', 0) * 5  # Antatt 5 års verdi
            
            budget_recommendations.append({
                'budget': float(budget),
                'recommendations': selected_recs,
                'total_cost': float(budget - remaining_budget),
                'total_benefit': float(total_benefit),
                'roi_percentage': float(total_benefit / (budget - remaining_budget) * 100 if (budget - remaining_budget) > 0 else 0)
            })
        
        return {
            'prioritized': all_recommendations[:5],  # Topp 5 anbefalinger
            'by_category': recommendations_by_category,
            'by_budget': budget_recommendations,
            'roi_optimized': sorted(all_recommendations, key=lambda x: x.get('roi_percentage', 0), reverse=True)[:5]
        }

    def _calculate_development_potential(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        structure_analysis: Dict[str, Any],
        layout_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Beregner utviklingspotensial for eiendommen"""
        development_options = []
        
        # Vurder utvidelsesepotensialet
        extension_potential = structure_analysis.get('modification_potential', {}).get('extension_potential', {})
        if extension_potential.get('category', 'low') in ['medium', 'high']:
            # Beregn omtrentlig potensielt areal
            area = dimension_analysis.get('room_measurements', {}).get('floor_area', 0) * 10
            potential_additional_area = area * 0.3  # Antar 30% utvidelse
            
            development_options.append({
                'type': 'extension',
                'description': 'Utvidelse av eksisterende bygningsmasse',
                'potential_area_increase': float(potential_additional_area),
                'estimated_cost': float(potential_additional_area * 25000),  # NOK per m²
                'value_increase': float(potential_additional_area * 40000),  # NOK per m²
                'roi_percentage': 60.0,
                'complexity': 'high',
                'timeline_months': 6
            })
        
        # Vurder loftsutbyggingspotensial
        if room_analysis.get('type') == 'attic' or 'attic' in [area.get('id') for area in layout_analysis.get('areas', [])]:
            potential_attic_area = 30  # m² (antatt)
            
            development_options.append({
                'type': 'attic_conversion',
                'description': 'Konvertering av loft til beboelsesrom',
                'potential_area_increase': float(potential_attic_area),
                'estimated_cost': float(potential_attic_area * 15000),  # NOK per m²
                'value_increase': float(potential_attic_area * 30000),  # NOK per m²
                'roi_percentage': 100.0,
                'complexity': 'medium',
                'timeline_months': 3
            })
        
        # Vurder underetasjeutbyggingspotensial
        if room_analysis.get('type') == 'basement' or 'basement' in [area.get('id') for area in layout_analysis.get('areas', [])]:
            potential_basement_area = 40  # m² (antatt)
            
            development_options.append({
                'type': 'basement_conversion',
                'description': 'Konvertering av kjeller til beboelsesrom',
                'potential_area_increase': float(potential_basement_area),
                'estimated_cost': float(potential_basement_area * 12000),  # NOK per m²
                'value_increase': float(potential_basement_area * 20000),  # NOK per m²
                'roi_percentage': 67.0,
                'complexity': 'medium',
                'timeline_months': 4
            })
        
        # Vurder garasjekonverteringspotensial
        if room_analysis.get('type') == 'garage' or 'garage' in [area.get('id') for area in layout_analysis.get('areas', [])]:
            potential_garage_area = 25  # m² (antatt)
            
            development_options.append({
                'type': 'garage_conversion',
                'description': 'Konvertering av garasje til beboelsesrom',
                'potential_area_increase': float(potential_garage_area),
                'estimated_cost': float(potential_garage_area * 10000),  # NOK per m²
                'value_increase': float(potential_garage_area * 18000),  # NOK per m²
                'roi_percentage': 80.0,
                'complexity': 'low',
                'timeline_months': 2
            })
        
        # Velg beste utviklingsalternativ basert på ROI
        best_option = None
        if development_options:
            best_option = max(development_options, key=lambda x: x.get('roi_percentage', 0))
        
        return {
            'development_options': development_options,
            'best_option': best_option,
            'total_potential_area_increase': float(sum(option.get('potential_area_increase', 0) for option in development_options)),
            'average_roi': float(sum(option.get('roi_percentage', 0) for option in development_options) / len(development_options) if development_options else 0)
        }

    def _generate_3d_model_parameters(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genererer parametre for 3D-modellering"""
        # Hent rommål
        room_dimensions = dimension_analysis.get('room_measurements', {})
        width = room_dimensions.get('width', 4)  # meter
        length = room_dimensions.get('length', 5)  # meter
        height = room_dimensions.get('ceiling_height', 2.4)  # meter
        
        # Hent objekter i rommet
        objects = dimension_analysis.get('detected_objects', [])
        
        # Konverter til 3D-modellparametere
        return {
            'dimensions': {
                'width': float(width),
                'length': float(length),
                'height': float(height)
            },
            'walls': [
                {'id': 'wall_1', 'start': [0, 0, 0], 'end': [width, 0, 0], 'height': height},
                {'id': 'wall_2', 'start': [width, 0, 0], 'end': [width, length, 0], 'height': height},
                {'id': 'wall_3', 'start': [width, length, 0], 'end': [0, length, 0], 'height': height},
                {'id': 'wall_4', 'start': [0, length, 0], 'end': [0, 0, 0], 'height': height}
            ],
            'objects': [self._convert_object_to_3d(obj, width, length) for obj in objects],
            'lighting': self._generate_lighting_parameters(room_analysis),
            'camera_positions': [
                {'id': 'front', 'position': [width/2, -3, height/2], 'target': [width/2, length/2, height/2]},
                {'id': 'top', 'position': [width/2, length/2, height+2], 'target': [width/2, length/2, 0]},
                {'id': 'corner', 'position': [-2, -2, height], 'target': [width/2, length/2, height/2]}
            ]
        }

    def _convert_object_to_3d(self, obj: Dict[str, Any], room_width: float, room_length: float) -> Dict[str, Any]:
        """Konverterer detektert objekt til 3D-parameter"""
        obj_type = obj.get('type', 'unknown')
        bbox = obj.get('bounding_box', [0, 0, 0, 0])
        
        # Konverter bounding box til 3D-koordinater
        x1, y1, x2, y2 = bbox
        obj_width = (x2 - x1) / 1000 * room_width  # Konverter fra piksel-koordinater til meter
        obj_length = (y2 - y1) / 1000 * room_length
        obj_x = x1 / 1000 * room_width
        obj_y = y1 / 1000 * room_length
        
        # Sett objekthøyde basert på type
        if obj_type == 'door':
            obj_height = 2.0
        elif obj_type == 'window':
            obj_height = 1.2
            obj_z = 1.0  # Startposisjon for vindu er typisk 1m over gulvet
        elif obj_type == 'furniture':
            obj_height = 0.8
            obj_z = 0
        else:
            obj_height = 0.5
            obj_z = 0
        
        return {
            'id': f'{obj_type}_{int(obj_x*100)}_{int(obj_y*100)}',
            'type': obj_type,
            'position': [float(obj_x), float(obj_y), float(obj_z if 'obj_z' in locals() else 0)],
            'dimensions': [float(obj_width), float(obj_length), float(obj_height)]
        }

    def _generate_material_mappings(self, material_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genererer materialtilordninger for 3D-modell"""
        # Konverter detekterte materialer til 3D-materialkart
        material_mappings = {
            'walls': {'type': 'paint', 'color': [1, 1, 1], 'roughness': 0.9},
            'floor': {'type': 'generic', 'color': [0.8, 0.8, 0.8], 'roughness': 0.7},
            'ceiling': {'type': 'paint', 'color': [1, 1, 1], 'roughness': 0.9}
        }
        
        # Oppdater med detekterte materialer
        for material in material_analysis.get('detected_materials', []):
            material_type = material.get('type', 'unknown')
            
            if material_type in ['drywall', 'brick', 'concrete', 'stone', 'wood']:
                material_mappings['walls'] = self._get_material_properties(material_type)
            elif material_type in ['hardwood', 'laminate', 'tile', 'carpet', 'vinyl', 'concrete']:
                material_mappings['floor'] = self._get_material_properties(material_type)
        
        return material_mappings

    def _get_material_properties(self, material_type: str) -> Dict[str, Any]:
        """Henter materialegenskaper for 3D-rendering"""
        material_properties = {
            'hardwood': {'type': 'wood', 'color': [0.6, 0.4, 0.2], 'roughness': 0.3, 'texture': 'wood_parquet'},
            'laminate': {'type': 'wood', 'color': [0.7, 0.5, 0.3], 'roughness': 0.4, 'texture': 'wood_laminate'},
            'tile': {'type': 'ceramic', 'color': [0.9, 0.9, 0.9], 'roughness': 0.1, 'texture': 'tile_ceramic'},
            'carpet': {'type': 'fabric', 'color': [0.5, 0.5, 0.5], 'roughness': 0.95, 'texture': 'fabric_carpet'},
            'vinyl': {'type': 'vinyl', 'color': [0.8, 0.8, 0.8], 'roughness': 0.6, 'texture': 'vinyl_floor'},
            'concrete': {'type': 'concrete', 'color': [0.7, 0.7, 0.7], 'roughness': 0.7, 'texture': 'concrete_smooth'},
            'drywall': {'type': 'paint', 'color': [1, 1, 1], 'roughness': 0.9, 'texture': 'wall_painted'},
            'brick': {'type': 'brick', 'color': [0.7, 0.3, 0.2], 'roughness': 0.8, 'texture': 'wall_brick'},
            'stone': {'type': 'stone', 'color': [0.6, 0.6, 0.6], 'roughness': 0.7, 'texture': 'wall_stone'},
            'wood': {'type': 'wood', 'color': [0.6, 0.4, 0.2], 'roughness': 0.5, 'texture': 'wall_wood'}
        }
        
        return material_properties.get(material_type, {'type': 'generic', 'color': [0.8, 0.8, 0.8], 'roughness': 0.5})

    def _analyze_lighting(
        self,
        room_analysis: Dict[str, Any],
        structure_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyserer lysforhold i rommet"""
        has_natural_light = 'natural_light' in room_analysis.get('features', [])
        
        # Antall vinduer
        window_objects = [obj for obj in structure_analysis.get('structural_elements', []) 
                         if obj.get('type') == 'window_opening']
        
        # Beregn lysintensitet basert på romtype og naturlig lysinnfall
        base_intensity = 1.0
        if not has_natural_light:
            base_intensity = 0.7
        
        # Juster intensitet basert på romtype
        room_type = room_analysis.get('type', 'unknown')
        type_intensity_map = {
            'living_room': 1.0,
            'kitchen': 0.9,
            'bedroom': 0.8,
            'bathroom': 0.7,
            'hallway': 0.6,
            'basement': 0.5,
            'attic': 0.5,
            'garage': 0.4
        }
        type_intensity = type_intensity_map.get(room_type, 0.8)
        
        light_intensity = base_intensity * type_intensity
        
        # Generer lyskilder
        light_sources = [
            {'type': 'ambient', 'intensity': max(0.3, light_intensity * 0.5), 'color': [1, 1, 1]}
        ]
        
        # Legg til tak-/hovedlys
        light_sources.append({
            'type': 'point',
            'position': [0.5, 0.5, 0.9],  # Relativ posisjon i rommet (midten av taket)
            'intensity': light_intensity,
            'color': [1, 0.95, 0.9],  # Litt varm hvit
            'falloff': 1.0
        })
        
        # Legg til vindusbelysning hvis det er naturlig lys
        if has_natural_light and window_objects:
            for i, window in enumerate(window_objects):
                # Antar at vinduer er på veggene (x=0, x=1, y=0, y=1)
                x1, y1, x2, y2 = window.get('bounding_box', [0, 0, 0, 0])
                window_center_x = (x1 + x2) / 2 / 1000
                window_center_y = (y1 + y2) / 2 / 1000
                
                # Juster posisjon for å være på en vegg
                position = [0.5, 0.5, 0.5]  # Standard midten av rommet
                
                if abs(window_center_x) < abs(window_center_x - 1):
                    # Nærmere x=0 veggen
                    position[0] = 0.05
                    position[1] = window_center_y
                elif abs(window_center_x - 1) < abs(window_center_x):
                    # Nærmere x=1 veggen
                    position[0] = 0.95
                    position[1] = window_center_y
                elif abs(window_center_y) < abs(window_center_y - 1):
                    # Nærmere y=0 veggen
                    position[0] = window_center_x
                    position[1] = 0.05
                else:
                    # Nærmere y=1 veggen
                    position[0] = window_center_x
                    position[1] = 0.95
                
                light_sources.append({
                    'type': 'directional',
                    'position': position,
                    'direction': [-0.3, -0.3, -0.5] if i % 2 == 0 else [0.3, -0.3, -0.5],  # Varierer litt retningen
                    'intensity': light_intensity * 1.5,
                    'color': [1, 0.98, 0.95],  # Dagslys
                    'falloff': 0.5
                })
        
        return {
            'natural_light': has_natural_light,
            'light_intensity': float(light_intensity),
            'light_quality': 'high' if light_intensity > 0.8 else ('medium' if light_intensity > 0.5 else 'low'),
            'light_sources': light_sources,
            'improvement_suggestions': self._generate_lighting_improvement_suggestions(
                has_natural_light, 
                light_intensity,
                room_analysis
            )
        }

    def _generate_lighting_improvement_suggestions(
        self,
        has_natural_light: bool,
        light_intensity: float,
        room_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Genererer forslag til forbedring av lysforhold"""
        suggestions = []
        
        if not has_natural_light:
            suggestions.append({
                'type': 'add_windows',
                'description': 'Legge til vinduer for bedre naturlig lysinnfall',
                'estimated_cost': 50000,
                'impact': 'high',
                'value_increase': 'medium'
            })
            
            suggestions.append({
                'type': 'add_skylight',
                'description': 'Installere takvinduer for økt naturlig lys',
                'estimated_cost': 30000,
                'impact': 'high',
                'value_increase': 'medium'
            })
        
        if light_intensity < 0.7:
            suggestions.append({
                'type': 'improve_artificial_lighting',
                'description': 'Oppgradere belysningsløsninger med strategisk plasserte lyskilder',
                'estimated_cost': 15000,
                'impact': 'medium',
                'value_increase': 'low'
            })
        
        # Forslag for spesifikke romtyper
        room_type = room_analysis.get('type', 'unknown')
        
        if room_type == 'kitchen':
            suggestions.append({
                'type': 'task_lighting',
                'description': 'Legge til arbeidsbelysning over benkeplater og matlaging',
                'estimated_cost': 8000,
                'impact': 'medium',
                'value_increase': 'low'
            })
        elif room_type == 'living_room':
            suggestions.append({
                'type': 'ambient_lighting',
                'description': 'Installere dimmbare lyskilder for variabel stemningsbelysning',
                'estimated_cost': 12000,
                'impact': 'medium',
                'value_increase': 'low'
            })
        
        return suggestions

    def _generate_lighting_parameters(self, room_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genererer lysparametere for 3D-visualisering"""
        # Forenklet implementasjon
        has_natural_light = 'natural_light' in room_analysis.get('features', [])
        room_type = room_analysis.get('type', 'unknown')
        
        # Basisintensitet basert på romtype
        intensity_map = {
            'living_room': 0.8,
            'kitchen': 0.9,
            'bedroom': 0.6,
            'bathroom': 0.7,
            'hallway': 0.5,
            'basement': 0.4,
            'attic': 0.5,
            'garage': 0.4
        }
        
        base_intensity = intensity_map.get(room_type, 0.7)
        if has_natural_light:
            base_intensity += 0.2
        
        # Generere lyskilder
        lights = [
            {
                'type': 'ambient',
                'intensity': max(0.2, base_intensity * 0.3),
                'color': [1, 1, 1]
            },
            {
                'type': 'directional',
                'direction': [-0.5, -0.5, -1],
                'intensity': base_intensity,
                'color': [1, 0.98, 0.92]
            }
        ]
        
        if has_natural_light:
            lights.append({
                'type': 'point',
                'position': [0.3, 0.3, 0.8],
                'intensity': 0.7,
                'color': [1, 0.98, 0.95],
                'falloff': 0.8
            })
        
        return {
            'lights': lights,
            'shadows': True,
            'ambient_occlusion': True,
            'exposure': 1.0
        }

    def _generate_renovation_visualization(
        self,
        recommendations: Dict[str, Any],
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genererer parametre for visualisering av renoveringsalternativer"""
        # Velg topp anbefalinger
        top_recommendations = recommendations.get('prioritized', [])[:3]
        
        visualizations = []
        for i, recommendation in enumerate(top_recommendations):
            rec_type = recommendation.get('type', '')
            
            # Lag visualiseringsparametere basert på anbefalingstype
            visualization = {
                'id': f'renovation_{i}',
                'title': recommendation.get('description', 'Renovering'),
                'before_after': True,
                'estimated_cost': recommendation.get('estimated_cost', 0),
                'roi_percentage': recommendation.get('roi_percentage', 0)
            }
            
            if 'wall' in rec_type:
                visualization['target'] = 'walls'
                visualization['material_change'] = {
                    'from': 'current',
                    'to': 'insulated_wall'
                }
            elif 'window' in rec_type:
                visualization['target'] = 'windows'
                visualization['object_change'] = {
                    'from': 'single_pane',
                    'to': 'energy_efficient'
                }
            elif 'floor' in rec_type:
                visualization['target'] = 'floor'
                visualization['material_change'] = {
                    'from': 'current',
                    'to': 'insulated_floor'
                }
            elif 'open_plan' in rec_type:
                visualization['target'] = 'layout'
                visualization['structure_change'] = {
                    'remove_walls': [0, 2]  # Eksempelvegger som fjernes
                }
            elif 'material_upgrade' in rec_type:
                visualization['target'] = 'materials'
                visualization['material_change'] = {
                    'from': 'current',
                    'to': 'upgraded'
                }
            
            visualizations.append(visualization)
        
        return {
            'room_parameters': self._generate_3d_model_parameters(room_analysis, dimension_analysis),
            'renovations': visualizations
        }

    def _calculate_confidence_score(
        self,
        room_analysis: Dict[str, Any],
        dimension_analysis: Dict[str, Any],
        material_analysis: Dict[str, Any],
        structure_analysis: Dict[str, Any]
    ) -> float:
        """Beregner en samlet konfidenscore for analysen"""
        # Samle konfidensscorer fra ulike analyser
        confidence_scores = []
        
        # Rom-analysekonfidens
        if 'confidence' in room_analysis:
            confidence_scores.append(room_analysis['confidence'])
        
        # Dimensjonsanalysekonfidens
        if 'detected_objects' in dimension_analysis:
            confidence_values = [obj.get('confidence', 0) for obj in dimension_analysis['detected_objects']]
            if confidence_values:
                confidence_scores.append(sum(confidence_values) / len(confidence_values))
        
        # Materialanalysekonfidens
        if 'detected_materials' in material_analysis:
            confidence_values = [material.get('confidence', 0) for material in material_analysis['detected_materials']]
            if confidence_values:
                confidence_scores.append(sum(confidence_values) / len(confidence_values))
        
        # Strukturanalysekonfidens
        if 'structural_elements' in structure_analysis:
            confidence_values = [element.get('confidence', 0) for element in structure_analysis['structural_elements']]
            if confidence_values:
                confidence_scores.append(sum(confidence_values) / len(confidence_values))
        
        # Beregn gjennomsnittlig konfidens
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.5  # Standard middelverdi
