import sys
import os
import unittest
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import numpy as np
import logging

# Legg til prosjektets rotmappe i PYTHONPATH for å kunne importere fra ai_modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_modules.AlterraML import (
    AlterraML, 
    PropertyData, 
    RegulationRule, 
    AnalysisResult, 
    BuildingPotential,
    EnergyProfile,
    TerrainData
)

# Test konfigurasjon
@pytest.fixture
def alterra_config():
    """Konfigurasjon for testing"""
    return {
        'model_path': tempfile.mkdtemp(),
        'cache_path': tempfile.mkdtemp(),
        'use_gpu': False,
        'lazy_loading': True
    }

@pytest.fixture
def alterra_instance(alterra_config):
    """Fixture for AlterraML-instans"""
    with patch('ai_modules.AlterraML.AlterraML._load_config', return_value=alterra_config):
        instance = AlterraML()
        yield instance
        # Rydde opp etter test
        for path in [alterra_config['model_path'], alterra_config['cache_path']]:
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path)

@pytest.fixture
def property_data():
    """Test property data"""
    return PropertyData(
        property_id="TEST-123",
        address="Testveien 1, 0123 Oslo",
        municipality_id="0301",
        zoning_category="residential",
        lot_size=500.0,
        current_utilization=0.3,
        building_height=10.0,
        floor_area_ratio=0.7
    )

@pytest.fixture
def terrain_data():
    """Test terrain data"""
    return TerrainData(
        property_id="TEST-123",
        width=100.0,
        depth=100.0,
        resolution=64,
        include_surroundings=True,
        include_buildings=True
    )

class TestAlterraML(unittest.TestCase):
    def setUp(self):
        """Sett opp testmiljøet før hver test."""
        self.alterra = AlterraML(use_gpu=False, model_path="local")
        
    def test_instance_creation(self):
        """Test at AlterraML-instansen opprettes riktig."""
        self.assertIsInstance(self.alterra, AlterraML)
        self.assertEqual(self.alterra.use_gpu, False)
        self.assertEqual(self.alterra.model_path, "local")
        
    @patch('ai_modules.AlterraML.AlterraML._load_regulation_model')
    def test_initialize(self, mock_load_model):
        """Test initialisering av modellen."""
        mock_load_model.return_value = MagicMock()
        self.alterra.initialize()
        self.assertTrue(self.alterra.is_initialized)
        mock_load_model.assert_called_once()
        
    def test_property_data_creation(self):
        """Test at PropertyData-objekter kan opprettes korrekt."""
        property_data = PropertyData(
            property_id="123",
            address="Testgata 1, Oslo",
            zoning_category="bolig",
            lot_size=500,
            current_utilization=200,
            building_height=8.5,
            floor_area_ratio=0.4
        )
        
        self.assertEqual(property_data.property_id, "123")
        self.assertEqual(property_data.address, "Testgata 1, Oslo")
        self.assertEqual(property_data.lot_size, 500)
        self.assertEqual(property_data.current_utilization, 200)
        
    @patch('ai_modules.AlterraML.AlterraML._analyze_property_regulations')
    @patch('ai_modules.AlterraML.AlterraML._calculate_building_potential')
    def test_analyze_property(self, mock_calc_potential, mock_analyze_regs):
        """Test analyse av eiendom."""
        # Opprett mock-objekter
        mock_analyze_regs.return_value = [
            RegulationRule(id="1", rule_type="max_height", value=10.0, description="Maks høyde")
        ]
        mock_calc_potential.return_value = BuildingPotential(
            max_buildable_area=300,
            max_height=10.0,
            max_units=3,
            optimal_configuration="3 leiligheter"
        )
        
        property_data = PropertyData(
            property_id="123",
            address="Testgata 1, Oslo",
            zoning_category="bolig",
            lot_size=500,
            current_utilization=200,
            building_height=8.5,
            floor_area_ratio=0.4
        )
        
        # Merk at AlterraML.initialize() normalt må kjøres før analyze_property
        self.alterra.is_initialized = True
        
        # Kjør funksjonen som skal testes
        result = self.alterra.analyze_property(property_data)
        
        # Verifiser resultatet
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.property_id, "123")
        mock_analyze_regs.assert_called_once()
        mock_calc_potential.assert_called_once()
        
    @patch('ai_modules.AlterraML.AlterraML._fetch_energy_data')
    def test_calculate_energy_profile(self, mock_fetch_energy):
        """Test beregning av energiprofil."""
        mock_fetch_energy.return_value = {
            "heating_demand": 150,
            "cooling_demand": 50,
            "primary_energy_source": "electricity",
            "energy_class": "C"
        }
        
        property_data = PropertyData(
            property_id="123",
            address="Testgata 1, Oslo",
            zoning_category="bolig",
            lot_size=500,
            current_utilization=200,
            building_height=8.5,
            floor_area_ratio=0.4
        )
        
        self.alterra.is_initialized = True
        energy_profile = self.alterra.calculate_energy_profile(property_data)
        
        self.assertIsInstance(energy_profile, EnergyProfile)
        self.assertEqual(energy_profile.energy_class, "C")
        mock_fetch_energy.assert_called_once()
        
    def test_validate_property_data(self):
        """Test validering av eiendomsdata."""
        # Gyldig eiendomsdata
        valid_data = PropertyData(
            property_id="123",
            address="Testgata 1, Oslo",
            zoning_category="bolig",
            lot_size=500,
            current_utilization=200,
            building_height=8.5,
            floor_area_ratio=0.4
        )
        
        # Test at ingen unntak kastes for gyldig data
        try:
            self.alterra._validate_property_data(valid_data)
        except ValueError:
            self.fail("_validate_property_data() raised ValueError unexpectedly!")
            
        # Ugyldig eiendomsdata (mangler adresse)
        invalid_data = PropertyData(
            property_id="123",
            address="",  # Tom adresse
            zoning_category="bolig",
            lot_size=500,
            current_utilization=200,
            building_height=8.5,
            floor_area_ratio=0.4
        )
        
        # Test at et unntak kastes for ugyldig data
        with self.assertRaises(ValueError):
            self.alterra._validate_property_data(invalid_data)

# Tester
@pytest.mark.asyncio
async def test_analyze_property(alterra_instance, property_data):
    """Test av eiendomsanalyse"""
    # Kjør analysen
    result = await alterra_instance.analyze_property(property_data)
    
    # Verifiser resultatet
    assert result is not None
    assert "property_id" in result
    assert result["property_id"] == property_data.property_id
    assert "address" in result
    assert result["address"] == property_data.address
    
    # Verifiser byggepotensial
    assert "building_potential" in result
    building_potential = result["building_potential"]
    assert hasattr(building_potential, "max_buildable_area")
    assert building_potential.max_buildable_area > 0
    
    # Test at ROI-beregningen er fornuftig
    assert "roi_estimate" in result
    assert isinstance(result["roi_estimate"], (int, float))
    assert 0 <= result["roi_estimate"] <= 1.0  # ROI bør være mellom 0 og 100%

@pytest.mark.asyncio
async def test_analyze_property_with_minimal_data(alterra_instance):
    """Test av eiendomsanalyse med minimale data"""
    # Opprett minimal property data
    minimal_data = PropertyData(
        address="Minimal test",
        lot_size=100.0,
        current_utilization=0.0,
        building_height=0.0,
        floor_area_ratio=0.0
    )
    
    # Kjør analysen
    result = await alterra_instance.analyze_property(minimal_data)
    
    # Verifiser at analysen gjennomføres uten feil
    assert result is not None
    assert "property_id" in result
    assert "building_potential" in result

@pytest.mark.asyncio
async def test_analyze_property_with_empty_data(alterra_instance):
    """Test av eiendomsanalyse med tomme data"""
    # Kjør analysen med None
    result = await alterra_instance.analyze_property(None)
    
    # Verifiser at fallback-analysen brukes
    assert result is not None
    assert "property_id" in result
    assert "building_potential" in result

@pytest.mark.asyncio
async def test_generate_terrain(alterra_instance, terrain_data):
    """Test av terrenggenerering"""
    # Lag midlertidige filer
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as heightmap_file, \
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as texture_file:
        
        heightmap_path = heightmap_file.name
        texture_path = texture_file.name
        
        try:
            # Kjør terrenggenerering
            result = await alterra_instance.generate_terrain(
                terrain_data, 
                heightmap_path, 
                texture_path, 
                texture_type="satellite"
            )
            
            # Verifiser resultatet
            assert result is not None
            assert "metadata" in result
            metadata = result["metadata"]
            assert metadata["property_id"] == terrain_data.property_id
            assert metadata["width"] == terrain_data.width
            assert metadata["depth"] == terrain_data.depth
            
            # Verifiser geografiske grenser
            assert "bounds" in result
            bounds = result["bounds"]
            assert "north" in bounds
            assert "south" in bounds
            assert "east" in bounds
            assert "west" in bounds
            
            # Verifiser at filene er opprettet
            assert os.path.exists(heightmap_path)
            assert os.path.exists(texture_path)
            assert os.path.getsize(heightmap_path) > 0
            assert os.path.getsize(texture_path) > 0
        finally:
            # Rydd opp
            for path in [heightmap_path, texture_path]:
                if os.path.exists(path):
                    os.unlink(path)

@pytest.mark.asyncio
async def test_generate_building(alterra_instance):
    """Test av bygningsgenerering"""
    # Enkel bygningsdata
    building_data = {
        "property_id": "TEST-123",
        "building_type": "residential",
        "floors": 2,
        "width": 10.0,
        "depth": 15.0,
        "height": 6.0
    }
    
    # Lag midlertidige filer
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as model_file, \
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as thumbnail_file:
        
        model_path = model_file.name
        thumbnail_path = thumbnail_file.name
        
        try:
            # Kjør bygningsgenerering
            result = await alterra_instance.generate_building(
                building_data, 
                model_path, 
                thumbnail_path
            )
            
            # Verifiser resultatet
            assert result is not None
            assert "metadata" in result
            metadata = result["metadata"]
            assert metadata["property_id"] == building_data["property_id"]
            assert metadata["building_type"] == building_data["building_type"]
            assert metadata["floors"] == building_data["floors"]
            
            # Verifiser at filene er opprettet
            assert os.path.exists(model_path)
            assert os.path.exists(thumbnail_path)
            assert os.path.getsize(model_path) > 0
            assert os.path.getsize(thumbnail_path) > 0
        finally:
            # Rydd opp
            for path in [model_path, thumbnail_path]:
                if os.path.exists(path):
                    os.unlink(path)

def test_initialization(alterra_config):
    """Test av initialisering"""
    with patch('ai_modules.AlterraML.AlterraML._load_config', return_value=alterra_config):
        # Test standard initialisering
        alterra = AlterraML()
        assert alterra.config == alterra_config
        
        # Test GPU-overstyring
        alterra = AlterraML(use_gpu=True)
        assert alterra.config['use_gpu'] is True
        
        alterra = AlterraML(use_gpu=False)
        assert alterra.config['use_gpu'] is False

@patch('ai_modules.AlterraML.get_onnxruntime')
def test_model_loading(mock_get_onnx, alterra_instance, alterra_config):
    """Test av modellinnlasting"""
    # Oppsett av mock for onnx
    mock_onnx = MagicMock()
    mock_session = MagicMock()
    mock_get_onnx.return_value = mock_onnx
    mock_onnx.InferenceSession.return_value = mock_session
    
    # Forbered en testmodell
    model_name = "test_model"
    model_file = "test_model.onnx"
    alterra_instance.config['models'] = {model_name: model_file}
    
    # Sikre at mappene eksisterer
    os.makedirs(alterra_config['model_path'], exist_ok=True)
    
    # Lag en dummy ONNX-fil
    model_path = os.path.join(alterra_config['model_path'], model_file)
    with open(model_path, 'wb') as f:
        f.write(b'dummy onnx content')
    
    # Test at _load_model fungerer
    with patch('ai_modules.AlterraML.AlterraML._get_onnx_metadata', return_value={}):
        model_data = alterra_instance._load_model(model_name)
        
        # Verifiser at modellen ble lastet
        assert model_data is not None
        assert model_data['type'] == 'onnx'
        assert model_data['model'] == mock_session
        
        # Verifiser at InferenceSession ble kalt med riktige parametre
        mock_onnx.InferenceSession.assert_called_once()
        args, kwargs = mock_onnx.InferenceSession.call_args
        assert args[0] == model_path
        
        # Verifiser at modellen ble cachet
        assert model_name in alterra_instance.models

@patch('requests.get')
def test_model_download(mock_get, alterra_instance, alterra_config):
    """Test av modellnedlasting"""
    # Oppsett av mock for requests
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content.return_value = [b'dummy onnx content']
    mock_get.return_value = mock_response
    
    # Forbered modellkonfigurasjon
    model_name = "test_model"
    model_file = "test_model.onnx"
    alterra_instance.config['models'] = {model_name: model_file}
    
    # Fjern eventuelle eksisterende filer
    model_path = os.path.join(alterra_config['model_path'], model_file)
    if os.path.exists(model_path):
        os.unlink(model_path)
    
    # Test at modellnedlasting forsøkes
    with patch('ai_modules.AlterraML.get_onnxruntime') as mock_get_onnx:
        # Sett opp mocks for videre operasjoner
        mock_onnx = MagicMock()
        mock_session = MagicMock()
        mock_get_onnx.return_value = mock_onnx
        mock_onnx.InferenceSession.return_value = mock_session
        
        with patch('ai_modules.AlterraML.AlterraML._get_onnx_metadata', return_value={}):
            # Forsøk å laste modell som ikke finnes
            with patch.dict(os.environ, {"MODEL_BASE_URL": "https://test.url"}):
                model_data = alterra_instance._load_model(model_name)
                
                # Verifiser at modellnedlasting ble forsøkt
                mock_get.assert_called_once_with(
                    f"https://test.url/{model_file}", 
                    stream=True
                )
                
                # Verifiser at modellen ble lastet
                assert model_data is not None
                assert model_data['type'] == 'onnx'
                
                # Verifiser at filen ble skrevet
                assert os.path.exists(model_path)

if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 