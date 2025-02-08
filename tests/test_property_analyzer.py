import pytest
from ai_modules.property_analyzer.property_analyzer import PropertyAnalyzer
from ai_modules.floor_plan_analyzer.src.floor_plan_analyzer import FloorPlanAnalyzer
from pathlib import Path
import numpy as np
import cv2
import json

class TestPropertyAnalyzer:
    @pytest.fixture
    def property_analyzer(self):
        return PropertyAnalyzer()
    
    @pytest.fixture
    def floor_plan_analyzer(self):
        return FloorPlanAnalyzer()
    
    @pytest.fixture
    def sample_floor_plan(self, tmp_path):
        """Create a sample floor plan image for testing"""
        image_path = tmp_path / "test_floor_plan.png"
        # Create a simple floor plan-like image
        img = np.zeros((800, 600, 3), dtype=np.uint8)
        # Draw some rooms
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), 2)
        cv2.rectangle(img, (350, 100), (500, 200), (255, 255, 255), 2)
        cv2.imwrite(str(image_path), img)
        return image_path
    
    @pytest.fixture
    def sample_property_data(self):
        return {
            "address": "Testveien 1",
            "municipality": "Drammen",
            "property_id": "1234/56",
            "area": 150.5,
            "floors": 2,
            "has_basement": True,
            "has_attic": True,
            "year_built": 1985,
            "current_usage": "residential",
            "lot_size": 500.0,
            "building_coverage": 30.0
        }
    
    def test_analyze_floor_plan(self, property_analyzer, sample_floor_plan):
        """Test floor plan analysis"""
        result = property_analyzer.analyze_floor_plan(sample_floor_plan)
        
        assert "rooms" in result
        assert "total_area" in result
        assert "room_dimensions" in result
        assert len(result["rooms"]) >= 2  # Should detect at least 2 rooms
        assert result["total_area"] > 0
    
    def test_calculate_rental_potential(self, property_analyzer, sample_property_data):
        """Test rental potential calculation"""
        potential = property_analyzer.calculate_rental_potential(sample_property_data)
        
        assert "potential_units" in potential
        assert "estimated_income" in potential
        assert "renovation_cost" in potential
        assert "roi" in potential
        assert potential["potential_units"] > 0
        assert potential["estimated_income"] > 0
    
    def test_analyze_zoning_regulations(self, property_analyzer, sample_property_data):
        """Test zoning regulations analysis"""
        regulations = property_analyzer.analyze_zoning_regulations(
            sample_property_data["municipality"],
            sample_property_data["property_id"]
        )
        
        assert "allowed_usage" in regulations
        assert "max_height" in regulations
        assert "max_coverage" in regulations
        assert "min_distance" in regulations
    
    def test_generate_development_options(self, property_analyzer, sample_property_data):
        """Test development options generation"""
        options = property_analyzer.generate_development_options(sample_property_data)
        
        assert len(options) > 0
        for option in options:
            assert "type" in option
            assert "description" in option
            assert "estimated_cost" in option
            assert "estimated_return" in option
    
    def test_analyze_building_structure(self, property_analyzer, sample_floor_plan):
        """Test building structure analysis"""
        structure = property_analyzer.analyze_building_structure(sample_floor_plan)
        
        assert "load_bearing_walls" in structure
        assert "modification_possibilities" in structure
        assert "structural_constraints" in structure
    
    def test_calculate_renovation_costs(self, property_analyzer, sample_property_data):
        """Test renovation cost calculation"""
        costs = property_analyzer.calculate_renovation_costs(
            sample_property_data,
            renovation_type="rental_conversion"
        )
        
        assert "total_cost" in costs
        assert "breakdown" in costs
        assert len(costs["breakdown"]) > 0
        assert costs["total_cost"] > 0
    
    def test_analyze_energy_efficiency(self, property_analyzer, sample_property_data):
        """Test energy efficiency analysis"""
        energy_analysis = property_analyzer.analyze_energy_efficiency(sample_property_data)
        
        assert "current_rating" in energy_analysis
        assert "potential_rating" in energy_analysis
        assert "recommended_improvements" in energy_analysis
        assert "estimated_savings" in energy_analysis
    
    def test_check_building_regulations(self, property_analyzer, sample_property_data):
        """Test building regulations compliance check"""
        compliance = property_analyzer.check_building_regulations(
            sample_property_data,
            municipality="Drammen"
        )
        
        assert "compliant" in compliance
        assert "violations" in compliance
        assert "required_permits" in compliance
    
    def test_estimate_market_value(self, property_analyzer, sample_property_data):
        """Test market value estimation"""
        valuation = property_analyzer.estimate_market_value(
            sample_property_data,
            include_potential=True
        )
        
        assert "current_value" in valuation
        assert "potential_value" in valuation
        assert "value_increase" in valuation
        assert valuation["current_value"] > 0
    
    def test_generate_3d_model(self, property_analyzer, sample_floor_plan):
        """Test 3D model generation"""
        model = property_analyzer.generate_3d_model(sample_floor_plan)
        
        assert "model_data" in model
        assert "textures" in model
        assert "dimensions" in model
    
    def test_analyze_property_division(self, property_analyzer, sample_property_data):
        """Test property division analysis"""
        division = property_analyzer.analyze_property_division(sample_property_data)
        
        assert "possible" in division
        assert "options" in division
        if division["possible"]:
            assert len(division["options"]) > 0
    
    def test_generate_documentation(self, property_analyzer, sample_property_data):
        """Test documentation generation"""
        docs = property_analyzer.generate_documentation(
            sample_property_data,
            document_type="building_application"
        )
        
        assert "application_forms" in docs
        assert "drawings" in docs
        assert "supporting_documents" in docs
    
    @pytest.mark.asyncio
    async def test_async_property_analysis(
        self,
        property_analyzer,
        sample_property_data,
        sample_floor_plan
    ):
        """Test complete async property analysis"""
        analysis = await property_analyzer.analyze_property_async(
            property_data=sample_property_data,
            floor_plan_path=sample_floor_plan
        )
        
        assert "rental_potential" in analysis
        assert "zoning_regulations" in analysis
        assert "development_options" in analysis
        assert "renovation_costs" in analysis
        assert "energy_analysis" in analysis
        assert "market_valuation" in analysis
        assert "3d_model" in analysis
    
    def test_error_handling(self, property_analyzer):
        """Test error handling"""
        # Test with invalid floor plan
        with pytest.raises(ValueError):
            property_analyzer.analyze_floor_plan("nonexistent_file.png")
        
        # Test with invalid property data
        with pytest.raises(ValueError):
            property_analyzer.calculate_rental_potential({})
        
        # Test with invalid municipality
        with pytest.raises(ValueError):
            property_analyzer.analyze_zoning_regulations("InvalidCity", "123/45")
    
    def test_result_validation(self, property_analyzer, sample_property_data):
        """Test result validation"""
        # Test rental potential validation
        potential = property_analyzer.calculate_rental_potential(sample_property_data)
        assert potential["estimated_income"] >= 0
        assert potential["renovation_cost"] >= 0
        assert 0 <= potential["roi"] <= 100
        
        # Test development options validation
        options = property_analyzer.generate_development_options(sample_property_data)
        for option in options:
            assert option["estimated_cost"] >= 0
            assert option["estimated_return"] >= 0
    
    @pytest.mark.parametrize("renovation_type,expected_cost_range", [
        ("basic", (50000, 200000)),
        ("moderate", (200000, 500000)),
        ("extensive", (500000, 2000000))
    ])
    def test_renovation_cost_ranges(
        self,
        property_analyzer,
        sample_property_data,
        renovation_type,
        expected_cost_range
    ):
        """Test renovation cost ranges for different types"""
        costs = property_analyzer.calculate_renovation_costs(
            sample_property_data,
            renovation_type=renovation_type
        )
        
        min_cost, max_cost = expected_cost_range
        assert min_cost <= costs["total_cost"] <= max_cost