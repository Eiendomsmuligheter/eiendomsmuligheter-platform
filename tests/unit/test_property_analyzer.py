import pytest
from backend.services.property_analyzer import PropertyAnalyzer
from backend.models.property import Property
from backend.models.analysis_result import AnalysisResult

@pytest.fixture
def property_analyzer():
    return PropertyAnalyzer()

@pytest.fixture
def sample_property():
    return Property(
        address="Storgata 1",
        municipality="Drammen",
        gnr=1,
        bnr=1,
        area=500.0,
        existing_buildings=[{
            "type": "house",
            "area": 150.0,
            "floors": 2,
            "usage": "residential"
        }]
    )

class TestPropertyAnalyzer:
    @pytest.mark.unit
    def test_analyze_property_basic(self, property_analyzer, sample_property):
        result = property_analyzer.analyze(sample_property)
        assert isinstance(result, AnalysisResult)
        assert result.property_id == sample_property.id
        assert result.status == "completed"

    @pytest.mark.unit
    def test_calculate_development_potential(self, property_analyzer, sample_property):
        potential = property_analyzer.calculate_development_potential(sample_property)
        assert isinstance(potential, dict)
        assert "max_utilization" in potential
        assert "current_utilization" in potential
        assert "potential_area" in potential
        assert potential["current_utilization"] < potential["max_utilization"]

    @pytest.mark.unit
    def test_validate_municipality_rules(self, property_analyzer, sample_property):
        rules = property_analyzer.get_municipality_rules(sample_property.municipality)
        assert isinstance(rules, dict)
        assert "zoning_plan" in rules
        assert "building_regulations" in rules

    @pytest.mark.integration
    def test_ocr_analysis(self, property_analyzer):
        with open("tests/data/sample_floor_plan.jpg", "rb") as f:
            result = property_analyzer.analyze_image(f.read())
        assert result["success"]
        assert "room_data" in result
        assert "total_area" in result

    @pytest.mark.integration
    def test_3d_model_generation(self, property_analyzer, sample_property):
        model = property_analyzer.generate_3d_model(sample_property)
        assert model["format"] == "gltf"
        assert "model_data" in model
        assert model["success"]

    @pytest.mark.security
    def test_input_validation(self, property_analyzer):
        with pytest.raises(ValueError):
            property_analyzer.analyze(None)
        with pytest.raises(ValueError):
            property_analyzer.analyze(Property(address="", municipality=""))

    @pytest.mark.performance
    def test_analysis_performance(self, property_analyzer, sample_property):
        import time
        start_time = time.time()
        property_analyzer.analyze(sample_property)
        duration = time.time() - start_time
        assert duration < 5.0  # Analyse bør ta mindre enn 5 sekunder

    @pytest.mark.unit
    def test_energy_analysis(self, property_analyzer, sample_property):
        energy_data = property_analyzer.analyze_energy_efficiency(sample_property)
        assert "current_rating" in energy_data
        assert "potential_rating" in energy_data
        assert "recommendations" in energy_data

    @pytest.mark.integration
    def test_enova_support_calculation(self, property_analyzer, sample_property):
        support_data = property_analyzer.calculate_enova_support(sample_property)
        assert "available_programs" in support_data
        assert "total_potential_support" in support_data
        assert isinstance(support_data["total_potential_support"], float)

    @pytest.mark.unit
    def test_building_code_compliance(self, property_analyzer, sample_property):
        compliance = property_analyzer.check_building_code_compliance(sample_property)
        assert "compliant" in compliance
        assert "violations" in compliance
        assert "recommendations" in compliance

    @pytest.mark.integration
    def test_document_generation(self, property_analyzer, sample_property):
        docs = property_analyzer.generate_documents(sample_property)
        assert "building_application" in docs
        assert "situation_plan" in docs
        assert "technical_drawings" in docs
        assert all(doc["generated"] for doc in docs.values())