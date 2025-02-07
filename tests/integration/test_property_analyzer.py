"""
Integrasjonstester for Property Analyzer
"""
import pytest
from backend.services.property_analyzer import PropertyAnalyzer
from backend.models.property import Property
from backend.models.analysis import Analysis

def test_property_analysis_flow(db_session, test_property_data):
    """Test hele analyseprosessen for en eiendom"""
    # Opprett property analyzer
    analyzer = PropertyAnalyzer()
    
    # Opprett test eiendom
    property = Property(**test_property_data)
    db_session.add(property)
    db_session.commit()
    
    # Test grunnleggende analyse
    analysis = analyzer.analyze_property(property.id)
    assert analysis.status == "completed"
    assert analysis.property_id == property.id
    assert "development_potential" in analysis.result
    
    # Test reguleringsplan analyse
    zoning_analysis = analyzer.analyze_zoning_regulations(property.id)
    assert "max_height" in zoning_analysis
    assert "max_bya" in zoning_analysis
    assert "allowed_usage" in zoning_analysis
    
    # Test utleiemuligheter
    rental_analysis = analyzer.analyze_rental_potential(property.id)
    assert "potential_units" in rental_analysis
    assert "estimated_income" in rental_analysis
    assert len(rental_analysis["potential_units"]) > 0

def test_floor_plan_analysis(db_session, test_property_data):
    """Test analyse av plantegninger"""
    analyzer = PropertyAnalyzer()
    
    # Opprett test eiendom med plantegning
    property = Property(**test_property_data)
    db_session.add(property)
    db_session.commit()
    
    # Test plantegningsanalyse
    floor_plan_analysis = analyzer.analyze_floor_plan(
        property.id,
        "test_floor_plan.pdf"
    )
    
    assert "total_area" in floor_plan_analysis
    assert "rooms" in floor_plan_analysis
    assert "modification_potential" in floor_plan_analysis
    
    # Sjekk romdetektering
    rooms = floor_plan_analysis["rooms"]
    assert len(rooms) > 0
    assert all("area" in room for room in rooms)
    assert all("type" in room for room in rooms)

def test_development_potential(db_session, test_property_data):
    """Test analyse av utviklingspotensial"""
    analyzer = PropertyAnalyzer()
    
    # Opprett test eiendom
    property = Property(**test_property_data)
    db_session.add(property)
    db_session.commit()
    
    # Test utviklingsanalyse
    development_analysis = analyzer.analyze_development_potential(property.id)
    
    assert "max_potential_area" in development_analysis
    assert "recommended_developments" in development_analysis
    assert "estimated_costs" in development_analysis
    assert "roi_estimate" in development_analysis
    
    # Sjekk anbefalinger
    recommendations = development_analysis["recommended_developments"]
    assert len(recommendations) > 0
    assert all("description" in rec for rec in recommendations)
    assert all("estimated_cost" in rec for rec in recommendations)
    assert all("potential_value" in rec for rec in recommendations)

def test_municipality_regulations(db_session, test_property_data):
    """Test sjekk av kommunale regler"""
    analyzer = PropertyAnalyzer()
    
    # Opprett test eiendom i Drammen
    property = Property(**test_property_data)
    db_session.add(property)
    db_session.commit()
    
    # Test reguleringsanalyse
    regulation_analysis = analyzer.analyze_municipality_regulations(property.id)
    
    assert "zoning_plan" in regulation_analysis
    assert "building_restrictions" in regulation_analysis
    assert "allowed_usage" in regulation_analysis
    
    # Sjekk Drammen-spesifikke regler
    drammen_rules = regulation_analysis["municipality_specific"]
    assert "parking_requirements" in drammen_rules
    assert "min_outdoor_area" in drammen_rules
    assert "max_bya_percentage" in drammen_rules

def test_3d_visualization(db_session, test_property_data):
    """Test 3D visualisering av eiendom"""
    analyzer = PropertyAnalyzer()
    
    # Opprett test eiendom
    property = Property(**test_property_data)
    db_session.add(property)
    db_session.commit()
    
    # Test 3D modellgenerering
    visualization = analyzer.generate_3d_model(property.id)
    
    assert "model_url" in visualization
    assert "textures" in visualization
    assert "scene_data" in visualization
    
    # Sjekk Omniverse-kompatibilitet
    assert visualization["format"] == "usd"  # Universal Scene Description
    assert "omniverse_metadata" in visualization
    assert visualization["scene_data"]["version"] >= 1.0