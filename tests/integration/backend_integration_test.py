import pytest
import aiohttp
import asyncio
from typing import Dict, Any
from datetime import datetime

class TestBackendIntegration:
    @pytest.fixture
    async def client(self):
        async with aiohttp.ClientSession() as session:
            yield session

    async def test_property_analysis_flow(self, client):
        """Test komplett analyseflyt"""
        # Test opplasting av bilde
        file_data = {
            'file': open('test_data/test_property.jpg', 'rb')
        }
        async with client.post('/api/property/analyze/upload', data=file_data) as response:
            assert response.status == 200
            upload_result = await response.json()
            assert 'file_id' in upload_result

        # Test analyse basert på opplastet fil
        analysis_data = {
            'file_id': upload_result['file_id']
        }
        async with client.post('/api/property/analyze', json=analysis_data) as response:
            assert response.status == 200
            analysis_result = await response.json()
            assert 'property_info' in analysis_result
            assert 'analysis_results' in analysis_result

        # Test henting av kommunale data
        property_id = analysis_result['property_info']['property_id']
        async with client.get(f'/api/municipality/property/{property_id}') as response:
            assert response.status == 200
            municipal_data = await response.json()
            assert 'regulations' in municipal_data
            assert 'property_history' in municipal_data

    async def test_energy_analysis_flow(self, client):
        """Test energianalyse og Enova-integrasjon"""
        # Test energianalyse
        property_data = {
            'address': 'Testveien 1, 3005 Drammen',
            'build_year': 1985,
            'size': 150
        }
        async with client.post('/api/property/energy-analysis', json=property_data) as response:
            assert response.status == 200
            energy_result = await response.json()
            assert 'current_rating' in energy_result
            assert 'potential_rating' in energy_result
            assert 'improvements' in energy_result

        # Test Enova-støtteberegning
        async with client.post('/api/enova/support-options', json=energy_result) as response:
            assert response.status == 200
            support_options = await response.json()
            assert len(support_options) > 0
            assert 'amount' in support_options[0]
            assert 'requirements' in support_options[0]

    async def test_payment_flow(self, client):
        """Test betalingsintegrasjon"""
        # Test opprettelse av betalingssesjon
        payment_data = {
            'amount': 299,
            'currency': 'nok',
            'email': 'test@example.com'
        }
        async with client.post('/api/payment/create-session', json=payment_data) as response:
            assert response.status == 200
            session_data = await response.json()
            assert 'session_id' in session_data
            assert 'client_secret' in session_data

        # Test verifisering av betaling
        verify_data = {
            'session_id': session_data['session_id']
        }
        async with client.post('/api/payment/verify', json=verify_data) as response:
            assert response.status == 200
            verify_result = await response.json()
            assert verify_result['status'] == 'complete'

    async def test_document_generation(self, client):
        """Test dokumentgenerering"""
        # Test generering av rapport
        report_data = {
            'property_id': 'test-123',
            'analysis_results': {},
            'energy_analysis': {},
            'municipal_data': {}
        }
        async with client.post('/api/documents/generate-report', json=report_data) as response:
            assert response.status == 200
            report_result = await response.json()
            assert 'report_url' in report_result
            assert 'report_id' in report_result

        # Test generering av byggesøknad
        application_data = {
            'property_id': 'test-123',
            'application_type': 'conversion',
            'project_details': {}
        }
        async with client.post('/api/documents/generate-application', json=application_data) as response:
            assert response.status == 200
            application_result = await response.json()
            assert 'application_url' in application_result
            assert 'application_id' in application_result

    async def test_error_handling(self, client):
        """Test feilhåndtering"""
        # Test ugyldig filformat
        file_data = {
            'file': open('test_data/invalid.txt', 'rb')
        }
        async with client.post('/api/property/analyze/upload', data=file_data) as response:
            assert response.status == 400
            error_result = await response.json()
            assert 'error' in error_result

        # Test ugyldig adresse
        analysis_data = {
            'address': 'Ikke en ekte adresse 123'
        }
        async with client.post('/api/property/analyze', json=analysis_data) as response:
            assert response.status == 400
            error_result = await response.json()
            assert 'error' in error_result

        # Test manglende betalingsinformasjon
        payment_data = {
            'amount': 299
            # Mangler påkrevd email
        }
        async with client.post('/api/payment/create-session', json=payment_data) as response:
            assert response.status == 400
            error_result = await response.json()
            assert 'error' in error_result