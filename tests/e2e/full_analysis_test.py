import pytest
from playwright.async_api import Page, expect
from typing import Dict, Any

class TestFullAnalysis:
    async def test_complete_property_analysis(self, page: Page):
        """Test fullstendig analyse av eiendom"""
        # Gå til hjemmesiden
        await page.goto('/')
        
        # Test opplasting av bilde
        await page.set_input_files('input[type="file"]', 'test_data/test_property.jpg')
        await expect(page.locator('.upload-status')).to_contain_text('Opplasting vellykket')

        # Test adressesøk
        search_input = page.locator('input[placeholder="Søk etter adresse"]')
        await search_input.fill('Testveien 1, 3005 Drammen')
        await search_input.press('Enter')
        await expect(page.locator('.property-info')).to_be_visible()

        # Test 3D-visualisering
        model_viewer = page.locator('.property-viewer')
        await expect(model_viewer).to_be_visible()
        
        # Test modellkontroller
        controls = page.locator('.model-controls')
        await controls.locator('button:has-text("Zoom inn")').click()
        await controls.locator('button:has-text("Rotér")').click()
        
        # Test analyseresultater
        results = page.locator('.analysis-results')
        await expect(results.locator('text=Utviklingspotensial')).to_be_visible()
        await expect(results.locator('text=Energianalyse')).to_be_visible()
        
        # Test Enova-integrasjon
        enova_section = results.locator('.enova-support')
        await expect(enova_section).to_be_visible()
        await expect(enova_section).to_contain_text('Støtteordninger')

        # Test dokumentgenerering
        await page.click('button:has-text("Generer rapport")')
        download_promise = page.wait_for_download()
        await page.click('button:has-text("Last ned PDF")')
        download = await download_promise
        assert download.suggested_filename.endswith('.pdf')

        # Test betalingsintegrasjon
        await page.click('button:has-text("Kjøp full rapport")')
        await expect(page.locator('iframe[name="stripe-frame"]')).to_be_visible()
        
        # Test at stripe-integrasjonen fungerer
        stripe_frame = await page.frame_locator('iframe[name="stripe-frame"]')
        await stripe_frame.locator('input[name="cardNumber"]').fill('4242424242424242')
        await stripe_frame.locator('input[name="cardExpiry"]').fill('12/25')
        await stripe_frame.locator('input[name="cardCvc"]').fill('123')
        await stripe_frame.locator('button[type="submit"]').click()
        
        # Verifiser at betalingen ble gjennomført
        await expect(page.locator('.payment-success')).to_be_visible()

async def test_municipality_integration(self, page: Page):
    """Test integrasjon mot kommunale systemer"""
    # Gå til søkesiden
    await page.goto('/search')
    
    # Test søk på gårds- og bruksnummer
    gnr_input = page.locator('input[name="gnr"]')
    bnr_input = page.locator('input[name="bnr"]')
    await gnr_input.fill('10')
    await bnr_input.fill('20')
    await page.click('button:has-text("Søk")')
    
    # Verifiser at kommunale data blir hentet
    await expect(page.locator('.municipal-data')).to_be_visible()
    await expect(page.locator('.property-history')).to_be_visible()
    
    # Test visning av reguleringsplan
    await page.click('button:has-text("Vis reguleringsplan")')
    await expect(page.locator('.zoning-map')).to_be_visible()
    
    # Test visning av tidligere byggesaker
    await page.click('button:has-text("Tidligere saker")')
    await expect(page.locator('.case-history')).to_be_visible()

async def test_energy_analysis(self, page: Page):
    """Test energianalyse og Enova-integrasjon"""
    # Gå til energianalysesiden
    await page.goto('/energy-analysis')
    
    # Last opp bygningsinformasjon
    await page.set_input_files('input[type="file"]', 'test_data/building_info.json')
    
    # Verifiser energianalyse
    energy_section = page.locator('.energy-analysis')
    await expect(energy_section).to_be_visible()
    await expect(energy_section).to_contain_text('Energikarakter')
    
    # Test Enova-støtteberegning
    await page.click('button:has-text("Beregn støtte")')
    support_section = page.locator('.enova-support')
    await expect(support_section).to_be_visible()
    await expect(support_section).to_contain_text('Tilgjengelige støtteordninger')
    
    # Test energitiltaksberegning
    await page.click('button:has-text("Beregn tiltak")')
    measures_section = page.locator('.energy-measures')
    await expect(measures_section).to_be_visible()
    await expect(measures_section).to_contain_text('Anbefalte tiltak')