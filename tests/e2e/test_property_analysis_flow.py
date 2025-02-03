import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

@pytest.fixture
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

@pytest.mark.e2e
class TestPropertyAnalysisFlow:
    def test_complete_property_analysis(self, driver):
        """
        Tester hele flyten fra innlogging til ferdig analyserapport
        """
        # 1. Åpne hjemmesiden
        driver.get("https://eiendomsmuligheter.no")
        assert "Eiendomsmuligheter" in driver.title

        # 2. Logg inn
        login_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "login-button"))
        )
        login_button.click()

        email_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "email"))
        )
        email_input.send_keys("test@example.com")

        password_input = driver.find_element(By.NAME, "password")
        password_input.send_keys("testpassword123")

        submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        submit_button.click()

        # 3. Naviger til analysesiden
        analyze_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "analyze-link"))
        )
        analyze_link.click()

        # 4. Fyll inn adresse
        address_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "address-input"))
        )
        address_input.send_keys("Storgata 1, Drammen")

        # 5. Last opp testfiler
        file_input = driver.find_element(By.ID, "file-upload")
        test_file_path = os.path.join(os.path.dirname(__file__), "../data/test_floor_plan.jpg")
        file_input.send_keys(test_file_path)

        # 6. Start analyse
        analyze_button = driver.find_element(By.ID, "start-analysis")
        analyze_button.click()

        # 7. Vent på analyseresultater
        results_container = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "analysis-results"))
        )

        # 8. Verifiser resultater
        assert "Analyseresultater" in results_container.text
        assert "Utviklingspotensial" in results_container.text
        assert "Enova-støtte" in results_container.text

        # 9. Sjekk 3D-visualisering
        model_viewer = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "model-viewer"))
        )
        assert model_viewer.is_displayed()

        # 10. Generer dokumenter
        generate_docs_button = driver.find_element(By.ID, "generate-documents")
        generate_docs_button.click()

        # 11. Verifiser dokumentgenerering
        download_links = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "document-download"))
        )
        assert len(download_links) >= 3  # Minst 3 dokumenter skal være generert

        # 12. Sjekk betalingsintegrasjon
        payment_button = driver.find_element(By.ID, "proceed-to-payment")
        payment_button.click()

        # 13. Verifiser Stripe iframe
        stripe_frame = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "stripe-iframe"))
        )
        assert stripe_frame.is_displayed()

    def test_error_handling(self, driver):
        """
        Tester feilhåndtering i systemet
        """
        driver.get("https://eiendomsmuligheter.no/analyze")

        # Test ugyldig adresse
        address_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "address-input"))
        )
        address_input.send_keys("Ikke en ekte adresse 123")
        
        analyze_button = driver.find_element(By.ID, "start-analysis")
        analyze_button.click()

        error_message = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "error-message"))
        )
        assert "Kunne ikke finne adressen" in error_message.text

    def test_performance(self, driver):
        """
        Tester ytelsen til systemet
        """
        start_time = time.time()
        
        driver.get("https://eiendomsmuligheter.no/analyze")
        
        # Mål tiden det tar å laste siden
        load_time = time.time() - start_time
        assert load_time < 3  # Siden bør laste på under 3 sekunder

        # Test responstid for adressesøk
        address_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "address-input"))
        )
        
        start_time = time.time()
        address_input.send_keys("Storg")
        
        # Vent på autocomplete-forslag
        suggestions = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "address-suggestions"))
        )
        
        autocomplete_time = time.time() - start_time
        assert autocomplete_time < 1  # Autocomplete bør respondere innen 1 sekund

    def test_responsive_design(self, driver):
        """
        Tester responsivt design
        """
        # Test mobil visning
        driver.set_window_size(375, 812)  # iPhone X dimensjoner
        driver.get("https://eiendomsmuligheter.no/analyze")
        
        # Sjekk at mobilmenyen er synlig
        mobile_menu = driver.find_element(By.CLASS_NAME, "mobile-menu")
        assert mobile_menu.is_displayed()

        # Test tablet visning
        driver.set_window_size(768, 1024)  # iPad dimensjoner
        
        # Sjekk at tablet-layouten er korrekt
        content_wrapper = driver.find_element(By.CLASS_NAME, "content-wrapper")
        assert "tablet-layout" in content_wrapper.get_attribute("class")