import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_property_analysis_workflow():
    """End-to-end test for property analysis workflow"""
    driver = webdriver.Firefox()
    try:
        # Navigate to homepage
        driver.get("https://eiendomsmuligheter.no")
        
        # Test address search
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "address-search"))
        )
        search_box.send_keys("Tollbugata 1, Drammen")
        search_box.submit()
        
        # Verify property info is loaded
        property_info = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "property-info"))
        )
        assert "Tollbugata 1" in property_info.text
        
        # Test file upload
        file_input = driver.find_element(By.ID, "file-upload")
        file_input.send_keys("/path/to/test/floorplan.pdf")
        
        # Start analysis
        analyze_button = driver.find_element(By.ID, "start-analysis")
        analyze_button.click()
        
        # Wait for analysis completion
        results = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
        )
        
        # Verify 3D model is generated
        model_viewer = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "omniverse-viewer"))
        )
        assert model_viewer.is_displayed()
        
        # Test document generation
        generate_docs_button = driver.find_element(By.ID, "generate-documents")
        generate_docs_button.click()
        
        # Verify documents are generated
        doc_list = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "document-list"))
        )
        assert len(doc_list.find_elements(By.CLASS_NAME, "document-item")) > 0
        
    finally:
        driver.quit()

def test_payment_integration():
    """Test Stripe payment integration"""
    driver = webdriver.Firefox()
    try:
        driver.get("https://eiendomsmuligheter.no/checkout")
        
        # Fill payment form
        card_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "card-element"))
        )
        # Use Stripe test card
        card_input.send_keys("4242424242424242")
        card_input.send_keys("1225")
        card_input.send_keys("123")
        
        # Submit payment
        pay_button = driver.find_element(By.ID, "submit-payment")
        pay_button.click()
        
        # Verify success
        success_message = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "payment-success"))
        )
        assert "Payment successful" in success_message.text
        
    finally:
        driver.quit()