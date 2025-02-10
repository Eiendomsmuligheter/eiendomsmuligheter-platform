import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

@pytest.fixture
def driver():
    driver = webdriver.Firefox()
    driver.implicitly_wait(10)
    yield driver
    driver.quit()

def test_property_upload_and_analysis(driver):
    # Test file upload functionality
    driver.get("http://localhost:8000")
    upload_btn = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "file-upload"))
    )
    upload_btn.send_keys("/path/to/test/property.jpg")
    
    # Test address search
    address_input = driver.find_element(By.ID, "address-search")
    address_input.send_keys("Testveien 1, Drammen")
    search_btn = driver.find_element(By.ID, "search-button")
    search_btn.click()
    
    # Verify analysis results
    results = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
    )
    assert "Eiendomsinformasjon" in results.text
    assert "Reguleringsdata" in results.text
    assert "Utviklingspotensial" in results.text

def test_payment_flow(driver):
    driver.get("http://localhost:8000/payment")
    
    # Select subscription plan
    plan_select = driver.find_element(By.ID, "plan-select")
    plan_select.click()
    basic_plan = driver.find_element(By.ID, "basic-plan")
    basic_plan.click()
    
    # Fill payment details
    card_input = driver.find_element(By.ID, "card-element")
    card_input.send_keys("4242424242424242")  # Test card number
    
    # Complete payment
    pay_button = driver.find_element(By.ID, "submit-payment")
    pay_button.click()
    
    # Verify success
    success_message = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "payment-success"))
    )
    assert "Betaling vellykket" in success_message.text