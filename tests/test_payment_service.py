import pytest
from fastapi import HTTPException
from core_modules.payment_service import PaymentService, PaymentPlan
import stripe
from datetime import datetime

pytestmark = pytest.mark.asyncio

async def test_create_customer(mock_payment_service):
    """Test customer creation"""
    customer = await mock_payment_service.create_customer(
        email="test@example.com",
        name="Test Customer"
    )
    
    assert customer["id"].startswith("cus_")
    assert customer["email"] == "test@example.com"
    assert customer["name"] == "Test Customer"

async def test_create_subscription(mock_payment_service):
    """Test subscription creation"""
    subscription = await mock_payment_service.create_subscription(
        customer_id="cus_test123",
        plan_id="price_test123"
    )
    
    assert subscription["id"].startswith("sub_")
    assert subscription["customer"] == "cus_test123"
    assert subscription["status"] == "active"

async def test_cancel_subscription(mock_payment_service):
    """Test subscription cancellation"""
    try:
        result = await mock_payment_service.cancel_subscription("sub_test123")
        assert result["status"] == "canceled"
    except stripe.error.StripeError as e:
        pytest.fail(f"Subscription cancellation failed: {str(e)}")

async def test_create_payment_intent(mock_payment_service):
    """Test payment intent creation"""
    intent = await mock_payment_service.create_payment_intent(
        amount=1000.0,
        currency="NOK"
    )
    
    assert intent["amount"] == 100000  # Amount in øre
    assert intent["currency"] == "NOK"
    assert "client_secret" in intent

async def test_get_subscription_status(mock_payment_service):
    """Test getting subscription status"""
    status = await mock_payment_service.get_subscription_status("sub_test123")
    
    assert "status" in status
    assert isinstance(status["current_period_end"], datetime)
    assert isinstance(status["cancel_at_period_end"], bool)

async def test_update_payment_method(mock_payment_service):
    """Test updating payment method"""
    customer = await mock_payment_service.update_payment_method(
        customer_id="cus_test123",
        payment_method_id="pm_test123"
    )
    
    assert customer["id"] == "cus_test123"
    assert customer["invoice_settings"]["default_payment_method"] == "pm_test123"

def test_get_available_plans(mock_payment_service):
    """Test getting available plans"""
    plans = mock_payment_service.get_available_plans()
    
    assert len(plans) == 3
    assert "basic" in plans
    assert "pro" in plans
    assert "enterprise" in plans
    
    # Test basic plan
    basic_plan = plans["basic"]
    assert isinstance(basic_plan, PaymentPlan)
    assert basic_plan.price == 299.0
    assert basic_plan.currency == "NOK"
    assert len(basic_plan.features) > 0
    
    # Test pro plan
    pro_plan = plans["pro"]
    assert pro_plan.price == 599.0
    assert "Avansert AI-analyse" in pro_plan.features
    
    # Test enterprise plan
    enterprise_plan = plans["enterprise"]
    assert enterprise_plan.price == 1499.0
    assert "API-tilgang" in enterprise_plan.features

async def test_create_portal_session(mock_payment_service):
    """Test creating billing portal session"""
    session = await mock_payment_service.create_portal_session(
        customer_id="cus_test123",
        return_url="https://eiendomsmuligheter.no/account"
    )
    
    assert "url" in session
    assert session["customer"] == "cus_test123"

async def test_handle_webhook(mock_payment_service):
    """Test webhook handling"""
    # Test successful payment
    payment_succeeded_event = {
        "id": "evt_test123",
        "type": "invoice.paid",
        "data": {
            "object": {
                "id": "in_test123",
                "customer": "cus_test123",
                "subscription": "sub_test123",
                "status": "paid"
            }
        }
    }
    
    try:
        event = await mock_payment_service.handle_webhook(
            payload=str.encode(json.dumps(payment_succeeded_event)),
            sig_header="test_signature",
            webhook_secret="test_secret"
        )
        assert event["type"] == "invoice.paid"
    except Exception as e:
        pytest.fail(f"Webhook handling failed: {str(e)}")
    
    # Test failed payment
    payment_failed_event = {
        "id": "evt_test456",
        "type": "invoice.payment_failed",
        "data": {
            "object": {
                "id": "in_test456",
                "customer": "cus_test123",
                "subscription": "sub_test123",
                "status": "failed"
            }
        }
    }
    
    try:
        event = await mock_payment_service.handle_webhook(
            payload=str.encode(json.dumps(payment_failed_event)),
            sig_header="test_signature",
            webhook_secret="test_secret"
        )
        assert event["type"] == "invoice.payment_failed"
    except Exception as e:
        pytest.fail(f"Webhook handling failed: {str(e)}")

def test_payment_plan_model():
    """Test PaymentPlan model"""
    plan = PaymentPlan(
        id="price_test123",
        name="Test Plan",
        price=399.0,
        currency="NOK",
        interval="month",
        features=["Feature 1", "Feature 2"]
    )
    
    assert plan.id == "price_test123"
    assert plan.name == "Test Plan"
    assert plan.price == 399.0
    assert plan.currency == "NOK"
    assert plan.interval == "month"
    assert len(plan.features) == 2

@pytest.mark.parametrize("amount,currency,expected_amount", [
    (100.0, "NOK", 10000),
    (99.99, "USD", 9999),
    (1000.0, "EUR", 100000),
])
async def test_payment_amount_conversion(
    mock_payment_service,
    amount,
    currency,
    expected_amount
):
    """Test payment amount conversion to minor units"""
    intent = await mock_payment_service.create_payment_intent(
        amount=amount,
        currency=currency
    )
    assert intent["amount"] == expected_amount

def test_payment_service_initialization():
    """Test PaymentService initialization"""
    # Test with missing API key
    with pytest.raises(ValueError):
        PaymentService()
    
    # Test with valid API key
    service = PaymentService(api_key="test_key")
    assert service.api_key == "test_key"
    assert stripe.api_key == "test_key"
    
    # Test plans initialization
    plans = service.get_available_plans()
    assert len(plans) == 3
    assert all(isinstance(plan, PaymentPlan) for plan in plans.values())