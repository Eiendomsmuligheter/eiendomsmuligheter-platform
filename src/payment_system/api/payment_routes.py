from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional, Dict, Any
from ..services.stripe_service import StripeService
from ..models.payment_model import (
    Customer, 
    PaymentTransaction,
    ProductType,
    PricingTier,
    PRODUCT_PRICING
)
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
stripe_service = StripeService(
    api_key=os.getenv("STRIPE_SECRET_KEY"),
    webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET")
)

@router.post("/customers", response_model=Dict[str, Any])
async def create_customer(customer: Customer):
    """
    Opprett en ny kunde
    """
    try:
        return await stripe_service.create_customer(customer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/payment-intent", response_model=Dict[str, Any])
async def create_payment_intent(
    product_type: ProductType,
    pricing_tier: PricingTier,
    customer_id: str,
    payment_method_id: Optional[str] = None
):
    """
    Opprett en betalingsintensjon
    """
    try:
        # Hent produktpris
        price = PRODUCT_PRICING[product_type][pricing_tier]
        
        return await stripe_service.create_payment_intent(
            amount=price.amount,
            currency="NOK",
            customer_id=customer_id,
            payment_method_id=payment_method_id,
            metadata={
                'product_type': product_type,
                'pricing_tier': pricing_tier
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/webhook")
async def stripe_webhook(
    payload: bytes,
    stripe_signature: str = Header(None)
):
    """
    Håndter Stripe webhook events
    """
    try:
        return await stripe_service.handle_webhook(payload, stripe_signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/refund/{payment_intent_id}")
async def refund_payment(
    payment_intent_id: str,
    amount: Optional[float] = None,
    reason: Optional[str] = None
):
    """
    Refunder en betaling
    """
    try:
        return await stripe_service.refund_payment(
            payment_intent_id=payment_intent_id,
            amount=amount,
            reason=reason
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/pricing")
async def get_pricing():
    """
    Hent prisliste for alle produkter og tjenester
    """
    return PRODUCT_PRICING

@router.post("/generate-invoice")
async def generate_invoice(transaction: PaymentTransaction):
    """
    Generer faktura for en transaksjon
    """
    try:
        invoice_url = await stripe_service.generate_invoice(transaction)
        return {"invoice_url": invoice_url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
