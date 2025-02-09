from fastapi import APIRouter, Depends, HTTPException
from ..services.payment_service import payment_service
from ..models.payment import Payment, Subscription
from ..schemas.payment import (
    PaymentIntentCreate,
    PaymentIntentResponse,
    SubscriptionCreate,
    SubscriptionResponse
)
from ..auth.dependencies import get_current_user
from typing import List

router = APIRouter(prefix="/api/payments", tags=["payments"])

@router.post("/create-payment-intent", response_model=PaymentIntentResponse)
async def create_payment_intent(
    payment_data: PaymentIntentCreate,
    current_user = Depends(get_current_user)
):
    """
    Opprett en betalingsintensjon for engangskjøp
    """
    result = await payment_service.create_payment_intent(
        amount=payment_data.amount,
        currency=payment_data.currency,
        customer_id=current_user.stripe_customer_id
    )
    return result

@router.post("/confirm-payment/{payment_intent_id}")
async def confirm_payment(
    payment_intent_id: str,
    current_user = Depends(get_current_user)
):
    """
    Bekreft en betaling etter vellykket kortbetaling
    """
    result = await payment_service.confirm_payment(payment_intent_id)
    return result

@router.post("/create-subscription", response_model=SubscriptionResponse)
async def create_subscription(
    subscription_data: SubscriptionCreate,
    current_user = Depends(get_current_user)
):
    """
    Opprett et nytt abonnement
    """
    result = await payment_service.create_subscription(
        customer_id=current_user.stripe_customer_id,
        price_id=subscription_data.price_id
    )
    return result

@router.delete("/cancel-subscription/{subscription_id}")
async def cancel_subscription(
    subscription_id: str,
    current_user = Depends(get_current_user)
):
    """
    Kanseller et eksisterende abonnement
    """
    result = await payment_service.cancel_subscription(subscription_id)
    return result

@router.get("/subscription-plans")
async def get_subscription_plans():
    """
    Hent alle tilgjengelige abonnementsplaner
    """
    return [
        {
            "id": "basic",
            "name": "Basis",
            "price": 99900,  # 999 NOK
            "currency": "nok",
            "interval": "monthly",
            "features": [
                "Grunnleggende eiendomsanalyse",
                "Plantegningsanalyse",
                "Reguleringsplansjekk",
                "PDF-rapport"
            ]
        },
        {
            "id": "pro",
            "name": "Professional",
            "price": 199900,  # 1999 NOK
            "currency": "nok",
            "interval": "monthly",
            "features": [
                "Alt i Basis",
                "3D-visualisering",
                "Energianalyse",
                "Automatisk byggesøknad",
                "Enova-støtteberegning"
            ]
        },
        {
            "id": "enterprise",
            "name": "Enterprise",
            "price": 499900,  # 4999 NOK
            "currency": "nok",
            "interval": "monthly",
            "features": [
                "Alt i Professional",
                "NVIDIA Omniverse integrasjon",
                "BIM-modellering",
                "API-tilgang",
                "Dedikert support"
            ]
        }
    ]