from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional, Dict
from sqlalchemy.orm import Session
from ...services.payment_service import PaymentService
from ...database.base import get_db
from ...core.security import get_current_user
from ...models.user import User
import os

router = APIRouter()

@router.post("/create-checkout-session")
async def create_checkout_session(
    plan: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Oppretter en Stripe checkout-sesjon for abonnementstegning
    """
    payment_service = PaymentService(db)
    
    success_url = f"{os.getenv('FRONTEND_URL')}/payment/success"
    cancel_url = f"{os.getenv('FRONTEND_URL')}/payment/cancelled"
    
    return await payment_service.create_checkout_session(
        current_user.id,
        plan,
        success_url,
        cancel_url
    )

@router.post("/webhook")
async def stripe_webhook(
    payload: Dict,
    stripe_signature: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Håndterer Stripe webhook-hendelser
    """
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Stripe-signatur mangler")

    payment_service = PaymentService(db)
    await payment_service.handle_webhook(payload, stripe_signature)
    return {"status": "success"}

@router.post("/create-portal-session")
async def create_portal_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Oppretter en Stripe kundeportal-sesjon
    """
    payment_service = PaymentService(db)
    return_url = f"{os.getenv('FRONTEND_URL')}/account"
    
    return await payment_service.create_portal_session(
        current_user.id,
        return_url
    )

@router.get("/subscription-status")
async def get_subscription_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Henter brukerens nåværende abonnementsstatus
    """
    payment_service = PaymentService(db)
    return await payment_service.get_subscription_status(current_user.id)