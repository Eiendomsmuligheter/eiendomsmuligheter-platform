from fastapi import APIRouter, Depends, HTTPException, Header, Request
from typing import Optional, Dict, List
from pydantic import BaseModel
import stripe
from core_modules.payment_service import PaymentService
from core_modules.auth_service import get_current_user
import os
import json

router = APIRouter(prefix="/api/payments", tags=["payments"])
payment_service = PaymentService(api_key=os.getenv("STRIPE_SECRET_KEY"))

class SubscriptionRequest(BaseModel):
    plan_id: str
    payment_method_id: str

class CustomerRequest(BaseModel):
    email: str
    name: str

@router.get("/plans")
async def get_plans():
    """Get all available subscription plans"""
    try:
        plans = payment_service.get_available_plans()
        return plans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-customer")
async def create_customer(
    customer_data: CustomerRequest,
    current_user = Depends(get_current_user)
):
    """Create a new customer in Stripe"""
    try:
        customer = await payment_service.create_customer(
            email=customer_data.email,
            name=customer_data.name
        )
        return customer
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/create-subscription")
async def create_subscription(
    subscription_data: SubscriptionRequest,
    current_user = Depends(get_current_user)
):
    """Create a new subscription for a customer"""
    try:
        subscription = await payment_service.create_subscription(
            customer_id=current_user.stripe_customer_id,
            plan_id=subscription_data.plan_id
        )
        return {
            "subscriptionId": subscription.id,
            "clientSecret": subscription.latest_invoice.payment_intent.client_secret
            if subscription.latest_invoice.payment_intent
            else None
        }
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cancel-subscription/{subscription_id}")
async def cancel_subscription(
    subscription_id: str,
    current_user = Depends(get_current_user)
):
    """Cancel a subscription"""
    try:
        subscription = await payment_service.cancel_subscription(subscription_id)
        return subscription
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/subscription-status/{subscription_id}")
async def get_subscription_status(
    subscription_id: str,
    current_user = Depends(get_current_user)
):
    """Get the status of a subscription"""
    try:
        status = await payment_service.get_subscription_status(subscription_id)
        return status
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/update-payment-method")
async def update_payment_method(
    payment_method_id: str,
    current_user = Depends(get_current_user)
):
    """Update a customer's default payment method"""
    try:
        customer = await payment_service.update_payment_method(
            customer_id=current_user.stripe_customer_id,
            payment_method_id=payment_method_id
        )
        return customer
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/create-portal-session")
async def create_portal_session(
    return_url: str,
    current_user = Depends(get_current_user)
):
    """Create a billing portal session for a customer"""
    try:
        session = await payment_service.create_portal_session(
            customer_id=current_user.stripe_customer_id,
            return_url=return_url
        )
        return session
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/webhook")
async def handle_webhook(
    request: Request,
    stripe_signature: str = Header(None)
):
    """Handle Stripe webhooks"""
    try:
        payload = await request.body()
        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        event = await payment_service.handle_webhook(
            payload=payload,
            sig_header=stripe_signature,
            webhook_secret=webhook_secret
        )
        return {"status": "success", "type": event.type}
        
    except (stripe.error.SignatureVerificationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))