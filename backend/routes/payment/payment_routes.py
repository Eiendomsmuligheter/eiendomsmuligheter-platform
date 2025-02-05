from fastapi import APIRouter, Depends, Header, Request, HTTPException
from typing import Optional
from ...services.payment.stripe_service import StripeService
from ...config.settings import Settings
from ...dependencies.auth import get_current_user
from pydantic import BaseModel

router = APIRouter()
settings = Settings()
stripe_service = StripeService(settings.STRIPE_SECRET_KEY)

class CheckoutSessionRequest(BaseModel):
    priceId: str
    customerEmail: Optional[str] = None

class WebhookResponse(BaseModel):
    success: bool
    message: str

@router.post("/create-checkout-session")
async def create_checkout_session(
    request: CheckoutSessionRequest,
    current_user = Depends(get_current_user)
):
    try:
        session = await stripe_service.create_checkout_session(
            price_id=request.priceId,
            customer_email=request.customerEmail or current_user.email
        )
        return session
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None)
):
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="No signature provided")

    try:
        payload = await request.body()
        event = await stripe_service.handle_webhook(
            payload=payload,
            sig_header=stripe_signature,
            webhook_secret=settings.STRIPE_WEBHOOK_SECRET
        )

        # Handle different event types
        if event['type'] == 'checkout.session.completed':
            # Handle successful payment
            session = event['data']['object']
            await handle_successful_payment(session)
        elif event['type'] == 'customer.subscription.created':
            # Handle subscription creation
            subscription = event['data']['object']
            await handle_subscription_created(subscription)
        elif event['type'] == 'customer.subscription.updated':
            # Handle subscription updates
            subscription = event['data']['object']
            await handle_subscription_updated(subscription)
        elif event['type'] == 'customer.subscription.deleted':
            # Handle subscription cancellation
            subscription = event['data']['object']
            await handle_subscription_cancelled(subscription)

        return WebhookResponse(success=True, message="Webhook handled successfully")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/verify-payment")
async def verify_payment(session_id: str, current_user = Depends(get_current_user)):
    try:
        session = await stripe_service.retrieve_session(session_id)
        if session.payment_status == 'paid':
            # Update user subscription status in database
            await update_user_subscription(current_user.id, session.subscription)
            return {"success": True}
        return {"success": False, "error": "Payment not completed"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def handle_successful_payment(session):
    # Implement payment success logic
    # Update user's subscription status
    # Send confirmation email
    # etc.
    pass

async def handle_subscription_created(subscription):
    # Implement subscription creation logic
    pass

async def handle_subscription_updated(subscription):
    # Implement subscription update logic
    pass

async def handle_subscription_cancelled(subscription):
    # Implement subscription cancellation logic
    pass

async def update_user_subscription(user_id: str, subscription_id: str):
    # Update user's subscription status in database
    pass