import stripe
from fastapi import HTTPException, Request
from models.subscription import Subscription
from models.user import User
from datetime import datetime
from config import settings

stripe.api_key = settings.STRIPE_SECRET_KEY

class WebhookHandler:
    @staticmethod
    async def handle_webhook(request: Request) -> dict:
        try:
            body = await request.body()
            sig_header = request.headers.get('stripe-signature')
            
            try:
                event = stripe.Webhook.construct_event(
                    body,
                    sig_header,
                    settings.STRIPE_WEBHOOK_SECRET
                )
            except stripe.error.SignatureVerificationError:
                raise HTTPException(status_code=400, detail="Ugyldig signatur")
            
            # Håndter ulike webhook events
            if event.type == 'customer.subscription.updated':
                await WebhookHandler._handle_subscription_updated(event.data.object)
            elif event.type == 'customer.subscription.deleted':
                await WebhookHandler._handle_subscription_deleted(event.data.object)
            elif event.type == 'invoice.payment_succeeded':
                await WebhookHandler._handle_payment_succeeded(event.data.object)
            elif event.type == 'invoice.payment_failed':
                await WebhookHandler._handle_payment_failed(event.data.object)
                
            return {"status": "success"}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def _handle_subscription_updated(subscription_object):
        subscription = await Subscription.get_by_stripe_id(subscription_object.id)
        if subscription:
            subscription.status = subscription_object.status
            subscription.current_period_end = datetime.fromtimestamp(
                subscription_object.current_period_end
            )
            if subscription_object.cancel_at:
                subscription.cancel_at = datetime.fromtimestamp(
                    subscription_object.cancel_at
                )
            await subscription.save()

    @staticmethod
    async def _handle_subscription_deleted(subscription_object):
        subscription = await Subscription.get_by_stripe_id(subscription_object.id)
        if subscription:
            subscription.status = "cancelled"
            subscription.ended_at = datetime.now()
            await subscription.save()

    @staticmethod
    async def _handle_payment_succeeded(invoice_object):
        subscription_id = invoice_object.subscription
        if subscription_id:
            subscription = await Subscription.get_by_stripe_id(subscription_id)
            if subscription:
                subscription.last_payment_status = "succeeded"
                subscription.last_payment_date = datetime.fromtimestamp(
                    invoice_object.created
                )
                await subscription.save()

    @staticmethod
    async def _handle_payment_failed(invoice_object):
        subscription_id = invoice_object.subscription
        if subscription_id:
            subscription = await Subscription.get_by_stripe_id(subscription_id)
            if subscription:
                subscription.last_payment_status = "failed"
                await subscription.save()
                
                # Send varsling til bruker om mislykket betaling
                user = await User.get(subscription.user_id)
                if user:
                    # TODO: Implementer varslingssystem
                    pass