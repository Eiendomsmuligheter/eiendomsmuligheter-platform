import stripe
from fastapi import HTTPException
from typing import Optional, Dict

class StripeService:
    def __init__(self, api_key: str):
        self.stripe = stripe
        self.stripe.api_key = api_key

    async def create_customer(self, email: str, name: Optional[str] = None) -> Dict:
        try:
            customer = self.stripe.Customer.create(
                email=email,
                name=name
            )
            return customer
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def create_subscription(self, customer_id: str, price_id: str) -> Dict:
        try:
            subscription = self.stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                payment_behavior="default_incomplete",
                expand=["latest_invoice.payment_intent"]
            )
            return subscription
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def create_checkout_session(self, price_id: str, customer_email: str) -> Dict:
        try:
            session = self.stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price": price_id,
                    "quantity": 1
                }],
                mode="subscription",
                success_url="https://eiendomsmuligheter.no/payment/success?session_id={CHECKOUT_SESSION_ID}",
                cancel_url="https://eiendomsmuligheter.no/payment/cancel",
                customer_email=customer_email
            )
            return session
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def handle_webhook(self, payload: bytes, sig_header: str, webhook_secret: str) -> Dict:
        try:
            event = self.stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
            return event
        except stripe.error.SignatureVerificationError as e:
            raise HTTPException(status_code=400, detail="Invalid signature")
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid payload")