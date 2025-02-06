import stripe
from fastapi import HTTPException
from typing import Dict, Optional
from datetime import datetime
import os
from ..models.user import User
from ..models.subscription import Subscription
from ..database import get_db
from sqlalchemy.orm import Session

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

class PaymentService:
    def __init__(self, db: Session):
        self.db = db
        self._setup_stripe_products()

    def _setup_stripe_products(self):
        """Setter opp produkter og priser i Stripe"""
        self.products = {
            "basic": {
                "name": "Basis Analyse",
                "price_id": os.getenv("STRIPE_BASIC_PRICE_ID"),
                "features": [
                    "Grunnleggende eiendomsanalyse",
                    "Plantegningsanalyse",
                    "Reguleringsdata",
                    "PDF-rapport"
                ]
            },
            "pro": {
                "name": "Pro Analyse",
                "price_id": os.getenv("STRIPE_PRO_PRICE_ID"),
                "features": [
                    "Alt i Basis",
                    "3D-visualisering",
                    "Utviklingspotensialanalyse",
                    "Automatisk byggesøknadsgenerering"
                ]
            },
            "enterprise": {
                "name": "Enterprise Løsning",
                "price_id": os.getenv("STRIPE_ENTERPRISE_PRICE_ID"),
                "features": [
                    "Alt i Pro",
                    "API-tilgang",
                    "Dedikert støtte",
                    "Tilpassede analyser"
                ]
            }
        }

    async def create_checkout_session(
        self, 
        user_id: str, 
        plan: str,
        success_url: str,
        cancel_url: str
    ) -> Dict:
        """Oppretter en Stripe Checkout-sesjon"""
        if plan not in self.products:
            raise HTTPException(status_code=400, detail="Ugyldig abonnementsplan")

        try:
            # Hent eller opprett Stripe-kunde
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="Bruker ikke funnet")

            if not user.stripe_customer_id:
                customer = stripe.Customer.create(
                    email=user.email,
                    metadata={"user_id": user_id}
                )
                user.stripe_customer_id = customer.id
                self.db.commit()

            # Opprett checkout-sesjon
            session = stripe.checkout.Session.create(
                customer=user.stripe_customer_id,
                payment_method_types=['card'],
                line_items=[{
                    'price': self.products[plan]['price_id'],
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    "user_id": user_id,
                    "plan": plan
                }
            )

            return {"session_id": session.id}

        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def handle_webhook(self, payload: Dict, sig_header: str) -> None:
        """Håndterer Stripe webhooks"""
        try:
            event = stripe.Webhook.construct_event(
                payload,
                sig_header,
                os.getenv("STRIPE_WEBHOOK_SECRET")
            )

            if event.type == "checkout.session.completed":
                session = event.data.object
                await self._handle_successful_subscription(session)
            
            elif event.type == "customer.subscription.deleted":
                subscription = event.data.object
                await self._handle_cancelled_subscription(subscription)

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def _handle_successful_subscription(self, session: Dict) -> None:
        """Håndterer vellykket abonnementstegning"""
        user_id = session.metadata.get("user_id")
        plan = session.metadata.get("plan")

        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Bruker ikke funnet")

        # Opprett eller oppdater abonnement
        subscription = self.db.query(Subscription).filter(
            Subscription.user_id == user_id
        ).first()

        if subscription:
            subscription.plan = plan
            subscription.status = "active"
            subscription.stripe_subscription_id = session.subscription
            subscription.updated_at = datetime.utcnow()
        else:
            subscription = Subscription(
                user_id=user_id,
                plan=plan,
                status="active",
                stripe_subscription_id=session.subscription
            )
            self.db.add(subscription)

        self.db.commit()

    async def _handle_cancelled_subscription(self, stripe_subscription: Dict) -> None:
        """Håndterer kansellert abonnement"""
        subscription = self.db.query(Subscription).filter(
            Subscription.stripe_subscription_id == stripe_subscription.id
        ).first()

        if subscription:
            subscription.status = "cancelled"
            subscription.updated_at = datetime.utcnow()
            self.db.commit()

    async def create_portal_session(self, user_id: str, return_url: str) -> Dict:
        """Oppretter en Stripe kundeportal-sesjon"""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or not user.stripe_customer_id:
            raise HTTPException(status_code=404, detail="Bruker ikke funnet eller mangler Stripe-kunde")

        try:
            session = stripe.billing_portal.Session.create(
                customer=user.stripe_customer_id,
                return_url=return_url
            )
            return {"url": session.url}

        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_subscription_status(self, user_id: str) -> Dict:
        """Henter abonnementsstatus for en bruker"""
        subscription = self.db.query(Subscription).filter(
            Subscription.user_id == user_id
        ).first()

        if not subscription:
            return {"status": "none", "plan": None}

        return {
            "status": subscription.status,
            "plan": subscription.plan,
            "created_at": subscription.created_at,
            "updated_at": subscription.updated_at
        }