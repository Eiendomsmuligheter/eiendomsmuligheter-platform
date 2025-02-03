import stripe
import os
from typing import Dict, Any, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session
from ..models.database import User, Payment, Subscription

class StripeService:
    def __init__(self):
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        self.public_key = os.getenv("STRIPE_PUBLIC_KEY")
        
    async def create_customer(self, user: User, db: Session) -> Dict[str, Any]:
        """
        Opprett en ny Stripe-kunde
        """
        try:
            customer = stripe.Customer.create(
                email=user.email,
                name=user.full_name,
                metadata={
                    "user_id": user.id
                }
            )
            
            return customer
            
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def create_subscription(
        self, 
        user: User,
        price_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Opprett et nytt abonnement
        """
        try:
            # Opprett eller hent Stripe-kunde
            if not user.stripe_customer_id:
                customer = await self.create_customer(user, db)
                user.stripe_customer_id = customer.id
                db.commit()
            
            # Opprett abonnement
            subscription = stripe.Subscription.create(
                customer=user.stripe_customer_id,
                items=[{"price": price_id}],
                payment_behavior="default_incomplete",
                expand=["latest_invoice.payment_intent"],
            )
            
            # Lagre abonnement i database
            db_subscription = Subscription(
                user_id=user.id,
                stripe_subscription_id=subscription.id,
                stripe_customer_id=user.stripe_customer_id,
                plan_id=price_id,
                status=subscription.status,
                current_period_start=subscription.current_period_start,
                current_period_end=subscription.current_period_end
            )
            db.add(db_subscription)
            db.commit()
            
            return {
                "subscriptionId": subscription.id,
                "clientSecret": subscription.latest_invoice.payment_intent.client_secret,
            }
            
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def handle_webhook(self, payload: Dict[str, Any], sig_header: str, db: Session) -> None:
        """
        Håndter Stripe webhook events
        """
        try:
            event = stripe.Webhook.construct_event(
                payload,
                sig_header,
                os.getenv("STRIPE_WEBHOOK_SECRET")
            )
            
            if event.type == "invoice.payment_succeeded":
                await self._handle_payment_succeeded(event.data.object, db)
                
            elif event.type == "invoice.payment_failed":
                await self._handle_payment_failed(event.data.object, db)
                
            elif event.type == "customer.subscription.deleted":
                await self._handle_subscription_deleted(event.data.object, db)
                
        except (stripe.error.SignatureVerificationError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def _handle_payment_succeeded(self, invoice: Dict[str, Any], db: Session) -> None:
        """
        Håndter vellykket betaling
        """
        # Registrer betaling
        payment = Payment(
            user_id=invoice.customer.metadata.user_id,
            stripe_payment_id=invoice.payment_intent,
            amount=invoice.amount_paid / 100,  # Konverter fra øre til kroner
            currency=invoice.currency,
            status="succeeded",
            payment_method=invoice.payment_method_types[0]
        )
        db.add(payment)
        
        # Oppdater abonnementsstatus
        subscription = db.query(Subscription).filter_by(
            stripe_subscription_id=invoice.subscription
        ).first()
        
        if subscription:
            subscription.status = "active"
            
        db.commit()
    
    async def _handle_payment_failed(self, invoice: Dict[str, Any], db: Session) -> None:
        """
        Håndter mislykket betaling
        """
        # Registrer mislykket betaling
        payment = Payment(
            user_id=invoice.customer.metadata.user_id,
            stripe_payment_id=invoice.payment_intent,
            amount=invoice.amount_due / 100,
            currency=invoice.currency,
            status="failed",
            payment_method=invoice.payment_method_types[0]
        )
        db.add(payment)
        
        # Oppdater abonnementsstatus
        subscription = db.query(Subscription).filter_by(
            stripe_subscription_id=invoice.subscription
        ).first()
        
        if subscription:
            subscription.status = "incomplete"
            
        db.commit()
    
    async def _handle_subscription_deleted(self, subscription_data: Dict[str, Any], db: Session) -> None:
        """
        Håndter kansellert abonnement
        """
        subscription = db.query(Subscription).filter_by(
            stripe_subscription_id=subscription_data.id
        ).first()
        
        if subscription:
            subscription.status = "canceled"
            db.commit()
    
    async def cancel_subscription(self, user: User, db: Session) -> Dict[str, Any]:
        """
        Kanseller et aktivt abonnement
        """
        try:
            subscription = db.query(Subscription).filter_by(
                user_id=user.id,
                status="active"
            ).first()
            
            if not subscription:
                raise HTTPException(status_code=404, detail="No active subscription found")
            
            canceled_subscription = stripe.Subscription.delete(
                subscription.stripe_subscription_id
            )
            
            subscription.status = "canceled"
            subscription.cancel_at_period_end = True
            db.commit()
            
            return {
                "status": "canceled",
                "effective_date": canceled_subscription.cancel_at
            }
            
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def update_payment_method(
        self,
        user: User,
        payment_method_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Oppdater betalingsmetode
        """
        try:
            if not user.stripe_customer_id:
                raise HTTPException(status_code=400, detail="No Stripe customer found")
            
            # Koble betalingsmetode til kunde
            payment_method = stripe.PaymentMethod.attach(
                payment_method_id,
                customer=user.stripe_customer_id,
            )
            
            # Sett som standard betalingsmetode
            stripe.Customer.modify(
                user.stripe_customer_id,
                invoice_settings={
                    "default_payment_method": payment_method.id
                },
            )
            
            return {"status": "success", "payment_method": payment_method.id}
            
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))