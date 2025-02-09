import stripe
from fastapi import HTTPException
from typing import Optional
from datetime import datetime
from ..config import settings
from ..models.payment import Payment
from ..database import get_db

stripe.api_key = settings.STRIPE_SECRET_KEY

class PaymentService:
    def __init__(self):
        self.db = next(get_db())

    async def create_payment_intent(self, amount: int, currency: str = 'nok', 
                                  customer_id: Optional[str] = None) -> dict:
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                customer=customer_id,
                payment_method_types=['card'],
                metadata={'integration_check': 'accept_a_payment'}
            )
            
            # Lagre betalingsintensjonen i databasen
            payment = Payment(
                stripe_payment_id=intent.id,
                amount=amount,
                currency=currency,
                customer_id=customer_id,
                status='pending',
                created_at=datetime.utcnow()
            )
            self.db.add(payment)
            self.db.commit()
            
            return {
                'clientSecret': intent.client_secret,
                'paymentId': payment.id
            }
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=400,
                detail=f'Stripe feil: {str(e)}'
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f'Serverfeil: {str(e)}'
            )

    async def confirm_payment(self, payment_intent_id: str) -> dict:
        try:
            intent = stripe.PaymentIntent.retrieve(payment_intent_id)
            
            # Oppdater betalingsstatus i databasen
            payment = self.db.query(Payment).filter(
                Payment.stripe_payment_id == payment_intent_id
            ).first()
            
            if payment:
                payment.status = intent.status
                payment.updated_at = datetime.utcnow()
                self.db.commit()
            
            return {
                'status': intent.status,
                'amount': intent.amount,
                'currency': intent.currency
            }
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=400,
                detail=f'Stripe feil: {str(e)}'
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f'Serverfeil: {str(e)}'
            )

    async def create_subscription(self, customer_id: str, price_id: str) -> dict:
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{'price': price_id}],
                payment_behavior='default_incomplete',
                expand=['latest_invoice.payment_intent']
            )
            
            return {
                'subscriptionId': subscription.id,
                'clientSecret': subscription.latest_invoice.payment_intent.client_secret
            }
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=400,
                detail=f'Stripe feil: {str(e)}'
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f'Serverfeil: {str(e)}'
            )

    async def cancel_subscription(self, subscription_id: str) -> dict:
        try:
            subscription = stripe.Subscription.delete(subscription_id)
            return {'status': subscription.status}
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=400,
                detail=f'Stripe feil: {str(e)}'
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f'Serverfeil: {str(e)}'
            )

payment_service = PaymentService()