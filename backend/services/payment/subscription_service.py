import stripe
from typing import Dict, Any
from datetime import datetime
from fastapi import HTTPException
from models.subscription import Subscription
from models.user import User
from config import settings

stripe.api_key = settings.STRIPE_SECRET_KEY

class SubscriptionService:
    @staticmethod
    async def create_subscription(user_id: int, plan_id: str, payment_method_id: str) -> Dict[str, Any]:
        try:
            # Hent bruker fra database
            user = await User.get(user_id)
            
            # Opprett eller hent stripe kunde
            if not user.stripe_customer_id:
                customer = stripe.Customer.create(
                    payment_method=payment_method_id,
                    email=user.email,
                    invoice_settings={
                        'default_payment_method': payment_method_id,
                    },
                )
                user.stripe_customer_id = customer.id
                await user.save()
            
            # Opprett abonnement
            subscription = stripe.Subscription.create(
                customer=user.stripe_customer_id,
                items=[{'price': plan_id}],
                expand=['latest_invoice.payment_intent'],
            )
            
            # Lagre abonnement i database
            db_subscription = await Subscription.create(
                user_id=user_id,
                stripe_subscription_id=subscription.id,
                plan_id=plan_id,
                status=subscription.status,
                current_period_end=datetime.fromtimestamp(subscription.current_period_end),
            )
            
            return {
                'subscriptionId': subscription.id,
                'clientSecret': subscription.latest_invoice.payment_intent.client_secret,
                'status': subscription.status,
            }
            
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="En feil oppstod ved opprettelse av abonnement")

    @staticmethod
    async def cancel_subscription(user_id: int, subscription_id: str) -> Dict[str, str]:
        try:
            subscription = await Subscription.get_by_stripe_id(subscription_id)
            if not subscription or subscription.user_id != user_id:
                raise HTTPException(status_code=404, detail="Abonnement ikke funnet")
            
            # Kanseller i Stripe
            stripe_sub = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True,
            )
            
            # Oppdater i database
            subscription.status = "canceling"
            subscription.cancel_at = datetime.fromtimestamp(stripe_sub.cancel_at)
            await subscription.save()
            
            return {"status": "Abonnement vil bli kansellert ved slutten av perioden"}
            
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="En feil oppstod ved kansellering av abonnement")

    @staticmethod
    async def get_subscription_status(user_id: int) -> Dict[str, Any]:
        try:
            subscription = await Subscription.get_active_by_user(user_id)
            if not subscription:
                return {"status": "inactive", "subscription": None}
            
            stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
            
            return {
                "status": stripe_sub.status,
                "subscription": {
                    "id": subscription.id,
                    "plan_id": subscription.plan_id,
                    "current_period_end": subscription.current_period_end.isoformat(),
                    "cancel_at": subscription.cancel_at.isoformat() if subscription.cancel_at else None,
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail="En feil oppstod ved henting av abonnementsstatus")