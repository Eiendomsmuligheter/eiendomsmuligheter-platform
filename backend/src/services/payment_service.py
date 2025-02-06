import os
import stripe
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class PaymentService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
        self.prices = {
            'basic': {
                'monthly': 'price_basic_monthly',
                'yearly': 'price_basic_yearly'
            },
            'pro': {
                'monthly': 'price_pro_monthly',
                'yearly': 'price_pro_yearly'
            },
            'enterprise': {
                'monthly': 'price_enterprise_monthly',
                'yearly': 'price_enterprise_yearly'
            }
        }

    async def create_checkout_session(
        self,
        price_id: str,
        customer_email: str,
        success_url: str,
        cancel_url: str
    ) -> Dict[str, Any]:
        """Oppretter en Stripe Checkout-sesjon"""
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                customer_email=customer_email
            )
            return {
                'session_id': session.id,
                'checkout_url': session.url
            }
        except Exception as e:
            self.logger.error(f"Error creating checkout session: {str(e)}")
            raise

    async def create_customer(
        self,
        email: str,
        name: str,
        payment_method_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Oppretter en ny kunde i Stripe"""
        try:
            customer_data = {
                'email': email,
                'name': name
            }
            if payment_method_id:
                customer_data['payment_method'] = payment_method_id

            customer = stripe.Customer.create(**customer_data)
            return {
                'customer_id': customer.id,
                'email': customer.email,
                'name': customer.name
            }
        except Exception as e:
            self.logger.error(f"Error creating customer: {str(e)}")
            raise

    async def create_subscription(
        self,
        customer_id: str,
        price_id: str
    ) -> Dict[str, Any]:
        """Oppretter et nytt abonnement"""
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{'price': price_id}],
                payment_behavior='default_incomplete',
                expand=['latest_invoice.payment_intent']
            )
            return {
                'subscription_id': subscription.id,
                'client_secret': subscription.latest_invoice.payment_intent.client_secret,
                'status': subscription.status
            }
        except Exception as e:
            self.logger.error(f"Error creating subscription: {str(e)}")
            raise

    async def cancel_subscription(
        self,
        subscription_id: str
    ) -> Dict[str, Any]:
        """Kansellerer et abonnement"""
        try:
            subscription = stripe.Subscription.delete(subscription_id)
            return {
                'subscription_id': subscription.id,
                'status': subscription.status,
                'canceled_at': datetime.fromtimestamp(subscription.canceled_at)
            }
        except Exception as e:
            self.logger.error(f"Error canceling subscription: {str(e)}")
            raise

    async def get_subscription_info(
        self,
        subscription_id: str
    ) -> Dict[str, Any]:
        """Henter informasjon om et abonnement"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return {
                'id': subscription.id,
                'status': subscription.status,
                'current_period_end': datetime.fromtimestamp(
                    subscription.current_period_end
                ),
                'plan': {
                    'id': subscription.plan.id,
                    'nickname': subscription.plan.nickname,
                    'amount': subscription.plan.amount,
                    'interval': subscription.plan.interval
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting subscription info: {str(e)}")
            raise

    async def update_subscription(
        self,
        subscription_id: str,
        new_price_id: str
    ) -> Dict[str, Any]:
        """Oppdaterer et abonnement med ny pris"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            updated = stripe.Subscription.modify(
                subscription_id,
                items=[{
                    'id': subscription['items']['data'][0].id,
                    'price': new_price_id,
                }]
            )
            return {
                'subscription_id': updated.id,
                'status': updated.status,
                'new_plan': {
                    'id': updated.plan.id,
                    'nickname': updated.plan.nickname,
                    'amount': updated.plan.amount,
                    'interval': updated.plan.interval
                }
            }
        except Exception as e:
            self.logger.error(f"Error updating subscription: {str(e)}")
            raise

    async def handle_webhook(self, payload: Dict[str, Any], sig_header: str) -> None:
        """Håndterer Stripe webhooks"""
        try:
            event = stripe.Webhook.construct_event(
                payload,
                sig_header,
                os.getenv('STRIPE_WEBHOOK_SECRET')
            )

            if event.type == 'customer.subscription.created':
                await self._handle_subscription_created(event.data.object)
            elif event.type == 'customer.subscription.updated':
                await self._handle_subscription_updated(event.data.object)
            elif event.type == 'customer.subscription.deleted':
                await self._handle_subscription_canceled(event.data.object)
            elif event.type == 'invoice.payment_succeeded':
                await self._handle_payment_succeeded(event.data.object)
            elif event.type == 'invoice.payment_failed':
                await self._handle_payment_failed(event.data.object)

        except Exception as e:
            self.logger.error(f"Error handling webhook: {str(e)}")
            raise

    async def _handle_subscription_created(self, subscription: Dict[str, Any]) -> None:
        """Håndterer nytt abonnement"""
        try:
            # Implementer logikk for å oppdatere lokal database
            pass
        except Exception as e:
            self.logger.error(f"Error handling subscription created: {str(e)}")
            raise

    async def _handle_subscription_updated(self, subscription: Dict[str, Any]) -> None:
        """Håndterer oppdatert abonnement"""
        try:
            # Implementer logikk for å oppdatere lokal database
            pass
        except Exception as e:
            self.logger.error(f"Error handling subscription updated: {str(e)}")
            raise

    async def _handle_subscription_canceled(self, subscription: Dict[str, Any]) -> None:
        """Håndterer kansellert abonnement"""
        try:
            # Implementer logikk for å oppdatere lokal database
            pass
        except Exception as e:
            self.logger.error(f"Error handling subscription canceled: {str(e)}")
            raise

    async def _handle_payment_succeeded(self, invoice: Dict[str, Any]) -> None:
        """Håndterer vellykket betaling"""
        try:
            # Implementer logikk for å oppdatere lokal database og sende kvittering
            pass
        except Exception as e:
            self.logger.error(f"Error handling payment succeeded: {str(e)}")
            raise

    async def _handle_payment_failed(self, invoice: Dict[str, Any]) -> None:
        """Håndterer mislykket betaling"""
        try:
            # Implementer logikk for å oppdatere lokal database og varsle kunde
            pass
        except Exception as e:
            self.logger.error(f"Error handling payment failed: {str(e)}")
            raise