import stripe
from typing import Dict, Optional
import logging
from datetime import datetime
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PaymentPlan:
    id: str
    name: str
    price: float
    currency: str
    interval: str
    features: list[str]

class PaymentService:
    def __init__(self, api_key: str = None):
        """Initialize the payment service with Stripe API key"""
        self.api_key = api_key or os.getenv('STRIPE_API_KEY')
        if not self.api_key:
            raise ValueError("Stripe API key is required")
        stripe.api_key = self.api_key
        
        # Define subscription plans
        self.plans = {
            'basic': PaymentPlan(
                id='price_basic',
                name='Basic',
                price=299.0,
                currency='NOK',
                interval='month',
                features=[
                    'Grunnleggende eiendomsanalyse',
                    'PDF-rapporter',
                    'Kommune-regelsjekk',
                    'Enkel 3D-visualisering'
                ]
            ),
            'pro': PaymentPlan(
                id='price_pro',
                name='Professional',
                price=599.0,
                currency='NOK',
                interval='month',
                features=[
                    'Alt i Basic',
                    'Avansert AI-analyse',
                    'Full 3D-visualisering',
                    'Automatisk dokumentgenerering',
                    'Reguleringsanalyse',
                    'Økonomisk analyse'
                ]
            ),
            'enterprise': PaymentPlan(
                id='price_enterprise',
                name='Enterprise',
                price=1499.0,
                currency='NOK',
                interval='month',
                features=[
                    'Alt i Professional',
                    'Ubegrenset analyser',
                    'API-tilgang',
                    'Dedikert support',
                    'Tilpassede rapporter',
                    'Team-samarbeid'
                ]
            )
        }

    async def create_customer(self, email: str, name: str) -> Dict:
        """Create a new customer in Stripe"""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name
            )
            logger.info(f"Created customer: {customer.id}")
            return customer
        except stripe.error.StripeError as e:
            logger.error(f"Error creating customer: {str(e)}")
            raise

    async def create_subscription(self, customer_id: str, plan_id: str) -> Dict:
        """Create a new subscription for a customer"""
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{'price': plan_id}],
                payment_behavior='default_incomplete',
                expand=['latest_invoice.payment_intent']
            )
            logger.info(f"Created subscription: {subscription.id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error creating subscription: {str(e)}")
            raise

    async def cancel_subscription(self, subscription_id: str) -> Dict:
        """Cancel a subscription"""
        try:
            subscription = stripe.Subscription.delete(subscription_id)
            logger.info(f"Cancelled subscription: {subscription_id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error cancelling subscription: {str(e)}")
            raise

    async def create_payment_intent(self, amount: float, currency: str = 'NOK') -> Dict:
        """Create a payment intent for one-time payments"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency
            )
            logger.info(f"Created payment intent: {intent.id}")
            return intent
        except stripe.error.StripeError as e:
            logger.error(f"Error creating payment intent: {str(e)}")
            raise

    async def get_subscription_status(self, subscription_id: str) -> Dict:
        """Get the status of a subscription"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return {
                'status': subscription.status,
                'current_period_end': datetime.fromtimestamp(subscription.current_period_end),
                'cancel_at_period_end': subscription.cancel_at_period_end
            }
        except stripe.error.StripeError as e:
            logger.error(f"Error retrieving subscription: {str(e)}")
            raise

    async def update_payment_method(self, customer_id: str, payment_method_id: str) -> Dict:
        """Update a customer's default payment method"""
        try:
            payment_method = stripe.PaymentMethod.attach(
                payment_method_id,
                customer=customer_id
            )
            customer = stripe.Customer.modify(
                customer_id,
                invoice_settings={'default_payment_method': payment_method_id}
            )
            logger.info(f"Updated payment method for customer: {customer_id}")
            return customer
        except stripe.error.StripeError as e:
            logger.error(f"Error updating payment method: {str(e)}")
            raise

    def get_available_plans(self) -> Dict[str, PaymentPlan]:
        """Get all available subscription plans"""
        return self.plans

    async def create_portal_session(self, customer_id: str, return_url: str) -> Dict:
        """Create a billing portal session for a customer"""
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url
            )
            logger.info(f"Created portal session for customer: {customer_id}")
            return session
        except stripe.error.StripeError as e:
            logger.error(f"Error creating portal session: {str(e)}")
            raise

    async def handle_webhook(self, payload: bytes, sig_header: str, webhook_secret: str) -> None:
        """Handle Stripe webhooks"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
            
            # Handle specific event types
            if event.type == 'invoice.paid':
                await self._handle_successful_payment(event.data.object)
            elif event.type == 'invoice.payment_failed':
                await self._handle_failed_payment(event.data.object)
            elif event.type == 'customer.subscription.deleted':
                await self._handle_subscription_cancelled(event.data.object)
                
            logger.info(f"Processed webhook event: {event.type}")
            return event
            
        except (stripe.error.SignatureVerificationError, ValueError) as e:
            logger.error(f"Error handling webhook: {str(e)}")
            raise

    async def _handle_successful_payment(self, invoice):
        """Handle successful payment webhook"""
        # Implement successful payment logic
        logger.info(f"Successfully processed payment for invoice: {invoice.id}")

    async def _handle_failed_payment(self, invoice):
        """Handle failed payment webhook"""
        # Implement failed payment logic
        logger.error(f"Failed payment for invoice: {invoice.id}")

    async def _handle_subscription_cancelled(self, subscription):
        """Handle subscription cancellation webhook"""
        # Implement subscription cancellation logic
        logger.info(f"Subscription cancelled: {subscription.id}")