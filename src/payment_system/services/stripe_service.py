import stripe
from typing import Optional, Dict, Any
from datetime import datetime
import logging
from ..models.payment_model import PaymentTransaction, Customer, ProductType, PricingTier, PaymentStatus

logger = logging.getLogger(__name__)

class StripeService:
    def __init__(self, api_key: str, webhook_secret: str):
        """
        Initialiserer Stripe-tjenesten med API-nøkler
        """
        self.stripe = stripe
        self.stripe.api_key = api_key
        self.webhook_secret = webhook_secret

    async def create_customer(self, customer: Customer) -> Dict[str, Any]:
        """
        Oppretter en ny kunde i Stripe
        """
        try:
            stripe_customer = await self.stripe.Customer.create(
                email=customer.email,
                name=customer.name,
                metadata={
                    'organization': customer.organization,
                    'vat_number': customer.vat_number
                }
            )
            return {
                'id': stripe_customer.id,
                'email': stripe_customer.email,
                'name': stripe_customer.name
            }
        except Exception as e:
            logger.error(f"Feil ved opprettelse av kunde i Stripe: {str(e)}")
            raise

    async def create_payment_intent(
        self,
        amount: float,
        currency: str,
        customer_id: str,
        payment_method_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Oppretter en betalingsintensjon i Stripe
        """
        try:
            intent_data = {
                'amount': int(amount * 100),  # Stripe bruker minste valutaenhet (øre)
                'currency': currency.lower(),
                'customer': customer_id,
                'metadata': metadata or {},
                'automatic_payment_methods': {'enabled': True}
            }
            
            if payment_method_id:
                intent_data['payment_method'] = payment_method_id
                intent_data['confirm'] = True
                intent_data['off_session'] = True

            payment_intent = await self.stripe.PaymentIntent.create(**intent_data)
            
            return {
                'id': payment_intent.id,
                'client_secret': payment_intent.client_secret,
                'status': payment_intent.status
            }
        except Exception as e:
            logger.error(f"Feil ved opprettelse av payment intent: {str(e)}")
            raise

    async def handle_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """
        Håndterer Stripe webhook events
        """
        try:
            event = self.stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )

            event_handlers = {
                'payment_intent.succeeded': self._handle_payment_success,
                'payment_intent.payment_failed': self._handle_payment_failure,
                'customer.subscription.created': self._handle_subscription_created,
                'customer.subscription.updated': self._handle_subscription_updated,
                'customer.subscription.deleted': self._handle_subscription_deleted
            }

            handler = event_handlers.get(event.type)
            if handler:
                return await handler(event.data.object)
            
            return {'status': 'ignored', 'type': event.type}

        except Exception as e:
            logger.error(f"Feil ved håndtering av webhook: {str(e)}")
            raise

    async def _handle_payment_success(self, payment_intent: Dict) -> Dict[str, Any]:
        """
        Håndterer vellykkede betalinger
        """
        try:
            transaction = PaymentTransaction(
                id=payment_intent.id,
                customer_id=payment_intent.customer,
                amount=payment_intent.amount / 100,
                currency=payment_intent.currency.upper(),
                status=PaymentStatus.COMPLETED,
                completed_at=datetime.fromtimestamp(payment_intent.created),
                metadata=payment_intent.metadata
            )

            return {
                'status': 'success',
                'transaction_id': transaction.id
            }
        except Exception as e:
            logger.error(f"Feil ved håndtering av vellykket betaling: {str(e)}")
            raise

    async def _handle_payment_failure(self, payment_intent: Dict) -> Dict[str, Any]:
        """
        Håndterer mislykkede betalinger
        """
        try:
            transaction = PaymentTransaction(
                id=payment_intent.id,
                customer_id=payment_intent.customer,
                amount=payment_intent.amount / 100,
                currency=payment_intent.currency.upper(),
                status=PaymentStatus.FAILED,
                metadata={
                    'error': payment_intent.last_payment_error.message if payment_intent.last_payment_error else 'Unknown error'
                }
            )

            return {
                'status': 'failed',
                'transaction_id': transaction.id,
                'error': transaction.metadata.get('error')
            }
        except Exception as e:
            logger.error(f"Feil ved håndtering av mislykket betaling: {str(e)}")
            raise

    async def generate_invoice(self, transaction: PaymentTransaction) -> str:
        """
        Genererer en faktura for en transaksjon
        """
        try:
            invoice = await self.stripe.Invoice.create(
                customer=transaction.customer_id,
                auto_advance=True,
                collection_method='charge_automatically',
                metadata={
                    'transaction_id': transaction.id,
                    'product_type': transaction.product_type,
                    'pricing_tier': transaction.pricing_tier
                }
            )

            await self.stripe.InvoiceItem.create(
                customer=transaction.customer_id,
                invoice=invoice.id,
                amount=int(transaction.amount * 100),
                currency=transaction.currency.lower(),
                description=f"{transaction.product_type} - {transaction.pricing_tier}"
            )

            finalized_invoice = await self.stripe.Invoice.finalize_invoice(invoice.id)
            return finalized_invoice.invoice_pdf

        except Exception as e:
            logger.error(f"Feil ved generering av faktura: {str(e)}")
            raise

    async def refund_payment(
        self,
        payment_intent_id: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refunderer en betaling helt eller delvis
        """
        try:
            refund_data = {
                'payment_intent': payment_intent_id,
                'reason': reason or 'requested_by_customer'
            }
            
            if amount:
                refund_data['amount'] = int(amount * 100)

            refund = await self.stripe.Refund.create(**refund_data)

            return {
                'id': refund.id,
                'status': refund.status,
                'amount': refund.amount / 100
            }

        except Exception as e:
            logger.error(f"Feil ved refundering av betaling: {str(e)}")
            raise
