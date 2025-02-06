import stripe
from typing import Dict, Optional, List
import logging
from datetime import datetime
import json
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class PaymentService:
    """
    Håndterer all betalingsfunksjonalitet via Stripe.
    Støtter:
    - Engangskjøp
    - Abonnementer
    - Fakturaer
    - Refusjoner
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        stripe.api_key = self.config["stripe"]["secret_key"]
        self.prices = self._initialize_price_plans()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Last betalingskonfigurasjon"""
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "stripe": {
                "secret_key": "${STRIPE_SECRET_KEY}",
                "webhook_secret": "${STRIPE_WEBHOOK_SECRET}",
                "public_key": "${STRIPE_PUBLIC_KEY}",
                "currency": "NOK",
                "payment_methods": ["card", "vipps"]
            },
            "subscriptions": {
                "basic": {
                    "price_id": "price_basic",
                    "features": [
                        "Grunnleggende eiendomsanalyse",
                        "3D-visualisering",
                        "Energianalyse"
                    ]
                },
                "pro": {
                    "price_id": "price_pro",
                    "features": [
                        "Alt i Basic",
                        "Avansert utviklingsanalyse",
                        "Automatisk byggesøknad",
                        "Direkte kommuneintegrasjon"
                    ]
                },
                "enterprise": {
                    "price_id": "price_enterprise",
                    "features": [
                        "Alt i Pro",
                        "Ubegrenset analyser",
                        "Dedikert support",
                        "API-tilgang"
                    ]
                }
            }
        }
        
    def _initialize_price_plans(self) -> Dict:
        """Initialiser priser og abonnementsplaner i Stripe"""
        try:
            prices = {}
            
            # Opprett produkter hvis de ikke eksisterer
            for plan, details in self.config["subscriptions"].items():
                product = stripe.Product.create(
                    name=f"Eiendomsmuligheter {plan.capitalize()}",
                    description="\n".join(details["features"])
                )
                
                # Opprett priser for produktet
                price = stripe.Price.create(
                    product=product.id,
                    unit_amount=self._get_plan_price(plan),
                    currency=self.config["stripe"]["currency"],
                    recurring={
                        "interval": "month"
                    }
                )
                
                prices[plan] = {
                    "product_id": product.id,
                    "price_id": price.id
                }
                
            return prices
            
        except stripe.error.StripeError as e:
            logger.error(f"Feil ved initialisering av priser: {str(e)}")
            raise
            
    def create_checkout_session(self,
                              plan_id: str,
                              customer_email: str,
                              success_url: str,
                              cancel_url: str) -> Dict:
        """
        Opprett en Stripe Checkout-sesjon for abonnementstegning
        """
        try:
            if plan_id not in self.prices:
                raise ValueError(f"Ugyldig abonnementsplan: {plan_id}")
                
            session = stripe.checkout.Session.create(
                customer_email=customer_email,
                payment_method_types=self.config["stripe"]["payment_methods"],
                line_items=[{
                    "price": self.prices[plan_id]["price_id"],
                    "quantity": 1
                }],
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    "plan": plan_id
                }
            )
            
            return {
                "session_id": session.id,
                "public_key": self.config["stripe"]["public_key"],
                "checkout_url": session.url
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Feil ved opprettelse av checkout-sesjon: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Kunne ikke opprette betalingssesjon: {str(e)}"
            )
            
    async def handle_webhook(self, payload: Dict, signature: str) -> Dict:
        """
        Håndter Stripe webhooks for hendelser som vellykkede betalinger,
        fornyelser, og feilede betalinger
        """
        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                self.config["stripe"]["webhook_secret"]
            )
            
            if event.type == "checkout.session.completed":
                await self._handle_successful_checkout(event.data.object)
                
            elif event.type == "customer.subscription.updated":
                await self._handle_subscription_update(event.data.object)
                
            elif event.type == "customer.subscription.deleted":
                await self._handle_subscription_cancellation(event.data.object)
                
            elif event.type == "invoice.payment_failed":
                await self._handle_failed_payment(event.data.object)
                
            return {"status": "success", "event_type": event.type}
            
        except stripe.error.SignatureVerificationError:
            logger.error("Ugyldig webhook-signatur")
            raise HTTPException(
                status_code=400,
                detail="Ugyldig webhook-signatur"
            )
            
        except Exception as e:
            logger.error(f"Feil ved håndtering av webhook: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Webhook-feil: {str(e)}"
            )
            
    async def create_portal_session(self, customer_id: str, return_url: str) -> Dict:
        """
        Opprett en Stripe Customer Portal-sesjon for håndtering av abonnement
        """
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url
            )
            
            return {
                "portal_url": session.url
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Feil ved opprettelse av portalsesjon: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Kunne ikke opprette portalsesjon: {str(e)}"
            )
            
    async def create_invoice(self,
                           customer_id: str,
                           items: List[Dict],
                           due_days: int = 14) -> Dict:
        """
        Opprett en faktura for ekstra tjenester
        """
        try:
            # Opprett fakturaposter
            invoice_items = []
            for item in items:
                invoice_item = stripe.InvoiceItem.create(
                    customer=customer_id,
                    amount=item["amount"],
                    currency=self.config["stripe"]["currency"],
                    description=item["description"]
                )
                invoice_items.append(invoice_item)
            
            # Opprett fakturaen
            invoice = stripe.Invoice.create(
                customer=customer_id,
                collection_method="send_invoice",
                due_date=int(datetime.now().timestamp()) + (due_days * 86400)
            )
            
            # Send fakturaen
            invoice.send_invoice()
            
            return {
                "invoice_id": invoice.id,
                "amount": invoice.total,
                "due_date": datetime.fromtimestamp(invoice.due_date),
                "invoice_url": invoice.hosted_invoice_url,
                "pdf_url": invoice.invoice_pdf
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Feil ved opprettelse av faktura: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Kunne ikke opprette faktura: {str(e)}"
            )
            
    def _get_plan_price(self, plan: str) -> int:
        """Hent pris for en abonnementsplan i øre"""
        prices = {
            "basic": 99900,  # 999 NOK
            "pro": 299900,   # 2999 NOK
            "enterprise": 999900  # 9999 NOK
        }
        return prices.get(plan, 99900)
        
    async def _handle_successful_checkout(self, session: Dict):
        """Håndter vellykket checkout"""
        try:
            # Aktiver kundens tilgang
            await self._activate_customer_access(
                session.customer,
                session.metadata.get("plan")
            )
            
            # Send velkomstepost
            await self._send_welcome_email(
                session.customer_email,
                session.metadata.get("plan")
            )
            
        except Exception as e:
            logger.error(f"Feil ved håndtering av vellykket checkout: {str(e)}")
            raise
            
    async def _handle_subscription_update(self, subscription: Dict):
        """Håndter abonnementsoppdateringer"""
        try:
            # Oppdater kundens tilgangsnivå
            await self._update_customer_access(
                subscription.customer,
                subscription.items.data[0].price.id
            )
            
        except Exception as e:
            logger.error(f"Feil ved håndtering av abonnementsoppdatering: {str(e)}")
            raise
            
    async def _handle_subscription_cancellation(self, subscription: Dict):
        """Håndter abonnementskansellering"""
        try:
            # Deaktiver kundens tilgang (ved periodeslutt)
            await self._schedule_access_deactivation(
                subscription.customer,
                subscription.current_period_end
            )
            
            # Send spørreundersøkelse
            await self._send_cancellation_survey(
                subscription.customer
            )
            
        except Exception as e:
            logger.error(f"Feil ved håndtering av abonnementskansellering: {str(e)}")
            raise
            
    async def _handle_failed_payment(self, invoice: Dict):
        """Håndter mislykket betaling"""
        try:
            # Send varsel til kunden
            await self._send_payment_failed_notification(
                invoice.customer_email,
                invoice.hosted_invoice_url
            )
            
            # Planer for ny betalingsforsøk
            if invoice.next_payment_attempt:
                await self._schedule_payment_retry(
                    invoice.id,
                    invoice.next_payment_attempt
                )
                
        except Exception as e:
            logger.error(f"Feil ved håndtering av mislykket betaling: {str(e)}")
            raise