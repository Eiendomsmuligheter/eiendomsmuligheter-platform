from fastapi import APIRouter, Depends, HTTPException, Request, Header
from typing import Dict, Optional
from ..services.payment_service import PaymentService
from ..models.payment import (
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    PortalSessionRequest,
    PortalSessionResponse,
    InvoiceRequest,
    InvoiceResponse,
    WebhookEvent
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/payments", tags=["payments"])
payment_service = PaymentService()

@router.post("/create-checkout-session",
            response_model=CheckoutSessionResponse)
async def create_checkout_session(
    request: CheckoutSessionRequest
) -> CheckoutSessionResponse:
    """
    Opprett en ny Stripe Checkout-sesjon for abonnementstegning
    """
    try:
        session = await payment_service.create_checkout_session(
            plan_id=request.plan_id,
            customer_email=request.customer_email,
            success_url=request.success_url,
            cancel_url=request.cancel_url
        )
        
        return CheckoutSessionResponse(
            session_id=session["session_id"],
            public_key=session["public_key"],
            checkout_url=session["checkout_url"]
        )
        
    except Exception as e:
        logger.error(f"Feil ved opprettelse av checkout-sesjon: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.post("/create-portal-session",
            response_model=PortalSessionResponse)
async def create_portal_session(
    request: PortalSessionRequest
) -> PortalSessionResponse:
    """
    Opprett en Stripe Customer Portal-sesjon for abonnementshåndtering
    """
    try:
        session = await payment_service.create_portal_session(
            customer_id=request.customer_id,
            return_url=request.return_url
        )
        
        return PortalSessionResponse(
            portal_url=session["portal_url"]
        )
        
    except Exception as e:
        logger.error(f"Feil ved opprettelse av portalsesjon: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.post("/create-invoice",
            response_model=InvoiceResponse)
async def create_invoice(
    request: InvoiceRequest
) -> InvoiceResponse:
    """
    Opprett en faktura for ekstra tjenester
    """
    try:
        invoice = await payment_service.create_invoice(
            customer_id=request.customer_id,
            items=request.items,
            due_days=request.due_days
        )
        
        return InvoiceResponse(
            invoice_id=invoice["invoice_id"],
            amount=invoice["amount"],
            due_date=invoice["due_date"],
            invoice_url=invoice["invoice_url"],
            pdf_url=invoice["pdf_url"]
        )
        
    except Exception as e:
        logger.error(f"Feil ved opprettelse av faktura: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None)
) -> Dict:
    """
    Håndter Stripe webhook-hendelser
    """
    try:
        # Les raw payload
        payload = await request.body()
        
        # Håndter webhook
        result = await payment_service.handle_webhook(
            payload=payload,
            signature=stripe_signature
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Feil ved håndtering av webhook: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )