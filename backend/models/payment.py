from pydantic import BaseModel, EmailStr, HttpUrl
from typing import List, Optional
from datetime import datetime

class CheckoutSessionRequest(BaseModel):
    """Request modell for å opprette en checkout-sesjon"""
    plan_id: str
    customer_email: EmailStr
    success_url: HttpUrl
    cancel_url: HttpUrl
    
class CheckoutSessionResponse(BaseModel):
    """Response modell for checkout-sesjon"""
    session_id: str
    public_key: str
    checkout_url: HttpUrl
    
class PortalSessionRequest(BaseModel):
    """Request modell for å opprette en portal-sesjon"""
    customer_id: str
    return_url: HttpUrl
    
class PortalSessionResponse(BaseModel):
    """Response modell for portal-sesjon"""
    portal_url: HttpUrl
    
class InvoiceItem(BaseModel):
    """Modell for en fakturalinje"""
    description: str
    amount: int  # Beløp i øre
    quantity: Optional[int] = 1
    
class InvoiceRequest(BaseModel):
    """Request modell for å opprette en faktura"""
    customer_id: str
    items: List[InvoiceItem]
    due_days: Optional[int] = 14
    
class InvoiceResponse(BaseModel):
    """Response modell for faktura"""
    invoice_id: str
    amount: int  # Totalbeløp i øre
    due_date: datetime
    invoice_url: HttpUrl
    pdf_url: HttpUrl
    
class WebhookEvent(BaseModel):
    """Modell for webhook-hendelser"""
    event_type: str
    data: dict