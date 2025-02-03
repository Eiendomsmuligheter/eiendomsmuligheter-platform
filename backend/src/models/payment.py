from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base
import enum
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class PaymentStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"

class PaymentType(enum.Enum):
    SUBSCRIPTION = "subscription"
    ONE_TIME = "one_time"
    ANALYSIS = "analysis"
    DOCUMENT = "document"

class PaymentMethod(enum.Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    VIPPS = "vipps"
    BANK_TRANSFER = "bank_transfer"
    INVOICE = "invoice"

class Payment(Base):
    __tablename__ = "payments"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    subscription_id = Column(String, ForeignKey("subscriptions.id"))
    
    # Betalingsinformasjon
    payment_type = Column(Enum(PaymentType), nullable=False)
    payment_method = Column(Enum(PaymentMethod))
    status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING)
    
    # Beløp og valuta
    amount = Column(Float, nullable=False)
    currency = Column(String, default="NOK")
    tax_amount = Column(Float)
    tax_rate = Column(Float)
    
    # Stripe-spesifikk informasjon
    stripe_payment_intent_id = Column(String)
    stripe_charge_id = Column(String)
    stripe_refund_id = Column(String)
    
    # Fakturainformasjon
    invoice_number = Column(String)
    invoice_reference = Column(String)
    due_date = Column(DateTime(timezone=True))
    
    # Betalingsdetaljer
    payment_details = Column(JSON)  # Detaljert betalingsinformasjon
    metadata = Column(JSON)  # Ekstra metadata
    
    # Refusjonsinformasjon
    refund_amount = Column(Float)
    refund_reason = Column(String)
    refund_date = Column(DateTime(timezone=True))
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relasjoner
    user = relationship("User", back_populates="payments")
    subscription = relationship("Subscription", back_populates="payments")
    
    def __repr__(self):
        return f"<Payment {self.id} - {self.amount} {self.currency}>"
        
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "subscription_id": self.subscription_id,
            "payment": {
                "type": self.payment_type.value,
                "method": self.payment_method.value if self.payment_method else None,
                "status": self.status.value,
                "amount": self.amount,
                "currency": self.currency,
                "tax": {
                    "amount": self.tax_amount,
                    "rate": self.tax_rate
                }
            },
            "stripe": {
                "payment_intent_id": self.stripe_payment_intent_id,
                "charge_id": self.stripe_charge_id,
                "refund_id": self.stripe_refund_id
            },
            "invoice": {
                "number": self.invoice_number,
                "reference": self.invoice_reference,
                "due_date": self.due_date
            },
            "refund": {
                "amount": self.refund_amount,
                "reason": self.refund_reason,
                "date": self.refund_date
            },
            "timestamps": {
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "completed_at": self.completed_at
            }
        }

    def process_payment(self, payment_details: dict):
        """Behandler en betaling"""
        try:
            self.status = PaymentStatus.PROCESSING
            self.payment_details = payment_details
            
            # Her ville vi integrere med Stripe eller annet betalingssystem
            # For nå, simulerer vi en vellykket betaling
            self.status = PaymentStatus.COMPLETED
            self.completed_at = func.now()
            
            return True
        except Exception as e:
            self.status = PaymentStatus.FAILED
            self.metadata = {
                "error": str(e),
                "error_time": func.now()
            }
            return False

    def process_refund(self, amount: float, reason: str):
        """Behandler en refusjon"""
        try:
            if amount > self.amount:
                raise ValueError("Refusjonsbeløp kan ikke være større enn betalingsbeløp")
                
            self.refund_amount = amount
            self.refund_reason = reason
            self.refund_date = func.now()
            
            if amount == self.amount:
                self.status = PaymentStatus.REFUNDED
            else:
                self.status = PaymentStatus.PARTIALLY_REFUNDED
                
            # Her ville vi integrere med Stripe eller annet betalingssystem
            return True
        except Exception as e:
            self.metadata = {
                **self.metadata,
                "refund_error": str(e),
                "refund_error_time": func.now()
            }
            return False

    def generate_invoice(self):
        """Genererer fakturainformasjon"""
        if not self.invoice_number:
            # Generer fakturanummer (dette er en forenklet versjon)
            year = func.now().year
            self.invoice_number = f"INV-{year}-{self.id[:8]}"
            
        if not self.due_date:
            # Sett forfallsdato til 14 dager frem i tid
            self.due_date = func.now() + datetime.timedelta(days=14)
            
        return {
            "invoice_number": self.invoice_number,
            "due_date": self.due_date,
            "amount": self.amount,
            "currency": self.currency,
            "tax_amount": self.tax_amount,
            "total_amount": self.amount + (self.tax_amount or 0)
        }