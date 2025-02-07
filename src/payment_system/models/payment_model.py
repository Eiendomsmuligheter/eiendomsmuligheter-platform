from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum

class PaymentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class ProductType(str, Enum):
    ANALYSIS_REPORT = "analysis_report"
    DRAWING_PACKAGE = "drawing_package"
    ENERGY_CONSULTATION = "energy_consultation"
    BUILDING_APPLICATION = "building_application"
    COMPLETE_PACKAGE = "complete_package"

class PricingTier(str, Enum):
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class ProductPrice(BaseModel):
    tier: PricingTier
    amount: float
    currency: str = "NOK"
    description: str
    features: List[str]

class PaymentMethod(BaseModel):
    type: str  # "card", "vipps", "bank_transfer"
    details: dict
    is_default: bool = False

class Customer(BaseModel):
    id: str
    email: str
    name: str
    organization: Optional[str]
    vat_number: Optional[str]
    payment_methods: List[PaymentMethod] = []

class PaymentTransaction(BaseModel):
    id: str = Field(..., description="Unik transaksjons-ID")
    customer_id: str
    product_type: ProductType
    pricing_tier: PricingTier
    amount: float
    currency: str = "NOK"
    status: PaymentStatus
    payment_method: str
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime]
    invoice_number: Optional[str]
    receipt_url: Optional[str]
    metadata: dict = {}

class Subscription(BaseModel):
    id: str
    customer_id: str
    pricing_tier: PricingTier
    status: str  # "active", "cancelled", "suspended"
    start_date: datetime
    end_date: Optional[datetime]
    auto_renew: bool = True
    payment_method_id: str
    last_payment_date: Optional[datetime]
    next_payment_date: Optional[datetime]

class Invoice(BaseModel):
    id: str
    transaction_id: str
    customer_id: str
    amount: float
    currency: str = "NOK"
    issue_date: datetime
    due_date: datetime
    status: str  # "paid", "unpaid", "overdue"
    line_items: List[dict]
    pdf_url: Optional[str]

# Prisliste for forskjellige produkter og tjenester
PRODUCT_PRICING = {
    ProductType.ANALYSIS_REPORT: {
        PricingTier.BASIC: ProductPrice(
            tier=PricingTier.BASIC,
            amount=1499.00,
            description="Grunnleggende eiendomsanalyse",
            features=[
                "Automatisk plantegningsanalyse",
                "Enkel ROI-kalkulator",
                "PDF-rapport"
            ]
        ),
        PricingTier.PROFESSIONAL: ProductPrice(
            tier=PricingTier.PROFESSIONAL,
            amount=2999.00,
            description="Profesjonell eiendomsanalyse",
            features=[
                "Alt i Basic",
                "Detaljert potensialanalyse",
                "3D-visualisering",
                "Reguleringssjekk",
                "Energianalyse"
            ]
        ),
        PricingTier.ENTERPRISE: ProductPrice(
            tier=PricingTier.ENTERPRISE,
            amount=4999.00,
            description="Enterprise eiendomsanalyse",
            features=[
                "Alt i Professional",
                "API-tilgang",
                "Dedikert støtte",
                "Ubegrenset analyser",
                "Prioritert behandling"
            ]
        )
    }
}
