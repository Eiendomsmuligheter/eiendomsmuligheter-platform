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
                "PDF-rapport",
                "Grunnleggende reguleringssjekk"
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
                "Omfattende reguleringssjekk",
                "Energianalyse",
                "Utleiepotensialanalyse"
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
                "Prioritert behandling",
                "Månedlige konsultasjoner"
            ]
        )
    },
    ProductType.DRAWING_PACKAGE: {
        PricingTier.BASIC: ProductPrice(
            tier=PricingTier.BASIC,
            amount=2999.00,
            description="Grunnleggende tegningspakke",
            features=[
                "2D plantegninger",
                "Enkle fasadetegninger",
                "Situasjonsplan",
                "PDF-format"
            ]
        ),
        PricingTier.PROFESSIONAL: ProductPrice(
            tier=PricingTier.PROFESSIONAL,
            amount=4999.00,
            description="Profesjonell tegningspakke",
            features=[
                "Alt i Basic",
                "3D-modellering",
                "Detaljerte fasadetegninger",
                "Tekniske spesifikasjoner",
                "Revit/AutoCAD-filer"
            ]
        ),
        PricingTier.ENTERPRISE: ProductPrice(
            tier=PricingTier.ENTERPRISE,
            amount=7999.00,
            description="Enterprise tegningspakke",
            features=[
                "Alt i Professional",
                "BIM-modeller",
                "VR-visualisering",
                "Ubegrensede revisjoner",
                "Komplett byggesøknadspakke"
            ]
        )
    },
    ProductType.ENERGY_CONSULTATION: {
        PricingTier.BASIC: ProductPrice(
            tier=PricingTier.BASIC,
            amount=1999.00,
            description="Grunnleggende energirådgivning",
            features=[
                "Energimerking",
                "Enkel energianalyse",
                "Grunnleggende tiltaksforslag",
                "Enova-støtteberegning"
            ]
        ),
        PricingTier.PROFESSIONAL: ProductPrice(
            tier=PricingTier.PROFESSIONAL,
            amount=3999.00,
            description="Profesjonell energirådgivning",
            features=[
                "Alt i Basic",
                "Detaljert energianalyse",
                "Termografering",
                "Komplett tiltaksplan",
                "Lønnsomhetsberegninger"
            ]
        ),
        PricingTier.ENTERPRISE: ProductPrice(
            tier=PricingTier.ENTERPRISE,
            amount=6999.00,
            description="Enterprise energirådgivning",
            features=[
                "Alt i Professional",
                "Bygningssimulering",
                "Klimaregnskap",
                "Søknadsassistanse Enova",
                "Årlig oppfølging"
            ]
        )
    },
    ProductType.BUILDING_APPLICATION: {
        PricingTier.BASIC: ProductPrice(
            tier=PricingTier.BASIC,
            amount=4999.00,
            description="Grunnleggende byggesøknad",
            features=[
                "Søknadsskjemaer",
                "Enkle tegninger",
                "Nabovarsel",
                "Digital innsending"
            ]
        ),
        PricingTier.PROFESSIONAL: ProductPrice(
            tier=PricingTier.PROFESSIONAL,
            amount=7999.00,
            description="Profesjonell byggesøknad",
            features=[
                "Alt i Basic",
                "Komplett tegningssett",
                "Teknisk beskrivelse",
                "Ansvarserklæringer",
                "Dokumenthåndtering"
            ]
        ),
        PricingTier.ENTERPRISE: ProductPrice(
            tier=PricingTier.ENTERPRISE,
            amount=12999.00,
            description="Enterprise byggesøknad",
            features=[
                "Alt i Professional",
                "Prosjektledelse",
                "Dispensasjonssøknader",
                "Møter med kommune",
                "Oppfølging til ferdigattest"
            ]
        )
    },
    ProductType.COMPLETE_PACKAGE: {
        PricingTier.BASIC: ProductPrice(
            tier=PricingTier.BASIC,
            amount=9999.00,
            description="Grunnleggende totalpakke",
            features=[
                "Eiendomsanalyse Basic",
                "Tegningspakke Basic",
                "Energirådgivning Basic",
                "Byggesøknad Basic",
                "10% rabatt på totalpris"
            ]
        ),
        PricingTier.PROFESSIONAL: ProductPrice(
            tier=PricingTier.PROFESSIONAL,
            amount=16999.00,
            description="Profesjonell totalpakke",
            features=[
                "Eiendomsanalyse Professional",
                "Tegningspakke Professional",
                "Energirådgivning Professional",
                "Byggesøknad Professional",
                "15% rabatt på totalpris"
            ]
        ),
        PricingTier.ENTERPRISE: ProductPrice(
            tier=PricingTier.ENTERPRISE,
            amount=27999.00,
            description="Enterprise totalpakke",
            features=[
                "Eiendomsanalyse Enterprise",
                "Tegningspakke Enterprise",
                "Energirådgivning Enterprise",
                "Byggesøknad Enterprise",
                "20% rabatt på totalpris",
                "Dedikert prosjektleder"
            ]
        )
    }
}
