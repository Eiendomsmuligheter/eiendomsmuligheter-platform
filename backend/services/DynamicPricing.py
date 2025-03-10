"""
DynamicPricing - Dynamisk prismodell for Eiendomsmuligheter Platform

Denne modulen håndterer dynamisk prissetting basert på:
- Kundens bruksmønster
- Datamengden som analyseres
- Kompleksiteten i analysen
- Ulike kundesegmenter
- Sesongvariasjoner
- Markedsetterspørsel
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math
import os
import random
from enum import Enum

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enum for kundetype
class CustomerType(str, Enum):
    INDIVIDUAL = "individual"     # Enkeltpersoner
    PROFESSIONAL = "professional" # Eiendomsmeglere, arkitekter
    ENTERPRISE = "enterprise"     # Entreprenører, utbyggere
    MUNICIPALITY = "municipality" # Kommuner og offentlige etater

# Enum for abonnementstype
class SubscriptionTier(str, Enum):
    FREE = "free"            # Gratis (begrenset)
    BASIC = "basic"          # Grunnleggende
    PROFESSIONAL = "professional" # Profesjonell
    ENTERPRISE = "enterprise"     # Bedrift
    CUSTOM = "custom"        # Skreddersydd

# Enum for valuta
class Currency(str, Enum):
    NOK = "NOK"
    EUR = "EUR"
    USD = "USD"

class DynamicPricing:
    """
    Hovedklasse for dynamisk prissetting i Eiendomsmuligheter Platform
    """
    _instance = None

    # Implementert som singleton
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DynamicPricing, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialiser prissettingsmodulen med standardverdier"""
        if self._initialized:
            return
            
        self._initialized = True
        self.base_prices = self._load_base_prices()
        self.price_history = []
        self.market_demand_factor = 1.0
        self.seasonal_factors = self._calculate_seasonal_factors()
        self.customer_discounts = {}
        self.promotional_codes = {}
        self.api_usage_stats = {}
        
        logger.info("DynamicPricing-modul initialisert")
        
    def _load_base_prices(self) -> Dict[str, Any]:
        """Last inn grunnpriser fra konfigurasjonsfil"""
        try:
            # Prøv å laste fra fil
            config_path = os.environ.get("PRICING_CONFIG_PATH", "config/pricing.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Hvis filen ikke finnes, bruk standardverdier
            return {
                "subscription": {
                    "free": {
                        "monthly_price": 0,
                        "annual_price": 0,
                        "included_requests": 10,
                        "included_properties": 3,
                        "max_request_complexity": 1
                    },
                    "basic": {
                        "monthly_price": 299,
                        "annual_price": 2990,
                        "included_requests": 50,
                        "included_properties": 20,
                        "max_request_complexity": 2
                    },
                    "professional": {
                        "monthly_price": 999,
                        "annual_price": 9990,
                        "included_requests": 500,
                        "included_properties": 100,
                        "max_request_complexity": 3
                    },
                    "enterprise": {
                        "monthly_price": 4999,
                        "annual_price": 49990,
                        "included_requests": 5000,
                        "included_properties": "unlimited",
                        "max_request_complexity": 5
                    }
                },
                "pay_per_use": {
                    "base_analysis_price": 99,
                    "property_size_factor": 0.02,  # Pris per kvm
                    "complexity_multipliers": {
                        "low": 1.0,
                        "medium": 1.5,
                        "high": 2.0,
                        "very_high": 3.0
                    },
                    "visualization_price": 49,
                    "report_generation": 79,
                    "regulatory_search": 59
                },
                "volume_discounts": {
                    "tier1": {"min": 10, "max": 50, "discount": 0.05},
                    "tier2": {"min": 51, "max": 100, "discount": 0.10},
                    "tier3": {"min": 101, "max": 500, "discount": 0.15},
                    "tier4": {"min": 501, "max": 1000, "discount": 0.20},
                    "tier5": {"min": 1001, "max": null, "discount": 0.25}
                },
                "customer_type_adjustments": {
                    "individual": 1.0,
                    "professional": 0.9,
                    "enterprise": 0.8,
                    "municipality": 0.7
                }
            }
            
        except Exception as e:
            logger.error(f"Feil ved lasting av prisdata: {str(e)}")
            # Returner forenklet standardprising ved feil
            return {
                "subscription": {
                    "free": {"monthly_price": 0, "annual_price": 0},
                    "basic": {"monthly_price": 299, "annual_price": 2990},
                    "professional": {"monthly_price": 999, "annual_price": 9990},
                    "enterprise": {"monthly_price": 4999, "annual_price": 49990}
                },
                "pay_per_use": {
                    "base_analysis_price": 99,
                    "complexity_multipliers": {"low": 1.0, "medium": 1.5, "high": 2.0}
                }
            }

    def _calculate_seasonal_factors(self) -> Dict[str, float]:
        """Beregn sesongvariasjoner i priser"""
        current_month = datetime.now().month
        
        # Høysesong (vår/tidlig sommer - mange eiendomstransaksjoner)
        high_season_months = [3, 4, 5, 6]
        # Lavsesong (vinter, sene sommermåneder - færre transaksjoner)
        low_season_months = [1, 2, 7, 8, 12]
        # Midtsesong (høst - moderat aktivitet)
        mid_season_months = [9, 10, 11]
        
        seasonal_factors = {}
        
        for month in range(1, 13):
            if month in high_season_months:
                seasonal_factors[str(month)] = 1.15  # 15% høyere pris i høysesong
            elif month in low_season_months:
                seasonal_factors[str(month)] = 0.9   # 10% lavere pris i lavsesong
            else:
                seasonal_factors[str(month)] = 1.0   # Normal pris i midtsesong
        
        return seasonal_factors
    
    def calculate_price(self, 
                       service_type: str, 
                       customer_data: Dict[str, Any],
                       request_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Beregn prisen for en tjeneste basert på kundedata og forespørselsdetaljer.
        
        Args:
            service_type: Type tjeneste ("property_analysis", "visualization", "report", etc.)
            customer_data: Data om kunden
            request_data: Detaljer om forespørselen
            
        Returns:
            Et dictionary med prisinformasjon
        """
        if not request_data:
            request_data = {}
            
        # Hent kundetype og abonnement
        customer_type = customer_data.get("customer_type", CustomerType.INDIVIDUAL)
        subscription_tier = customer_data.get("subscription_tier", SubscriptionTier.FREE)
        customer_id = customer_data.get("customer_id", "anonymous")
        
        # Sjekk om kunden har brukt opp sine inkluderte forespørsler
        subscription_usage = self._get_customer_usage(customer_id)
        subscription_limit = self._get_subscription_limit(subscription_tier)
        
        # Standardvaluta
        currency = customer_data.get("preferred_currency", Currency.NOK)
        
        # Grunnpris for tjenesten
        base_price = self._get_base_price_for_service(service_type)
        
        # Beregn kompleksitetsfaktor
        complexity = request_data.get("complexity", "low")
        complexity_factor = self.base_prices["pay_per_use"]["complexity_multipliers"].get(complexity, 1.0)
        
        # Juster for property_size hvis det er relevant
        property_size = request_data.get("property_size", 0)
        property_size_adjustment = 0
        
        if service_type == "property_analysis" and property_size > 0:
            property_size_factor = self.base_prices["pay_per_use"].get("property_size_factor", 0)
            property_size_adjustment = property_size * property_size_factor
        
        # Juster for kundetypen
        customer_adjustment = self.base_prices["customer_type_adjustments"].get(customer_type, 1.0)
        
        # Juster for sesong
        current_month = str(datetime.now().month)
        seasonal_adjustment = self.seasonal_factors.get(current_month, 1.0)
        
        # Juster for markedsetterspørsel
        demand_adjustment = self.market_demand_factor
        
        # Juster for volum-rabatter
        volume_discount = self._calculate_volume_discount(customer_id)
        
        # Spesifikke kundeavtaler
        customer_discount = self.customer_discounts.get(customer_id, 0.0)
        
        # Kampanjekoder
        promo_code = request_data.get("promo_code", None)
        promo_discount = self._get_promo_discount(promo_code)
        
        # Beregn grunnprisen justert for alle faktorer
        adjusted_base_price = base_price * complexity_factor * customer_adjustment * seasonal_adjustment * demand_adjustment
        
        # Legg til størrelsesjustering hvis relevant
        adjusted_price = adjusted_base_price + property_size_adjustment
        
        # Trekk fra rabatter
        final_price = adjusted_price * (1 - volume_discount) * (1 - customer_discount) * (1 - promo_discount)
        
        # Avrund til nærmeste heltall
        final_price = math.ceil(final_price)
        
        # Sjekk om prisen er inkludert i abonnementet
        price_to_charge = 0 if subscription_usage < subscription_limit else final_price
        
        # Logg prissettingsinformasjon for analyse
        self._log_price_calculation(
            customer_id=customer_id,
            service_type=service_type,
            base_price=base_price,
            adjustments={
                "complexity": complexity_factor,
                "customer_type": customer_adjustment,
                "seasonal": seasonal_adjustment,
                "demand": demand_adjustment,
                "volume_discount": volume_discount,
                "customer_discount": customer_discount,
                "promo_discount": promo_discount
            },
            final_price=final_price,
            price_to_charge=price_to_charge,
            currency=currency
        )
        
        # Returner detaljert prisinformasjon
        return {
            "service_type": service_type,
            "base_price": base_price,
            "adjusted_price": adjusted_price,
            "final_price": final_price,
            "price_to_charge": price_to_charge,
            "currency": currency,
            "subscription_usage": subscription_usage,
            "subscription_limit": subscription_limit,
            "is_included_in_subscription": price_to_charge == 0,
            "adjustments": {
                "complexity_factor": complexity_factor,
                "property_size_adjustment": property_size_adjustment,
                "customer_adjustment": customer_adjustment,
                "seasonal_adjustment": seasonal_adjustment,
                "demand_adjustment": demand_adjustment,
                "volume_discount": volume_discount,
                "customer_discount": customer_discount,
                "promo_discount": promo_discount
            }
        }
    
    def get_subscription_pricing(self, 
                               subscription_tier: SubscriptionTier, 
                               billing_cycle: str = "monthly", 
                               currency: Currency = Currency.NOK) -> Dict[str, Any]:
        """
        Hent prissetting for et abonnement.
        
        Args:
            subscription_tier: Abonnementsnivå
            billing_cycle: Faktureringssyklus ("monthly" eller "annual")
            currency: Ønsket valuta
            
        Returns:
            Prisinformasjon for abonnementet
        """
        tier_info = self.base_prices["subscription"].get(subscription_tier, {})
        
        if not tier_info:
            return {"error": f"Ukjent abonnementsnivå: {subscription_tier}"}
        
        price_key = "annual_price" if billing_cycle == "annual" else "monthly_price"
        base_price = tier_info.get(price_key, 0)
        
        # Juster for sesong hvis vi er i et kampanjeområde
        current_month = str(datetime.now().month)
        seasonal_adjustment = self.seasonal_factors.get(current_month, 1.0)
        
        # Juster for markedsetterspørsel
        demand_adjustment = self.market_demand_factor
        
        # Beregn justert pris
        adjusted_price = base_price * seasonal_adjustment * demand_adjustment
        final_price = math.ceil(adjusted_price)  # Avrund til nærmeste heltall
        
        # Konverter valuta hvis nødvendig
        if currency != Currency.NOK:
            final_price = self._convert_currency(final_price, Currency.NOK, currency)
        
        return {
            "subscription_tier": subscription_tier,
            "billing_cycle": billing_cycle,
            "base_price": base_price,
            "adjusted_price": final_price,
            "currency": currency,
            "adjustments": {
                "seasonal_adjustment": seasonal_adjustment,
                "demand_adjustment": demand_adjustment
            },
            "features": {
                "included_requests": tier_info.get("included_requests", 0),
                "included_properties": tier_info.get("included_properties", 0),
                "max_request_complexity": tier_info.get("max_request_complexity", 1)
            }
        }
    
    def register_usage(self, customer_id: str, service_type: str, complexity: str) -> None:
        """
        Registrer bruk av en tjeneste for en kunde.
        
        Args:
            customer_id: Kunde-ID
            service_type: Type tjeneste som ble brukt
            complexity: Kompleksitetsnivå
        """
        timestamp = datetime.now().isoformat()
        
        if customer_id not in self.api_usage_stats:
            self.api_usage_stats[customer_id] = []
            
        self.api_usage_stats[customer_id].append({
            "timestamp": timestamp,
            "service_type": service_type,
            "complexity": complexity
        })
        
        # Oppdater markedsetterspørselsfaktor basert på total bruk
        total_requests = sum(len(usage) for usage in self.api_usage_stats.values())
        
        # Juster markedsetterspørselsfaktor (forenklet eksempel)
        if total_requests > 10000:
            self.market_demand_factor = 1.1  # Høy etterspørsel
        elif total_requests > 5000:
            self.market_demand_factor = 1.05  # Moderat etterspørsel
        else:
            self.market_demand_factor = 1.0  # Normal etterspørsel
    
    def add_customer_discount(self, customer_id: str, discount_percentage: float, 
                             expiry_date: Optional[datetime] = None) -> None:
        """
        Legg til en kundespesifikk rabatt.
        
        Args:
            customer_id: Kunde-ID
            discount_percentage: Rabattprosent (0.0 - 1.0)
            expiry_date: Utløpsdato for rabatten
        """
        self.customer_discounts[customer_id] = {
            "discount": min(max(discount_percentage, 0.0), 1.0),  # Begrens til 0-100%
            "expiry_date": expiry_date
        }
        
        logger.info(f"Lagt til rabatt på {discount_percentage*100}% for kunde {customer_id}")
    
    def add_promo_code(self, code: str, discount_percentage: float, 
                      expiry_date: datetime, max_uses: int = 1,
                      valid_services: Optional[List[str]] = None) -> None:
        """
        Legg til en kampanjekode.
        
        Args:
            code: Kampanjekoden
            discount_percentage: Rabattprosent (0.0 - 1.0)
            expiry_date: Utløpsdato
            max_uses: Maksimalt antall ganger koden kan brukes
            valid_services: Tjenester koden er gyldig for
        """
        normalized_code = code.upper()
        
        self.promotional_codes[normalized_code] = {
            "discount": min(max(discount_percentage, 0.0), 1.0),
            "expiry_date": expiry_date,
            "max_uses": max_uses,
            "current_uses": 0,
            "valid_services": valid_services or []
        }
        
        logger.info(f"Lagt til kampanjekode {normalized_code} med {discount_percentage*100}% rabatt")
    
    def update_base_prices(self, new_prices: Dict[str, Any]) -> None:
        """
        Oppdater grunnprisene.
        
        Args:
            new_prices: Nye prisdata
        """
        # Lagre gammel prising for historikk
        self.price_history.append({
            "timestamp": datetime.now().isoformat(),
            "prices": self.base_prices
        })
        
        # Oppdater med nye priser, men behold eksisterende struktur
        for category, values in new_prices.items():
            if category in self.base_prices:
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, dict) and key in self.base_prices[category]:
                            self.base_prices[category][key].update(value)
                        else:
                            self.base_prices[category][key] = value
                else:
                    self.base_prices[category] = values
            else:
                self.base_prices[category] = values
        
        logger.info("Grunnpriser oppdatert")
    
    def get_price_history(self, 
                         start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Hent prishistorikk innenfor et datointervall.
        
        Args:
            start_date: Startdato for historikken
            end_date: Sluttdato for historikken
            
        Returns:
            Liste med prishistorikk-elementer
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)  # 1 år tilbake
            
        if not end_date:
            end_date = datetime.now()
            
        filtered_history = []
        
        for entry in self.price_history:
            entry_date = datetime.fromisoformat(entry["timestamp"])
            if start_date <= entry_date <= end_date:
                filtered_history.append(entry)
                
        return filtered_history
    
    def generate_price_quote(self, 
                           customer_id: str, 
                           services: List[Dict[str, Any]],
                           subscription_change: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generer et pristilbud for en kunde.
        
        Args:
            customer_id: Kunde-ID
            services: Liste med tjenester som skal inkluderes i tilbudet
            subscription_change: Eventuelle endringer i abonnement
            
        Returns:
            Pristilbud med detaljert informasjon
        """
        customer_data = self._get_customer_data(customer_id)
        
        total_price = 0
        service_prices = []
        
        # Beregn pris for hver tjeneste
        for service in services:
            price_info = self.calculate_price(
                service_type=service["type"],
                customer_data=customer_data,
                request_data=service.get("details", {})
            )
            
            service_prices.append(price_info)
            total_price += price_info["price_to_charge"]
        
        # Håndter abonnementsendringer
        subscription_price_info = None
        if subscription_change:
            new_tier = subscription_change.get("new_tier")
            billing_cycle = subscription_change.get("billing_cycle", "monthly")
            
            if new_tier:
                subscription_price_info = self.get_subscription_pricing(
                    subscription_tier=new_tier,
                    billing_cycle=billing_cycle,
                    currency=customer_data.get("preferred_currency", Currency.NOK)
                )
                
                total_price += subscription_price_info["adjusted_price"]
        
        # Generer tilbudsinformasjon
        quote_id = f"QUOTE-{customer_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        valid_until = datetime.now() + timedelta(days=30)  # Gyldig i 30 dager
        
        return {
            "quote_id": quote_id,
            "customer_id": customer_id,
            "created_at": datetime.now().isoformat(),
            "valid_until": valid_until.isoformat(),
            "total_price": total_price,
            "currency": customer_data.get("preferred_currency", Currency.NOK),
            "service_details": service_prices,
            "subscription_changes": subscription_price_info,
            "terms": {
                "payment_terms": "30 dager",
                "cancellation_policy": "Refusjon innen 14 dager hvis tjenesten ikke møter forventningene."
            }
        }
    
    # Private hjelpemetoder
    def _get_base_price_for_service(self, service_type: str) -> float:
        """Hent grunnpris for en tjeneste"""
        pay_per_use = self.base_prices.get("pay_per_use", {})
        
        if service_type == "property_analysis":
            return pay_per_use.get("base_analysis_price", 99)
        elif service_type == "visualization":
            return pay_per_use.get("visualization_price", 49)
        elif service_type == "report":
            return pay_per_use.get("report_generation", 79)
        elif service_type == "regulatory_search":
            return pay_per_use.get("regulatory_search", 59)
        else:
            # Standard pris for ukjente tjenester
            return pay_per_use.get("base_analysis_price", 99)
    
    def _get_customer_usage(self, customer_id: str) -> int:
        """Hent kundens bruk så langt denne måneden"""
        if customer_id not in self.api_usage_stats:
            return 0
            
        # Tell kun bruk fra inneværende måned
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        monthly_usage = 0
        for usage in self.api_usage_stats[customer_id]:
            usage_date = datetime.fromisoformat(usage["timestamp"])
            if usage_date.month == current_month and usage_date.year == current_year:
                monthly_usage += 1
                
        return monthly_usage
    
    def _get_subscription_limit(self, subscription_tier: SubscriptionTier) -> int:
        """Hent grensen for inkluderte forespørsler i abonnementet"""
        tier_info = self.base_prices["subscription"].get(subscription_tier, {})
        return tier_info.get("included_requests", 0)
    
    def _calculate_volume_discount(self, customer_id: str) -> float:
        """Beregn volumrabatt basert på kundens bruksmønster"""
        if customer_id not in self.api_usage_stats:
            return 0.0
            
        # Tell total bruk de siste 30 dagene
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        usage_count = 0
        for usage in self.api_usage_stats[customer_id]:
            usage_date = datetime.fromisoformat(usage["timestamp"])
            if usage_date >= thirty_days_ago:
                usage_count += 1
        
        # Finn riktig rabattnivå basert på bruk
        volume_discounts = self.base_prices.get("volume_discounts", {})
        
        for tier_name, tier_info in volume_discounts.items():
            min_usage = tier_info.get("min", 0)
            max_usage = tier_info.get("max", float('inf'))
            
            if min_usage <= usage_count <= (max_usage or float('inf')):
                return tier_info.get("discount", 0.0)
                
        return 0.0
    
    def _get_promo_discount(self, promo_code: Optional[str]) -> float:
        """Hent rabatt for en kampanjekode"""
        if not promo_code:
            return 0.0
            
        normalized_code = promo_code.upper()
        
        if normalized_code not in self.promotional_codes:
            return 0.0
            
        promo_info = self.promotional_codes[normalized_code]
        
        # Sjekk om koden er utgått
        if promo_info["expiry_date"] < datetime.now():
            return 0.0
            
        # Sjekk om koden har nådd maksimalt antall bruk
        if promo_info["current_uses"] >= promo_info["max_uses"]:
            return 0.0
            
        # Inkrementere bruken av koden
        promo_info["current_uses"] += 1
        
        return promo_info["discount"]
    
    def _get_customer_data(self, customer_id: str) -> Dict[str, Any]:
        """
        Hent kundedata (i en faktisk implementasjon ville dette komme fra en database)
        """
        # Dette er bare dummy-data for testing
        if customer_id == "anonymous":
            return {
                "customer_id": "anonymous",
                "customer_type": CustomerType.INDIVIDUAL,
                "subscription_tier": SubscriptionTier.FREE,
                "preferred_currency": Currency.NOK
            }
            
        # Simuler henting av kundedata
        customer_types = list(CustomerType)
        subscription_tiers = list(SubscriptionTier)
        
        # Deterministisk "tilfeldig" kundedata basert på kunde-ID
        hash_val = sum(ord(c) for c in customer_id)
        
        return {
            "customer_id": customer_id,
            "customer_type": customer_types[hash_val % len(customer_types)],
            "subscription_tier": subscription_tiers[hash_val % len(subscription_tiers)],
            "preferred_currency": Currency.NOK,
            "signup_date": (datetime.now() - timedelta(days=hash_val % 365)).isoformat()
        }
    
    def _convert_currency(self, amount: float, from_currency: Currency, to_currency: Currency) -> float:
        """
        Konverterer beløp mellom valutaer (forenklet for demonstrasjon)
        """
        if from_currency == to_currency:
            return amount
            
        # Forenklet valutakurser (i praksis ville dette hentes fra en API)
        exchange_rates = {
            "NOK_TO_EUR": 0.085,
            "NOK_TO_USD": 0.093,
            "EUR_TO_NOK": 11.76,
            "EUR_TO_USD": 1.09,
            "USD_TO_NOK": 10.75,
            "USD_TO_EUR": 0.92
        }
        
        # Konverter til ønsket valuta
        if from_currency == Currency.NOK and to_currency == Currency.EUR:
            return amount * exchange_rates["NOK_TO_EUR"]
        elif from_currency == Currency.NOK and to_currency == Currency.USD:
            return amount * exchange_rates["NOK_TO_USD"]
        elif from_currency == Currency.EUR and to_currency == Currency.NOK:
            return amount * exchange_rates["EUR_TO_NOK"]
        elif from_currency == Currency.EUR and to_currency == Currency.USD:
            return amount * exchange_rates["EUR_TO_USD"]
        elif from_currency == Currency.USD and to_currency == Currency.NOK:
            return amount * exchange_rates["USD_TO_NOK"]
        elif from_currency == Currency.USD and to_currency == Currency.EUR:
            return amount * exchange_rates["USD_TO_EUR"]
            
        return amount  # Fallback
    
    def _log_price_calculation(self, customer_id: str, service_type: str,
                              base_price: float, adjustments: Dict[str, float],
                              final_price: float, price_to_charge: float,
                              currency: Currency) -> None:
        """
        Logg detaljer om prisberegning for senere analyse
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "service_type": service_type,
            "base_price": base_price,
            "adjustments": adjustments,
            "final_price": final_price,
            "price_to_charge": price_to_charge,
            "currency": currency
        }
        
        logger.debug(f"Prisberegning: {json.dumps(log_entry)}")
        
        # I en faktisk implementasjon ville dette bli lagret i en database
        # for analyse og optimalisering av prismodellen
    
# Eksempel på bruk
if __name__ == "__main__":
    # Opprett instans av DynamicPricing
    pricing = DynamicPricing()
    
    # Eksempelbruk: Beregn pris for en eiendomsanalyse
    customer_data = {
        "customer_id": "CUST123",
        "customer_type": CustomerType.PROFESSIONAL,
        "subscription_tier": SubscriptionTier.BASIC,
        "preferred_currency": Currency.NOK
    }
    
    request_data = {
        "complexity": "medium",
        "property_size": 150,  # 150 kvm
        "promo_code": "WELCOME10"
    }
    
    # Legg til en kampanjekode
    pricing.add_promo_code(
        code="WELCOME10",
        discount_percentage=0.1,  # 10% rabatt
        expiry_date=datetime.now() + timedelta(days=30),
        max_uses=100,
        valid_services=["property_analysis", "visualization"]
    )
    
    # Beregn pris
    price_info = pricing.calculate_price(
        service_type="property_analysis",
        customer_data=customer_data,
        request_data=request_data
    )
    
    print(f"Prisdetaljer: {json.dumps(price_info, indent=2)}")
    
    # Hent abonnementsprising
    subscription_info = pricing.get_subscription_pricing(
        subscription_tier=SubscriptionTier.PROFESSIONAL,
        billing_cycle="annual",
        currency=Currency.NOK
    )
    
    print(f"Abonnementsinformasjon: {json.dumps(subscription_info, indent=2)}") 