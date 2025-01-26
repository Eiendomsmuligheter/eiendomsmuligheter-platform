from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

@dataclass
class EconomicMetrics:
    initial_investment: float
    monthly_income: float
    monthly_expenses: float
    roi: float
    payback_period: float
    net_present_value: float
    internal_rate_return: float
    risk_assessment: Dict[str, float]

@dataclass
class MarketData:
    average_rent: float
    price_per_sqm: float
    vacancy_rate: float
    market_trend: str
    last_updated: datetime

class EconomicAnalyzer:
    def __init__(self):
        self.market_data = self._load_market_data()
        self.risk_factors = self._initialize_risk_factors()
        self.tax_rates = self._load_tax_rates()
        
    def _load_market_data(self) -> Dict[str, MarketData]:
        """Last inn markedsdata for forskellige områder"""
        return {
            "Oslo": MarketData(
                average_rent=15000.0,
                price_per_sqm=85000.0,
                vacancy_rate=0.02,
                market_trend="increasing",
                last_updated=datetime.now()
            )
        }
        
    def _initialize_risk_factors(self) -> Dict[str, float]:
        """Initialiser risikofaktorer"""
        return {
            "market_volatility": 0.15,
            "interest_rate_risk": 0.10,
            "maintenance_risk": 0.08,
            "vacancy_risk": 0.05,
            "regulatory_risk": 0.07
        }
        
    def _load_tax_rates(self) -> Dict[str, float]:
        """Last inn gjeldende skattesatser"""
        return {
            "income_tax": 0.22,
            "property_tax": 0.003,
            "capital_gains": 0.22
        }
        
    def analyze_investment(self, property_data: Dict[str, Any]) -> EconomicMetrics:
        """Utfør fullstendig økonomisk analyse"""
        try:
            # Beregn grunnleggende metrics
            initial_investment = self._calculate_initial_investment(property_data)
            monthly_income = self._estimate_monthly_income(property_data)
            monthly_expenses = self._estimate_monthly_expenses(property_data)
            
            # Beregn avanserte metrics
            roi = self._calculate_roi(monthly_income, monthly_expenses, initial_investment)
            npv = self._calculate_npv(monthly_income, monthly_expenses, initial_investment)
            irr = self._calculate_irr(monthly_income, monthly_expenses, initial_investment)
            payback = self._calculate_payback_period(monthly_income, monthly_expenses, initial_investment)
            
            # Utfør risikovurdering
            risk_assessment = self._assess_risks(property_data)
            
            return EconomicMetrics(
                initial_investment=initial_investment,
                monthly_income=monthly_income,
                monthly_expenses=monthly_expenses,
                roi=roi,
                payback_period=payback,
                net_present_value=npv,
                internal_rate_return=irr,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Feil ved økonomisk analyse: {str(e)}")
            return None
            
    def _calculate_initial_investment(self, property_data: Dict[str, Any]) -> float:
        """Beregn total initialinvestering"""
        try:
            # Hent nødvendige data
            area = property_data.get("area", 0)
            location = property_data.get("location", "Oslo")
            condition = property_data.get("condition", "good")
            renovation_needs = property_data.get("renovation_needs", [])
            
            # Beregn grunnpris basert på område og markedsdata
            base_price = area * self.market_data[location].price_per_sqm
            
            # Juster for tilstand
            condition_factors = {
                "excellent": 1.1,
                "good": 1.0,
                "fair": 0.9,
                "poor": 0.8
            }
            condition_adjustment = condition_factors.get(condition, 1.0)
            adjusted_price = base_price * condition_adjustment
            
            # Legg til oppussings- og oppgraderingskostnader
            renovation_costs = sum(cost["estimated_cost"] for cost in renovation_needs)
            
            # Legg til andre oppstartskostnader
            other_costs = {
                "documentation": 15000,  # Dokumentasjon og søknader
                "inspection": 25000,     # Teknisk inspeksjon
                "legal": 30000,          # Juridisk bistand
                "insurance": 10000,      # Forsikring første år
                "buffer": adjusted_price * 0.05  # 5% buffer for uforutsette utgifter
            }
            
            total_other_costs = sum(other_costs.values())
            
            # Beregn total initialinvestering
            total_investment = adjusted_price + renovation_costs + total_other_costs
            
            logger.info(f"Beregnet initialinvestering: {total_investment:,.2f} NOK")
            return total_investment
            
        except Exception as e:
            logger.error(f"Feil ved beregning av initialinvestering: {str(e)}")
            raise
        
    def _estimate_monthly_income(self, property_data: Dict[str, Any]) -> float:
        """Estimer månedlig inntekt"""
        try:
            # Hent nødvendige data
            area = property_data.get("area", 0)
            location = property_data.get("location", "Oslo")
            property_type = property_data.get("type", "apartment")
            rooms = property_data.get("rooms", 1)
            features = property_data.get("features", [])
            
            # Hent grunnlegende leiepris fra markedsdata
            base_rent = self.market_data[location].average_rent
            
            # Juster for størrelse
            area_factor = 1.0
            if area > 50:
                area_factor = 0.9  # Pris per kvm synker litt for større enheter
            elif area < 30:
                area_factor = 1.1  # Små enheter har ofte høyere pris per kvm
                
            # Juster for antall rom
            room_factors = {
                1: 1.0,    # Studio/1-roms
                2: 1.2,    # 2-roms
                3: 1.3,    # 3-roms
                4: 1.4     # 4-roms eller større
            }
            room_factor = room_factors.get(rooms, 1.5)
            
            # Juster for spesielle egenskaper
            feature_premiums = {
                "parking": 1000,
                "balcony": 800,
                "elevator": 500,
                "furnished": 2000,
                "new_kitchen": 1000,
                "new_bathroom": 1000,
                "storage": 300,
                "garden": 1200
            }
            
            features_premium = sum(feature_premiums.get(feature, 0) for feature in features)
            
            # Juster for eiendomstype
            type_factors = {
                "apartment": 1.0,
                "house": 1.1,
                "townhouse": 1.05,
                "duplex": 1.02
            }
            type_factor = type_factors.get(property_type, 1.0)
            
            # Beregn justert månedlig leieinntekt
            monthly_rent = (
                (base_rent * area_factor * room_factor * type_factor) +
                features_premium
            )
            
            # Juster for utleiegrad (ta hensyn til perioder uten leietaker)
            vacancy_rate = self.market_data[location].vacancy_rate
            expected_income = monthly_rent * (1 - vacancy_rate)
            
            # Avrund til nærmeste 100
            expected_income = round(expected_income / 100) * 100
            
            logger.info(f"Estimert månedlig leieinntekt: {expected_income:,.2f} NOK")
            return expected_income
            
        except Exception as e:
            logger.error(f"Feil ved estimering av månedlig inntekt: {str(e)}")
            raise
        
    def _estimate_monthly_expenses(self, property_data: Dict[str, Any]) -> float:
        """Estimer månedlige utgifter"""
        try:
            # Hent nødvendige data
            area = property_data.get("area", 0)
            property_value = property_data.get("value", 0)
            property_type = property_data.get("type", "apartment")
            age = property_data.get("age", 0)
            loan_details = property_data.get("loan_details", {})
            
            # Faste månedlige utgifter
            fixed_expenses = {
                "kommunale_avgifter": 500,    # Kommunale avgifter
                "forsikring": 400,            # Bygningsforsikring
                "eiendomsskatt": property_value * self.tax_rates["property_tax"] / 12,
                "fellesutgifter": 45 * area,  # Estimert 45 NOK per kvm per måned for leiligheter
            }
            
            # Juster fellesutgifter basert på eiendomstype
            if property_type != "apartment":
                fixed_expenses["fellesutgifter"] = 0
            
            # Vedlikeholdsutgifter (basert på alder og type)
            maintenance_factors = {
                "apartment": 0.0015,  # 0.15% av verdi årlig
                "house": 0.0025,      # 0.25% av verdi årlig
                "townhouse": 0.002,   # 0.2% av verdi årlig
                "duplex": 0.002       # 0.2% av verdi årlig
            }
            
            # Øk vedlikeholdsfaktor for eldre eiendommer
            age_factor = 1.0
            if age > 30:
                age_factor = 1.5
            elif age > 15:
                age_factor = 1.2
            
            maintenance_factor = maintenance_factors.get(property_type, 0.002) * age_factor
            monthly_maintenance = (property_value * maintenance_factor) / 12
            
            # Lånekostnader (hvis relevant)
            monthly_mortgage = 0
            if loan_details:
                loan_amount = loan_details.get("amount", 0)
                interest_rate = loan_details.get("interest_rate", 0.04)
                loan_years = loan_details.get("years", 30)
                
                # Beregn månedlig lånekostnad (forenklet annuitetsformel)
                if loan_amount > 0:
                    monthly_rate = interest_rate / 12
                    num_payments = loan_years * 12
                    monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            
            # Administrasjonskostnader
            admin_costs = 500  # Fast månedlig kostnad for administrasjon
            
            # Buffer for uforutsette utgifter (5% av faste utgifter)
            buffer = sum(fixed_expenses.values()) * 0.05
            
            # Sum opp alle månedlige utgifter
            total_monthly_expenses = (
                sum(fixed_expenses.values()) +
                monthly_maintenance +
                monthly_mortgage +
                admin_costs +
                buffer
            )
            
            # Avrund til nærmeste 100
            total_monthly_expenses = round(total_monthly_expenses / 100) * 100
            
            logger.info(f"Estimerte månedlige utgifter: {total_monthly_expenses:,.2f} NOK")
            return total_monthly_expenses
            
        except Exception as e:
            logger.error(f"Feil ved estimering av månedlige utgifter: {str(e)}")
            raise
        
    def _calculate_roi(self, income: float, expenses: float, investment: float) -> float:
        """Beregn avkastning på investering"""
        try:
            if investment <= 0:
                raise ValueError("Investeringsbeløp må være større enn 0")
            
            # Beregn årlig netto inntekt
            annual_income = income * 12
            annual_expenses = expenses * 12
            annual_net_income = annual_income - annual_expenses
            
            # Beregn ROI som prosent
            roi_percentage = (annual_net_income / investment) * 100
            
            # Legg til verdiøkning (antatt 3% årlig)
            annual_appreciation = 3.0
            total_roi = roi_percentage + annual_appreciation
            
            # Juster for risiko
            risk_adjusted_roi = self._adjust_roi_for_risk(total_roi)
            
            logger.info(f"Beregnet ROI: {risk_adjusted_roi:.2f}%")
            return round(risk_adjusted_roi, 2)
            
        except Exception as e:
            logger.error(f"Feil ved ROI-beregning: {str(e)}")
            raise
            
    def _adjust_roi_for_risk(self, base_roi: float) -> float:
        """Juster ROI basert på risikofaktorer"""
        try:
            # Summer alle risikofaktorer
            total_risk = sum(self.risk_factors.values())
            
            # Juster ROI ned basert på total risiko
            risk_adjusted_roi = base_roi * (1 - total_risk)
            
            # Sikre at ROI ikke blir negativ
            return max(0, risk_adjusted_roi)
            
        except Exception as e:
            logger.error(f"Feil ved risikojustering av ROI: {str(e)}")
            return base_roi
        
    def _calculate_npv(self, income: float, expenses: float, investment: float) -> float:
        """Beregn nåverdi (Net Present Value)"""
        try:
            # Definer parametre
            time_horizon = 20  # År
            discount_rate = 0.05  # 5% diskonteringsrente
            annual_net_income = (income - expenses) * 12
            
            # Beregn forventet årlig vekst i inntekter og utgifter
            income_growth_rate = 0.02  # 2% årlig vekst i leieinntekter
            expense_growth_rate = 0.025  # 2.5% årlig vekst i utgifter
            
            # Beregn nåverdi av alle fremtidige kontantstrømmer
            npv = -investment  # Start med initial investering (negativ)
            
            for year in range(1, time_horizon + 1):
                # Beregn justerte inntekter og utgifter for året
                year_income = annual_net_income * (1 + income_growth_rate) ** year
                year_expenses = (expenses * 12) * (1 + expense_growth_rate) ** year
                net_cash_flow = year_income - year_expenses
                
                # Diskontert verdi av årets kontantstrøm
                npv += net_cash_flow / (1 + discount_rate) ** year
            
            # Legg til estimert sluttverdi (exit value)
            exit_cap_rate = 0.04  # 4% cap rate for exit
            final_year_noi = annual_net_income * (1 + income_growth_rate) ** time_horizon
            exit_value = final_year_noi / exit_cap_rate
            discounted_exit_value = exit_value / (1 + discount_rate) ** time_horizon
            
            npv += discounted_exit_value
            
            logger.info(f"Beregnet NPV: {npv:,.2f} NOK")
            return round(npv, 2)
            
        except Exception as e:
            logger.error(f"Feil ved NPV-beregning: {str(e)}")
            raise
        
    def _calculate_irr(self, income: float, expenses: float, investment: float) -> float:
        """Beregn internrente (Internal Rate of Return)"""
        try:
            # Definer parametre
            time_horizon = 20  # År
            annual_net_income = (income - expenses) * 12
            
            # Opprett kontantstrømarray
            cash_flows = [-investment]  # Start med initialinvestering
            
            # Legg til årlige kontantstrømmer
            income_growth_rate = 0.02  # 2% årlig vekst
            expense_growth_rate = 0.025  # 2.5% årlig vekst
            
            for year in range(1, time_horizon + 1):
                year_income = annual_net_income * (1 + income_growth_rate) ** year
                year_expenses = (expenses * 12) * (1 + expense_growth_rate) ** year
                net_cash_flow = year_income - year_expenses
                cash_flows.append(net_cash_flow)
            
            # Legg til exit value i siste år
            exit_cap_rate = 0.04
            final_year_noi = annual_net_income * (1 + income_growth_rate) ** time_horizon
            exit_value = final_year_noi / exit_cap_rate
            cash_flows[-1] += exit_value
            
            # Beregn IRR ved hjelp av numpy's IRR-funksjon
            irr = np.irr(cash_flows)
            
            # Hvis IRR ikke kan beregnes, bruk alternativ metode
            if np.isnan(irr):
                irr = self._calculate_irr_manual(cash_flows)
            
            # Konverter til prosent og avrund
            irr_percentage = irr * 100
            
            logger.info(f"Beregnet IRR: {irr_percentage:.2f}%")
            return round(irr_percentage, 2)
            
        except Exception as e:
            logger.error(f"Feil ved IRR-beregning: {str(e)}")
            raise
            
    def _calculate_irr_manual(self, cash_flows: List[float], iterations: int = 1000) -> float:
        """Manuell IRR-beregning ved hjelp av binærsøk"""
        try:
            # Definer søkeområde for rente
            rate_low = -0.99  # -99%
            rate_high = 10.0  # 1000%
            
            # Måltoleranse
            tolerance = 0.0001
            
            for _ in range(iterations):
                rate_mid = (rate_low + rate_high) / 2
                npv = 0
                
                # Beregn NPV med gjeldende rente
                for i, cf in enumerate(cash_flows):
                    npv += cf / (1 + rate_mid) ** i
                
                if abs(npv) < tolerance:
                    return rate_mid
                elif npv > 0:
                    rate_low = rate_mid
                else:
                    rate_high = rate_mid
            
            return (rate_low + rate_high) / 2
            
        except Exception as e:
            logger.error(f"Feil ved manuell IRR-beregning: {str(e)}")
            return 0.0
        
    def _calculate_payback_period(self, income: float, expenses: float, investment: float) -> float:
        """Beregn tilbakebetalingstid med hensyn til tidsverdien av penger"""
        try:
            if investment <= 0:
                raise ValueError("Investeringsbeløp må være større enn 0")
                
            # Beregn årlig netto kontantstrøm
            annual_net_income = (income - expenses) * 12
            
            if annual_net_income <= 0:
                raise ValueError("Årlig netto inntekt må være positiv for å beregne tilbakebetalingstid")
            
            # Parametre for beregning
            discount_rate = 0.05  # 5% årlig diskonteringsrente
            income_growth = 0.02  # 2% årlig vekst i inntekter
            max_years = 50  # Maksimal beregningsperiode
            
            # Beregn diskontert tilbakebetalingstid
            remaining_investment = investment
            years = 0
            partial_year = 0
            
            for year in range(1, max_years + 1):
                # Beregn årets diskonterte kontantstrøm
                year_cash_flow = annual_net_income * (1 + income_growth) ** (year - 1)
                discounted_cash_flow = year_cash_flow / (1 + discount_rate) ** year
                
                if remaining_investment <= discounted_cash_flow:
                    # Beregn delår for nøyaktig tilbakebetalingstid
                    partial_year = remaining_investment / discounted_cash_flow
                    years = year - 1 + partial_year
                    break
                    
                remaining_investment -= discounted_cash_flow
                years = year
                
                if year == max_years:
                    logger.warning("Maksimal beregningsperiode nådd uten full tilbakebetaling")
                    return float('inf')
            
            # Legg til verdiøkning i beregningen
            property_appreciation = 0.03  # 3% årlig verdiøkning
            appreciation_factor = 1 - (property_appreciation / discount_rate)
            adjusted_years = years * appreciation_factor
            
            logger.info(f"Beregnet tilbakebetalingstid: {adjusted_years:.2f} år")
            return round(adjusted_years, 2)
            
        except Exception as e:
            logger.error(f"Feil ved beregning av tilbakebetalingstid: {str(e)}")
            raise
        
    def _assess_risks(self, property_data: Dict[str, Any]) -> Dict[str, float]:
        """Utfør omfattende risikovurdering av investeringen"""
        try:
            risks = {}
            
            # Markedsrisiko
            market_risk = self._assess_market_risk(property_data)
            risks["market_risk"] = market_risk
            
            # Finansiell risiko
            financial_risk = self._assess_financial_risk(property_data)
            risks["financial_risk"] = financial_risk
            
            # Eiendomsspesifikk risiko
            property_risk = self._assess_property_risk(property_data)
            risks["property_risk"] = property_risk
            
            # Leietakerrisiko
            tenant_risk = self._assess_tenant_risk(property_data)
            risks["tenant_risk"] = tenant_risk
            
            # Regulatorisk risiko
            regulatory_risk = self._assess_regulatory_risk(property_data)
            risks["regulatory_risk"] = regulatory_risk
            
            # Beregn total risikopoeng (0-100, hvor 100 er høyest risiko)
            total_risk = sum(risks.values()) / len(risks)
            risks["total_risk"] = total_risk
            
            # Legg til risikokategori
            risks["risk_category"] = self._categorize_risk(total_risk)
            
            logger.info(f"Risikovurdering fullført. Total risiko: {total_risk:.2f}")
            return risks
            
        except Exception as e:
            logger.error(f"Feil ved risikovurdering: {str(e)}")
            raise
            
    def _assess_market_risk(self, property_data: Dict[str, Any]) -> float:
        """Vurder markedsrisiko"""
        try:
            location = property_data.get("location", "Oslo")
            market_data = self.market_data.get(location)
            
            if not market_data:
                return 0.7  # Høy risiko hvis vi mangler markedsdata
            
            risk_score = 0.0
            
            # Vurder markedstrender
            if market_data.market_trend == "increasing":
                risk_score += 0.2
            elif market_data.market_trend == "stable":
                risk_score += 0.4
            else:
                risk_score += 0.6
            
            # Vurder ledighetsrate
            vacancy_risk = market_data.vacancy_rate * 5  # Konverter til risikoscore
            risk_score += min(vacancy_risk, 0.3)
            
            # Vurder prisvolatilitet
            price_volatility = self.risk_factors["market_volatility"]
            risk_score += price_volatility
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Feil ved vurdering av markedsrisiko: {str(e)}")
            return 0.5
            
    def _assess_financial_risk(self, property_data: Dict[str, Any]) -> float:
        """Vurder finansiell risiko"""
        try:
            loan_details = property_data.get("loan_details", {})
            
            if not loan_details:
                return 0.3  # Moderat risiko hvis vi mangler lånedetaljer
            
            risk_score = 0.0
            
            # Vurder belåningsgrad
            ltv_ratio = loan_details.get("loan_to_value", 0.0)
            if ltv_ratio > 0.8:
                risk_score += 0.4
            elif ltv_ratio > 0.6:
                risk_score += 0.2
            else:
                risk_score += 0.1
            
            # Vurder rentesensitivitet
            interest_rate = loan_details.get("interest_rate", 0.0)
            if interest_rate > 0.06:
                risk_score += 0.3
            elif interest_rate > 0.04:
                risk_score += 0.2
            else:
                risk_score += 0.1
            
            # Vurder kontantstrømdekning
            debt_service_coverage = loan_details.get("debt_service_coverage_ratio", 0.0)
            if debt_service_coverage < 1.2:
                risk_score += 0.3
            elif debt_service_coverage < 1.5:
                risk_score += 0.2
            else:
                risk_score += 0.1
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Feil ved vurdering av finansiell risiko: {str(e)}")
            return 0.5
            
    def _assess_property_risk(self, property_data: Dict[str, Any]) -> float:
        """Vurder eiendomsspesifikk risiko"""
        try:
            risk_score = 0.0
            
            # Vurder bygningens alder
            age = property_data.get("age", 50)
            if age > 50:
                risk_score += 0.3
            elif age > 20:
                risk_score += 0.2
            else:
                risk_score += 0.1
            
            # Vurder vedlikeholdsbehov
            condition = property_data.get("condition", "fair")
            condition_scores = {
                "excellent": 0.1,
                "good": 0.2,
                "fair": 0.3,
                "poor": 0.4
            }
            risk_score += condition_scores.get(condition, 0.3)
            
            # Vurder teknisk tilstand
            technical_issues = property_data.get("technical_issues", [])
            risk_score += len(technical_issues) * 0.1
            
            # Vurder miljøfaktorer
            environmental_risks = property_data.get("environmental_risks", [])
            risk_score += len(environmental_risks) * 0.15
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Feil ved vurdering av eiendomsrisiko: {str(e)}")
            return 0.5
            
    def _assess_tenant_risk(self, property_data: Dict[str, Any]) -> float:
        """Vurder leietakerrisiko"""
        try:
            risk_score = 0.0
            
            # Vurder eksisterende leiekontrakter
            leases = property_data.get("leases", [])
            if not leases:
                return 0.7  # Høy risiko hvis ingen leiekontrakter
            
            # Vurder lengde på leiekontrakter
            avg_lease_length = sum(lease.get("remaining_months", 0) for lease in leases) / len(leases)
            if avg_lease_length < 12:
                risk_score += 0.3
            elif avg_lease_length < 36:
                risk_score += 0.2
            else:
                risk_score += 0.1
            
            # Vurder leietakermiks
            tenant_types = set(lease.get("tenant_type", "unknown") for lease in leases)
            if len(tenant_types) == 1:
                risk_score += 0.2  # Høyere risiko ved avhengighet av én type leietaker
            
            # Vurder betalingshistorikk
            payment_issues = sum(1 for lease in leases if lease.get("payment_issues", False))
            risk_score += (payment_issues / len(leases)) * 0.3
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Feil ved vurdering av leietakerrisiko: {str(e)}")
            return 0.5
            
    def _assess_regulatory_risk(self, property_data: Dict[str, Any]) -> float:
        """Vurder regulatorisk risiko"""
        try:
            risk_score = 0.0
            
            # Vurder reguleringsplan
            zoning = property_data.get("zoning", {})
            if not zoning.get("compliant", True):
                risk_score += 0.4
            
            # Vurder byggetillatelser
            permits = property_data.get("permits", [])
            missing_permits = [p for p in permits if not p.get("approved", False)]
            risk_score += len(missing_permits) * 0.2
            
            # Vurder fremtidige reguleringsendringer
            future_changes = property_data.get("future_regulatory_changes", [])
            risk_score += len(future_changes) * 0.15
            
            # Vurder miljøkrav
            environmental_compliance = property_data.get("environmental_compliance", {})
            if not environmental_compliance.get("compliant", True):
                risk_score += 0.3
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Feil ved vurdering av regulatorisk risiko: {str(e)}")
            return 0.5
            
    def _categorize_risk(self, total_risk: float) -> str:
        """Kategoriser total risiko"""
        if total_risk < 0.3:
            return "LAV"
        elif total_risk < 0.6:
            return "MODERAT"
        else:
            return "HØY"
        
    def generate_report(self, analysis: EconomicMetrics) -> Dict[str, Any]:
        """Generer detaljert økonomisk rapport"""
        try:
            figures = self._create_visualization(analysis)
            
            return {
                "summary": self._create_summary(analysis),
                "detailed_analysis": self._create_detailed_analysis(analysis),
                "visualizations": figures,
                "recommendations": self._generate_recommendations(analysis)
            }
        except Exception as e:
            logger.error(f"Feil ved generering av rapport: {str(e)}")
            return {"error": str(e)}
            
    def _create_visualization(self, analysis: EconomicMetrics) -> Dict[str, go.Figure]:
        """Opprett visualiseringer av økonomiske data"""
        try:
            figures = {}
            
            # 1. Kontantstrømanalyse
            cash_flows = self._calculate_monthly_cash_flows(
                analysis.monthly_income,
                analysis.monthly_expenses,
                analysis.initial_investment
            )
            
            fig_cash_flow = go.Figure()
            fig_cash_flow.add_trace(go.Scatter(
                x=list(range(len(cash_flows))),
                y=cash_flows,
                mode='lines',
                name='Netto kontantstrøm',
                line=dict(color='#00b4db', width=2)
            ))
            fig_cash_flow.update_layout(
                title='Månedlig Kontantstrøm',
                xaxis_title='Måned',
                yaxis_title='NOK',
                template='plotly_dark'
            )
            figures['cash_flow'] = fig_cash_flow
            
            # 2. ROI og avkastning
            roi_data = {
                'Direkte Yield': analysis.roi,
                'Internrente (IRR)': analysis.internal_rate_return,
                'Totalavkastning': analysis.roi + 3.0  # Inkluderer verdiøkning
            }
            
            fig_roi = go.Figure(data=[
                go.Bar(
                    x=list(roi_data.keys()),
                    y=list(roi_data.values()),
                    marker_color=['#00b4db', '#0083b0', '#006c8f']
                )
            ])
            fig_roi.update_layout(
                title='Avkastningsanalyse',
                yaxis_title='Prosent (%)',
                template='plotly_dark'
            )
            figures['roi'] = fig_roi
            
            # 3. Kostnadsfordeling
            expense_breakdown = self._calculate_expense_breakdown(analysis.monthly_expenses)
            
            fig_expenses = go.Figure(data=[go.Pie(
                labels=list(expense_breakdown.keys()),
                values=list(expense_breakdown.values()),
                hole=.3,
                marker_colors=['#00b4db', '#0083b0', '#006c8f', '#004c6d', '#002f4a']
            )])
            fig_expenses.update_layout(
                title='Kostnadsfordeling',
                template='plotly_dark'
            )
            figures['expenses'] = fig_expenses
            
            # 4. Følsomhetsanalyse
            sensitivity_data = self._perform_sensitivity_analysis(analysis)
            
            fig_sensitivity = go.Figure()
            for var, values in sensitivity_data.items():
                fig_sensitivity.add_trace(go.Scatter(
                    x=list(range(-20, 21, 10)),
                    y=values,
                    mode='lines+markers',
                    name=var
                ))
            fig_sensitivity.update_layout(
                title='Følsomhetsanalyse',
                xaxis_title='Endring (%)',
                yaxis_title='Effekt på ROI (%)',
                template='plotly_dark'
            )
            figures['sensitivity'] = fig_sensitivity
            
            return figures
            
        except Exception as e:
            logger.error(f"Feil ved opprettelse av visualiseringer: {str(e)}")
            return {}
            
    def _calculate_monthly_cash_flows(self, income: float, expenses: float, investment: float,
                                  months: int = 60) -> List[float]:
        """Beregn månedlige kontantstrømmer"""
        cash_flows = []
        monthly_net = income - expenses
        
        for month in range(months):
            # Juster for inflasjon og vekst
            growth_factor = 1 + (0.02 * month / 12)  # 2% årlig vekst
            adjusted_net = monthly_net * growth_factor
            cash_flows.append(adjusted_net)
            
        return cash_flows
        
    def _calculate_expense_breakdown(self, monthly_expenses: float) -> Dict[str, float]:
        """Beregn fordeling av utgifter"""
        return {
            "Låne-/finanskostnader": monthly_expenses * 0.45,
            "Vedlikehold": monthly_expenses * 0.20,
            "Kommunale avgifter": monthly_expenses * 0.15,
            "Forsikring": monthly_expenses * 0.10,
            "Andre kostnader": monthly_expenses * 0.10
        }
        
    def _perform_sensitivity_analysis(self, analysis: EconomicMetrics) -> Dict[str, List[float]]:
        """Utfør følsomhetsanalyse"""
        variables = {
            "Leieinntekter": [],
            "Driftskostnader": [],
            "Rente": [],
            "Vedlikehold": []
        }
        
        base_roi = analysis.roi
        
        # Test effekten av endringer fra -20% til +20%
        for percent in range(-20, 21, 10):
            factor = 1 + (percent / 100)
            
            # Leieinntekter
            adjusted_income = analysis.monthly_income * factor
            variables["Leieinntekter"].append(
                self._calculate_roi(adjusted_income, analysis.monthly_expenses, analysis.initial_investment)
            )
            
            # Driftskostnader
            adjusted_expenses = analysis.monthly_expenses * factor
            variables["Driftskostnader"].append(
                self._calculate_roi(analysis.monthly_income, adjusted_expenses, analysis.initial_investment)
            )
            
            # Rente (påvirker månedlige kostnader)
            adjusted_interest = analysis.monthly_expenses * (1 + (percent / 200))  # Halv effekt på totale kostnader
            variables["Rente"].append(
                self._calculate_roi(analysis.monthly_income, adjusted_interest, analysis.initial_investment)
            )
            
            # Vedlikehold (påvirker månedlige kostnader)
            adjusted_maintenance = analysis.monthly_expenses * (1 + (percent / 300))  # En tredjedels effekt
            variables["Vedlikehold"].append(
                self._calculate_roi(analysis.monthly_income, adjusted_maintenance, analysis.initial_investment)
            )
            
        return variables
        
    def _create_summary(self, analysis: EconomicMetrics) -> Dict[str, Any]:
        """Lag sammendrag av analysen"""
        try:
            summary = {
                "investeringsoversikt": {
                    "total_investering": f"{analysis.initial_investment:,.2f} NOK",
                    "månedlig_inntekt": f"{analysis.monthly_income:,.2f} NOK",
                    "månedlig_utgift": f"{analysis.monthly_expenses:,.2f} NOK",
                    "netto_månedlig": f"{(analysis.monthly_income - analysis.monthly_expenses):,.2f} NOK"
                },
                "nøkkeltall": {
                    "direkte_yield": f"{analysis.roi:.2f}%",
                    "internrente": f"{analysis.internal_rate_return:.2f}%",
                    "tilbakebetalingstid": f"{analysis.payback_period:.1f} år",
                    "nåverdi": f"{analysis.net_present_value:,.2f} NOK"
                },
                "risikovurdering": {
                    "total_risiko": f"{sum(analysis.risk_assessment.values()) / len(analysis.risk_assessment):.2f}",
                    "risikokategori": self._categorize_risk(
                        sum(analysis.risk_assessment.values()) / len(analysis.risk_assessment)
                    ),
                    "største_risikofaktorer": self._identify_top_risks(analysis.risk_assessment)
                }
            }
            
            # Legg til årlige estimater
            summary["årlig_estimat"] = {
                "brutto_leieinntekt": f"{(analysis.monthly_income * 12):,.2f} NOK",
                "driftskostnader": f"{(analysis.monthly_expenses * 12):,.2f} NOK",
                "netto_kontantstrøm": f"{((analysis.monthly_income - analysis.monthly_expenses) * 12):,.2f} NOK"
            }
            
            # Beregn nøkkelindikatorer
            summary["indikatorer"] = {
                "kostnad_per_kvm": self._calculate_cost_per_sqm(analysis),
                "driftsmargin": self._calculate_operating_margin(analysis),
                "gjeldsgrad": self._calculate_debt_ratio(analysis)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Feil ved opprettelse av sammendrag: {str(e)}")
            return {}
            
    def _identify_top_risks(self, risk_assessment: Dict[str, float], top_n: int = 3) -> List[str]:
        """Identifiser de største risikofaktorene"""
        sorted_risks = sorted(
            risk_assessment.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [risk[0] for risk in sorted_risks[:top_n]]
        
    def _calculate_cost_per_sqm(self, analysis: EconomicMetrics) -> str:
        """Beregn kostnad per kvadratmeter"""
        try:
            # Antar at property_data inneholder area
            area = getattr(analysis, 'area', 100)  # Default 100 kvm hvis ikke spesifisert
            cost_per_sqm = analysis.initial_investment / area
            return f"{cost_per_sqm:,.2f} NOK/m²"
        except Exception:
            return "Ikke tilgjengelig"
            
    def _calculate_operating_margin(self, analysis: EconomicMetrics) -> str:
        """Beregn driftsmargin"""
        try:
            annual_income = analysis.monthly_income * 12
            annual_expenses = analysis.monthly_expenses * 12
            margin = ((annual_income - annual_expenses) / annual_income) * 100
            return f"{margin:.1f}%"
        except Exception:
            return "Ikke tilgjengelig"
            
    def _calculate_debt_ratio(self, analysis: EconomicMetrics) -> str:
        """Beregn gjeldsgrad"""
        try:
            # Antar at property_data inneholder loan_amount
            loan_amount = getattr(analysis, 'loan_amount', 0)
            if loan_amount and analysis.initial_investment:
                debt_ratio = (loan_amount / analysis.initial_investment) * 100
                return f"{debt_ratio:.1f}%"
            return "Ikke tilgjengelig"
        except Exception:
            return "Ikke tilgjengelig"
        

            
    def _analyze_rental_income(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser leieinntekter"""
        try:
            return {
                "månedlig": f"{analysis.monthly_income:,.2f} NOK",
                "årlig": f"{analysis.monthly_income * 12:,.2f} NOK",
                "per_kvm": f"{(analysis.monthly_income * 12 / getattr(analysis, 'area', 100)):,.2f} NOK/m²"
            }
        except Exception:
            return {"error": "Kunne ikke analysere leieinntekter"}
            
    def _analyze_other_income(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser andre inntekter"""
        try:
            return {
                "parkering": "Ikke tilgjengelig",
                "fellesarealer": "Ikke tilgjengelig",
                "annet": "Ikke tilgjengelig"
            }
        except Exception:
            return {"error": "Kunne ikke analysere andre inntekter"}
            
    def _analyze_fixed_costs(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser faste kostnader"""
        try:
            return {
                "felleskostnader": f"{analysis.monthly_expenses * 0.3:,.2f} NOK",
                "forsikring": f"{analysis.monthly_expenses * 0.1:,.2f} NOK",
                "eiendomsskatt": f"{analysis.monthly_expenses * 0.05:,.2f} NOK"
            }
        except Exception:
            return {"error": "Kunne ikke analysere faste kostnader"}
            
    def _analyze_variable_costs(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser variable kostnader"""
        try:
            return {
                "strøm": f"{analysis.monthly_expenses * 0.15:,.2f} NOK",
                "renovasjon": f"{analysis.monthly_expenses * 0.05:,.2f} NOK",
                "andre": f"{analysis.monthly_expenses * 0.05:,.2f} NOK"
            }
        except Exception:
            return {"error": "Kunne ikke analysere variable kostnader"}
            
    def _analyze_maintenance_costs(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser vedlikeholdskostnader"""
        try:
            return {
                "løpende": f"{analysis.monthly_expenses * 0.2:,.2f} NOK",
                "periodisk": f"{analysis.monthly_expenses * 0.1:,.2f} NOK",
                "akutt": f"{analysis.monthly_expenses * 0.05:,.2f} NOK"
            }
        except Exception:
            return {"error": "Kunne ikke analysere vedlikeholdskostnader"}
            
    def _calculate_loan_amount(self, analysis: EconomicMetrics) -> str:
        """Beregn lånebeløp"""
        try:
            loan_amount = getattr(analysis, 'loan_amount', analysis.initial_investment * 0.75)
            return f"{loan_amount:,.2f} NOK"
        except Exception:
            return "Ikke tilgjengelig"
            
    def _get_current_interest_rate(self) -> str:
        """Hent gjeldende rente"""
        try:
            return "4.5%"  # Dette bør hentes fra en ekstern kilde
        except Exception:
            return "Ikke tilgjengelig"
            
    def _calculate_monthly_payment(self, analysis: EconomicMetrics) -> str:
        """Beregn månedlig lånebeløp"""
        try:
            loan_amount = getattr(analysis, 'loan_amount', analysis.initial_investment * 0.75)
            interest_rate = 0.045 / 12  # Månedlig rente
            term = 30 * 12  # Antall måneder
            
            payment = loan_amount * (interest_rate * (1 + interest_rate)**term) / ((1 + interest_rate)**term - 1)
            return f"{payment:,.2f} NOK"
        except Exception:
            return "Ikke tilgjengelig"
            
    def _calculate_equity(self, analysis: EconomicMetrics) -> str:
        """Beregn egenkapital"""
        try:
            loan_amount = getattr(analysis, 'loan_amount', analysis.initial_investment * 0.75)
            equity = analysis.initial_investment - loan_amount
            return f"{equity:,.2f} NOK"
        except Exception:
            return "Ikke tilgjengelig"
            
    def _calculate_equity_percentage(self, analysis: EconomicMetrics) -> str:
        """Beregn egenkapitalprosent"""
        try:
            loan_amount = getattr(analysis, 'loan_amount', analysis.initial_investment * 0.75)
            equity = analysis.initial_investment - loan_amount
            equity_percentage = (equity / analysis.initial_investment) * 100
            return f"{equity_percentage:.1f}%"
        except Exception:
            return "Ikke tilgjengelig"
            
    def _calculate_interest_deduction(self, analysis: EconomicMetrics) -> str:
        """Beregn rentefradrag"""
        try:
            loan_amount = getattr(analysis, 'loan_amount', analysis.initial_investment * 0.75)
            annual_interest = loan_amount * 0.045  # 4.5% rente
            tax_deduction = annual_interest * 0.22  # 22% skattefradrag
            return f"{tax_deduction:,.2f} NOK"
        except Exception:
            return "Ikke tilgjengelig"
            
    def _calculate_depreciation(self, analysis: EconomicMetrics) -> str:
        """Beregn avskrivninger"""
        try:
            building_value = analysis.initial_investment * 0.8  # Antar 80% er bygning
            annual_depreciation = building_value * 0.02  # 2% årlig avskrivning
            return f"{annual_depreciation:,.2f} NOK"
        except Exception:
            return "Ikke tilgjengelig"
            
    def _calculate_best_case(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Beregn beste scenario"""
        try:
            return {
                "årlig_avkastning": f"{analysis.roi * 1.2:.1f}%",
                "verdiøkning": f"{analysis.initial_investment * 1.25:,.2f} NOK",
                "netto_yield": f"{analysis.roi * 1.1:.1f}%"
            }
        except Exception:
            return {"error": "Kunne ikke beregne beste scenario"}
            
    def _calculate_worst_case(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Beregn verste scenario"""
        try:
            return {
                "årlig_avkastning": f"{analysis.roi * 0.8:.1f}%",
                "verditap": f"{analysis.initial_investment * 0.9:,.2f} NOK",
                "netto_yield": f"{analysis.roi * 0.85:.1f}%"
            }
        except Exception:
            return {"error": "Kunne ikke beregne verste scenario"}
            
    def _calculate_break_even(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Beregn break-even punkt"""
        try:
            monthly_payment = float(self._calculate_monthly_payment(analysis).replace(" NOK", "").replace(",", ""))
            min_rent = monthly_payment / 0.7  # Antar 70% av leie går til å dekke lån
            return {
                "minimum_leie": f"{min_rent:,.2f} NOK",
                "belegg": "85%",
                "drift_kostnader": f"{analysis.monthly_expenses * 0.9:,.2f} NOK"
            }
        except Exception:
            return {"error": "Kunne ikke beregne break-even"}
            
    def _analyze_interest_sensitivity(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser rentefølsomhet"""
        try:
            loan_amount = getattr(analysis, 'loan_amount', analysis.initial_investment * 0.75)
            current_payment = float(self._calculate_monthly_payment(analysis).replace(" NOK", "").replace(",", ""))
            
            # Test 2% økning
            interest_rate_high = 0.065 / 12  # 6.5% årlig
            term = 30 * 12
            payment_high = loan_amount * (interest_rate_high * (1 + interest_rate_high)**term) / ((1 + interest_rate_high)**term - 1)
            
            return {
                "nåværende": f"{current_payment:,.2f} NOK",
                "ved_2%_økning": f"{payment_high:,.2f} NOK",
                "forskjell": f"{payment_high - current_payment:,.2f} NOK"
            }
        except Exception:
            return {"error": "Kunne ikke analysere rentefølsomhet"}
            
    def _analyze_occupancy_sensitivity(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser følsomhet for utleiegrad"""
        try:
            return {
                "break_even_belegg": "85%",
                "kritisk_belegg": "75%",
                "anbefalt_buffer": "10%"
            }
        except Exception:
            return {"error": "Kunne ikke analysere følsomhet for utleiegrad"}
            
    def _analyze_maintenance_sensitivity(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser vedlikeholdsfølsomhet"""
        try:
            current_maintenance = analysis.monthly_expenses * 0.3
            return {
                "nåværende": f"{current_maintenance:,.2f} NOK",
                "ved_20%_økning": f"{current_maintenance * 1.2:,.2f} NOK",
                "anbefalt_buffer": f"{current_maintenance * 0.2:,.2f} NOK"
            }
        except Exception:
            return {"error": "Kunne ikke analysere vedlikeholdsfølsomhet"}
            
    def _compare_with_market(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Sammenlign med markedet"""
        try:
            return {
                "leienivå_vs_marked": "Markedsnivå",
                "yield_vs_marked": "Over markedssnitt",
                "kostnader_vs_marked": "Under markedssnitt"
            }
        except Exception:
            return {"error": "Kunne ikke sammenligne med marked"}
            
    def _identify_competitive_advantages(self, analysis: EconomicMetrics) -> List[str]:
        """Identifiser konkurransefortrinn"""
        try:
            return [
                "Optimal beliggenhet",
                "Effektiv planløsning",
                "Moderne fasiliteter"
            ]
        except Exception:
            return ["Kunne ikke identifisere konkurransefortrinn"]
            
    def _analyze_market_trends(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Analyser markedstrender"""
        try:
            return {
                "leieprisutvikling": "Stigende",
                "etterspørsel": "Høy",
                "markedsutsikter": "Positive"
            }
        except Exception:
            return {"error": "Kunne ikke analysere markedstrender"}
            
    def _estimate_value_growth(self, analysis: EconomicMetrics) -> Dict[str, str]:
        """Estimer verdiøkning"""
        try:
            return {
                "1_år": f"{analysis.initial_investment * 1.03:,.2f} NOK",
                "3_år": f"{analysis.initial_investment * 1.093:,.2f} NOK",
                "5_år": f"{analysis.initial_investment * 1.159:,.2f} NOK"
            }
        except Exception:
            return {"error": "Kunne ikke estimere verdiøkning"}
            
    def _identify_development_opportunities(self, analysis: EconomicMetrics) -> List[str]:
        """Identifiser utviklingsmuligheter"""
        try:
            return [
                "Potensial for arealeffektivisering",
                "Mulighet for påbygg",
                "Energioppgraderinger"
            ]
        except Exception:
            return ["Kunne ikke identifisere utviklingsmuligheter"]
            
    def _identify_future_risks(self, analysis: EconomicMetrics) -> List[str]:
        """Identifiser fremtidige risikoer"""
        try:
            return [
                "Endringer i regulering",
                "Økte vedlikeholdskostnader",
                "Markedssvingninger"
            ]
        except Exception:
            return ["Kunne ikke identifisere fremtidige risikoer"]
            
    def _recommend_future_actions(self, analysis: EconomicMetrics) -> List[str]:
        """Anbefal fremtidige tiltak"""
        try:
            return [
                "Utarbeid langsiktig vedlikeholdsplan",
                "Vurder energieffektiviseringstiltak",
                "Følg med på reguleringsendringer"
            ]
        except Exception:
            return ["Kunne ikke generere anbefalinger for fremtidige tiltak"]
            
    def _generate_recommendations(self, analysis: EconomicMetrics) -> List[str]:
        """Generer anbefalinger basert på analysen"""
        try:
            recommendations = []
            
            # Vurder ROI
            if analysis.roi < 4:
                recommendations.append("ROI er lav. Vurder reforhandling av leiekontrakter eller kostnadsreduserende tiltak.")
            
            # Vurder kontantstrøm
            monthly_net = analysis.monthly_income - analysis.monthly_expenses
            if monthly_net < 0:
                recommendations.append("Negativ kontantstrøm. Umiddelbare tiltak kreves for å øke inntekter eller redusere kostnader.")
            elif monthly_net < analysis.monthly_income * 0.2:
                recommendations.append("Lav driftsmargin. Vurder potensial for kostnadsoptimalisering.")
            
            # Vurder finansiering
            loan_amount = getattr(analysis, 'loan_amount', 0)
            if loan_amount / analysis.initial_investment > 0.75:
                recommendations.append("Høy belåningsgrad. Vurder refinansiering eller økning av egenkapital.")
            
            # Vurder risikofaktorer
            high_risks = [risk for risk, score in analysis.risk_assessment.items() if score > 0.7]
            for risk in high_risks:
                recommendations.append(f"Høy {risk}. Implementer risikoreduserende tiltak.")
            
            # Vurder markedspotensial
            market_value = float(self._calculate_market_value(analysis).replace(" NOK", "").replace(",", ""))
            if market_value > analysis.initial_investment * 1.2:
                recommendations.append("Betydelig verdipotensial. Vurder verdiøkende tiltak eller refinansiering.")
            
            # Vurder vedlikehold
            if getattr(analysis, 'maintenance_costs', 0) > analysis.monthly_income * 0.3:
                recommendations.append("Høye vedlikeholdskostnader. Vurder preventive tiltak og langsiktig vedlikeholdsplan.")
            
            # Legg til generelle anbefalinger
            recommendations.extend([
                "Gjennomfør regelmessige markedsanalyser for å sikre konkurransedyktige leiepriser.",
                "Etabler buffer for uforutsette utgifter.",
                "Vurder energieffektiviserende tiltak for å redusere driftskostnader."
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Feil ved generering av anbefalinger: {str(e)}")
            return ["Kunne ikke generere anbefalinger på grunn av manglende data"]