const { MaterialDatabase } = require('../database/materialDb');
const { LaborCostCalculator } = require('../utils/laborCalculator');
const { MarketDataAnalyzer } = require('../utils/marketAnalyzer');
const { RiskAnalyzer } = require('../utils/riskAnalyzer');

class AdvancedCostCalculatorService {
    constructor() {
        this.materialDb = new MaterialDatabase();
        this.laborCalculator = new LaborCostCalculator();
        this.marketAnalyzer = new MarketDataAnalyzer();
        this.riskAnalyzer = new RiskAnalyzer();
    }

    async calculateDetailedCosts(propertyData, projectSpecs) {
        try {
            // Hent all nødvendig kostnadsinformasjon
            const [
                materialCosts,
                laborCosts,
                equipmentCosts,
                permitCosts,
                marketFactors,
                riskAssessment
            ] = await Promise.all([
                this.calculateMaterialCosts(projectSpecs),
                this.calculateLaborCosts(projectSpecs),
                this.calculateEquipmentCosts(projectSpecs),
                this.calculatePermitCosts(propertyData, projectSpecs),
                this.analyzeMarketFactors(propertyData.location),
                this.analyzeProjectRisks(projectSpecs)
            ]);

            // Beregn totale kostnader med alle faktorer
            const totalCost = await this.computeTotalCost({
                materialCosts,
                laborCosts,
                equipmentCosts,
                permitCosts,
                marketFactors,
                riskAssessment
            });

            return {
                overview: this.generateCostOverview(totalCost),
                breakdown: this.generateDetailedBreakdown(totalCost),
                timeline: this.generateCostTimeline(totalCost),
                risks: this.assessCostRisks(totalCost, riskAssessment),
                optimizations: this.suggestCostOptimizations(totalCost),
                financing: await this.analyzeFinancingOptions(totalCost)
            };
        } catch (error) {
            console.error('Feil i kostnadsberegning:', error);
            throw new Error('Kunne ikke fullføre kostnadsberegning');
        }
    }

    async calculateMaterialCosts(projectSpecs) {
        const materials = await this.getMaterialRequirements(projectSpecs);
        const costs = {};
        let totalMaterialCost = 0;

        for (const material of materials) {
            const cost = await this.calculateIndividualMaterialCost(material);
            costs[material.type] = cost;
            totalMaterialCost += cost.total;
        }

        return {
            total: totalMaterialCost,
            breakdown: costs,
            wastageAllowance: this.calculateWastageAllowance(materials),
            qualityUpgrades: this.calculateQualityUpgrades(materials),
            bulkDiscounts: this.calculateBulkDiscounts(materials)
        };
    }

    async calculateLaborCosts(projectSpecs) {
        const laborNeeds = await this.analyzeLaborRequirements(projectSpecs);
        
        return {
            skilled: this.calculateSkilledLabor(laborNeeds),
            unskilled: this.calculateUnskilledLabor(laborNeeds),
            specialist: this.calculateSpecialistLabor(laborNeeds),
            supervision: this.calculateSupervisionCosts(laborNeeds),
            overtime: this.estimateOvertimeCosts(laborNeeds),
            contractors: await this.estimateContractorCosts(laborNeeds)
        };
    }

    async analyzeProjectRisks(projectSpecs) {
        const risks = await this.riskAnalyzer.analyzeProject(projectSpecs);
        
        return {
            identification: this.identifyRisks(risks),
            quantification: this.quantifyRisks(risks),
            mitigation: this.developMitigationStrategies(risks),
            contingency: this.calculateRiskContingency(risks),
            insurance: this.recommendInsuranceCoverage(risks)
        };
    }

    async computeTotalCost(costs) {
        // Beregn grunnkostnad
        const baseCost = this.calculateBaseCost(costs);

        // Juster for markedsfaktorer
        const marketAdjusted = this.applyMarketFactors(baseCost, costs.marketFactors);

        // Legg til risikopåslag
        const riskAdjusted = this.applyRiskFactors(marketAdjusted, costs.riskAssessment);

        // Beregn usikkerhetsmargin
        const uncertainty = this.calculateUncertainty(riskAdjusted, costs);

        return {
            baseCost,
            marketAdjusted,
            riskAdjusted,
            uncertainty,
            total: riskAdjusted * (1 + uncertainty)
        };
    }

    generateCostOverview(totalCost) {
        return {
            summary: this.createCostSummary(totalCost),
            keyMetrics: this.calculateKeyMetrics(totalCost),
            comparisons: this.generateCostComparisons(totalCost),
            scenarios: this.generateCostScenarios(totalCost)
        };
    }

    generateDetailedBreakdown(totalCost) {
        return {
            byCategory: this.breakdownByCategory(totalCost),
            byPhase: this.breakdownByPhase(totalCost),
            byRiskLevel: this.breakdownByRiskLevel(totalCost),
            byTimeframe: this.breakdownByTimeframe(totalCost)
        };
    }

    async analyzeFinancingOptions(totalCost) {
        return {
            loanOptions: await this.analyzeLoanOptions(totalCost),
            paymentSchedule: this.createPaymentSchedule(totalCost),
            cashFlowAnalysis: this.analyzeCashFlow(totalCost),
            investmentReturn: this.calculateROI(totalCost),
            fundingSources: this.identifyFundingSources(totalCost)
        };
    }

    suggestCostOptimizations(totalCost) {
        return {
            materialAlternatives: this.findMaterialAlternatives(totalCost),
            laborOptimizations: this.optimizeLabor(totalCost),
            timelineAdjustments: this.optimizeTimeline(totalCost),
            bulkPurchasing: this.analyzeBulkPurchasing(totalCost),
            contractorNegotiations: this.suggestNegotiationPoints(totalCost)
        };
    }

    // Implementer alle nødvendige hjelpemetoder...
    async getMaterialRequirements(projectSpecs) {
        // Implementer material krav analyse
    }

    async calculateIndividualMaterialCost(material) {
        // Implementer individuell material kostnadsberegning
    }

    calculateWastageAllowance(materials) {
        // Implementer kapp beregning
    }

    // ... flere hjelpemetoder
}

module.exports = new AdvancedCostCalculatorService();