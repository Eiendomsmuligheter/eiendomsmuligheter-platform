class RegulatoryChecker {
    constructor() {
        this.apiHandler = new APIHandler();
        this.municipalityData = null;
        this.zoneRegulations = null;
        this.buildingCodes = null;
        this.propertyData = null;
    }

    async initialize(municipalityCode) {
        try {
            // Load regulatory data for the municipality
            this.municipalityData = await this.apiHandler.getMunicipalityData(municipalityCode);
            this.zoneRegulations = await this.apiHandler.getZoneRegulations(municipalityCode);
            this.buildingCodes = await this.apiHandler.getBuildingCodes();
        } catch (error) {
            console.error('Failed to initialize regulatory checker:', error);
            throw error;
        }
    }

    async checkCompliance(propertyData) {
        this.propertyData = propertyData;

        const complianceReport = {
            zoning: await this.checkZoningCompliance(),
            buildingCodes: await this.checkBuildingCodeCompliance(),
            parking: await this.checkParkingRequirements(),
            accessibility: await this.checkAccessibilityRequirements(),
            fire: await this.checkFireSafetyRequirements(),
            environmental: await this.checkEnvironmentalRequirements(),
            overall: {
                compliant: true,
                violations: [],
                recommendations: []
            }
        };

        // Calculate overall compliance
        complianceReport.overall = this.calculateOverallCompliance(complianceReport);

        return complianceReport;
    }

    async checkZoningCompliance() {
        const zoneData = await this.getZoneData(this.propertyData.location);
        const compliance = {
            compliant: true,
            violations: [],
            allowedUses: zoneData.allowedUses,
            restrictions: [],
            recommendations: []
        };

        // Check building height
        if (this.propertyData.height > zoneData.maxHeight) {
            compliance.compliant = false;
            compliance.violations.push({
                type: 'height',
                current: this.propertyData.height,
                maximum: zoneData.maxHeight,
                description: 'Byggehøyde overskrider maksimal tillatt høyde'
            });
        }

        // Check plot ratio (utnyttelsesgrad)
        const plotRatio = this.calculatePlotRatio();
        if (plotRatio > zoneData.maxPlotRatio) {
            compliance.compliant = false;
            compliance.violations.push({
                type: 'plotRatio',
                current: plotRatio,
                maximum: zoneData.maxPlotRatio,
                description: 'Utnyttelsesgrad overskrider maksimal tillatt grad'
            });
        }

        // Check setback requirements
        const setbackViolations = this.checkSetbackRequirements(zoneData.setbackRules);
        if (setbackViolations.length > 0) {
            compliance.compliant = false;
            compliance.violations.push(...setbackViolations);
        }

        // Check permitted use
        if (!this.isPermittedUse(this.propertyData.intendedUse, zoneData.allowedUses)) {
            compliance.compliant = false;
            compliance.violations.push({
                type: 'use',
                current: this.propertyData.intendedUse,
                allowed: zoneData.allowedUses,
                description: 'Planlagt bruk er ikke tillatt i denne sonen'
            });
        }

        // Add zone-specific restrictions
        compliance.restrictions = this.getZoneRestrictions(zoneData);

        // Generate recommendations
        compliance.recommendations = this.generateZoningRecommendations(compliance.violations);

        return compliance;
    }

    async checkBuildingCodeCompliance() {
        const compliance = {
            compliant: true,
            violations: [],
            requirements: {},
            recommendations: []
        };

        // Check structural requirements
        const structuralCompliance = await this.checkStructuralRequirements();
        if (!structuralCompliance.compliant) {
            compliance.compliant = false;
            compliance.violations.push(...structuralCompliance.violations);
        }

        // Check safety requirements
        const safetyCompliance = await this.checkSafetyRequirements();
        if (!safetyCompliance.compliant) {
            compliance.compliant = false;
            compliance.violations.push(...safetyCompliance.violations);
        }

        // Check technical requirements
        const technicalCompliance = await this.checkTechnicalRequirements();
        if (!technicalCompliance.compliant) {
            compliance.compliant = false;
            compliance.violations.push(...technicalCompliance.violations);
        }

        // Store all applicable requirements
        compliance.requirements = {
            structural: structuralCompliance.requirements,
            safety: safetyCompliance.requirements,
            technical: technicalCompliance.requirements
        };

        // Generate recommendations
        compliance.recommendations = this.generateBuildingCodeRecommendations(compliance.violations);

        return compliance;
    }

    async checkParkingRequirements() {
        const parkingRules = await this.getParkingRules();
        const compliance = {
            compliant: true,
            violations: [],
            requirements: parkingRules,
            currentSpaces: this.propertyData.parkingSpaces,
            recommendations: []
        };

        // Calculate required parking spaces
        const requiredSpaces = this.calculateRequiredParkingSpaces(parkingRules);
        compliance.requiredSpaces = requiredSpaces;

        // Check if current spaces meet requirements
        if (this.propertyData.parkingSpaces < requiredSpaces.total) {
            compliance.compliant = false;
            compliance.violations.push({
                type: 'insufficientParking',
                current: this.propertyData.parkingSpaces,
                required: requiredSpaces.total,
                description: 'Utilstrekkelig antall parkeringsplasser'
            });
        }

        // Check accessibility requirements for parking
        if (this.propertyData.handicapParkingSpaces < requiredSpaces.handicap) {
            compliance.compliant = false;
            compliance.violations.push({
                type: 'insufficientHandicapParking',
                current: this.propertyData.handicapParkingSpaces,
                required: requiredSpaces.handicap,
                description: 'Utilstrekkelig antall HC-parkeringsplasser'
            });
        }

        // Check electric vehicle charging requirements
        if (this.propertyData.evChargingSpaces < requiredSpaces.evCharging) {
            compliance.compliant = false;
            compliance.violations.push({
                type: 'insufficientEVCharging',
                current: this.propertyData.evChargingSpaces,
                required: requiredSpaces.evCharging,
                description: 'Utilstrekkelig antall ladeplasser for elbil'
            });
        }

        // Generate recommendations
        compliance.recommendations = this.generateParkingRecommendations(compliance.violations);

        return compliance;
    }

    async checkAccessibilityRequirements() {
        const accessibilityRules = await this.getAccessibilityRules();
        const compliance = {
            compliant: true,
            violations: [],
            requirements: accessibilityRules,
            recommendations: []
        };

        // Check entrance accessibility
        if (!this.checkEntranceAccessibility(accessibilityRules)) {
            compliance.compliant = false;
            compliance.violations.push({
                type: 'entrance',
                description: 'Inngangsparti oppfyller ikke tilgjengelighetskrav'
            });
        }

        // Check door widths
        const doorViolations = this.checkDoorWidths(accessibilityRules);
        if (doorViolations.length > 0) {
            compliance.compliant = false;
            compliance.violations.push(...doorViolations);
        }

        // Check ramps and elevators
        const verticalAccessViolations = this.checkVerticalAccess(accessibilityRules);
        if (verticalAccessViolations.length > 0) {
            compliance.compliant = false;
            compliance.violations.push(...verticalAccessViolations);
        }

        // Check bathroom accessibility
        if (!this.checkBathroomAccessibility(accessibilityRules)) {
            compliance.compliant = false;
            compliance.violations.push({
                type: 'bathroom',
                description: 'Bad/toalett oppfyller ikke tilgjengelighetskrav'
            });
        }

        // Generate recommendations
        compliance.recommendations = this.generateAccessibilityRecommendations(compliance.violations);

        return compliance;
    }

    async checkFireSafetyRequirements() {
        const fireSafetyRules = await this.getFireSafetyRules();
        const compliance = {
            compliant: true,
            violations: [],
            requirements: fireSafetyRules,
            recommendations: []
        };

        // Check fire resistance ratings
        const fireResistanceViolations = this.checkFireResistance(fireSafetyRules);
        if (fireResistanceViolations.length > 0) {
            compliance.compliant = false;
            compliance.violations.push(...fireResistanceViolations);
        }

        // Check escape routes
        const escapeRouteViolations = this.checkEscapeRoutes(fireSafetyRules);
        if (escapeRouteViolations.length > 0) {
            compliance.compliant = false;
            compliance.violations.push(...escapeRouteViolations);
        }

        // Check fire detection and suppression systems
        const systemViolations = this.checkFireSystems(fireSafetyRules);
        if (systemViolations.length > 0) {
            compliance.compliant = false;
            compliance.violations.push(...systemViolations);
        }

        // Generate recommendations
        compliance.recommendations = this.generateFireSafetyRecommendations(compliance.violations);

        return compliance;
    }

    calculateOverallCompliance(complianceReport) {
        const overall = {
            compliant: true,
            violations: [],
            recommendations: [],
            riskLevel: 'low'
        };

        // Collect all violations
        Object.entries(complianceReport).forEach(([category, report]) => {
            if (category !== 'overall' && !report.compliant) {
                overall.compliant = false;
                report.violations.forEach(violation => {
                    overall.violations.push({
                        category,
                        ...violation
                    });
                });
            }
        });

        // Assess risk level
        if (overall.violations.length > 0) {
            const criticalViolations = overall.violations.filter(v => 
                v.category === 'fire' || v.category === 'structural');
            
            if (criticalViolations.length > 0) {
                overall.riskLevel = 'high';
            } else if (overall.violations.length > 5) {
                overall.riskLevel = 'medium';
            }
        }

        // Generate comprehensive recommendations
        overall.recommendations = this.generateOverallRecommendations(overall.violations);

        return overall;
    }

    // Helper methods
    calculatePlotRatio() {
        const totalFloorArea = this.propertyData.floors.reduce((sum, floor) => sum + floor.area, 0);
        return totalFloorArea / this.propertyData.plotArea;
    }

    isPermittedUse(intendedUse, allowedUses) {
        return allowedUses.some(use => 
            use.toLowerCase() === intendedUse.toLowerCase() ||
            use.toLowerCase().includes(intendedUse.toLowerCase())
        );
    }

    checkSetbackRequirements(rules) {
        const violations = [];
        const boundaries = this.propertyData.boundaries;

        rules.forEach(rule => {
            const actualSetback = this.calculateMinimumSetback(boundaries, rule.direction);
            if (actualSetback < rule.minimum) {
                violations.push({
                    type: 'setback',
                    direction: rule.direction,
                    current: actualSetback,
                    required: rule.minimum,
                    description: `Utilstrekkelig avstand til ${rule.direction}`
                });
            }
        });

        return violations;
    }

    generateZoningRecommendations(violations) {
        return violations.map(violation => {
            switch (violation.type) {
                case 'height':
                    return {
                        priority: 'high',
                        description: 'Reduser byggehøyde eller søk dispensasjon',
                        details: `Nåværende høyde: ${violation.current}m, Maksimal tillatt: ${violation.maximum}m`
                    };
                case 'plotRatio':
                    return {
                        priority: 'high',
                        description: 'Reduser bygningsvolum eller søk dispensasjon',
                        details: `Nåværende utnyttelsesgrad: ${violation.current}, Maksimal tillatt: ${violation.maximum}`
                    };
                case 'setback':
                    return {
                        priority: 'medium',
                        description: `Øk avstand til ${violation.direction}`,
                        details: `Nåværende avstand: ${violation.current}m, Minimum påkrevd: ${violation.required}m`
                    };
                default:
                    return {
                        priority: 'medium',
                        description: 'Adresser reguleringsavvik',
                        details: violation.description
                    };
            }
        });
    }

    generateOverallRecommendations(violations) {
        const recommendations = [];
        const violationsByCategory = this.groupViolationsByCategory(violations);

        // Generate prioritized recommendations
        Object.entries(violationsByCategory).forEach(([category, categoryViolations]) => {
            const categoryRec = this.generateCategoryRecommendations(category, categoryViolations);
            recommendations.push(...categoryRec);
        });

        // Sort recommendations by priority
        return this.prioritizeRecommendations(recommendations);
    }

    groupViolationsByCategory(violations) {
        return violations.reduce((groups, violation) => {
            if (!groups[violation.category]) {
                groups[violation.category] = [];
            }
            groups[violation.category].push(violation);
            return groups;
        }, {});
    }

    prioritizeRecommendations(recommendations) {
        const priorityOrder = {
            'critical': 0,
            'high': 1,
            'medium': 2,
            'low': 3
        };

        return recommendations.sort((a, b) => 
            priorityOrder[a.priority] - priorityOrder[b.priority]
        );
    }
}