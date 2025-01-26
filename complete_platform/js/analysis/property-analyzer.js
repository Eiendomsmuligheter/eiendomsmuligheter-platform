class PropertyAnalyzer {
    constructor() {
        this.propertyData = null;
        this.floorPlans = new Map();
        this.regulatoryData = null;
        this.analysisResults = null;
        
        // Initialize API handlers
        this.apiHandler = new APIHandler();
    }

    async analyzeProperty(input, type = 'address') {
        try {
            // Show loading overlay
            this.showLoading('Analyserer eiendom...');

            // Get property data based on input type
            switch (type) {
                case 'address':
                    this.propertyData = await this.apiHandler.getPropertyDataByAddress(input);
                    break;
                case 'url':
                    this.propertyData = await this.apiHandler.getPropertyDataByUrl(input);
                    break;
                case 'floorplan':
                    this.propertyData = await this.processFloorPlan(input);
                    break;
                default:
                    throw new Error('Invalid input type');
            }

            // Get regulatory data
            this.regulatoryData = await this.apiHandler.getRegulatoryData(this.propertyData.location);

            // Perform comprehensive analysis
            this.analysisResults = await this.performAnalysis();

            // Hide loading overlay
            this.hideLoading();

            // Return results
            return this.analysisResults;

        } catch (error) {
            this.hideLoading();
            console.error('Analysis error:', error);
            throw error;
        }
    }

    async processFloorPlan(fileInput) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = async (e) => {
                try {
                    // Process image data
                    const imageData = e.target.result;
                    
                    // Send to API for processing
                    const processedData = await this.apiHandler.processFloorPlanImage(imageData);
                    
                    resolve(processedData);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(fileInput);
        });
    }

    async performAnalysis() {
        const analysis = {
            propertyDetails: await this.analyzePropertyDetails(),
            spatialAnalysis: await this.analyzeSpatialData(),
            regulatoryCompliance: await this.analyzeRegulatoryCompliance(),
            developmentPotential: await this.analyzeDevelopmentPotential(),
            marketAnalysis: await this.analyzeMarketConditions(),
            sustainabilityScore: await this.analyzeSustainability(),
            recommendations: []
        };

        // Generate recommendations based on analysis
        analysis.recommendations = this.generateRecommendations(analysis);

        return analysis;
    }

    async analyzePropertyDetails() {
        return {
            propertyType: this.propertyData.type,
            totalArea: this.calculateTotalArea(),
            livingArea: this.calculateLivingArea(),
            floors: this.analyzeFloors(),
            rooms: this.analyzeRooms(),
            buildYear: this.propertyData.buildYear,
            condition: await this.assessCondition(),
            technicalStatus: await this.analyzeTechnicalStatus()
        };
    }

    async analyzeSpatialData() {
        return {
            floorPlanAnalysis: this.analyzeFloorPlans(),
            spaceUtilization: this.analyzeSpaceUtilization(),
            accessibilityScore: this.analyzeAccessibility(),
            naturalLight: this.analyzeNaturalLight(),
            ventilation: this.analyzeVentilation(),
            acoustics: this.analyzeAcoustics()
        };
    }

    async analyzeRegulatoryCompliance() {
        return {
            zoning: this.checkZoningCompliance(),
            buildingCodes: this.checkBuildingCodeCompliance(),
            permitRequirements: this.analyzePermitRequirements(),
            restrictions: this.analyzeRestrictions(),
            parkingRequirements: this.analyzeParkingRequirements()
        };
    }

    async analyzeDevelopmentPotential() {
        return {
            expansionPotential: this.analyzeExpansionPotential(),
            renovationPotential: this.analyzeRenovationPotential(),
            landUtilization: this.analyzeLandUtilization(),
            valueAddOpportunities: this.identifyValueAddOpportunities()
        };
    }

    async analyzeMarketConditions() {
        return {
            marketValue: await this.estimateMarketValue(),
            rentalPotential: await this.analyzeRentalPotential(),
            marketTrends: await this.analyzeMarketTrends(),
            demographicAnalysis: await this.analyzeDemographics(),
            competitiveAnalysis: await this.analyzeCompetition()
        };
    }

    async analyzeSustainability() {
        return {
            energyEfficiency: this.analyzeEnergyEfficiency(),
            environmentalImpact: this.analyzeEnvironmentalImpact(),
            sustainableMaterials: this.analyzeMaterials(),
            greenCertifications: this.analyzeGreenCertifications()
        };
    }

    // Utility methods for calculations and analysis
    calculateTotalArea() {
        let totalArea = 0;
        this.floorPlans.forEach(floor => {
            totalArea += this.calculateFloorArea(floor);
        });
        return totalArea;
    }

    calculateLivingArea() {
        let livingArea = 0;
        this.floorPlans.forEach(floor => {
            livingArea += this.calculateHabitableArea(floor);
        });
        return livingArea;
    }

    analyzeFloors() {
        const floorAnalysis = {};
        this.floorPlans.forEach((floor, level) => {
            floorAnalysis[level] = {
                area: this.calculateFloorArea(floor),
                rooms: this.countRooms(floor),
                windows: this.countWindows(floor),
                doors: this.countDoors(floor),
                ceiling_height: this.measureCeilingHeight(floor)
            };
        });
        return floorAnalysis;
    }

    analyzeRooms() {
        const rooms = [];
        this.floorPlans.forEach(floor => {
            floor.rooms.forEach(room => {
                rooms.push({
                    name: room.name,
                    area: room.area,
                    windows: this.countRoomWindows(room),
                    doors: this.countRoomDoors(room),
                    natural_light: this.assessNaturalLight(room)
                });
            });
        });
        return rooms;
    }

    async assessCondition() {
        // Analyze various aspects of the property's condition
        const structuralCondition = await this.analyzeStructuralCondition();
        const surfaceCondition = this.analyzeSurfaceCondition();
        const technicalCondition = await this.analyzeTechnicalSystems();

        return {
            overall: this.calculateOverallCondition(structuralCondition, surfaceCondition, technicalCondition),
            structural: structuralCondition,
            surface: surfaceCondition,
            technical: technicalCondition
        };
    }

    generateRecommendations(analysis) {
        const recommendations = [];

        // Analyze development potential
        if (analysis.developmentPotential.expansionPotential.score > 0.7) {
            recommendations.push({
                type: 'development',
                priority: 'high',
                description: 'Betydelig potensial for utvidelse av eiendommen',
                details: analysis.developmentPotential.expansionPotential.details
            });
        }

        // Analyze regulatory compliance
        const nonCompliantItems = this.findNonCompliantItems(analysis.regulatoryCompliance);
        if (nonCompliantItems.length > 0) {
            recommendations.push({
                type: 'regulatory',
                priority: 'critical',
                description: 'Regulatoriske avvik som må addresseres',
                items: nonCompliantItems
            });
        }

        // Analyze sustainability
        if (analysis.sustainabilityScore.energyEfficiency.rating < 'C') {
            recommendations.push({
                type: 'sustainability',
                priority: 'medium',
                description: 'Forbedringspotensial for energieffektivitet',
                suggestions: this.generateEnergySuggestions(analysis.sustainabilityScore)
            });
        }

        // Analyze market conditions
        if (analysis.marketAnalysis.rentalPotential.score > 0.8) {
            recommendations.push({
                type: 'market',
                priority: 'high',
                description: 'Sterkt potensial for utleie',
                details: analysis.marketAnalysis.rentalPotential.details
            });
        }

        return recommendations;
    }

    findNonCompliantItems(compliance) {
        const nonCompliant = [];
        
        // Check zoning compliance
        if (!compliance.zoning.compliant) {
            nonCompliant.push({
                category: 'Regulering',
                issue: compliance.zoning.issues,
                solution: compliance.zoning.remediation
            });
        }

        // Check building codes
        compliance.buildingCodes.violations.forEach(violation => {
            nonCompliant.push({
                category: 'Byggekrav',
                issue: violation.description,
                solution: violation.remediation
            });
        });

        // Check parking requirements
        if (!compliance.parkingRequirements.compliant) {
            nonCompliant.push({
                category: 'Parkering',
                issue: compliance.parkingRequirements.deficit,
                solution: compliance.parkingRequirements.suggestions
            });
        }

        return nonCompliant;
    }

    generateEnergySuggestions(sustainabilityScore) {
        const suggestions = [];
        
        // Analyze insulation
        if (sustainabilityScore.insulation.rating < 'B') {
            suggestions.push({
                component: 'Isolasjon',
                current: sustainabilityScore.insulation.current,
                target: sustainabilityScore.insulation.recommended,
                roi: sustainabilityScore.insulation.roi
            });
        }

        // Analyze heating system
        if (sustainabilityScore.heating.efficiency < 0.8) {
            suggestions.push({
                component: 'Varmesystem',
                current: sustainabilityScore.heating.type,
                recommended: sustainabilityScore.heating.recommendation,
                savings: sustainabilityScore.heating.potentialSavings
            });
        }

        // Analyze windows
        if (sustainabilityScore.windows.rating < 'B') {
            suggestions.push({
                component: 'Vinduer',
                current: sustainabilityScore.windows.type,
                recommended: sustainabilityScore.windows.recommendation,
                roi: sustainabilityScore.windows.roi
            });
        }

        return suggestions;
    }

    showLoading(message) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingText = loadingOverlay.querySelector('.loading-text');
        
        if (loadingText) {
            loadingText.textContent = message;
        }
        
        loadingOverlay.classList.add('active');
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        loadingOverlay.classList.remove('active');
    }
}