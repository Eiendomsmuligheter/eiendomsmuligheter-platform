const axios = require('axios');
const { parseString } = require('xml2js');
const { promisify } = require('util');
const parseXMLAsync = promisify(parseString);

class MunicipalityService {
    constructor() {
        this.baseUrls = {
            planRegistry: process.env.PLAN_REGISTRY_API_URL,
            buildingRegistry: process.env.BUILDING_REGISTRY_API_URL,
            mapData: process.env.MAP_DATA_API_URL,
            propertyRegistry: process.env.PROPERTY_REGISTRY_API_URL
        };
        
        this.apiKeys = {
            planRegistry: process.env.PLAN_REGISTRY_API_KEY,
            buildingRegistry: process.env.BUILDING_REGISTRY_API_KEY,
            mapData: process.env.MAP_DATA_API_KEY,
            propertyRegistry: process.env.PROPERTY_REGISTRY_API_KEY
        };
    }

    async getPropertyInformation(address) {
        try {
            const [
                propertyData,
                planData,
                buildingData,
                restrictionsData
            ] = await Promise.all([
                this.getPropertyData(address),
                this.getPlanData(address),
                this.getBuildingData(address),
                this.getRestrictions(address)
            ]);

            return {
                property: propertyData,
                planning: planData,
                building: buildingData,
                restrictions: restrictionsData,
                summary: this.generateSummary({
                    propertyData,
                    planData,
                    buildingData,
                    restrictionsData
                })
            };
        } catch (error) {
            console.error('Feil ved henting av eiendomsinformasjon:', error);
            throw new Error('Kunne ikke hente komplett eiendomsinformasjon');
        }
    }

    async getPropertyData(address) {
        try {
            const response = await axios.get(
                `${this.baseUrls.propertyRegistry}/property/search`,
                {
                    params: { address },
                    headers: { 'X-API-Key': this.apiKeys.propertyRegistry }
                }
            );

            return this.processPropertyData(response.data);
        } catch (error) {
            console.error('Feil ved henting av eiendomsdata:', error);
            return null;
        }
    }

    async getPlanData(address) {
        try {
            const response = await axios.get(
                `${this.baseUrls.planRegistry}/plans`,
                {
                    params: { address },
                    headers: { 'X-API-Key': this.apiKeys.planRegistry }
                }
            );

            return this.processPlanData(response.data);
        } catch (error) {
            console.error('Feil ved henting av plandata:', error);
            return null;
        }
    }

    async getBuildingData(address) {
        try {
            const response = await axios.get(
                `${this.baseUrls.buildingRegistry}/buildings`,
                {
                    params: { address },
                    headers: { 'X-API-Key': this.apiKeys.buildingRegistry }
                }
            );

            return this.processBuildingData(response.data);
        } catch (error) {
            console.error('Feil ved henting av bygningsdata:', error);
            return null;
        }
    }

    async getRestrictions(address) {
        try {
            const response = await axios.get(
                `${this.baseUrls.planRegistry}/restrictions`,
                {
                    params: { address },
                    headers: { 'X-API-Key': this.apiKeys.planRegistry }
                }
            );

            return this.processRestrictions(response.data);
        } catch (error) {
            console.error('Feil ved henting av restriksjoner:', error);
            return null;
        }
    }

    async getZoningRegulations(municipalityCode, zoneId) {
        try {
            const response = await axios.get(
                `${this.baseUrls.planRegistry}/regulations`,
                {
                    params: { 
                        municipalityCode,
                        zoneId
                    },
                    headers: { 'X-API-Key': this.apiKeys.planRegistry }
                }
            );

            return this.processRegulations(response.data);
        } catch (error) {
            console.error('Feil ved henting av reguleringsbestemmelser:', error);
            return null;
        }
    }

    async getPropertyBoundaries(cadastralNumber) {
        try {
            const response = await axios.get(
                `${this.baseUrls.mapData}/boundaries`,
                {
                    params: { cadastralNumber },
                    headers: { 'X-API-Key': this.apiKeys.mapData }
                }
            );

            return this.processBoundaryData(response.data);
        } catch (error) {
            console.error('Feil ved henting av eiendomsgrenser:', error);
            return null;
        }
    }

    // Data prosessering metoder
    processPropertyData(data) {
        return {
            cadastralNumber: data.cadastralNumber,
            area: data.area,
            ownership: data.ownership,
            buildingTypes: data.buildingTypes,
            lastUpdated: data.lastUpdated
        };
    }

    processPlanData(data) {
        return {
            zoning: data.zoning,
            planId: data.planId,
            planName: data.planName,
            planStatus: data.planStatus,
            allowedUses: data.allowedUses,
            restrictions: data.restrictions
        };
    }

    processBuildingData(data) {
        return {
            buildingId: data.buildingId,
            buildingType: data.buildingType,
            constructionYear: data.constructionYear,
            totalArea: data.totalArea,
            floors: data.floors,
            technical: {
                energyRating: data.energyRating,
                foundation: data.foundation,
                constructionMethod: data.constructionMethod
            }
        };
    }

    processRestrictions(data) {
        return {
            cultural: data.culturalHeritage || [],
            environmental: data.environmental || [],
            infrastructure: data.infrastructure || [],
            other: data.other || []
        };
    }

    processRegulations(data) {
        return {
            buildingHeight: data.buildingHeight,
            floorAreaRatio: data.floorAreaRatio,
            coverage: data.coverage,
            setbacks: data.setbacks,
            parkingRequirements: data.parkingRequirements
        };
    }

    processBoundaryData(data) {
        return {
            type: 'FeatureCollection',
            features: [
                {
                    type: 'Feature',
                    geometry: data.geometry,
                    properties: data.properties
                }
            ]
        };
    }

    // Sammendrag og analyse
    generateSummary(data) {
        const summary = {
            developmentPotential: this.assessDevelopmentPotential(data),
            restrictions: this.summarizeRestrictions(data.restrictions),
            recommendations: this.generateRecommendations(data)
        };

        return summary;
    }

    assessDevelopmentPotential(data) {
        // Vurder utviklingspotensial basert på all tilgjengelig data
        const potential = {
            buildingExpansion: this.calculateExpansionPotential(data),
            propertyDivision: this.assessDivisionPossibility(data),
            useChange: this.evaluateUseChangePotential(data)
        };

        return potential;
    }

    calculateExpansionPotential(data) {
        // Beregn potensial for utbygging
        return {
            possibleExpansion: true,
            maxAdditionalArea: 100,
            restrictions: []
        };
    }

    assessDivisionPossibility(data) {
        // Vurder mulighet for deling av eiendom
        return {
            possible: true,
            minimumSize: 600,
            requirements: []
        };
    }

    evaluateUseChangePotential(data) {
        // Vurder potensial for bruksendring
        return {
            possible: true,
            allowedUses: [],
            requirements: []
        };
    }

    summarizeRestrictions(restrictions) {
        // Oppsummer alle relevante restriksjoner
        return {
            critical: [],
            significant: [],
            minor: []
        };
    }

    generateRecommendations(data) {
        // Generer anbefalinger basert på all data
        return {
            immediate: [],
            shortTerm: [],
            longTerm: []
        };
    }
}

module.exports = new MunicipalityService();