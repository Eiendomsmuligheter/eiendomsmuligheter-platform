class APIHandler {
    constructor() {
        this.baseUrl = '/api';
        this.endpoints = {
            property: '/property',
            regulatory: '/regulatory',
            analysis: '/analysis',
            floorplan: '/floorplan',
            municipality: '/municipality'
        };
    }

    async getPropertyDataByAddress(address) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.property}/address`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ address })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching property data:', error);
            throw error;
        }
    }

    async getPropertyDataByUrl(url) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.property}/url`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching property data from URL:', error);
            throw error;
        }
    }

    async processFloorPlanImage(imageData) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.floorplan}/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ imageData })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error processing floorplan:', error);
            throw error;
        }
    }

    async getRegulatoryData(location) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.regulatory}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ location })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching regulatory data:', error);
            throw error;
        }
    }

    async getMunicipalityData(municipalityCode) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.municipality}/${municipalityCode}`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching municipality data:', error);
            throw error;
        }
    }

    async getZoneRegulations(municipalityCode) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.regulatory}/zone/${municipalityCode}`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching zone regulations:', error);
            throw error;
        }
    }

    async getBuildingCodes() {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.regulatory}/building-codes`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching building codes:', error);
            throw error;
        }
    }

    async analyzeProperty(propertyData) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.analysis}/property`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(propertyData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error analyzing property:', error);
            throw error;
        }
    }

    async getPropertyValue(propertyData) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.analysis}/value`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(propertyData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting property value:', error);
            throw error;
        }
    }

    async getMarketAnalysis(location) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.analysis}/market`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ location })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting market analysis:', error);
            throw error;
        }
    }

    async submitPropertyReport(reportData) {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.analysis}/report`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(reportData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error submitting property report:', error);
            throw error;
        }
    }

    // Error handling helper
    handleError(error, customMessage = 'An error occurred') {
        console.error(customMessage, error);
        throw new Error(`${customMessage}: ${error.message}`);
    }
}