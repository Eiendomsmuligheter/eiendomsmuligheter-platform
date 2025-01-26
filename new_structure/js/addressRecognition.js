class AddressRecognition {
    constructor() {
        this.kartverketAPI = "https://ws.geonorge.no/adresser/v1";
        this.kommuneAPI = "https://api.kommune.no/api/v1";
        this.cache = new Map();
    }

    async searchAddress(query) {
        if (query.length < 3) return [];
        if (this.cache.has(query)) return this.cache.get(query);

        try {
            const response = await fetch(`${this.kartverketAPI}/sok?sok=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            const suggestions = data.adresser.map(addr => ({
                address: addr.adressetekst,
                postalCode: addr.postnummer,
                municipality: addr.kommunenavn,
                coordinates: {
                    lat: addr.representasjonspunkt.lat,
                    lon: addr.representasjonspunkt.lon
                },
                propertyInfo: {
                    gnr: addr.gardsnummer,
                    bnr: addr.bruksnummer
                }
            }));

            this.cache.set(query, suggestions);
            return suggestions;
        } catch (error) {
            console.error('Address search failed:', error);
            return [];
        }
    }

    async getMunicipalityRegulations(municipalityName) {
        try {
            const response = await fetch(`${this.kommuneAPI}/regulations/${municipalityName}`);
            const data = await response.json();
            
            return {
                zoning: data.zoning,
                buildingRegulations: data.buildingRegulations,
                parkingRequirements: data.parkingRequirements,
                specialAreas: data.specialAreas,
                contactInfo: data.contactInfo
            };
        } catch (error) {
            console.error('Municipality regulations fetch failed:', error);
            return null;
        }
    }

    async getPropertyDetails(gnr, bnr, municipality) {
        try {
            const response = await fetch(
                `${this.kommuneAPI}/property/${municipality}/${gnr}/${bnr}`
            );
            const data = await response.json();
            
            return {
                area: data.area,
                buildYear: data.buildYear,
                propertyType: data.propertyType,
                zoningPlan: data.zoningPlan,
                restrictions: data.restrictions,
                previousApplications: data.previousApplications
            };
        } catch (error) {
            console.error('Property details fetch failed:', error);
            return null;
        }
    }

    async validateAddress(address) {
        try {
            const response = await fetch(
                `${this.kartverketAPI}/validate?adresse=${encodeURIComponent(address)}`
            );
            return await response.json();
        } catch (error) {
            console.error('Address validation failed:', error);
            return { valid: false, error: error.message };
        }
    }
}