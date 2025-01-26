// Main Application Logic
class EiendomsAI {
    constructor() {
        this.propertyAnalyzer = new PropertyAnalyzer();
        this.regulatoryChecker = new RegulatoryChecker();
        this.formGenerator = new FormGenerator();
        this.visualizer = new Visualizer3D();
        this.currentAnalysis = null;
    }

    async initialize() {
        this.setupEventListeners();
        this.initializeUI();
        await this.loadMunicipalityData();
    }

    setupEventListeners() {
        // Address Search
        document.getElementById('addressSearch').addEventListener('input', 
            debounce(this.handleAddressSearch.bind(this), 300));

        // File Upload
        document.getElementById('fileUpload').addEventListener('change', 
            this.handleFileUpload.bind(this));

        // Analysis Start
        document.getElementById('startAnalysis').addEventListener('click', 
            this.startAnalysis.bind(this));
    }

    async handleAddressSearch(event) {
        const address = event.target.value;
        if (address.length < 3) return;

        try {
            const suggestions = await this.getAddressSuggestions(address);
            this.displayAddressSuggestions(suggestions);
        } catch (error) {
            console.error('Address search failed:', error);
        }
    }

    async getAddressSuggestions(address) {
        // Implementer API-kall til kartverket/kommune
        return [];
    }

    async startAnalysis() {
        try {
            this.showLoadingState('Starter analyse...');
            
            // Hent adresse og plantegning
            const address = document.getElementById('addressSearch').value;
            const floorPlan = await this.getFloorPlan();
            
            // Kjør analysene
            const municipalityRules = await this.regulatoryChecker.checkMunicipalRegulations(address);
            const analysis = await this.propertyAnalyzer.analyzeProperty(address, floorPlan);
            
            // Generer 3D-modell
            const model = await this.visualizer.generate3DModel(floorPlan, analysis.recommendations);
            
            // Generer skjemaer
            const forms = await this.formGenerator.generateForms(analysis, municipalityRules);
            
            // Vis resultater
            this.displayResults(analysis, model, forms);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError('Kunne ikke fullføre analysen. Vennligst prøv igjen.');
        } finally {
            this.hideLoadingState();
        }
    }

    async displayResults(analysis, model, forms) {
        // Vis 3D-modell
        this.visualizer.displayModel(model, 'visualizationContainer');
        
        // Vis analyserapport
        this.displayAnalysisReport(analysis);
        
        // Vis skjemaer
        this.displayForms(forms);
        
        // Aktiver nedlastingsknapper
        this.enableDownloadButtons(analysis, model, forms);
    }

    displayAnalysisReport(analysis) {
        const container = document.getElementById('analysisResults');
        container.innerHTML = `
            <div class="analysis-section glass-effect">
                <h3>Mulige Utleieenheter</h3>
                ${this.generateUnitsSummary(analysis.possibilities)}
            </div>
            
            <div class="analysis-section glass-effect">
                <h3>Tekniske Krav</h3>
                ${this.generateRequirementsList(analysis.requirements)}
            </div>
            
            <div class="analysis-section glass-effect">
                <h3>Kostnadsestimater</h3>
                ${this.generateCostBreakdown(analysis.costs)}
            </div>
            
            <div class="analysis-section glass-effect">
                <h3>Neste Steg</h3>
                ${this.generateNextSteps(analysis.recommendations)}
            </div>
        `;
    }

    generateUnitsSummary(possibilities) {
        return possibilities.map(unit => `
            <div class="unit-card">
                <h4>${unit.type}</h4>
                <p>Areal: ${unit.area}m²</p>
                <p>Antall rom: ${unit.rooms}</p>
                <p>Estimert leieinntekt: ${unit.estimatedRent} kr/mnd</p>
            </div>
        `).join('');
    }

    generateRequirementsList(requirements) {
        return Object.entries(requirements).map(([category, details]) => `
            <div class="requirement-category">
                <h4>${category}</h4>
                <ul>
                    ${Object.entries(details).map(([req, status]) => `
                        <li class="requirement-item ${status.met ? 'met' : 'not-met'}">
                            ${req}: ${status.description}
                            ${status.met ? '✓' : '⚠'}
                        </li>
                    `).join('')}
                </ul>
            </div>
        `).join('');
    }

    generateCostBreakdown(costs) {
        return `
            <div class="costs-summary">
                <div class="total-cost">
                    <h4>Total estimert kostnad</h4>
                    <p class="cost-amount">${costs.total.toLocaleString()} kr</p>
                </div>
                <div class="cost-breakdown">
                    ${Object.entries(costs.breakdown).map(([category, amount]) => `
                        <div class="cost-item">
                            <span>${category}</span>
                            <span>${amount.toLocaleString()} kr</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    generateNextSteps(recommendations) {
        return `
            <div class="steps-list">
                ${recommendations.map((step, index) => `
                    <div class="step-item">
                        <div class="step-number">${index + 1}</div>
                        <div class="step-content">
                            <h5>${step.title}</h5>
                            <p>${step.description}</p>
                            ${step.action ? `
                                <button onclick="handleStepAction('${step.action}')" 
                                        class="btn btn-primary btn-sm">
                                    ${step.actionText}
                                </button>
                            ` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Hjelpefunksjoner
    showLoadingState(message) {
        const loader = document.createElement('div');
        loader.className = 'loading-overlay';
        loader.innerHTML = `
            <div class="loading-content">
                <div class="spinner"></div>
                <p>${message}</p>
            </div>
        `;
        document.body.appendChild(loader);
    }

    hideLoadingState() {
        const loader = document.querySelector('.loading-overlay');
        if (loader) {
            loader.remove();
        }
    }

    showError(message) {
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger animate__animated animate__fadeIn';
        alert.textContent = message;
        document.getElementById('alertContainer').appendChild(alert);
        
        setTimeout(() => {
            alert.classList.add('animate__fadeOut');
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    }
}

// Helper Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    const app = new EiendomsAI();
    app.initialize();
});