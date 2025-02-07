const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');
const Chart = require('chart.js');
const { createCanvas } = require('canvas');

class ReportGenerator {
    constructor() {
        this.doc = null;
        this.currentPage = 1;
        this.margins = {
            top: 50,
            bottom: 50,
            left: 50,
            right: 50
        };
    }

    async generateReport(analysisData, outputPath) {
        this.doc = new PDFDocument({
            autoFirstPage: true,
            size: 'A4',
            margins: this.margins
        });

        // Opprett output stream
        const stream = fs.createWriteStream(outputPath);
        this.doc.pipe(stream);

        // Generer rapport
        await this.generateCoverPage(analysisData);
        this.generateTableOfContents(analysisData);
        await this.generateExecutiveSummary(analysisData);
        await this.generatePropertyAnalysis(analysisData);
        await this.generateFinancialAnalysis(analysisData);
        await this.generateRegulatoryAnalysis(analysisData);
        await this.generateRecommendations(analysisData);
        this.generateAppendices(analysisData);

        // Finaliser dokument
        this.doc.end();

        return new Promise((resolve, reject) => {
            stream.on('finish', () => resolve(outputPath));
            stream.on('error', reject);
        });
    }

    async generateCoverPage(analysisData) {
        // Logo og header
        this.doc.image('path/to/logo.png', 50, 50, { width: 150 });
        
        this.doc.fontSize(24)
            .text('Eiendomsanalyse Rapport', {
                align: 'center',
                margin: 50
            });

        // Eiendomsinfo
        this.doc.fontSize(16)
            .text(`${analysisData.address}`, {
                align: 'center',
                margin: 20
            });

        // Dato og referanse
        this.doc.fontSize(12)
            .text(`Generert: ${new Date().toLocaleDateString('nb-NO')}`, {
                align: 'center'
            })
            .text(`Referanse: ${analysisData.id}`, {
                align: 'center'
            });

        this.doc.addPage();
    }

    generateTableOfContents(analysisData) {
        this.doc.fontSize(16)
            .text('Innholdsfortegnelse', {
                underline: true
            });

        const sections = [
            'Sammendrag',
            'Eiendomsanalyse',
            'Økonomisk Analyse',
            'Regulatorisk Analyse',
            'Anbefalinger',
            'Vedlegg'
        ];

        let y = 100;
        sections.forEach((section, index) => {
            this.doc.fontSize(12)
                .text(section, 50, y)
                .text((index + 2).toString(), 500, y);
            y += 20;
        });

        this.doc.addPage();
    }

    async generateExecutiveSummary(analysisData) {
        this.doc.fontSize(16)
            .text('Sammendrag', {
                underline: true
            });

        const summary = this.generateSummaryText(analysisData);
        this.doc.fontSize(12)
            .text(summary, {
                align: 'justify'
            });

        // Nøkkeltall
        await this.addKeyMetrics(analysisData);

        this.doc.addPage();
    }

    async generatePropertyAnalysis(analysisData) {
        this.doc.fontSize(16)
            .text('Eiendomsanalyse', {
                underline: true
            });

        // Egenskapsdetaljer
        this.addPropertyDetails(analysisData);

        // Tilstandsvurdering
        await this.addConditionAssessment(analysisData);

        // Utviklingsmuligheter
        this.addDevelopmentOpportunities(analysisData);

        this.doc.addPage();
    }

    async generateFinancialAnalysis(analysisData) {
        this.doc.fontSize(16)
            .text('Økonomisk Analyse', {
                underline: true
            });

        // ROI-beregninger
        await this.addROICalculations(analysisData);

        // Kostnadsestimater
        await this.addCostEstimates(analysisData);

        // Markedsanalyse
        await this.addMarketAnalysis(analysisData);

        this.doc.addPage();
    }

    async generateRegulatoryAnalysis(analysisData) {
        this.doc.fontSize(16)
            .text('Regulatorisk Analyse', {
                underline: true
            });

        // Reguleringsstatus
        this.addZoningInformation(analysisData);

        // Byggetillatelser
        this.addPermitRequirements(analysisData);

        // Restriksjoner
        this.addRestrictions(analysisData);

        this.doc.addPage();
    }

    async generateRecommendations(analysisData) {
        this.doc.fontSize(16)
            .text('Anbefalinger', {
                underline: true
            });

        // Hovedanbefalinger
        this.addMainRecommendations(analysisData);

        // Prioritert tiltaksliste
        await this.addActionPlan(analysisData);

        // Risikovurdering
        await this.addRiskAssessment(analysisData);

        this.doc.addPage();
    }

    generateAppendices(analysisData) {
        this.doc.fontSize(16)
            .text('Vedlegg', {
                underline: true
            });

        // Tekniske spesifikasjoner
        this.addTechnicalSpecifications(analysisData);

        // Relevante dokumenter
        this.addRelevantDocuments(analysisData);

        // Metodikk og kilder
        this.addMethodologyAndSources();
    }

    // Hjelpemetoder for grafgenerering
    async generateChart(data, type, options) {
        const canvas = createCanvas(600, 400);
        const ctx = canvas.getContext('2d');
        
        new Chart(ctx, {
            type: type,
            data: data,
            options: options
        });

        return canvas;
    }

    // Hjelpemetoder for tekstgenerering
    generateSummaryText(analysisData) {
        return `
            Basert på vår omfattende analyse av eiendommen ${analysisData.address}, 
            har vi identifisert flere lovende utviklingsmuligheter. Hovedfunnene inkluderer:

            ${this.formatKeyFindings(analysisData.possibilities)}

            Den estimerte totalverdien av disse forbedringene er ${
                new Intl.NumberFormat('nb-NO', { 
                    style: 'currency', 
                    currency: 'NOK' 
                }).format(analysisData.aiAnalysis.potentialValue)
            }.
        `;
    }

    formatKeyFindings(possibilities) {
        return possibilities
            .map(p => `- ${p.type}: ${p.description}`)
            .join('\n');
    }

    // Diverse hjelpemetoder for spesifikke seksjoner
    async addKeyMetrics(analysisData) {
        // Implementer nøkkeltall-visualisering
    }

    addPropertyDetails(analysisData) {
        // Implementer eiendomsdetaljer
    }

    async addConditionAssessment(analysisData) {
        // Implementer tilstandsvurdering
    }

    addDevelopmentOpportunities(analysisData) {
        // Implementer utviklingsmuligheter
    }

    async addROICalculations(analysisData) {
        // Implementer ROI-beregninger
    }

    async addCostEstimates(analysisData) {
        // Implementer kostnadsestimater
    }

    async addMarketAnalysis(analysisData) {
        // Implementer markedsanalyse
    }

    addZoningInformation(analysisData) {
        // Implementer reguleringsinformasjon
    }

    addPermitRequirements(analysisData) {
        // Implementer tillatelseskrav
    }

    addRestrictions(analysisData) {
        // Implementer restriksjoner
    }

    addMainRecommendations(analysisData) {
        // Implementer hovedanbefalinger
    }

    async addActionPlan(analysisData) {
        // Implementer tiltaksplan
    }

    async addRiskAssessment(analysisData) {
        // Implementer risikovurdering
    }

    addTechnicalSpecifications(analysisData) {
        // Implementer tekniske spesifikasjoner
    }

    addRelevantDocuments(analysisData) {
        // Implementer relevante dokumenter
    }

    addMethodologyAndSources() {
        // Implementer metodikk og kilder
    }
}

module.exports = new ReportGenerator();