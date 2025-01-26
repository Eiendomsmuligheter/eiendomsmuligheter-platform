class ReportGenerator {
    constructor() {
        this.emojis = {
            success: '✅',
            warning: '⚠️',
            error: '❌',
            info: 'ℹ️',
            money: '💰',
            house: '🏠',
            tools: '🛠️',
            document: '📄',
            light: '💡',
            fire: '🔥',
            door: '🚪',
            window: '🪟',
            ruler: '📏',
            check: '✔️',
            cross: '❌',
            warning: '⚠️',
            air: '💨',
            sound: '🔊',
            temp: '🌡️',
            calendar: '📅',
            clock: '🕒'
        };
    }

    generateUserFriendlyReport(analysis) {
        const sections = [
            this.generateSummarySection(analysis),
            this.generateRequirementsSection(analysis),
            this.generateCostsSection(analysis),
            this.generateTimelineSection(analysis)
        ];

        return sections.join('\n\n');
    }

    generateSummarySection(analysis) {
        return `
        ${this.emojis.house} BOLIGANALYSE - SAMMENDRAG
        ═══════════════════════════════

        ${this.emojis.info} Adresse: ${analysis.address}
        ${this.emojis.ruler} Totalt areal: ${analysis.totalArea}m²
        ${this.emojis.money} Estimert utleiepotensial: ${analysis.estimatedRent} kr/mnd

        UTLEIEMULIGHETER:
        ${analysis.rentalUnits.map(unit => `
        ${this.emojis.door} Enhet ${unit.number}:
        • Areal: ${unit.area}m²
        • Rom: ${unit.rooms}
        • Estimert leie: ${unit.rent} kr/mnd
        `).join('\n')}
        `;
    }

    generateRequirementsSection(analysis) {
        return `
        ${this.emojis.document} TEKNISKE KRAV
        ═══════════════════

        ${this.emojis.fire} BRANNSIKKERHET
        ${this.getStatusIcon(analysis.requirements.fire.status)}
        • Rømningsveier: ${this.getStatusIcon(analysis.requirements.fire.escape)}
        • Brannskiller: ${this.getStatusIcon(analysis.requirements.fire.barriers)}
        • Brannvarsling: ${this.getStatusIcon(analysis.requirements.fire.alarms)}

        ${this.emojis.window} LYS OG VENTILASJON
        ${this.getStatusIcon(analysis.requirements.ventilation.status)}
        • Dagslys: ${this.getStatusIcon(analysis.requirements.ventilation.daylight)}
        • Luftkvalitet: ${this.getStatusIcon(analysis.requirements.ventilation.airQuality)}
        • Vindusstørrelse: ${this.getStatusIcon(analysis.requirements.ventilation.windowSize)}

        ${this.emojis.ruler} ROMKRAV
        ${this.getStatusIcon(analysis.requirements.rooms.status)}
        • Takhøyde: ${this.getStatusIcon(analysis.requirements.rooms.height)}
        • Romareal: ${this.getStatusIcon(analysis.requirements.rooms.area)}
        • Adkomst: ${this.getStatusIcon(analysis.requirements.rooms.access)}

        ${this.emojis.sound} LYDKRAV
        ${this.getStatusIcon(analysis.requirements.sound.status)}
        • Vegger: ${this.getStatusIcon(analysis.requirements.sound.walls)}
        • Etasjeskiller: ${this.getStatusIcon(analysis.requirements.sound.floors)}
        `;
    }

    generateCostsSection(analysis) {
        return `
        ${this.emojis.money} KOSTNADSESTIMAT
        ═══════════════════

        TOTALT: ${analysis.costs.total.toLocaleString()} kr

        Fordeling:
        ${Object.entries(analysis.costs.breakdown).map(([category, cost]) => 
            `${this.emojis.tools} ${category}: ${cost.toLocaleString()} kr`
        ).join('\n')}

        ${this.emojis.light} INNTEKTSPOTENSIAL
        • Månedlig: ${analysis.income.monthly.toLocaleString()} kr
        • Årlig: ${analysis.income.yearly.toLocaleString()} kr
        • ROI: ${analysis.income.roi} måneder
        `;
    }

    generateTimelineSection(analysis) {
        return `
        ${this.emojis.calendar} TIDSLINJE
        ═══════════════

        ${analysis.timeline.map((step, index) => `
        ${this.emojis.clock} Steg ${index + 1}: ${step.title}
        • Varighet: ${step.duration}
        • Kostnad: ${step.cost.toLocaleString()} kr
        • Status: ${this.getStatusIcon(step.status)}
        `).join('\n')}
        `;
    }

    getStatusIcon(status) {
        switch(status.toLowerCase()) {
            case 'ok':
            case 'godkjent':
                return this.emojis.success;
            case 'warning':
            case 'advarsel':
                return this.emojis.warning;
            case 'error':
            case 'feil':
                return this.emojis.error;
            default:
                return this.emojis.info;
        }
    }

    generateFormPackage(analysis) {
        return {
            bruksendring: this.generateBruksendringForm(analysis),
            nabovarsel: this.generateNabovarselForm(analysis),
            tekniskDokumentasjon: this.generateTekniskDokumentasjon(analysis)
        };
    }

    generateBruksendringForm(analysis) {
        return {
            skjemaType: 'Søknad om bruksendring',
            skjemanummer: '5153',
            utfyltData: {
                eiendom: {
                    adresse: analysis.address,
                    gnr: analysis.propertyInfo.gnr,
                    bnr: analysis.propertyInfo.bnr,
                    kommune: analysis.propertyInfo.municipality
                },
                tiltakshaver: {
                    navn: analysis.owner.name,
                    kontaktinfo: analysis.owner.contact
                },
                tiltaket: {
                    beskrivelse: `Bruksendring av ${analysis.conversionDetails.fromType} til ${analysis.conversionDetails.toType}`,
                    areal: analysis.conversionDetails.area,
                    etasje: analysis.conversionDetails.floor
                }
            },
            vedlegg: this.generateAttachmentsList(analysis)
        };
    }

    generateNabovarselForm(analysis) {
        return {
            skjemaType: 'Nabovarsel',
            skjemanummer: '5154',
            naboer: analysis.neighbors.map(nabo => ({
                adresse: nabo.address,
                gnrBnr: nabo.propertyId,
                kontaktinfo: nabo.contact
            }))
        };
    }

    generateTekniskDokumentasjon(analysis) {
        return {
            brannkonsept: {
                rømningsveier: analysis.technical.fireEscape,
                brannmotstand: analysis.technical.fireResistance
            },
            ventilasjon: {
                luftmengder: analysis.technical.airFlow,
                ventilasjonstype: analysis.technical.ventilationType
            },
            lyd: {
                lydmålinger: analysis.technical.soundMeasurements,
                isolasjonsverdi: analysis.technical.soundInsulation
            }
        };
    }

    generateAttachmentsList(analysis) {
        return [
            {
                type: 'Plantegning',
                status: 'Vedlagt',
                format: 'PDF',
                nummer: '1'
            },
            {
                type: 'Brannkonsept',
                status: 'Vedlagt',
                format: 'PDF',
                nummer: '2'
            },
            {
                type: 'Ventilasjonsplan',
                status: 'Vedlagt',
                format: 'PDF',
                nummer: '3'
            }
        ];
    }
}