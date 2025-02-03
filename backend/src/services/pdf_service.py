from typing import Dict, List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
import json
import logging
from pathlib import Path
import qrcode
import io

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Oppretter tilpassede stiler for PDF"""
        custom = {
            'Heading1': ParagraphStyle(
                'CustomHeading1',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            ),
            'Heading2': ParagraphStyle(
                'CustomHeading2',
                parent=self.styles['Heading2'],
                fontSize=18,
                spaceAfter=20
            ),
            'Normal': ParagraphStyle(
                'CustomNormal',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=12
            )
        }
        return custom
        
    async def generate_analysis_report(
        self,
        property_data: Dict,
        analysis_results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """Genererer en detaljert PDF-rapport fra analysen"""
        if not output_path:
            output_path = f"/tmp/analyse_rapport_{property_data['id']}.pdf"
            
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            
            # Tittel
            story.append(Paragraph(
                "Eiendomsanalyse Rapport",
                self.custom_styles['Heading1']
            ))
            
            # Eiendomsinformasjon
            story.append(Paragraph(
                "Eiendomsinformasjon",
                self.custom_styles['Heading2']
            ))
            
            property_info = [
                ["Adresse:", property_data.get('address', '')],
                ["Gnr/Bnr:", f"{property_data.get('gnr', '')}/{property_data.get('bnr', '')}"],
                ["Kommune:", property_data.get('municipality', '')],
                ["Tomteareal:", f"{property_data.get('plot_size', '')} m²"],
                ["BRA:", f"{property_data.get('bra', '')} m²"]
            ]
            
            table = Table(property_info)
            table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Reguleringsbestemmelser
            story.append(Paragraph(
                "Reguleringsbestemmelser",
                self.custom_styles['Heading2']
            ))
            
            for reg in analysis_results.get('regulations', []):
                story.append(Paragraph(
                    f"<b>{reg['title']}</b>",
                    self.custom_styles['Normal']
                ))
                story.append(Paragraph(
                    reg['description'],
                    self.custom_styles['Normal']
                ))
                
            # Utviklingspotensial
            story.append(Paragraph(
                "Utviklingspotensial",
                self.custom_styles['Heading2']
            ))
            
            for option in analysis_results.get('potential', {}).get('options', []):
                story.append(Paragraph(
                    f"<b>{option['title']}</b>",
                    self.custom_styles['Normal']
                ))
                story.append(Paragraph(
                    option['description'],
                    self.custom_styles['Normal']
                ))
                story.append(Paragraph(
                    f"Estimert kostnad: {option['estimatedCost']} NOK",
                    self.custom_styles['Normal']
                ))
                story.append(Paragraph(
                    f"Potensiell verdiøkning: {option['potentialValue']} NOK",
                    self.custom_styles['Normal']
                ))
                story.append(Spacer(1, 12))
                
            # Energianalyse
            story.append(Paragraph(
                "Energianalyse",
                self.custom_styles['Heading2']
            ))
            
            energy = analysis_results.get('energyAnalysis', {})
            story.append(Paragraph(
                f"Energimerking: {energy.get('rating', '')}",
                self.custom_styles['Normal']
            ))
            story.append(Paragraph(
                f"Årlig energiforbruk: {energy.get('consumption', '')} kWh/år",
                self.custom_styles['Normal']
            ))
            
            # Enova-støtte
            story.append(Paragraph(
                "Enova-støttemuligheter",
                self.custom_styles['Heading2']
            ))
            
            for support in energy.get('enovaSupport', []):
                story.append(Paragraph(
                    f"<b>{support['title']}</b>",
                    self.custom_styles['Normal']
                ))
                story.append(Paragraph(
                    support['description'],
                    self.custom_styles['Normal']
                ))
                story.append(Paragraph(
                    f"Støttebeløp: {support['amount']} NOK",
                    self.custom_styles['Normal']
                ))
                
            # QR-kode for digital versjon
            qr = qrcode.QRCode(
                version=1,
                box_size=10,
                border=5
            )
            qr.add_data(f"https://eiendomsmuligheter.no/rapport/{property_data['id']}")
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # Konverter QR-kode til ReportLab-format
            qr_buffer = io.BytesIO()
            qr_img.save(qr_buffer, format='PNG')
            qr_buffer.seek(0)
            
            story.append(Spacer(1, 30))
            story.append(Image(qr_buffer, width=100, height=100))
            story.append(Paragraph(
                "Scan QR-koden for digital versjon av rapporten",
                self.custom_styles['Normal']
            ))
            
            # Generer PDF
            doc.build(story)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Feil ved generering av PDF: {e}")
            raise
            
    async def generate_building_application(
        self,
        property_data: Dict,
        application_data: Dict
    ) -> str:
        """Genererer byggesøknad basert på analyse og brukerdata"""
        output_path = f"/tmp/byggesoknad_{property_data['id']}.pdf"
        
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            
            # Implementer byggesøknadsgenerering her
            # Dette vil variere basert på kommunens krav
            
            return output_path
            
        except Exception as e:
            logger.error(f"Feil ved generering av byggesøknad: {e}")
            raise