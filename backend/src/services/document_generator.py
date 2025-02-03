import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
import json

class DocumentGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = "generated_documents"
        os.makedirs(self.output_dir, exist_ok=True)

    async def generate_documents(
        self,
        property_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate all necessary documents for the property"""
        try:
            documents = []
            
            # Generate building application
            building_app = await self.generate_building_application(
                property_info,
                analysis_results
            )
            if building_app:
                documents.append(building_app)
            
            # Generate analysis report
            analysis_report = await self.generate_analysis_report(
                property_info,
                analysis_results
            )
            if analysis_report:
                documents.append(analysis_report)
            
            # Generate technical drawings
            drawings = await self.generate_technical_drawings(
                property_info,
                analysis_results
            )
            documents.extend(drawings)
            
            return documents
        except Exception as e:
            self.logger.error(f"Error generating documents: {str(e)}")
            raise

    async def generate_building_application(
        self,
        property_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate building application document"""
        try:
            filename = f"byggersøknad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            normal_style = styles['Normal']
            
            # Build content
            content = []
            
            # Title
            content.append(Paragraph("BYGGESØKNAD", title_style))
            content.append(Spacer(1, 12))
            
            # Property information
            content.append(Paragraph("1. Eiendomsinformasjon", styles['Heading2']))
            content.append(Spacer(1, 6))
            
            property_table_data = [
                ["Adresse:", property_info.get("address", "")],
                ["Gnr/Bnr:", property_info.get("property_id", "")],
                ["Tomtestørrelse:", f"{property_info.get('plot_size', 0)} m²"],
                ["BRA:", f"{property_info.get('bra', 0)} m²"]
            ]
            
            property_table = Table(
                property_table_data,
                colWidths=[100, 300]
            )
            property_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            
            content.append(property_table)
            content.append(Spacer(1, 12))
            
            # Tiltakshaver information
            content.append(Paragraph("2. Tiltakshaver", styles['Heading2']))
            content.append(Spacer(1, 6))
            
            # Add more sections...
            
            # Generate PDF
            doc.build(content)
            
            return {
                "title": "Byggesøknad",
                "description": "Komplett byggesøknad klar for innsending",
                "url": filepath,
                "type": "application"
            }
        except Exception as e:
            self.logger.error(f"Error generating building application: {str(e)}")
            return None

    async def generate_analysis_report(
        self,
        property_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate detailed analysis report"""
        try:
            filename = f"analyse_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            content = []
            
            # Title
            content.append(Paragraph("ANALYSERAPPORT", styles['Heading1']))
            content.append(Spacer(1, 12))
            
            # Property summary
            content.append(Paragraph("Eiendomssammendrag", styles['Heading2']))
            content.append(Spacer(1, 6))
            
            summary_text = f"""
            Adressen {property_info.get('address', '')} har blitt analysert for
            utviklingspotensial. Eiendommen er {property_info.get('plot_size', 0)} m²
            og har en BRA på {property_info.get('bra', 0)} m².
            """
            content.append(Paragraph(summary_text, styles['Normal']))
            content.append(Spacer(1, 12))
            
            # Development potential
            content.append(Paragraph("Utviklingspotensial", styles['Heading2']))
            content.append(Spacer(1, 6))
            
            for option in analysis_results.get('potential', {}).get('options', []):
                content.append(
                    Paragraph(f"• {option['title']}", styles['Heading3'])
                )
                content.append(
                    Paragraph(option['description'], styles['Normal'])
                )
                content.append(
                    Paragraph(
                        f"Estimert kostnad: {option['estimatedCost']} NOK",
                        styles['Normal']
                    )
                )
                content.append(
                    Paragraph(
                        f"Potensiell verdiøkning: {option['potentialValue']} NOK",
                        styles['Normal']
                    )
                )
                content.append(Spacer(1, 12))
            
            # Generate PDF
            doc.build(content)
            
            return {
                "title": "Analyserapport",
                "description": "Detaljert analyse av utviklingsmuligheter",
                "url": filepath,
                "type": "report"
            }
        except Exception as e:
            self.logger.error(f"Error generating analysis report: {str(e)}")
            return None

    async def generate_technical_drawings(
        self,
        property_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate technical drawings"""
        try:
            drawings = []
            
            # Generate floor plan
            floor_plan = await self._generate_floor_plan(
                property_info,
                analysis_results
            )
            if floor_plan:
                drawings.append(floor_plan)
            
            # Generate facade drawings
            facade_drawings = await self._generate_facade_drawings(
                property_info,
                analysis_results
            )
            drawings.extend(facade_drawings)
            
            # Generate situation plan
            situation_plan = await self._generate_situation_plan(
                property_info,
                analysis_results
            )
            if situation_plan:
                drawings.append(situation_plan)
            
            return drawings
        except Exception as e:
            self.logger.error(f"Error generating technical drawings: {str(e)}")
            return []

    async def _generate_floor_plan(
        self,
        property_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate detailed floor plan"""
        try:
            filename = f"plantegning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # This would typically use CAD or similar software
            # For now, create a simple PDF
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            content = []
            
            content.append(Paragraph("PLANTEGNING", styles['Heading1']))
            content.append(Spacer(1, 12))
            
            # Add floor plan details...
            
            doc.build(content)
            
            return {
                "title": "Plantegning",
                "description": "Detaljert plantegning med mål",
                "url": filepath,
                "type": "drawing"
            }
        except Exception as e:
            self.logger.error(f"Error generating floor plan: {str(e)}")
            return None

    async def _generate_facade_drawings(
        self,
        property_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate facade drawings"""
        try:
            drawings = []
            
            for direction in ['nord', 'sør', 'øst', 'vest']:
                filename = f"fasade_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                filepath = os.path.join(self.output_dir, filename)
                
                doc = SimpleDocTemplate(
                    filepath,
                    pagesize=A4,
                    rightMargin=72,
                    leftMargin=72,
                    topMargin=72,
                    bottomMargin=72
                )
                
                styles = getSampleStyleSheet()
                content = []
                
                content.append(
                    Paragraph(f"FASADETEGNING {direction.upper()}", styles['Heading1'])
                )
                content.append(Spacer(1, 12))
                
                # Add facade details...
                
                doc.build(content)
                
                drawings.append({
                    "title": f"Fasadetegning {direction}",
                    "description": f"Fasadetegning sett fra {direction}",
                    "url": filepath,
                    "type": "drawing"
                })
            
            return drawings
        except Exception as e:
            self.logger.error(f"Error generating facade drawings: {str(e)}")
            return []

    async def _generate_situation_plan(
        self,
        property_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate situation plan"""
        try:
            filename = f"situasjonsplan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            content = []
            
            content.append(Paragraph("SITUASJONSPLAN", styles['Heading1']))
            content.append(Spacer(1, 12))
            
            # Add situation plan details...
            
            doc.build(content)
            
            return {
                "title": "Situasjonsplan",
                "description": "Situasjonsplan med avstander og høyder",
                "url": filepath,
                "type": "drawing"
            }
        except Exception as e:
            self.logger.error(f"Error generating situation plan: {str(e)}")
            return None

    async def generate_specific_document(
        self,
        property_id: str,
        document_type: str
    ) -> Dict[str, Any]:
        """Generate a specific type of document"""
        try:
            # Get property info and analysis results
            # In a real implementation, this would fetch from a database
            property_info = {}  # Mock data
            analysis_results = {}  # Mock data
            
            if document_type == "building_application":
                return await self.generate_building_application(
                    property_info,
                    analysis_results
                )
            elif document_type == "analysis_report":
                return await self.generate_analysis_report(
                    property_info,
                    analysis_results
                )
            elif document_type == "floor_plan":
                return await self._generate_floor_plan(
                    property_info,
                    analysis_results
                )
            elif document_type == "situation_plan":
                return await self._generate_situation_plan(
                    property_info,
                    analysis_results
                )
            else:
                raise ValueError(f"Unknown document type: {document_type}")
        except Exception as e:
            self.logger.error(
                f"Error generating specific document: {str(e)}"
            )
            raise