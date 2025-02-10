from typing import Dict, List, Optional, Tuple
import pdfkit
import json
import os
from pathlib import Path
import jinja2
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import numpy as np
import cv2

class CompleteDocumentGenerator:
    def __init__(self):
        """Initialiserer dokumentgenerator"""
        self.templates_dir = "templates"
        self.forms_dir = "forms"
        self.output_dir = "output"
        self.template_env = self._setup_jinja_environment()
        
    def _setup_jinja_environment(self) -> jinja2.Environment:
        """Setter opp Jinja2 miljø for templating"""
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=True
        )
        
    def generate_all_documents(self,
                             property_data: Dict,
                             analysis_results: Dict,
                             municipality: str) -> Dict[str, str]:
        """
        Genererer alle nødvendige dokumenter for byggesøknad
        
        Args:
            property_data: Data om eiendommen
            analysis_results: Resultater fra eiendomsanalysen
            municipality: Kommune
            
        Returns:
            Dict med stier til alle genererte dokumenter
        """
        # Opprett output-mappe
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        documents = {
            # Hovedsøknadsskjema
            'application_form': self._generate_application_form(
                property_data,
                analysis_results,
                municipality
            ),
            
            # Tekniske tegninger
            'technical_drawings': self._generate_technical_drawings(
                property_data,
                analysis_results
            ),
            
            # Situasjonsplan
            'site_plan': self._generate_site_plan(
                property_data,
                analysis_results
            ),
            
            # Gjennomføringsplan
            'implementation_plan': self._generate_implementation_plan(
                property_data,
                analysis_results
            ),
            
            # Nabovarsel
            'neighbor_notification': self._generate_neighbor_notification(
                property_data,
                analysis_results
            ),
            
            # Ansvarsrett-erklæringer
            'responsibility_declarations': self._generate_responsibility_declarations(
                property_data
            ),
            
            # Enova-søknad
            'enova_application': self._generate_enova_application(
                property_data,
                analysis_results
            ),
            
            # Analyserapport
            'analysis_report': self._generate_analysis_report(
                property_data,
                analysis_results
            )
        }
        
        return documents
        
    def _generate_application_form(self,
                                 property_data: Dict,
                                 analysis_results: Dict,
                                 municipality: str) -> str:
        """Genererer hovedsøknadsskjema"""
        template = self.template_env.get_template('application_form.html')
        
        # Fyll ut søknadsskjema
        content = template.render(
            property=property_data,
            analysis=analysis_results,
            municipality=municipality,
            date=datetime.now().strftime("%d.%m.%Y")
        )
        
        # Konverter til PDF
        output_path = os.path.join(self.output_dir, 'application_form.pdf')
        pdfkit.from_string(content, output_path)
        
        return output_path
        
    def _generate_technical_drawings(self,
                                   property_data: Dict,
                                   analysis_results: Dict) -> Dict[str, str]:
        """Genererer tekniske tegninger"""
        drawings = {}
        
        # Plantegninger
        drawings['floor_plans'] = self._generate_floor_plans(
            property_data,
            analysis_results
        )
        
        # Fasadetegninger
        drawings['facades'] = self._generate_facade_drawings(
            property_data,
            analysis_results
        )
        
        # Snittegninger
        drawings['sections'] = self._generate_section_drawings(
            property_data,
            analysis_results
        )
        
        return drawings
        
    def _generate_site_plan(self,
                           property_data: Dict,
                           analysis_results: Dict) -> str:
        """Genererer situasjonsplan"""
        # Opprett canvas
        output_path = os.path.join(self.output_dir, 'site_plan.pdf')
        c = canvas.Canvas(output_path, pagesize=A4)
        
        # Tegn situasjonsplan
        self._draw_site_plan(c, property_data, analysis_results)
        
        c.save()
        return output_path
        
    def _generate_implementation_plan(self,
                                    property_data: Dict,
                                    analysis_results: Dict) -> str:
        """Genererer gjennomføringsplan"""
        template = self.template_env.get_template('implementation_plan.html')
        
        content = template.render(
            property=property_data,
            analysis=analysis_results,
            date=datetime.now().strftime("%d.%m.%Y")
        )
        
        output_path = os.path.join(self.output_dir, 'implementation_plan.pdf')
        pdfkit.from_string(content, output_path)
        
        return output_path
        
    def _generate_neighbor_notification(self,
                                      property_data: Dict,
                                      analysis_results: Dict) -> str:
        """Genererer nabovarsel"""
        template = self.template_env.get_template('neighbor_notification.html')
        
        content = template.render(
            property=property_data,
            analysis=analysis_results,
            date=datetime.now().strftime("%d.%m.%Y")
        )
        
        output_path = os.path.join(self.output_dir, 'neighbor_notification.pdf')
        pdfkit.from_string(content, output_path)
        
        return output_path
        
    def _generate_responsibility_declarations(self,
                                           property_data: Dict) -> Dict[str, str]:
        """Genererer ansvarsrett-erklæringer"""
        declarations = {}
        
        # Generer erklæringer for hver ansvarlig
        for role in ['SØK', 'PRO', 'UTF']:
            declarations[role] = self._generate_single_declaration(
                property_data,
                role
            )
            
        return declarations
        
    def _generate_enova_application(self,
                                  property_data: Dict,
                                  analysis_results: Dict) -> str:
        """Genererer Enova-søknad"""
        template = self.template_env.get_template('enova_application.html')
        
        content = template.render(
            property=property_data,
            analysis=analysis_results,
            energy_analysis=analysis_results.get('energy_analysis', {}),
            date=datetime.now().strftime("%d.%m.%Y")
        )
        
        output_path = os.path.join(self.output_dir, 'enova_application.pdf')
        pdfkit.from_string(content, output_path)
        
        return output_path
        
    def _generate_analysis_report(self,
                                property_data: Dict,
                                analysis_results: Dict) -> str:
        """Genererer detaljert analyserapport"""
        doc = SimpleDocTemplate(
            os.path.join(self.output_dir, 'analysis_report.pdf'),
            pagesize=A4
        )
        
        # Bygger rapportinnhold
        story = []
        styles = getSampleStyleSheet()
        
        # Tittel
        story.append(Paragraph("Eiendomsanalyse", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Eiendomsinformasjon
        story.append(Paragraph("Eiendomsinformasjon", styles['Heading2']))
        story.append(self._create_property_info_table(property_data))
        story.append(Spacer(1, 12))
        
        # Analyse av utviklingspotensial
        story.append(Paragraph("Utviklingspotensial", styles['Heading2']))
        story.append(self._create_potential_analysis_table(analysis_results))
        story.append(Spacer(1, 12))
        
        # Reguleringsbestemmelser
        story.append(Paragraph("Reguleringsbestemmelser", styles['Heading2']))
        story.append(self._create_regulations_table(analysis_results))
        story.append(Spacer(1, 12))
        
        # Anbefalinger
        story.append(Paragraph("Anbefalinger", styles['Heading2']))
        story.append(self._create_recommendations_section(analysis_results))
        
        # Bygg dokumentet
        doc.build(story)
        
        return doc.filename
        
    def _create_property_info_table(self, property_data: Dict) -> Table:
        """Oppretter tabell med eiendomsinformasjon"""
        data = [
            ['Eiendomstype', property_data.get('type', '')],
            ['Adresse', property_data.get('address', '')],
            ['Gnr/Bnr', property_data.get('property_id', '')],
            ['Tomteareal', f"{property_data.get('plot_size', '')} m²"],
            ['BRA', f"{property_data.get('floor_area', '')} m²"],
            ['BYA', f"{property_data.get('footprint_area', '')} m²"]
        ]
        
        return self._create_formatted_table(data)
        
    def _create_potential_analysis_table(self, analysis_results: Dict) -> Table:
        """Oppretter tabell med analyse av utviklingspotensial"""
        potential = analysis_results.get('potential', {})
        data = [
            ['Mulighet', 'Status', 'Estimert kostnad', 'Estimert verdiøkning']
        ]
        
        for option, details in potential.items():
            data.append([
                option,
                'Mulig' if details.get('feasible', False) else 'Ikke mulig',
                f"{details.get('estimated_cost', 0):,} kr",
                f"{details.get('estimated_value_increase', 0):,} kr"
            ])
            
        return self._create_formatted_table(data, header=True)
        
    def _create_formatted_table(self,
                              data: List[List[str]],
                              header: bool = False) -> Table:
        """Oppretter formatert tabell"""
        table = Table(data)
        style = [
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]
        
        if header:
            style.append(('BACKGROUND', (0, 0), (-1, 0), colors.grey))
            
        table.setStyle(TableStyle(style))
        return table