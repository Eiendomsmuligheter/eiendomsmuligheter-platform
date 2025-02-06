from typing import Dict, List, Optional
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import os
import tempfile
from datetime import datetime
from weasyprint import HTML, CSS
import json

class DocumentGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.temp_dir = tempfile.mkdtemp()
        
        # Last inn maler
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """
        Laster inn dokumentmaler
        """
        return {
            "byggesoknad": self._load_template("byggesoknad"),
            "situasjonsplan": self._load_template("situasjonsplan"),
            "teknisk_beskrivelse": self._load_template("teknisk_beskrivelse"),
            "nabovarsel": self._load_template("nabovarsel")
        }
    
    def _load_template(self, template_name: str) -> Dict:
        """
        Laster inn en spesifikk mal
        """
        template_path = os.path.join(
            os.path.dirname(__file__),
            f"../templates/{template_name}.json"
        )
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Advarsel: Mal ikke funnet: {template_name}")
            return {}
    
    async def generate_building_application(
        self,
        property_data: Dict,
        analysis_results: Dict
    ) -> str:
        """
        Genererer komplett byggesøknad med alle vedlegg
        """
        try:
            # Opprett midlertidig mappe for dokumenter
            application_dir = os.path.join(self.temp_dir, "byggesoknad")
            os.makedirs(application_dir, exist_ok=True)
            
            # Generer hoveddokumenter
            main_application = await self._generate_main_application(
                property_data,
                analysis_results
            )
            
            technical_description = await self._generate_technical_description(
                property_data,
                analysis_results
            )
            
            situation_plan = await self._generate_situation_plan(
                property_data,
                analysis_results
            )
            
            neighbor_notice = await self._generate_neighbor_notice(
                property_data,
                analysis_results
            )
            
            # Kombiner alle dokumenter
            output_path = os.path.join(application_dir, "komplett_byggesoknad.pdf")
            self._merge_pdfs(
                [
                    main_application,
                    technical_description,
                    situation_plan,
                    neighbor_notice
                ],
                output_path
            )
            
            return output_path
            
        except Exception as e:
            print(f"Feil ved generering av byggesøknad: {str(e)}")
            return None
    
    async def _generate_main_application(
        self,
        property_data: Dict,
        analysis_results: Dict
    ) -> str:
        """
        Genererer hovedsøknadsskjema
        """
        output_path = os.path.join(self.temp_dir, "soknad.pdf")
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )
        
        # Hent maldata
        template = self.templates["byggesoknad"]
        
        # Bygg dokumentinnhold
        story = []
        
        # Tittel
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        )
        story.append(Paragraph("BYGGESØKNAD", title_style))
        
        # Eiendomsinformasjon
        story.append(Paragraph("1. Eiendomsinformasjon", self.styles['Heading2']))
        property_info = [
            ["Eiendom", f"{property_data['address']}"],
            ["Gnr/Bnr", f"{property_data['gnr']}/{property_data['bnr']}"],
            ["Kommune", property_data['municipality']]
        ]
        story.append(self._create_info_table(property_info))
        story.append(Spacer(1, 20))
        
        # Tiltakshaver
        story.append(Paragraph("2. Tiltakshaver", self.styles['Heading2']))
        owner_info = [
            ["Navn", property_data['owner_name']],
            ["Adresse", property_data['owner_address']],
            ["Telefon", property_data['owner_phone']],
            ["E-post", property_data['owner_email']]
        ]
        story.append(self._create_info_table(owner_info))
        story.append(Spacer(1, 20))
        
        # Tiltakets art
        story.append(Paragraph("3. Tiltakets art", self.styles['Heading2']))
        project_description = analysis_results.get('development_potential', {}).get(
            'recommendations', []
        )
        for desc in project_description:
            story.append(Paragraph(f"• {desc}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Reguleringsmessige forhold
        story.append(Paragraph("4. Reguleringsmessige forhold", self.styles['Heading2']))
        regulation_info = [
            ["Plantype", analysis_results['regulations']['zoning_plan']['plan_type']],
            ["Formål", analysis_results['regulations']['zoning_plan']['purpose']],
            ["Utnyttingsgrad", f"{analysis_results['regulations']['max_utilization']['max_bya']}% BYA"]
        ]
        story.append(self._create_info_table(regulation_info))
        story.append(Spacer(1, 20))
        
        # Arealdisponering
        story.append(Paragraph("5. Arealdisponering", self.styles['Heading2']))
        area_info = [
            ["Tomteareal", f"{property_data.get('plot_area', 0)} m²"],
            ["Bebygd areal", f"{property_data.get('built_area', 0)} m²"],
            ["Nytt bebygd areal", f"{analysis_results.get('new_area', 0)} m²"]
        ]
        story.append(self._create_info_table(area_info))
        
        # Bygg dokumentet
        doc.build(story)
        
        return output_path
    
    async def _generate_technical_description(
        self,
        property_data: Dict,
        analysis_results: Dict
    ) -> str:
        """
        Genererer teknisk beskrivelse
        """
        output_path = os.path.join(self.temp_dir, "teknisk_beskrivelse.pdf")
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )
        
        # Bygg dokumentinnhold
        story = []
        
        # Tittel
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        )
        story.append(Paragraph("TEKNISK BESKRIVELSE", title_style))
        
        # Konstruksjon
        story.append(Paragraph("1. Konstruksjon", self.styles['Heading2']))
        construction_info = [
            ["Fundament", "Såle av armert betong"],
            ["Yttervegger", "Bindingsverk i tre, 198mm isolasjon"],
            ["Tak", "Salttak med takstein"],
            ["Etasjeskiller", "Trebjelkelag"]
        ]
        story.append(self._create_info_table(construction_info))
        story.append(Spacer(1, 20))
        
        # Tekniske installasjoner
        story.append(Paragraph("2. Tekniske installasjoner", self.styles['Heading2']))
        technical_info = [
            ["Ventilasjon", "Balansert ventilasjon med varmegjenvinning"],
            ["Oppvarming", "Elektrisk og varmepumpe"],
            ["Brannsikring", "Brannvarslere og slukkeutstyr iht. TEK17"]
        ]
        story.append(self._create_info_table(technical_info))
        
        # Bygg dokumentet
        doc.build(story)
        
        return output_path
    
    async def _generate_situation_plan(
        self,
        property_data: Dict,
        analysis_results: Dict
    ) -> str:
        """
        Genererer situasjonsplan
        """
        # Dette er en forenklet implementasjon - må integreres med faktisk karttjeneste
        output_path = os.path.join(self.temp_dir, "situasjonsplan.pdf")
        
        c = canvas.Canvas(output_path, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(30*mm, 280*mm, "SITUASJONSPLAN")
        
        # Tegn kartdata (forenklet)
        c.setStrokeColorRGB(0, 0, 0)
        c.setFillColorRGB(0.9, 0.9, 0.9)
        c.rect(30*mm, 50*mm, 150*mm, 200*mm, fill=1)
        
        # Tegn målestokk
        c.setFont("Helvetica", 10)
        c.drawString(30*mm, 40*mm, "Målestokk 1:500")
        
        c.save()
        return output_path
    
    async def _generate_neighbor_notice(
        self,
        property_data: Dict,
        analysis_results: Dict
    ) -> str:
        """
        Genererer nabovarsel
        """
        output_path = os.path.join(self.temp_dir, "nabovarsel.pdf")
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )
        
        # Bygg dokumentinnhold
        story = []
        
        # Tittel
        story.append(Paragraph("NABOVARSEL", self.styles['Heading1']))
        story.append(Spacer(1, 20))
        
        # Prosjektinformasjon
        story.append(Paragraph(
            f"Det varsles herved om byggetiltak på eiendom {property_data['address']}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 10))
        
        # Tiltaksbeskrivelse
        story.append(Paragraph("Beskrivelse av tiltaket:", self.styles['Heading2']))
        for desc in analysis_results.get('development_potential', {}).get(
            'recommendations', []
        ):
            story.append(Paragraph(f"• {desc}", self.styles['Normal']))
        
        # Bygg dokumentet
        doc.build(story)
        
        return output_path
    
    def _create_info_table(self, data: List[List[str]]) -> Table:
        """
        Oppretter formatert informasjonstabell
        """
        table = Table(data, colWidths=[80*mm, 80*mm])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        return table
    
    def _merge_pdfs(self, pdf_list: List[str], output_path: str):
        """
        Slår sammen flere PDF-filer til én
        """
        result = fitz.open()
        
        for pdf in pdf_list:
            with fitz.open(pdf) as doc:
                result.insert_pdf(doc)
        
        result.save(output_path)
        result.close()
    
    def cleanup(self):
        """
        Rydder opp temporære filer
        """
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)