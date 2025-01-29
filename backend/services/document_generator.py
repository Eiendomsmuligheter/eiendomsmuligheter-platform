from typing import Dict, List
import jinja2
import pdfkit
import os
from datetime import datetime
import json

class DocumentGenerator:
    def __init__(self):
        self.template_dir = "templates"
        self.output_dir = "generated_documents"
        self.env = self._setup_jinja()

    def _setup_jinja(self) -> jinja2.Environment:
        """
        Set up Jinja2 environment with custom filters and templates
        """
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=True
        )
        
        # Add custom filters
        env.filters['format_date'] = lambda d: d.strftime('%d.%m.%Y')
        env.filters['format_number'] = lambda n: f"{n:,.2f}"
        
        return env

    async def generate_documents(
        self,
        property_info: Dict,
        regulations: Dict,
        development_potential: Dict,
        energy_analysis: Dict
    ) -> List[str]:
        """
        Generate all necessary documents for the property
        """
        generated_files = []

        try:
            # Generate building application
            building_app = await self._generate_building_application(
                property_info,
                regulations
            )
            generated_files.append(building_app)

            # Generate situation plan
            situation_plan = await self._generate_situation_plan(
                property_info,
                regulations
            )
            generated_files.append(situation_plan)

            # Generate technical drawings
            drawings = await self._generate_technical_drawings(
                property_info,
                development_potential
            )
            generated_files.extend(drawings)

            # Generate analysis report
            analysis_report = await self._generate_analysis_report(
                property_info,
                regulations,
                development_potential,
                energy_analysis
            )
            generated_files.append(analysis_report)

        except Exception as e:
            print(f"Error generating documents: {str(e)}")
            raise

        return generated_files

    async def _generate_building_application(
        self,
        property_info: Dict,
        regulations: Dict
    ) -> str:
        """
        Generate building application documents
        """
        template = self.env.get_template('building_application.html')
        
        context = {
            'property': property_info,
            'regulations': regulations,
            'date': datetime.now(),
            'application_type': self._determine_application_type(property_info)
        }
        
        html = template.render(context)
        
        # Generate PDF
        output_path = os.path.join(
            self.output_dir,
            f"byggesoknad_{property_info['gnr']}_{property_info['bnr']}.pdf"
        )
        
        pdfkit.from_string(html, output_path)
        
        return output_path

    async def _generate_situation_plan(
        self,
        property_info: Dict,
        regulations: Dict
    ) -> str:
        """
        Generate situation plan
        """
        template = self.env.get_template('situation_plan.html')
        
        context = {
            'property': property_info,
            'regulations': regulations,
            'measurements': self._calculate_measurements(property_info),
            'distances': self._calculate_distances(property_info)
        }
        
        html = template.render(context)
        
        # Generate PDF
        output_path = os.path.join(
            self.output_dir,
            f"situasjonsplan_{property_info['gnr']}_{property_info['bnr']}.pdf"
        )
        
        pdfkit.from_string(html, output_path)
        
        return output_path

    async def _generate_technical_drawings(
        self,
        property_info: Dict,
        development_potential: Dict
    ) -> List[str]:
        """
        Generate technical drawings (floor plans, facades, sections)
        """
        generated_files = []
        
        # Floor plans
        floor_plans = await self._generate_floor_plans(
            property_info,
            development_potential
        )
        generated_files.extend(floor_plans)
        
        # Facades
        facades = await self._generate_facades(
            property_info,
            development_potential
        )
        generated_files.extend(facades)
        
        # Sections
        sections = await self._generate_sections(
            property_info,
            development_potential
        )
        generated_files.extend(sections)
        
        return generated_files

    async def _generate_analysis_report(
        self,
        property_info: Dict,
        regulations: Dict,
        development_potential: Dict,
        energy_analysis: Dict
    ) -> str:
        """
        Generate comprehensive analysis report
        """
        template = self.env.get_template('analysis_report.html')
        
        context = {
            'property': property_info,
            'regulations': regulations,
            'development': development_potential,
            'energy': energy_analysis,
            'date': datetime.now(),
            'recommendations': self._generate_recommendations(
                development_potential,
                energy_analysis
            )
        }
        
        html = template.render(context)
        
        # Generate PDF
        output_path = os.path.join(
            self.output_dir,
            f"analyserapport_{property_info['gnr']}_{property_info['bnr']}.pdf"
        )
        
        pdfkit.from_string(html, output_path)
        
        return output_path

    def _determine_application_type(self, property_info: Dict) -> str:
        """
        Determine the type of building application needed
        """
        if property_info.get('development_type') == 'new_building':
            return 'new_building'
        elif property_info.get('development_type') == 'extension':
            return 'extension'
        elif property_info.get('development_type') == 'renovation':
            return 'renovation'
        else:
            return 'unknown'

    def _calculate_measurements(self, property_info: Dict) -> Dict:
        """
        Calculate necessary measurements for the situation plan
        """
        return {
            'total_area': property_info.get('area', 0),
            'building_area': property_info.get('building_area', 0),
            'utilization_rate': (
                property_info.get('building_area', 0) /
                property_info.get('area', 1) * 100
            ),
            'heights': {
                'ground_level': property_info.get('ground_level', 0),
                'building_height': property_info.get('building_height', 0),
                'terrain_points': property_info.get('terrain_points', [])
            }
        }

    def _calculate_distances(self, property_info: Dict) -> Dict:
        """
        Calculate distances to property lines and neighboring buildings
        """
        return {
            'to_neighbor': property_info.get('distance_to_neighbor', 0),
            'to_road': property_info.get('distance_to_road', 0),
            'to_property_line': property_info.get('distance_to_property_line', 0)
        }

    async def _generate_floor_plans(
        self,
        property_info: Dict,
        development_potential: Dict
    ) -> List[str]:
        """
        Generate floor plan drawings
        """
        floor_plans = []
        template = self.env.get_template('floor_plan.html')
        
        for floor in property_info.get('floors', []):
            context = {
                'floor': floor,
                'measurements': self._get_floor_measurements(floor),
                'rooms': self._get_room_details(floor),
                'development': development_potential
            }
            
            html = template.render(context)
            
            output_path = os.path.join(
                self.output_dir,
                f"plantegning_{floor['level']}_{property_info['gnr']}_{property_info['bnr']}.pdf"
            )
            
            pdfkit.from_string(html, output_path)
            floor_plans.append(output_path)
        
        return floor_plans

    def _get_floor_measurements(self, floor: Dict) -> Dict:
        """
        Get detailed measurements for a floor
        """
        return {
            'area': floor.get('area', 0),
            'height': floor.get('height', 0),
            'windows': self._calculate_window_area(floor),
            'doors': self._get_door_details(floor)
        }

    def _get_room_details(self, floor: Dict) -> List[Dict]:
        """
        Get detailed information about rooms on a floor
        """
        rooms = []
        for room in floor.get('rooms', []):
            rooms.append({
                'name': room.get('name', ''),
                'area': room.get('area', 0),
                'height': room.get('height', 0),
                'windows': self._calculate_window_area(room),
                'usage': room.get('usage', ''),
                'requirements': self._get_room_requirements(room)
            })
        return rooms

    def _calculate_window_area(self, space: Dict) -> float:
        """
        Calculate total window area for a space
        """
        return sum(
            window.get('width', 0) * window.get('height', 0)
            for window in space.get('windows', [])
        )

    def _get_door_details(self, space: Dict) -> List[Dict]:
        """
        Get details about doors in a space
        """
        return [{
            'width': door.get('width', 0),
            'height': door.get('height', 0),
            'type': door.get('type', ''),
            'direction': door.get('direction', '')
        } for door in space.get('doors', [])]

    def _get_room_requirements(self, room: Dict) -> Dict:
        """
        Get building requirements for a specific room type
        """
        room_type = room.get('usage', '').lower()
        
        requirements = {
            'bedroom': {
                'min_area': 7,
                'min_height': 2.4,
                'min_window_area': 1.0,
                'ventilation': 'mechanical'
            },
            'living_room': {
                'min_area': 15,
                'min_height': 2.4,
                'min_window_area': 1.5,
                'ventilation': 'mechanical'
            },
            'bathroom': {
                'min_area': 4,
                'min_height': 2.2,
                'waterproof': True,
                'ventilation': 'mechanical'
            },
            'kitchen': {
                'min_area': 6,
                'min_height': 2.4,
                'min_window_area': 1.0,
                'ventilation': 'mechanical_hood'
            }
        }
        
        return requirements.get(room_type, {})