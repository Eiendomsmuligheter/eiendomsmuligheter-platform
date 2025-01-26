import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Room3D:
    dimensions: Tuple[float, float, float]  # length, width, height
    position: Tuple[float, float, float]    # x, y, z coordinates
    room_type: str
    features: List[str]
    windows: List[Dict[str, Any]]
    doors: List[Dict[str, Any]]

@dataclass
class VisualizationSettings:
    quality: str = "high"
    lighting: str = "natural"
    texture_level: str = "detailed"
    show_measurements: bool = True
    show_annotations: bool = True

class VisualizationEngine:
    def __init__(self):
        self.settings = VisualizationSettings()
        self.scene = None
        self.materials_db = self._load_materials_db()
        
    def _load_materials_db(self) -> Dict[str, Dict[str, Any]]:
        """Last inn database med materialer og teksturer"""
        return {
            "wall": {
                "standard": {"color": [1, 1, 1], "roughness": 0.9},
                "brick": {"texture": "brick.jpg", "roughness": 0.7},
                "wood": {"texture": "wood.jpg", "roughness": 0.5}
            },
            "floor": {
                "hardwood": {"texture": "hardwood.jpg", "roughness": 0.3},
                "tile": {"texture": "tile.jpg", "roughness": 0.1},
                "carpet": {"texture": "carpet.jpg", "roughness": 0.95}
            }
        }
        
    def create_3d_model(self, floor_plan: Dict[str, Any]) -> None:
        """Opprett 3D-modell fra plantegning"""
        try:
            rooms = self._process_floor_plan(floor_plan)
            self.scene = self._generate_scene(rooms)
            return self.scene
        except Exception as e:
            logger.error(f"Feil ved opprettelse av 3D-modell: {str(e)}")
            return None
            
    def _process_floor_plan(self, floor_plan: Dict[str, Any]) -> List[Room3D]:
        """Prosesser plantegning til 3D-rom"""
        rooms = []
        for room_data in floor_plan.get("rooms", []):
            # Hent romegenskaper
            dimensions = (
                float(room_data.get("length", 0)),
                float(room_data.get("width", 0)),
                float(room_data.get("height", 2.4))  # Standard takhøyde hvis ikke spesifisert
            )
            
            position = (
                float(room_data.get("x", 0)),
                float(room_data.get("y", 0)),
                float(room_data.get("z", 0))
            )
            
            # Opprett Room3D objekt
            room = Room3D(
                dimensions=dimensions,
                position=position,
                room_type=room_data.get("type", "undefined"),
                features=room_data.get("features", []),
                windows=room_data.get("windows", []),
                doors=room_data.get("doors", [])
            )
            rooms.append(room)
        
        return rooms
        
    def _generate_scene(self, rooms: List[Room3D]) -> o3d.geometry.TriangleMesh:
        """Generer 3D-scene fra rom"""
        scene = o3d.geometry.TriangleMesh()
        
        for room in rooms:
            # Opprett romgeometri
            room_mesh = self._create_room_mesh(room)
            
            # Legg til vinduer og dører
            window_meshes = self._create_window_meshes(room)
            door_meshes = self._create_door_meshes(room)
            
            # Kombiner alle mesh-elementer
            room_mesh += window_meshes
            room_mesh += door_meshes
            
            # Anvend materialer og teksturer
            self._apply_materials(room_mesh, room.room_type)
            
            # Legg til i hovedscenen
            scene += room_mesh
        
        # Optimaliser scene
        scene.compute_vertex_normals()
        scene.compute_triangle_normals()
        
        return scene
    
    def _create_room_mesh(self, room: Room3D) -> o3d.geometry.TriangleMesh:
        """Opprett mesh for et enkelt rom"""
        l, w, h = room.dimensions
        x, y, z = room.position
        
        # Opprett boks for rommet
        mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=l)
        mesh.translate([x, y, z])
        
        return mesh
    
    def _create_window_meshes(self, room: Room3D) -> o3d.geometry.TriangleMesh:
        """Opprett mesh for vinduer"""
        combined_mesh = o3d.geometry.TriangleMesh()
        
        for window in room.windows:
            w = float(window.get("width", 1.0))
            h = float(window.get("height", 1.2))
            pos_x = float(window.get("x", 0))
            pos_y = float(window.get("y", 1.0))
            pos_z = float(window.get("z", 0))
            
            window_mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=0.1)
            window_mesh.translate([pos_x, pos_y, pos_z])
            combined_mesh += window_mesh
            
        return combined_mesh
    
    def _create_door_meshes(self, room: Room3D) -> o3d.geometry.TriangleMesh:
        """Opprett mesh for dører"""
        combined_mesh = o3d.geometry.TriangleMesh()
        
        for door in room.doors:
            w = float(door.get("width", 0.9))
            h = float(door.get("height", 2.1))
            pos_x = float(door.get("x", 0))
            pos_y = float(door.get("y", 0))
            pos_z = float(door.get("z", 0))
            
            door_mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=0.1)
            door_mesh.translate([pos_x, pos_y, pos_z])
            combined_mesh += door_mesh
            
        return combined_mesh
    
    def _apply_materials(self, mesh: o3d.geometry.TriangleMesh, room_type: str) -> None:
        """Anvend materialer på mesh basert på romtype"""
        if room_type in self.materials_db:
            material = self.materials_db[room_type]
            if "color" in material:
                mesh.paint_uniform_color(material["color"])
        
    def generate_walkthrough(self) -> List[Dict[str, Any]]:
        """Generer virtuell omvisning"""
        try:
            if not self.scene:
                raise ValueError("Ingen scene er lastet")
            
            walkthrough = []
            
            # Definer kameraposisjoner for omvisningen
            camera_positions = self._calculate_camera_positions()
            
            for pos in camera_positions:
                # Opprett visning fra hver posisjon
                view = {
                    "position": pos["position"],
                    "target": pos["target"],
                    "up": [0, 1, 0],
                    "field_of_view": 60,
                    "near": 0.1,
                    "far": 100.0
                }
                
                # Render bilde fra denne posisjonen
                image = self._render_from_position(view)
                
                walkthrough.append({
                    "view": view,
                    "image": image,
                    "description": self._generate_view_description(pos)
                })
            
            return walkthrough
        except Exception as e:
            logger.error(f"Feil ved generering av omvisning: {str(e)}")
            return []
            
    def _calculate_camera_positions(self) -> List[Dict[str, Any]]:
        """Beregn optimale kameraposisjoner for omvisning"""
        if not self.scene:
            return []
            
        # Beregn scene bounds
        min_bound = self.scene.get_min_bound()
        max_bound = self.scene.get_max_bound()
        center = (min_bound + max_bound) / 2
        
        # Definer standard kameraposisjoner
        positions = []
        
        # Oversiktsbilde
        positions.append({
            "position": [center[0], center[1] + 5, center[2] + 5],
            "target": center,
            "type": "overview"
        })
        
        # Posisjoner for hvert rom
        for room in self._get_rooms():
            room_center = room.position + np.array(room.dimensions) / 2
            positions.extend([
                {
                    "position": [room_center[0], room_center[1], room_center[2] + 2],
                    "target": room_center,
                    "type": "room",
                    "room": room
                },
                {
                    "position": [room_center[0] + 2, room_center[1], room_center[2]],
                    "target": room_center,
                    "type": "room_side",
                    "room": room
                }
            ])
            
        return positions
        
    def _render_from_position(self, view: Dict[str, Any]) -> np.ndarray:
        """Render bilde fra gitt kameraposisjon"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(self.scene)
        
        # Sett kameraparametere
        ctr = vis.get_view_control()
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        cam.extrinsic = np.array([
            [1, 0, 0, -view["position"][0]],
            [0, 1, 0, -view["position"][1]],
            [0, 0, 1, -view["position"][2]],
            [0, 0, 0, 1]
        ])
        ctr.convert_from_pinhole_camera_parameters(cam)
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer()
        vis.destroy_window()
        
        return np.asarray(image)
        
    def _generate_view_description(self, position: Dict[str, Any]) -> str:
        """Generer beskrivelse av visningen"""
        if position["type"] == "overview":
            return "Oversiktsbilde av hele boligen"
        elif position["type"] == "room":
            room = position["room"]
            return f"Visning av {room.room_type} ({room.dimensions[0]}x{room.dimensions[1]} m)"
        elif position["type"] == "room_side":
            room = position["room"]
            return f"Sidevisning av {room.room_type}"
        return "Standard visning"
        
    def create_interactive_visualization(self) -> Dict[str, Any]:
        """Lag interaktiv visualisering"""
        try:
            if not self.scene:
                raise ValueError("Ingen scene er lastet")
                
            # Konverter scene til Plotly-format
            vertices = np.asarray(self.scene.vertices)
            triangles = np.asarray(self.scene.triangles)
            colors = np.asarray(self.scene.vertex_colors)
            
            # Opprett Plotly-figur
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    vertexcolor=colors,
                    lighting=dict(
                        ambient=0.8,
                        diffuse=0.9,
                        fresnel=0.2,
                        specular=0.5,
                        roughness=0.5
                    ),
                    lightposition=dict(
                        x=100,
                        y=200,
                        z=150
                    )
                )
            ])
            
            # Konfigurer layout
            fig.update_layout(
                scene=dict(
                    camera=dict(
                        up=dict(x=0, y=1, z=0),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )
            
            return {
                "figure": fig,
                "controls": self._generate_controls(),
                "annotations": self._generate_annotations()
            }
        except Exception as e:
            logger.error(f"Feil ved opprettelse av visualisering: {str(e)}")
            return {"error": str(e)}
            
    def _generate_controls(self) -> Dict[str, Any]:
        """Generer kontroller for visualiseringen"""
        return {
            "camera": {
                "zoom": {
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 1.0
                },
                "rotation": {
                    "enabled": True,
                    "speed": 1.0,
                    "default_angle": 0
                }
            },
            "lighting": {
                "intensity": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "default": 0.8
                },
                "position": {
                    "adjustable": True,
                    "default": {"x": 100, "y": 200, "z": 150}
                },
                "color": {
                    "options": ["white", "warm", "cool"],
                    "default": "white"
                }
            },
            "display": {
                "wireframe": {
                    "enabled": True,
                    "default": False
                },
                "transparency": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "default": 0.0
                },
                "measurements": {
                    "enabled": True,
                    "default": True
                }
            }
        }
        
    def _generate_annotations(self) -> List[Dict[str, Any]]:
        """Generer annoteringer for visualiseringen"""
        annotations = []
        
        if not self.scene:
            return annotations
            
        # Legg til dimensjoner og målinger
        if self.settings.show_measurements:
            for room in self._get_rooms():
                room_center = room.position + np.array(room.dimensions) / 2
                annotations.append({
                    "type": "dimension",
                    "position": room_center.tolist(),
                    "text": f"{room.dimensions[0]}m x {room.dimensions[1]}m",
                    "show_lines": True
                })
        
        # Legg til romnavn og typer
        if self.settings.show_annotations:
            for room in self._get_rooms():
                annotations.append({
                    "type": "label",
                    "position": (room.position + [0, 0, room.dimensions[2]]).tolist(),
                    "text": room.room_type.capitalize(),
                    "style": {
                        "font_size": 14,
                        "color": "#ffffff"
                    }
                })
        
        return annotations
        
    def _get_rooms(self) -> List[Room3D]:
        """Hent alle rom i scenen"""
        # Dette er en hjelpefunksjon som må implementeres basert på hvordan
        # rommene er lagret i scenen
        return []
        
    def export_model(self, format: str = "glb") -> bytes:
        """Eksporter 3D-modell til spesifisert format"""
        try:
            if not self.scene:
                raise ValueError("Ingen scene er lastet")
            
            # Konverter til ønsket format
            if format.lower() == "glb":
                # Eksporter til GLB format (for web-visning)
                return self._export_to_glb()
            elif format.lower() == "obj":
                # Eksporter til OBJ format
                return self._export_to_obj()
            elif format.lower() == "fbx":
                # Eksporter til FBX format
                return self._export_to_fbx()
            else:
                raise ValueError(f"Ukjent eksportformat: {format}")
                
        except Exception as e:
            logger.error(f"Feil ved eksport av modell: {str(e)}")
            return None
            
    def _export_to_glb(self) -> bytes:
        """Eksporter scene til GLB format"""
        # Konverter Open3D scene til GLB
        temp_path = "temp_model.glb"
        o3d.io.write_triangle_mesh(temp_path, self.scene)
        
        with open(temp_path, "rb") as f:
            data = f.read()
            
        # Fjern midlertidig fil
        import os
        os.remove(temp_path)
        
        return data
        
    def _export_to_obj(self) -> bytes:
        """Eksporter scene til OBJ format"""
        temp_path = "temp_model.obj"
        o3d.io.write_triangle_mesh(temp_path, self.scene)
        
        with open(temp_path, "rb") as f:
            data = f.read()
            
        # Fjern midlertidig fil
        import os
        os.remove(temp_path)
        
        return data
        
    def _export_to_fbx(self) -> bytes:
        """Eksporter scene til FBX format"""
        # Dette krever en FBX SDK eller lignende bibliotek
        raise NotImplementedError("FBX export er ikke implementert ennå")