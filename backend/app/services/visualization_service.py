import omni.kit.commands
from pxr import Usd, UsdGeom, Sdf, Gf
import numpy as np
from typing import Dict, List, Optional
import trimesh
import open3d as o3d
from pathlib import Path
import tempfile
import os

class VisualizationService:
    def __init__(self):
        self.stage = None
        self.temp_dir = tempfile.mkdtemp()
        self._initialize_omniverse()
    
    def _initialize_omniverse(self):
        """
        Initialiserer Omniverse-miljøet
        """
        # Sett opp Omniverse-tilkobling
        self.stage = Usd.Stage.CreateNew("temp.usda")
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        
        # Opprett standard scene-elementer
        self._setup_lighting()
        self._setup_ground_plane()
    
    def _setup_lighting(self):
        """
        Setter opp standard belysning
        """
        # Opprett hovedlys
        light_path = "/World/MainLight"
        light = UsdGeom.DistantLight.Define(self.stage, light_path)
        light.CreateIntensityAttr(500)
        light.CreateAngleAttr(0.53)
        light.AddTranslateOp().Set(Gf.Vec3d(0, 5, 0))
        light.AddRotateXYZOp().Set(Gf.Vec3d(-45, 0, 0))
        
        # Opprett ambient lys
        ambient_path = "/World/AmbientLight"
        ambient = UsdGeom.DomeLight.Define(self.stage, ambient_path)
        ambient.CreateIntensityAttr(100)
    
    def _setup_ground_plane(self):
        """
        Oppretter et grunnplan
        """
        ground_path = "/World/Ground"
        ground = UsdGeom.Mesh.Define(self.stage, ground_path)
        
        points = [(-50, 0, -50), (50, 0, -50), (50, 0, 50), (-50, 0, 50)]
        normals = [(0, 1, 0)] * 4
        indices = [0, 1, 2, 0, 2, 3]
        
        ground.CreatePointsAttr(points)
        ground.CreateNormalsAttr(normals)
        ground.CreateFaceVertexIndicesAttr(indices)
        ground.CreateFaceVertexCountsAttr([3, 3])
    
    async def create_3d_model_from_floorplan(
        self,
        floor_plan_path: str,
        height: float = 2.4
    ) -> str:
        """
        Genererer en 3D-modell fra en plantegning
        """
        try:
            # Last inn plantegning
            room_contours = self._extract_room_contours(floor_plan_path)
            
            # Opprett 3D-geometri
            meshes = []
            for contour in room_contours:
                mesh = self._create_room_mesh(contour, height)
                meshes.append(mesh)
            
            # Kombiner alle mesh
            combined_mesh = trimesh.util.concatenate(meshes)
            
            # Eksporter til USD
            output_path = os.path.join(self.temp_dir, "building.usda")
            self._export_to_usd(combined_mesh, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Feil ved generering av 3D-modell: {str(e)}")
            return None
    
    def _extract_room_contours(self, floor_plan_path: str) -> List[np.ndarray]:
        """
        Ekstraherer romkonturer fra plantegning
        """
        # Last inn bilde
        image = o3d.io.read_image(floor_plan_path)
        
        # Konverter til gråtoner og finn konturer
        gray = np.asarray(image)
        if len(gray.shape) > 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        
        # Finn konturer
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrer og prosesser konturer
        valid_contours = []
        min_area = 1000
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                # Forenkle kontur
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                valid_contours.append(approx)
        
        return valid_contours
    
    def _create_room_mesh(self, contour: np.ndarray, height: float) -> trimesh.Trimesh:
        """
        Oppretter 3D mesh for et rom
        """
        # Konverter kontur til 2D punkter
        points_2d = contour.squeeze()
        
        # Opprett gulv og tak
        floor_points = np.hstack([points_2d, np.zeros((len(points_2d), 1))])
        ceiling_points = np.hstack([points_2d, np.full((len(points_2d), 1), height)])
        
        # Opprett vegger
        wall_vertices = []
        wall_faces = []
        
        for i in range(len(points_2d)):
            next_i = (i + 1) % len(points_2d)
            
            # Legg til veggpunkter
            base_idx = len(wall_vertices)
            wall_vertices.extend([
                floor_points[i],
                floor_points[next_i],
                ceiling_points[next_i],
                ceiling_points[i]
            ])
            
            # Legg til veggflater
            wall_faces.extend([
                [base_idx, base_idx + 1, base_idx + 2],
                [base_idx, base_idx + 2, base_idx + 3]
            ])
        
        # Kombiner alle vertices og faces
        vertices = np.vstack([floor_points, ceiling_points, wall_vertices])
        
        # Opprett faces for gulv og tak
        floor_faces = [[i for i in range(len(points_2d))]]
        ceiling_faces = [[i + len(points_2d) for i in range(len(points_2d))]]
        
        # Kombiner alle faces
        faces = np.vstack([floor_faces, ceiling_faces, wall_faces])
        
        # Opprett mesh
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _export_to_usd(self, mesh: trimesh.Trimesh, output_path: str):
        """
        Eksporterer 3D-modell til USD-format
        """
        # Opprett ny USD-stage
        stage = Usd.Stage.CreateNew(output_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        
        # Opprett mesh i USD
        mesh_path = "/World/Building"
        usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        
        # Sett mesh-data
        usd_mesh.CreatePointsAttr(mesh.vertices)
        usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.flatten())
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        
        # Legg til materialer
        self._add_materials(stage, mesh_path)
        
        # Lagre USD-fil
        stage.Save()
    
    def _add_materials(self, stage: Usd.Stage, mesh_path: str):
        """
        Legger til standard materialer
        """
        # Opprett material
        material_path = "/World/Building/Material"
        material = UsdShade.Material.Define(stage, material_path)
        
        # Opprett PBR shader
        pbr = UsdShade.Shader.Define(stage, material_path + "/PBRShader")
        pbr.CreateIdAttr("UsdPreviewSurface")
        
        # Sett standardverdier
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.8, 0.8))
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        
        # Koble material til mesh
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(mesh_path)).Bind(material)
    
    async def create_visualization(
        self,
        model_path: str,
        output_path: str,
        camera_settings: Optional[Dict] = None
    ) -> str:
        """
        Oppretter en visualisering med Omniverse
        """
        try:
            # Last inn 3D-modell
            stage = Usd.Stage.Open(model_path)
            
            # Sett opp kamera
            if camera_settings:
                self._setup_camera(stage, camera_settings)
            else:
                self._setup_default_camera(stage)
            
            # Render scene
            self._render_scene(stage, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Feil ved visualisering: {str(e)}")
            return None
    
    def _setup_default_camera(self, stage: Usd.Stage):
        """
        Setter opp standard kamera
        """
        camera_path = "/World/Camera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        
        # Sett kameraposisjon og orientering
        camera.AddTranslateOp().Set(Gf.Vec3d(0, 5, 10))
        camera.AddRotateXYZOp().Set(Gf.Vec3d(-30, 0, 0))
        
        # Sett kameraegenskaper
        camera.CreateFocalLengthAttr(50)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(24)
    
    def _render_scene(self, stage: Usd.Stage, output_path: str):
        """
        Rendrer scenen med Omniverse
        """
        # Sett opp render-innstillinger
        render_settings = {
            "resolution": (1920, 1080),
            "samples_per_pixel": 64,
            "denoising": True,
            "quality": "production"
        }
        
        # Start rendering
        omni.kit.commands.execute(
            'RenderCommand',
            stage=stage,
            output_path=output_path,
            settings=render_settings
        )
    
    def cleanup(self):
        """
        Rydder opp temporære filer
        """
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)