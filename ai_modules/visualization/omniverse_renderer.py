import omni.kit.commands
import omni.usd
import omni.renderer
import omni.physx
import omni.timeline
import omni.kit.viewport
import omni.graph.core as og
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Gf, Sdf, Vt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import logging
import asyncio
import json
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid
import base64

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RenderingConfig:
    """Konfigurasjon for rendering med avanserte innstillinger"""
    def __init__(self, **kwargs):
        # Renderingskvalitet
        self.render_quality = kwargs.get("render_quality", "high")
        self.raytracing_enabled = kwargs.get("raytracing_enabled", True)
        self.samples_per_pixel = kwargs.get("samples_per_pixel", 1024)
        self.max_bounces = kwargs.get("max_bounces", 8)
        self.denoising_enabled = kwargs.get("denoising_enabled", True)
        self.caustics_enabled = kwargs.get("caustics_enabled", True)
        
        # Oppløsning
        resolution = kwargs.get("resolution", {"width": 1920, "height": 1080})
        self.width = resolution.get("width", 1920)
        self.height = resolution.get("height", 1080)
        
        # Avanserte materialinnstillinger
        self.subsurface_scattering = kwargs.get("subsurface_scattering", True)
        self.thin_film_interference = kwargs.get("thin_film_interference", True)
        self.material_quality = kwargs.get("material_quality", "physically_based")
        
        # Belysningsinnstillinger
        self.lighting_quality = kwargs.get("lighting_quality", "dynamic")
        self.ambient_occlusion = kwargs.get("ambient_occlusion", True)
        self.indirect_lighting_quality = kwargs.get("indirect_lighting_quality", "high")
        self.environment_lighting = kwargs.get("environment_lighting", True)
        
        # Kamerai nnstillinger
        self.camera_type = kwargs.get("camera_type", "perspective")
        self.depth_of_field = kwargs.get("depth_of_field", True)
        self.motion_blur = kwargs.get("motion_blur", False)
        
        # Ytelsesrelaterte innstillinger
        self.gpu_acceleration = kwargs.get("gpu_acceleration", True)
        self.multi_gpu = kwargs.get("multi_gpu", True)
        self.caching_enabled = kwargs.get("caching_enabled", True)
        self.max_memory_usage_gb = kwargs.get("max_memory_usage_gb", 16)
        
        # Utdatainnstillinger
        self.output_formats = kwargs.get("output_formats", ["png", "usd", "glb"])
        self.animation_enabled = kwargs.get("animation_enabled", False)
        self.animation_frames = kwargs.get("animation_frames", 0)
        self.animation_fps = kwargs.get("animation_fps", 30)
        
        # BIM-relaterte innstillinger
        self.bim_metadata_enabled = kwargs.get("bim_metadata_enabled", True)
        self.ifc_compatibility = kwargs.get("ifc_compatibility", True)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'RenderingConfig':
        """Laster konfigurasjon fra JSON-fil"""
        if not os.path.exists(json_path):
            logger.warning(f"Config file {json_path} not found. Using default settings.")
            return cls()
        
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Error loading config from {json_path}: {str(e)}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Konverterer konfigurasjon til ordbok"""
        return {k: v for k, v in self.__dict__.items()}
    
    def to_json(self, json_path: str):
        """Lagrer konfigurasjon til JSON-fil"""
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config to {json_path}: {str(e)}")

class MaterialLibrary:
    """Bibliotek for materialer med avanserte egenskaper"""
    def __init__(self):
        self.materials = {}
        self._initialize_standard_materials()
    
    def _initialize_standard_materials(self):
        """Initialiser standard materialbibliotek"""
        # Byggematerialer
        self.add_material("concrete", {
            "type": "physically_based",
            "base_color": (0.7, 0.7, 0.7),
            "roughness": 0.9,
            "metallic": 0.0,
            "normal_strength": 1.0,
            "displacement_strength": 0.05,
            "textures": {
                "albedo": "textures/concrete_albedo.png",
                "roughness": "textures/concrete_roughness.png",
                "normal": "textures/concrete_normal.png",
                "displacement": "textures/concrete_displacement.png"
            }
        })
        
        self.add_material("wood_oak", {
            "type": "physically_based",
            "base_color": (0.6, 0.4, 0.2),
            "roughness": 0.7,
            "metallic": 0.0,
            "subsurface_scattering": 0.1,
            "textures": {
                "albedo": "textures/wood_oak_albedo.png",
                "roughness": "textures/wood_oak_roughness.png",
                "normal": "textures/wood_oak_normal.png"
            }
        })
        
        self.add_material("marble", {
            "type": "physically_based",
            "base_color": (0.9, 0.9, 0.9),
            "roughness": 0.2,
            "metallic": 0.0,
            "subsurface_scattering": 0.3,
            "textures": {
                "albedo": "textures/marble_albedo.png",
                "roughness": "textures/marble_roughness.png",
                "normal": "textures/marble_normal.png"
            }
        })
        
        # Glass og metaller
        self.add_material("glass_clear", {
            "type": "glass",
            "base_color": (1.0, 1.0, 1.0),
            "roughness": 0.05,
            "ior": 1.52,
            "transmission": 0.95,
            "thin_walled": False
        })
        
        self.add_material("aluminum", {
            "type": "metal",
            "base_color": (0.9, 0.9, 0.9),
            "roughness": 0.2,
            "metallic": 1.0,
            "anisotropy": 0.0
        })
        
        # Overflatebehandlinger
        self.add_material("paint_white", {
            "type": "physically_based",
            "base_color": (0.95, 0.95, 0.95),
            "roughness": 0.9,
            "metallic": 0.0
        })
        
        self.add_material("paint_eggshell", {
            "type": "physically_based",
            "base_color": (0.95, 0.95, 0.9),
            "roughness": 0.7,
            "metallic": 0.0
        })
        
        # Utendørsmaterialer
        self.add_material("grass", {
            "type": "physically_based",
            "base_color": (0.3, 0.5, 0.2),
            "roughness": 1.0,
            "metallic": 0.0,
            "subsurface_scattering": 0.2,
            "textures": {
                "albedo": "textures/grass_albedo.png",
                "roughness": "textures/grass_roughness.png",
                "normal": "textures/grass_normal.png"
            }
        })
        
        self.add_material("water", {
            "type": "physically_based",
            "base_color": (0.0, 0.1, 0.2),
            "roughness": 0.05,
            "metallic": 0.0,
            "transmission": 0.95,
            "ior": 1.33
        })
    
    def add_material(self, name: str, properties: Dict[str, Any]):
        """Legg til materiale i biblioteket"""
        self.materials[name] = properties
    
    def get_material(self, name: str) -> Dict[str, Any]:
        """Hent materiale fra biblioteket"""
        if name not in self.materials:
            logger.warning(f"Material '{name}' not found in library. Using default.")
            return self.materials.get("paint_white", {})
        return self.materials[name]
    
    def create_usd_material(self, stage: Usd.Stage, name: str, properties: Dict[str, Any]) -> UsdShade.Material:
        """Opprett USD-materiale fra egenskaper"""
        material_path = f"/Materials/{name}"
        material = UsdShade.Material.Define(stage, material_path)
        
        # Opprett PBR shader
        pbr_shader = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
        pbr_shader.CreateIdAttr("UsdPreviewSurface")
        
        # Sett basis shader-egenskaper
        pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*properties.get("base_color", (0.8, 0.8, 0.8))))
        pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(properties.get("roughness", 0.5))
        pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(properties.get("metallic", 0.0))
        
        # Sett avanserte egenskaper basert på materialtype
        if properties.get("type") == "glass":
            pbr_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.1)
            pbr_shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(properties.get("ior", 1.5))
            pbr_shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
            pbr_shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(properties.get("roughness", 0.05))
            
        elif properties.get("type") == "metal":
            pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
            if "anisotropy" in properties:
                pbr_shader.CreateInput("specularAnisotropy", Sdf.ValueTypeNames.Float).Set(properties.get("anisotropy"))
        
        # Håndter teksturer
        textures = properties.get("textures", {})
        for tex_type, tex_path in textures.items():
            self._add_texture_to_shader(stage, pbr_shader, material_path, tex_type, tex_path)
        
        # Koble shader til material
        material.CreateSurfaceOutput().ConnectToSource(pbr_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        
        return material
    
    def _add_texture_to_shader(self, stage: Usd.Stage, shader: UsdShade.Shader, material_path: str, 
                              tex_type: str, tex_path: str):
        """Legg til tekstur til shader"""
        # Opprett teksturnodeordbok
        texture_inputs = {
            "albedo": {"shader_input": "diffuseColor", "type": Sdf.ValueTypeNames.Color3f, "wrap": "repeat", "scale": (1, 1)},
            "roughness": {"shader_input": "roughness", "type": Sdf.ValueTypeNames.Float, "wrap": "repeat", "scale": (1, 1)},
            "normal": {"shader_input": "normal", "type": Sdf.ValueTypeNames.Normal3f, "wrap": "repeat", "scale": (1, 1)},
            "metallic": {"shader_input": "metallic", "type": Sdf.ValueTypeNames.Float, "wrap": "repeat", "scale": (1, 1)},
            "displacement": {"shader_input": "displacement", "type": Sdf.ValueTypeNames.Float, "wrap": "repeat", "scale": (1, 1)},
            "occlusion": {"shader_input": "occlusion", "type": Sdf.ValueTypeNames.Float, "wrap": "repeat", "scale": (1, 1)},
            "emissive": {"shader_input": "emissiveColor", "type": Sdf.ValueTypeNames.Color3f, "wrap": "repeat", "scale": (1, 1)}
        }
        
        if tex_type not in texture_inputs:
            logger.warning(f"Unknown texture type '{tex_type}'. Skipping.")
            return
        
        # Opprett teksturnode
        tex_info = texture_inputs[tex_type]
        tex_shader = UsdShade.Shader.Define(stage, f"{material_path}/Texture_{tex_type}")
        tex_shader.CreateIdAttr("UsdUVTexture")
        
        # Sett teksturfilsti
        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
        
        # Sett wrapping og scaling
        tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(tex_info["wrap"])
        tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(tex_info["wrap"])
        tex_shader.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*tex_info["scale"]))
        
        # For normalmap, legg til normalmap-shader
        if tex_type == "normal":
            normal_reader = UsdShade.Shader.Define(stage, f"{material_path}/NormalMap")
            normal_reader.CreateIdAttr("UsdTransform2d")
            
            # Koble teksturoutput til normalmap-reader
            tex_output = tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            normal_input = normal_reader.CreateInput("in", Sdf.ValueTypeNames.Float3)
            normal_input.ConnectToSource(tex_output)
            
            # Koble normalmap til shader
            normal_output = normal_reader.CreateOutput("normal", Sdf.ValueTypeNames.Normal3f)
            shader.CreateInput(tex_info["shader_input"], tex_info["type"]).ConnectToSource(normal_output)
        else:
            # Koble tekstur direkte til shader for andre teksturtyper
            output_name = "r" if tex_info["type"] == Sdf.ValueTypeNames.Float else "rgb"
            tex_output = tex_shader.CreateOutput(output_name, tex_info["type"])
            shader.CreateInput(tex_info["shader_input"], tex_info["type"]).ConnectToSource(tex_output)

class OmniverseRenderer:
    """
    Avansert klasse for å håndtere 3D-visualisering med NVIDIA Omniverse.
    Støtter:
    - Fotorealistisk rendering
    - Materiale-simulering med PBR
    - Fysikkbasert belysning og skyggeberegning
    - BIM-integrasjon med metadata
    - USD-pipeline og eksport til flere formater
    - RTX ray-tracing for realtidsvisualisering
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = RenderingConfig.from_json(config_path) if config_path else RenderingConfig()
        self.stage = None
        self.renderer = None
        self.material_library = MaterialLibrary()
        self.timeline = None
        self.viewport = None
        self.camera = None
        
        # Caching for optimal ytelse
        self.asset_cache = {}
        self.render_cache = {}
        
        # Statistikk og diagnostikk
        self.stats = {
            "model_complexity": {},
            "render_times": [],
            "memory_usage": []
        }
        
        # Initialiser Omniverse-miljø
        self._initialize_omniverse()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Last konfigurasjon for Omniverse"""
        if not config_path or not os.path.exists(config_path):
            logger.info("Using default renderer configuration")
            return RenderingConfig().to_dict()
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return RenderingConfig().to_dict()
    
    def _initialize_omniverse(self):
        """Initialiser Omniverse og sett opp rendering-miljø"""
        try:
            # Initialiser Omniverse Kit
            logger.info("Initializing Omniverse environment")
            omni.kit.commands.execute('Create New Stage')
            
            # Hent aktiv stage
            self.stage = omni.usd.get_context().get_stage()
            
            # Sett opp renderer
            self.renderer = omni.renderer.create()
            self._configure_renderer()
            
            # Sett opp timeline for animasjon
            self.timeline = omni.timeline.get_timeline_interface()
            
            # Sett opp viewport
            self.viewport = omni.kit.viewport.get_viewport_interface()
            
            # Sett opp physics engine
            self._setup_physics()
            
            # Sett opp standard kamera
            self._setup_default_camera()
            
            logger.info("Omniverse environment initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Omniverse: {str(e)}")
            raise
    
    def _configure_renderer(self):
        """Konfigurer renderingsmotoren med avanserte innstillinger"""
        logger.info(f"Configuring renderer with quality: {self.config.render_quality}")
        
        # Konfigurer grunnleggende renderingsinnstillinger
        if self.config.raytracing_enabled:
            self.renderer.set_raytracing_enabled(True)
            self.renderer.set_samples_per_pixel(self.config.samples_per_pixel)
            self.renderer.set_max_bounces(self.config.max_bounces)
        
        # Konfigurer avanserte renderingsinnstillinger
        render_settings = {
            "rtx": {
                "pathtracing": {
                    "enabled": True,
                    "spp": self.config.samples_per_pixel,
                    "maxBounces": self.config.max_bounces,
                    "denoise": self.config.denoising_enabled,
                    "caustics": self.config.caustics_enabled
                },
                "ambient": {
                    "enabled": self.config.ambient_occlusion,
                    "intensity": 1.0,
                    "quality": 1.0 if self.config.render_quality == "high" else 0.5
                },
                "raytracing": {
                    "enabled": self.config.raytracing_enabled,
                    "reflections": True,
                    "shadows": True,
                    "transparencies": True
                }
            },
            "renderer": {
                "resolution": {
                    "width": self.config.width,
                    "height": self.config.height
                },
                "quality": {
                    "level": self.config.render_quality,
                    "subsurfaceScattering": self.config.subsurface_scattering,
                    "thinFilm": self.config.thin_film_interference,
                    "motionBlur": self.config.motion_blur,
                    "depthOfField": self.config.depth_of_field
                },
                "performance": {
                    "gpuAcceleration": self.config.gpu_acceleration,
                    "multiGpu": self.config.multi_gpu,
                    "caching": self.config.caching_enabled,
                    "maxMemoryUsageGb": self.config.max_memory_usage_gb
                }
            }
        }
        
        # Bruk omni.kit.commands for å angi renderer-innstillinger
        omni.kit.commands.execute('SetRenderSettings', settings=render_settings)
        
        logger.info("Renderer configured successfully")
    
    def _setup_physics(self):
        """Sett opp fysikkmotor for simulering"""
        if not hasattr(omni, 'physx'):
            logger.warning("PhysX module not available, skipping physics setup")
            return
        
        try:
            # Opprett physics scene
            scene = UsdPhysics.Scene.Define(self.stage, '/World/PhysicsScene')
            
            # Sett opp gravitasjon
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)  # Standard gravitasjonskonstant
            
            # Aktiver fysikksimulering
            omni.physx.get_physx_interface().enable_physics()
            
            logger.info("Physics engine initialized")
        except Exception as e:
            logger.error(f"Error setting up physics: {str(e)}")
    
    def _setup_default_camera(self):
        """Sett opp standard renderingskamera"""
        try:
            # Opprett kamera
            camera_path = '/World/Camera'
            camera_prim = self.stage.DefinePrim(camera_path, 'Camera')
            self.camera = UsdGeom.Camera(camera_prim)
            
            # Sett plassering og orientering
            xform = UsdGeom.Xformable(camera_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(0, 1.7, 5))
            xform.AddRotateYOp().Set(180)
            
            # Sett kameraegenskaper
            if self.config.camera_type == "perspective":
                self.camera.CreateProjectionAttr('perspective')
                self.camera.CreateFocalLengthAttr(35)  # 35mm fokallengde
                self.camera.CreateFStopAttr(5.6)
                self.camera.CreateFocusDistanceAttr(5)
            else:
                self.camera.CreateProjectionAttr('orthographic')
                self.camera.CreateHorizontalApertureAttr(10)  # Ortografisk bredde
            
            # Aktiver dybdeskarphet hvis konfigurert
            if self.config.depth_of_field:
                self.camera.CreateDepthOfFieldEnabledAttr(True)
            
            # Sett kamera aktivt i viewport
            self.viewport.set_active_camera(camera_path)
            
            logger.info("Default camera set up at position (0, 1.7, 5)")
        except Exception as e:
            logger.error(f"Error setting up default camera: {str(e)}")
    
    async def generate_3d_model(self, 
                              floor_plan_data: Dict[str, Any],
                              structural_data: Dict[str, Any],
                              materials_data: Dict[str, Any],
                              **kwargs) -> Dict[str, Any]:
        """
        Generer en fotorealistisk 3D-modell basert på plantegning og strukturdata med forbedret pipeline
        
        Args:
            floor_plan_data: Data for plantegninger og etasjer
            structural_data: Data for strukturelle elementer
            materials_data: Data for materialer og teksturer
            **kwargs: Tilleggsparametre for rendering
        
        Returns:
            Dict med modell-URL, forhåndsvisningsbilder og metadata
        """
        try:
            # Generer unik modell-ID for sporing
            model_id = self._generate_model_id(floor_plan_data, structural_data)
            
            # Sjekk om modellen er i cache
            if self.config.caching_enabled and model_id in self.render_cache:
                logger.info(f"Using cached model {model_id}")
                return self.render_cache[model_id]
            
            logger.info(f"Generating 3D model with ID: {model_id}")
            
            # Opprett grunnleggende geometri asynkront
            await self._create_base_geometry(floor_plan_data)
            
            # Legg til strukturelle elementer parallelt
            await self._add_structural_elements(structural_data)
            
            # Appliser materialer med forbedret shader-støtte
            await self._apply_materials(materials_data)
            
            # Sett opp realistisk belysning
            await self._setup_lighting(floor_plan_data.get("lighting", {}))
            
            # Legg til miljø og omgivelser
            await self._add_environment(kwargs.get("environment", {}))
            
            # Legg til BIM-metadata hvis aktivert
            if self.config.bim_metadata_enabled:
                self._add_bim_metadata(floor_plan_data, structural_data)
            
            # Optimaliser modell for rendering
            self._optimize_model()
            
            # Render modellen med høy kvalitet
            render_result = await self._render_model(
                view_positions=kwargs.get("camera_positions"),
                output_path=kwargs.get("output_path")
            )
            
            # Samle og beregn modellstatistikk
            stats = self._collect_model_statistics()
            
            # Opprett resultatpakke
            result = {
                "model_id": model_id,
                "model_url": render_result.get("model_url"),
                "preview_images": render_result.get("previews", []),
                "metadata": self._generate_metadata(stats),
                "statistics": stats
            }
            
            # Lagre i cache for senere bruk
            if self.config.caching_enabled:
                self.render_cache[model_id] = result
            
            logger.info(f"3D model generation completed for {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error during 3D model generation: {str(e)}")
            raise
    
    def _generate_model_id(self, floor_plan_data: Dict[str, Any], structural_data: Dict[str, Any]) -> str:
        """Generer unik ID for modellen basert på input data"""
        # Opprett en hash av de viktigste attributtene for å identifisere modellen
        model_hash = hashlib.md5()
        
        # Legg til floor plan data
        if "floors" in floor_plan_data:
            model_hash.update(str(len(floor_plan_data["floors"])).encode())
        
        # Legg til structural data
        if "walls" in structural_data:
            model_hash.update(str(len(structural_data["walls"])).encode())
        
        # Legg til timestamp
        model_hash.update(str(time.time()).encode())
        
        # Returner hash som base64-streng
        return base64.urlsafe_b64encode(model_hash.digest()).decode()[:12]
    
    async def _create_base_geometry(self, floor_plan_data: Dict[str, Any]):
        """Opprett grunnleggende geometri fra plantegning med forbedret geometribehandling"""
        try:
            logger.info("Creating base geometry from floor plan")
            
            # Opprett hierarki for bygningsgeometri
            building_prim = self.stage.DefinePrim("/World/Building", "Xform")
            
            # Opprett koordinatsystemer og referanseplan
            self._create_reference_planes(building_prim.GetPath())
            
            # Opprett terreng hvis tilgjengelig
            if "terrain" in floor_plan_data:
                await self._create_terrain(floor_plan_data["terrain"], building_prim.GetPath())
            
            # Opprett etasjer
            floor_prims = []
            for i, floor in enumerate(floor_plan_data.get("floors", [])):
                floor_prim = await self._create_floor(floor, building_prim.GetPath(), i)
                floor_prims.append(floor_prim)
            
            # Opprett vegger for hver etasje
            wall_tasks = []
            for floor_idx, floor_prim in enumerate(floor_prims):
                floor = floor_plan_data.get("floors", [])[floor_idx]
                for wall in floor.get("walls", []):
                    wall_tasks.append(self._create_wall(wall, floor_prim.GetPath(), floor_idx))
            
            # Vent på at alle veggene skal bli opprettet
            await asyncio.gather(*wall_tasks)
            
            # Opprett tak
            if "roof" in floor_plan_data:
                await self._create_roof(floor_plan_data["roof"], building_prim.GetPath())
            
            logger.info(f"Base geometry created with {len(floor_prims)} floors")
            
        except Exception as e:
            logger.error(f"Error creating base geometry: {str(e)}")
            raise
    
    def _create_reference_planes(self, parent_path: Sdf.Path):
        """Opprett referanseplan for modellen"""
        ground_plane = self.stage.DefinePrim(f"{parent_path}/RefPlanes/Ground", "Plane")
        
        # Sett størrelse til 50x50 meter
        UsdGeom.Plane(ground_plane).CreateSizeAttr(50)
        
        # Juster plassering
        xform = UsdGeom.Xformable(ground_plane)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xform.AddRotateXOp().Set(90)  # Rotere for å være horisontalt
        
        # Gjør planet usynlig i rendering, men synlig i viewport
        UsdGeom.Imageable(ground_plane).CreatePurposeAttr("guide")
    
    async def _create_terrain(self, terrain_data: Dict[str, Any], parent_path: Sdf.Path) -> UsdGeom.Mesh:
        """Opprett terreng basert på høydedata"""
        terrain_prim_path = f"{parent_path}/Terrain"
        
        # Opprett terrengmesh
        terrain_prim = self.stage.DefinePrim(terrain_prim_path, "Mesh")
        terrain_mesh = UsdGeom.Mesh(terrain_prim)
        
        # Hent terrengdata
        width = terrain_data.get("width", 50)
        length = terrain_data.get("length", 50)
        resolution = terrain_data.get("resolution", 20)
        height_map = terrain_data.get("height_map", [])
        
        # Hvis høydekart ikke er gitt, generer flatt terreng
        if not height_map:
            # Opprett et flatt terreng med lett tekstur
            vertices, faces, uvs = self._generate_flat_terrain(width, length, resolution)
        else:
            # Opprett terreng basert på høydekart
            vertices, faces, uvs = self._generate_terrain_from_height_map(
                width, length, resolution, height_map
            )
        
        # Sett vertekser og ansikter
        terrain_mesh.CreatePointsAttr(vertices)
        terrain_mesh.CreateFaceVertexIndicesAttr(faces)
        terrain_mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 3))  # Trekanter
        
        # Sett UV-koordinater
        terrain_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        # Beregn normaler
        normals = self._calculate_mesh_normals(vertices, faces)
        terrain_mesh.CreateNormalsAttr(normals)
        
        # Sett extent
        min_point = Gf.Vec3f(min(v[0] for v in vertices), min(v[1] for v in vertices), min(v[2] for v in vertices))
        max_point = Gf.Vec3f(max(v[0] for v in vertices), max(v[1] for v in vertices), max(v[2] for v in vertices))
        terrain_mesh.CreateExtentAttr([min_point, max_point])
        
        return terrain_mesh
    
    def _generate_flat_terrain(self, width: float, length: float, resolution: int) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer et flatt terreng"""
        vertices = []
        faces = []
        uvs = []
        
        # Opprett vertekser i et grid
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = (i / resolution - 0.5) * width
                z = (j / resolution - 0.5) * length
                y = 0.0  # Flat terreng
                
                vertices.append(Gf.Vec3f(x, y, z))
                uvs.append(Gf.Vec2f(i / resolution, j / resolution))
        
        # Opprett trekangler
        for i in range(resolution):
            for j in range(resolution):
                v0 = i * (resolution + 1) + j
                v1 = v0 + 1
                v2 = v0 + (resolution + 1)
                v3 = v2 + 1
                
                # Første trekant
                faces.extend([v0, v1, v2])
                
                # Andre trekant
                faces.extend([v2, v1, v3])
        
        return vertices, faces, uvs
    
    def _generate_terrain_from_height_map(self, width: float, length: float, resolution: int, 
                                         height_map: List[float]) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer terreng fra høydekart"""
        vertices = []
        faces = []
        uvs = []
        
        # Sjekk at høydekartet har riktig størrelse
        expected_size = (resolution + 1) * (resolution + 1)
        if len(height_map) != expected_size:
            logger.warning(f"Height map size mismatch. Expected {expected_size}, got {len(height_map)}. Resizing.")
            if len(height_map) > expected_size:
                height_map = height_map[:expected_size]
            else:
                height_map.extend([0.0] * (expected_size - len(height_map)))
        
        # Opprett vertekser i et grid
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = (i / resolution - 0.5) * width
                z = (j / resolution - 0.5) * length
                idx = i * (resolution + 1) + j
                y = height_map[idx] if idx < len(height_map) else 0.0
                
                vertices.append(Gf.Vec3f(x, y, z))
                uvs.append(Gf.Vec2f(i / resolution, j / resolution))
        
        # Opprett trekangler
        for i in range(resolution):
            for j in range(resolution):
                v0 = i * (resolution + 1) + j
                v1 = v0 + 1
                v2 = v0 + (resolution + 1)
                v3 = v2 + 1
                
                # Første trekant
                faces.extend([v0, v1, v2])
                
                # Andre trekant
                faces.extend([v2, v1, v3])
        
        return vertices, faces, uvs
    
    def _calculate_mesh_normals(self, vertices: List[Gf.Vec3f], faces: List[int]) -> List[Gf.Vec3f]:
        """Beregn normaler for et mesh basert på vertekser og ansikter"""
        normals = [Gf.Vec3f(0, 0, 0)] * len(vertices)
        
        # Beregn normaler for hver trekant og akkumuler
        for i in range(0, len(faces), 3):
            if i + 2 < len(faces):
                v0_idx = faces[i]
                v1_idx = faces[i + 1]
                v2_idx = faces[i + 2]
                
                if all(idx < len(vertices) for idx in [v0_idx, v1_idx, v2_idx]):
                    v0 = vertices[v0_idx]
                    v1 = vertices[v1_idx]
                    v2 = vertices[v2_idx]
                    
                    # Beregn trekantnormal med kryssprodukt
                    edge1 = Gf.Vec3f(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
                    edge2 = Gf.Vec3f(v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
                    normal = Gf.Vec3f.Cross(edge1, edge2)
                    
                    # Normaliser hvis mulig
                    length = normal.GetLength()
                    if length > 0.0001:
                        normal /= length
                    
                    # Akkumuler normalen til hver verteks
                    normals[v0_idx] += normal
                    normals[v1_idx] += normal
                    normals[v2_idx] += normal
        
        # Normaliser akkumulerte normaler
        for i in range(len(normals)):
            length = normals[i].GetLength()
            if length > 0.0001:
                normals[i] /= length
            else:
                normals[i] = Gf.Vec3f(0, 1, 0)  # Standard opp-normal
        
        return normals
    
    async def _create_floor(self, floor_data: Dict[str, Any], parent_path: Sdf.Path, floor_index: int) -> UsdGeom.Xform:
        """Opprett en etasje med gulv og struktur"""
        floor_height = floor_data.get("height", 3.0)  # Standard etasjehøyde i meter
        elevation = floor_data.get("elevation", floor_index * floor_height)
        
        # Opprett etasjeprim
        floor_prim_path = f"{parent_path}/Floor_{floor_index}"
        floor_prim = self.stage.DefinePrim(floor_prim_path, "Xform")
        
        # Plasser etasjen på riktig høyde
        xform = UsdGeom.Xformable(floor_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, elevation, 0))
        
        # Opprett gulvplate
        floor_slab = self.stage.DefinePrim(f"{floor_prim_path}/Slab", "Mesh")
        slab_mesh = UsdGeom.Mesh(floor_slab)
        
        # Hent etasjegeometri
        outline = floor_data.get("outline", [[0, 0], [10, 0], [10, 10], [0, 10]])
        
        # Opprett vertekser
        vertices = []
        for point in outline:
            vertices.append(Gf.Vec3f(point[0], 0, point[1]))  # Gulvets Y er 0 i lokal etasje-koordinat
        
        # Triangulering av polygon (forenklet)
        faces = self._triangulate_polygon(vertices)
        
        # Sett mesh-data
        slab_mesh.CreatePointsAttr(vertices)
        slab_mesh.CreateFaceVertexIndicesAttr(faces)
        slab_mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 3))  # Trekanter
        
        # Opprett etasje-metadata
        UsdGeom.Xform(floor_prim).GetPrim().CreateAttribute(
            "floor:index", Sdf.ValueTypeNames.Int).Set(floor_index)
        UsdGeom.Xform(floor_prim).GetPrim().CreateAttribute(
            "floor:elevation", Sdf.ValueTypeNames.Float).Set(elevation)
        UsdGeom.Xform(floor_prim).GetPrim().CreateAttribute(
            "floor:height", Sdf.ValueTypeNames.Float).Set(floor_height)
        
        # Opprett rom hvis definert
        room_tasks = []
        for room in floor_data.get("rooms", []):
            room_tasks.append(self._create_room(room, floor_prim.GetPath()))
        
        if room_tasks:
            await asyncio.gather(*room_tasks)
        
        return UsdGeom.Xform(floor_prim)
    
    def _triangulate_polygon(self, vertices: List[Gf.Vec3f]) -> List[int]:
        """Triangulerer et polygon (forenklet implementasjon for konvekse polygon)"""
        # For ikke-konvekse polygon bør en mer robust algoritme brukes
        faces = []
        n = len(vertices)
        
        if n < 3:
            return faces
        
        # Enkel vifte-triangulering fra første punkt
        for i in range(1, n - 1):
            faces.extend([0, i, i + 1])
        
        return faces
    
    async def _create_room(self, room_data: Dict[str, Any], parent_path: Sdf.Path) -> UsdGeom.Xform:
        """Opprett et rom med gulv, vegger og metadata"""
        room_id = room_data.get("id", "room")
        room_name = room_data.get("name", f"Room_{room_id}")
        
        # Opprett rom-prim
        room_prim_path = f"{parent_path}/Rooms/{room_name}"
        room_prim = self.stage.DefinePrim(room_prim_path, "Xform")
        
        # Sett metadata
        room_prim.GetPrim().CreateAttribute("room:id", Sdf.ValueTypeNames.String).Set(room_id)
        room_prim.GetPrim().CreateAttribute("room:type", Sdf.ValueTypeNames.String).Set(
            room_data.get("type", "generic"))
        room_prim.GetPrim().CreateAttribute("room:area", Sdf.ValueTypeNames.Float).Set(
            room_data.get("area", 0.0))
        
        # Opprett gulv for rommet
        floor_mesh = self.stage.DefinePrim(f"{room_prim_path}/Floor", "Mesh")
        
        # Hent gulvgeometri
        outline = room_data.get("outline", [[0, 0], [4, 0], [4, 4], [0, 4]])
        
        # Opprett vertekser
        vertices = []
        for point in outline:
            vertices.append(Gf.Vec3f(point[0], 0, point[1]))
        
        # Triangulering
        faces = self._triangulate_polygon(vertices)
        
        # Sett mesh-data
        UsdGeom.Mesh(floor_mesh).CreatePointsAttr(vertices)
        UsdGeom.Mesh(floor_mesh).CreateFaceVertexIndicesAttr(faces)
        UsdGeom.Mesh(floor_mesh).CreateFaceVertexCountsAttr([3] * (len(faces) // 3))
        
        # Beregn UV-koordinater
        uvs = self._generate_uvs_for_floor(vertices)
        UsdGeom.Mesh(floor_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        return UsdGeom.Xform(room_prim)
    
    def _generate_uvs_for_floor(self, vertices: List[Gf.Vec3f]) -> List[Gf.Vec2f]:
        """Generer UV-koordinater for et gulv"""
        # Finn min/max X og Z koordinater
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_z = min(v[2] for v in vertices)
        max_z = max(v[2] for v in vertices)
        
        # Bredde og dybde
        width = max_x - min_x
        depth = max_z - min_z
        
        if width < 0.001 or depth < 0.001:
            return [Gf.Vec2f(0, 0)] * len(vertices)
        
        # Opprett UV-koordinater, skalert for å gjenta mønsteret hver 2 meter
        uvs = []
        for v in vertices:
            u = (v[0] - min_x) / 2.0  # Repetering hver 2 meter i X-retning
            v = (v[2] - min_z) / 2.0  # Repetering hver 2 meter i Z-retning
            uvs.append(Gf.Vec2f(u, v))
        
        return uvs
    
    async def _create_wall(self, wall_data: Dict[str, Any], parent_path: Sdf.Path, floor_index: int) -> UsdGeom.Mesh:
        """Opprett en vegg mellom to punkter"""
        start = wall_data.get("start", [0, 0])
        end = wall_data.get("end", [5, 0])
        height = wall_data.get("height", 2.7)  # Standard romhøyde
        thickness = wall_data.get("thickness", 0.2)  # Standard veggtykkelse
        
        # Opprett vegg-prim
        wall_id = wall_data.get("id", f"wall_{floor_index}_{len(start)}_{len(end)}")
        wall_prim_path = f"{parent_path}/Walls/{wall_id}"
        wall_prim = self.stage.DefinePrim(wall_prim_path, "Mesh")
        
        # Beregn vegggeometri
        vertices, faces, uvs = self._generate_wall_geometry(start, end, height, thickness)
        
        # Sett mesh-data
        wall_mesh = UsdGeom.Mesh(wall_prim)
        wall_mesh.CreatePointsAttr(vertices)
        wall_mesh.CreateFaceVertexIndicesAttr(faces)
        wall_mesh.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))  # Kvadrater
        wall_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        # Sett metadata
        wall_prim.GetPrim().CreateAttribute("wall:id", Sdf.ValueTypeNames.String).Set(wall_id)
        wall_prim.GetPrim().CreateAttribute("wall:type", Sdf.ValueTypeNames.String).Set(
            wall_data.get("type", "interior"))
        wall_prim.GetPrim().CreateAttribute("wall:structural", Sdf.ValueTypeNames.Bool).Set(
            wall_data.get("structural", False))
        
        # Legg til åpninger (vinduer, dører)
        for opening in wall_data.get("openings", []):
            await self._create_wall_opening(opening, wall_prim.GetPath(), start, end, height)
        
        return wall_mesh
    
    def _generate_wall_geometry(self, start: List[float], end: List[float], height: float, 
                               thickness: float) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer vegggeometri mellom to punkter"""
        # Beregn vektorer langs veggen
        wall_vec = [end[0] - start[0], end[1] - start[1]]
        length = (wall_vec[0]**2 + wall_vec[1]**2)**0.5
        
        if length < 0.001:
            return [], [], []
        
        # Normaliser
        wall_dir = [wall_vec[0] / length, wall_vec[1] / length]
        
        # Beregn normalen (90 grader rotasjon mot klokken)
        normal = [-wall_dir[1], wall_dir[0]]
        
        # Beregn de fire hjørnene for veggbasis
        half_thickness = thickness / 2
        c1 = [start[0] + normal[0] * half_thickness, start[1] + normal[1] * half_thickness]
        c2 = [end[0] + normal[0] * half_thickness, end[1] + normal[1] * half_thickness]
        c3 = [end[0] - normal[0] * half_thickness, end[1] - normal[1] * half_thickness]
        c4 = [start[0] - normal[0] * half_thickness, start[1] - normal[1] * half_thickness]
        
        # Opprett vertekser (8 vertekser - 4 nede, 4 oppe)
        vertices = [
            # Bunn
            Gf.Vec3f(c1[0], 0, c1[1]),
            Gf.Vec3f(c2[0], 0, c2[1]),
            Gf.Vec3f(c3[0], 0, c3[1]),
            Gf.Vec3f(c4[0], 0, c4[1]),
            # Topp
            Gf.Vec3f(c1[0], height, c1[1]),
            Gf.Vec3f(c2[0], height, c2[1]),
            Gf.Vec3f(c3[0], height, c3[1]),
            Gf.Vec3f(c4[0], height, c4[1])
        ]
        
        # Sett ansiktsindekser (6 sider, hver med 4 vertekser)
        faces = [
            # Forside
            0, 1, 5, 4,
            # Høyre side
            1, 2, 6, 5,
            # Bakside
            2, 3, 7, 6,
            # Venstre side
            3, 0, 4, 7,
            # Topp
            4, 5, 6, 7,
            # Bunn
            3, 2, 1, 0
        ]
        
        # Generer UV-koordinater
        uvs = []
        # Forenklet UV-mapping: skaler basert på størrelse
        for i in range(len(vertices)):
            if i < 4:  # Bunn
                uvs.append(Gf.Vec2f(vertices[i][0] / 2, vertices[i][2] / 2))
            else:  # Topp
                uvs.append(Gf.Vec2f(vertices[i][0] / 2, vertices[i][2] / 2))
        
        return vertices, faces, uvs
    
    async def _create_wall_opening(self, opening_data: Dict[str, Any], wall_path: Sdf.Path, 
                                  wall_start: List[float], wall_end: List[float], wall_height: float) -> UsdGeom.Mesh:
        """Opprett en åpning i en vegg (vindu eller dør)"""
        opening_type = opening_data.get("type", "window")
        position = opening_data.get("position", 0.5)  # Relativ posisjon langs vegg (0-1)
        width = opening_data.get("width", 1.0)
        height = opening_data.get("height", 1.2)
        sill_height = opening_data.get("sill_height", 0.9)  # Høyde fra gulv til underkant
        
        # Beregn plassering langs veggen
        wall_vec = [wall_end[0] - wall_start[0], wall_end[1] - wall_start[1]]
        wall_length = (wall_vec[0]**2 + wall_vec[1]**2)**0.5
        
        # Skalert posisjon
        pos_along_wall = position * wall_length
        
        # Midtpunkt for åpningen
        mid_x = wall_start[0] + (wall_vec[0] / wall_length) * pos_along_wall
        mid_z = wall_start[1] + (wall_vec[1] / wall_length) * pos_along_wall
        
        # Opprett åpning prim
        opening_id = opening_data.get("id", f"{opening_type}_{position:.1f}")
        opening_prim_path = f"{wall_path}/Openings/{opening_id}"
        opening_prim = self.stage.DefinePrim(opening_prim_path, "Xform")
        
        # Plasser åpningen
        xform = UsdGeom.Xformable(opening_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(mid_x, sill_height, mid_z))
        
        # Beregn rotasjon for å innrette med veggen
        angle = np.degrees(np.arctan2(wall_vec[1], wall_vec[0]))
        xform.AddRotateYOp().Set(angle)
        
        # Opprett åpningsgeometri basert på type
        if opening_type == "window":
            await self._create_window(opening_prim.GetPath(), width, height)
        elif opening_type == "door":
            await self._create_door(opening_prim.GetPath(), width, height)
        
        # Sett metadata
        opening_prim.GetPrim().CreateAttribute("opening:type", Sdf.ValueTypeNames.String).Set(opening_type)
        opening_prim.GetPrim().CreateAttribute("opening:width", Sdf.ValueTypeNames.Float).Set(width)
        opening_prim.GetPrim().CreateAttribute("opening:height", Sdf.ValueTypeNames.Float).Set(height)
        
        return UsdGeom.Xform(opening_prim)
    
    async def _create_window(self, parent_path: Sdf.Path, width: float, height: float) -> UsdGeom.Mesh:
        """Opprett et vindu med glass og karm"""
        # Opprett vinduskarm
        frame_prim_path = f"{parent_path}/Frame"
        frame_prim = self.stage.DefinePrim(frame_prim_path, "Mesh")
        
        # Opprett glassrute
        glass_prim_path = f"{parent_path}/Glass"
        glass_prim = self.stage.DefinePrim(glass_prim_path, "Mesh")
        
        # Beregn geometri (forenklet boksmodell)
        frame_thickness = 0.1
        glass_thickness = 0.02
        
        # Vinduskarm
        frame_vertices, frame_faces, frame_uvs = self._generate_window_frame_geometry(
            width, height, frame_thickness)
        
        # Glassrute
        glass_vertices, glass_faces, glass_uvs = self._generate_glass_geometry(
            width - frame_thickness*2, height - frame_thickness*2, glass_thickness)
        
        # Sett mesh-data for karm
        frame_mesh = UsdGeom.Mesh(frame_prim)
        frame_mesh.CreatePointsAttr(frame_vertices)
        frame_mesh.CreateFaceVertexIndicesAttr(frame_faces)
        frame_mesh.CreateFaceVertexCountsAttr([4] * (len(frame_faces) // 4))
        frame_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(frame_uvs)
        
        # Sett mesh-data for glass
        glass_mesh = UsdGeom.Mesh(glass_prim)
        glass_mesh.CreatePointsAttr(glass_vertices)
        glass_mesh.CreateFaceVertexIndicesAttr(glass_faces)
        glass_mesh.CreateFaceVertexCountsAttr([4] * (len(glass_faces) // 4))
        glass_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(glass_uvs)
        
        # Plasser glassrute midt i karmen
        glass_xform = UsdGeom.Xformable(glass_prim)
        glass_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, frame_thickness/2))
        
        return UsdGeom.Xform(frame_prim)
    
    def _generate_window_frame_geometry(self, width: float, height: float, 
                                       thickness: float) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer geometri for vinduskarm"""
        # Forenklet boksmodell med hull for glass
        outer_half_width = width / 2
        outer_half_height = height / 2
        inner_half_width = outer_half_width - thickness
        inner_half_height = outer_half_height - thickness
        
        # Vertekser for ytre og indre boks
        vertices = [
            # Ytre front (+Z)
            Gf.Vec3f(-outer_half_width, -outer_half_height, thickness),   # 0
            Gf.Vec3f(outer_half_width, -outer_half_height, thickness),    # 1
            Gf.Vec3f(outer_half_width, outer_half_height, thickness),     # 2
            Gf.Vec3f(-outer_half_width, outer_half_height, thickness),    # 3
            
            # Ytre bak (-Z)
            Gf.Vec3f(-outer_half_width, -outer_half_height, 0),           # 4
            Gf.Vec3f(outer_half_width, -outer_half_height, 0),            # 5
            Gf.Vec3f(outer_half_width, outer_half_height, 0),             # 6
            Gf.Vec3f(-outer_half_width, outer_half_height, 0),            # 7
            
            # Indre front (+Z)
            Gf.Vec3f(-inner_half_width, -inner_half_height, thickness),   # 8
            Gf.Vec3f(inner_half_width, -inner_half_height, thickness),    # 9
            Gf.Vec3f(inner_half_width, inner_half_height, thickness),     # 10
            Gf.Vec3f(-inner_half_width, inner_half_height, thickness),    # 11
            
            # Indre bak (-Z)
            Gf.Vec3f(-inner_half_width, -inner_half_height, 0),           # 12
            Gf.Vec3f(inner_half_width, -inner_half_height, 0),            # 13
            Gf.Vec3f(inner_half_width, inner_half_height, 0),             # 14
            Gf.Vec3f(-inner_half_width, inner_half_height, 0)             # 15
        ]
        
        # Ansikter (kvads)
        faces = [
            # Ytre front
            0, 1, 2, 3,
            
            # Ytre sidene
            0, 4, 5, 1,  # Bunn
            1, 5, 6, 2,  # Høyre
            2, 6, 7, 3,  # Topp
            3, 7, 4, 0,  # Venstre
            
            # Ytre bak
            4, 7, 6, 5,
            
            # Indre front
            11, 10, 9, 8,
            
            # Indre sidene
            8, 9, 13, 12,  # Bunn
            9, 10, 14, 13,  # Høyre
            10, 11, 15, 14,  # Topp
            11, 8, 12, 15,  # Venstre
            
            # Indre bak
            12, 13, 14, 15
        ]
        
        # UV-koordinater
        uvs = []
        for v in vertices:
            # Normaliserte UV-koordinater
            u = (v[0] / width) + 0.5  # 0-1 fra venstre til høyre
            v = (v[1] / height) + 0.5  # 0-1 fra bunn til topp
            uvs.append(Gf.Vec2f(u, v))
        
        return vertices, faces, uvs
    
    def _generate_glass_geometry(self, width: float, height: float, 
                               thickness: float) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer geometri for glass"""
        half_width = width / 2
        half_height = height / 2
        half_thickness = thickness / 2
        
        # Vertekser for glasspanel (enkel boks)
        vertices = [
            # Front (+Z)
            Gf.Vec3f(-half_width, -half_height, half_thickness),
            Gf.Vec3f(half_width, -half_height, half_thickness),
            Gf.Vec3f(half_width, half_height, half_thickness),
            Gf.Vec3f(-half_width, half_height, half_thickness),
            
            # Bak (-Z)
            Gf.Vec3f(-half_width, -half_height, -half_thickness),
            Gf.Vec3f(half_width, -half_height, -half_thickness),
            Gf.Vec3f(half_width, half_height, -half_thickness),
            Gf.Vec3f(-half_width, half_height, -half_thickness)
        ]
        
        # Ansikter (kvads)
        faces = [
            # Front
            0, 1, 2, 3,
            # Sidene
            0, 4, 5, 1,  # Bunn
            1, 5, 6, 2,  # Høyre
            2, 6, 7, 3,  # Topp
            3, 7, 4, 0,  # Venstre
            # Bak
            4, 7, 6, 5
        ]
        
        # UV-koordinater
        uvs = []
        for v in vertices:
            u = (v[0] / width) + 0.5
            v = (v[1] / height) + 0.5
            uvs.append(Gf.Vec2f(u, v))
        
        return vertices, faces, uvs
    
    async def _create_door(self, parent_path: Sdf.Path, width: float, height: float) -> UsdGeom.Mesh:
        """Opprett en dør med karm"""
        # Opprett dørkarm
        frame_prim_path = f"{parent_path}/Frame"
        frame_prim = self.stage.DefinePrim(frame_prim_path, "Mesh")
        
        # Opprett dørblad
        door_prim_path = f"{parent_path}/Door"
        door_prim = self.stage.DefinePrim(door_prim_path, "Mesh")
        
        # Beregn geometri
        frame_thickness = 0.1
        door_thickness = 0.04
        
        # Dørkarm
        frame_vertices, frame_faces, frame_uvs = self._generate_door_frame_geometry(
            width, height, frame_thickness)
        
        # Dørblad
        door_vertices, door_faces, door_uvs = self._generate_door_geometry(
            width - frame_thickness*2, height - frame_thickness, door_thickness)
        
        # Sett mesh-data for karm
        frame_mesh = UsdGeom.Mesh(frame_prim)
        frame_mesh.CreatePointsAttr(frame_vertices)
        frame_mesh.CreateFaceVertexIndicesAttr(frame_faces)
        frame_mesh.CreateFaceVertexCountsAttr([4] * (len(frame_faces) // 4))
        frame_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(frame_uvs)
        
        # Sett mesh-data for dørblad
        door_mesh = UsdGeom.Mesh(door_prim)
        door_mesh.CreatePointsAttr(door_vertices)
        door_mesh.CreateFaceVertexIndicesAttr(door_faces)
        door_mesh.CreateFaceVertexCountsAttr([4] * (len(door_faces) // 4))
        door_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(door_uvs)
        
        # Plasser dørblad i karmen
        door_xform = UsdGeom.Xformable(door_prim)
        door_xform.AddTranslateOp().Set(Gf.Vec3d(-width/2 + frame_thickness, 0, frame_thickness/2))
        
        # Sett dørens rotasjonspunkt (hengsel)
        pivot = UsdGeom.Xformable(door_prim).AddTransformOp()
        pivot.Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 0)))
        
        return UsdGeom.Xform(frame_prim)
    
    def _generate_door_frame_geometry(self, width: float, height: float, 
                                    thickness: float) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer geometri for dørkarm"""
        # Lignende som vinduskarm, men uten kant i bunnen
        outer_half_width = width / 2
        inner_half_width = outer_half_width - thickness
        
        # Vertekser
        vertices = [
            # Ytre front (+Z)
            Gf.Vec3f(-outer_half_width, 0, thickness),                  # 0
            Gf.Vec3f(outer_half_width, 0, thickness),                   # 1
            Gf.Vec3f(outer_half_width, height, thickness),              # 2
            Gf.Vec3f(-outer_half_width, height, thickness),             # 3
            
            # Ytre bak (-Z)
            Gf.Vec3f(-outer_half_width, 0, 0),                          # 4
            Gf.Vec3f(outer_half_width, 0, 0),                           # 5
            Gf.Vec3f(outer_half_width, height, 0),                      # 6
            Gf.Vec3f(-outer_half_width, height, 0),                     # 7
            
            # Indre front (+Z)
            Gf.Vec3f(-inner_half_width, 0, thickness),                  # 8
            Gf.Vec3f(inner_half_width, 0, thickness),                   # 9
            Gf.Vec3f(inner_half_width, height - thickness, thickness),  # 10
            Gf.Vec3f(-inner_half_width, height - thickness, thickness), # 11
            
            # Indre bak (-Z)
            Gf.Vec3f(-inner_half_width, 0, 0),                          # 12
            Gf.Vec3f(inner_half_width, 0, 0),                           # 13
            Gf.Vec3f(inner_half_width, height - thickness, 0),          # 14
            Gf.Vec3f(-inner_half_width, height - thickness, 0)          # 15
        ]
        
        # Ansikter (kvads) - lignende som for vinduskarm
        faces = [
            # Ytre front
            0, 1, 2, 3,
            
            # Ytre sidene
            0, 4, 5, 1,  # Bunn
            1, 5, 6, 2,  # Høyre
            2, 6, 7, 3,  # Topp
            3, 7, 4, 0,  # Venstre
            
            # Ytre bak
            4, 7, 6, 5,
            
            # Indre front
            11, 10, 9, 8,
            
            # Indre sidene
            8, 9, 13, 12,  # Bunn
            9, 10, 14, 13,  # Høyre
            10, 11, 15, 14,  # Topp
            11, 8, 12, 15,  # Venstre
            
            # Indre bak
            12, 13, 14, 15,
            
            # Toppkarm
            3, 2, 10, 11,  # Front
            2, 6, 14, 10,  # Høyre
            6, 7, 15, 14,  # Bak
            7, 3, 11, 15   # Venstre
        ]
        
        # UV-koordinater
        uvs = []
        for v in vertices:
            u = (v[0] / width) + 0.5
            v = (v[1] / height)
            uvs.append(Gf.Vec2f(u, v))
        
        return vertices, faces, uvs
    
    def _generate_door_geometry(self, width: float, height: float, 
                              thickness: float) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer geometri for dørblad"""
        half_width = width / 2
        half_thickness = thickness / 2
        
        # Vertekser for dørblad (enkel boks)
        vertices = [
            # Front (+Z)
            Gf.Vec3f(0, 0, half_thickness),            # 0
            Gf.Vec3f(width, 0, half_thickness),        # 1
            Gf.Vec3f(width, height, half_thickness),   # 2
            Gf.Vec3f(0, height, half_thickness),       # 3
            
            # Bak (-Z)
            Gf.Vec3f(0, 0, -half_thickness),           # 4
            Gf.Vec3f(width, 0, -half_thickness),       # 5
            Gf.Vec3f(width, height, -half_thickness),  # 6
            Gf.Vec3f(0, height, -half_thickness)       # 7
        ]
        
        # Ansikter (kvads)
        faces = [
            # Front
            0, 1, 2, 3,
            # Sidene
            0, 4, 5, 1,  # Bunn
            1, 5, 6, 2,  # Høyre
            2, 6, 7, 3,  # Topp
            3, 7, 4, 0,  # Venstre
            # Bak
            4, 7, 6, 5
        ]
        
        # UV-koordinater
        uvs = [
            Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1),
            Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)
        ]
        
        return vertices, faces, uvs
    
    async def _create_roof(self, roof_data: Dict[str, Any], parent_path: Sdf.Path) -> UsdGeom.Xform:
        """Opprett et tak basert på takdata"""
        roof_type = roof_data.get("type", "flat")
        outline = roof_data.get("outline", [])
        height = roof_data.get("height", 1.0)  # Høyde fra takfot til møne for skråtak
        overhang = roof_data.get("overhang", 0.5)  # Takutstikk
        elevation = roof_data.get("elevation", 3.0)  # Høyde fra bakken
        
        # Opprett tak-prim
        roof_prim_path = f"{parent_path}/Roof"
        roof_prim = self.stage.DefinePrim(roof_prim_path, "Xform")
        
        # Plasser taket
        xform = UsdGeom.Xformable(roof_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, elevation, 0))
        
        # Opprett takgeometri basert på type
        if roof_type == "flat":
            await self._create_flat_roof(roof_prim.GetPath(), outline, overhang)
        elif roof_type == "gable":
            await self._create_gable_roof(roof_prim.GetPath(), outline, height, overhang)
        elif roof_type == "hip":
            await self._create_hip_roof(roof_prim.GetPath(), outline, height, overhang)
        else:
            logger.warning(f"Unsupported roof type: {roof_type}. Using flat roof.")
            await self._create_flat_roof(roof_prim.GetPath(), outline, overhang)
        
        # Sett metadata
        roof_prim.GetPrim().CreateAttribute("roof:type", Sdf.ValueTypeNames.String).Set(roof_type)
        roof_prim.GetPrim().CreateAttribute("roof:height", Sdf.ValueTypeNames.Float).Set(height)
        roof_prim.GetPrim().CreateAttribute("roof:overhang", Sdf.ValueTypeNames.Float).Set(overhang)
        
        return UsdGeom.Xform(roof_prim)
    
    async def _create_flat_roof(self, parent_path: Sdf.Path, outline: List[List[float]], 
                              overhang: float) -> UsdGeom.Mesh:
        """Opprett et flatt tak"""
        # Ekspander omriss med takutstikk
        expanded_outline = self._expand_outline(outline, overhang)
        
        # Opprett takprim
        roof_mesh_path = f"{parent_path}/RoofMesh"
        roof_mesh_prim = self.stage.DefinePrim(roof_mesh_path, "Mesh")
        roof_mesh = UsdGeom.Mesh(roof_mesh_prim)
        
        # Opprett vertekser
        vertices = []
        for point in expanded_outline:
            vertices.append(Gf.Vec3f(point[0], 0, point[1]))  # Y er opp
        
        # Triangulering
        faces = self._triangulate_polygon(vertices)
        
        # Sett mesh-data
        roof_mesh.CreatePointsAttr(vertices)
        roof_mesh.CreateFaceVertexIndicesAttr(faces)
        roof_mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 3))
        
        # Opprett UV-koordinater
        uvs = self._generate_uvs_for_floor(vertices)
        roof_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        return roof_mesh
    
    async def _create_gable_roof(self, parent_path: Sdf.Path, outline: List[List[float]], 
                               height: float, overhang: float) -> UsdGeom.Xform:
        """Opprett et saltak"""
        # Finn lengste akse for takryggen
        min_x = min(p[0] for p in outline)
        max_x = max(p[0] for p in outline)
        min_z = min(p[1] for p in outline)
        max_z = max(p[1] for p in outline)
        
        width = max_x - min_x
        depth = max_z - min_z
        
        # Ekspander omriss med takutstikk
        expanded_outline = self._expand_outline(outline, overhang)
        
        # Midtpunkt for takryggen
        mid_x = (min_x + max_x) / 2
        mid_z = (min_z + max_z) / 2
        
        # Opprett takxform
        roof_xform_path = f"{parent_path}/GableRoof"
        roof_xform_prim = self.stage.DefinePrim(roof_xform_path, "Xform")
        
        # Opprett takflater
        roof_parts = []
        
        # Bestem retning for takrygg
        if width > depth:
            # Takrygg langs X-aksen
            ridge_start = [mid_x - width/2 - overhang, height, mid_z]
            ridge_end = [mid_x + width/2 + overhang, height, mid_z]
            
            # Opprett to takflater (front og bak)
            front_vertices = [
                Gf.Vec3f(min_x - overhang, 0, min_z - overhang),
                Gf.Vec3f(max_x + overhang, 0, min_z - overhang),
                Gf.Vec3f(max_x + overhang, height, mid_z),
                Gf.Vec3f(min_x - overhang, height, mid_z)
            ]
            
            back_vertices = [
                Gf.Vec3f(min_x - overhang, 0, max_z + overhang),
                Gf.Vec3f(max_x + overhang, 0, max_z + overhang),
                Gf.Vec3f(max_x + overhang, height, mid_z),
                Gf.Vec3f(min_x - overhang, height, mid_z)
            ]
        else:
            # Takrygg langs Z-aksen
            ridge_start = [mid_x, height, mid_z - depth/2 - overhang]
            ridge_end = [mid_x, height, mid_z + depth/2 + overhang]
            
            # Opprett to takflater (venstre og høyre)
            left_vertices = [
                Gf.Vec3f(min_x - overhang, 0, min_z - overhang),
                Gf.Vec3f(mid_x, height, min_z - overhang),
                Gf.Vec3f(mid_x, height, max_z + overhang),
                Gf.Vec3f(min_x - overhang, 0, max_z + overhang)
            ]
            
            right_vertices = [
                Gf.Vec3f(max_x + overhang, 0, min_z - overhang),
                Gf.Vec3f(mid_x, height, min_z - overhang),
                Gf.Vec3f(mid_x, height, max_z + overhang),
                Gf.Vec3f(max_x + overhang, 0, max_z + overhang)
            ]
            
            front_vertices = left_vertices
            back_vertices = right_vertices
        
        # Opprett takflater
        front_roof_mesh = self.stage.DefinePrim(f"{roof_xform_path}/FrontSlope", "Mesh")
        back_roof_mesh = self.stage.DefinePrim(f"{roof_xform_path}/BackSlope", "Mesh")
        
        # Sett mesh-data for front takflate
        front_faces = [0, 1, 2, 3]  # Enkel kvadrilateral
        front_uvs = [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
        
        UsdGeom.Mesh(front_roof_mesh).CreatePointsAttr(front_vertices)
        UsdGeom.Mesh(front_roof_mesh).CreateFaceVertexIndicesAttr(front_faces)
        UsdGeom.Mesh(front_roof_mesh).CreateFaceVertexCountsAttr([4])
        UsdGeom.Mesh(front_roof_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(front_uvs)
        
        # Sett mesh-data for bakre takflate
        back_faces = [0, 3, 2, 1]  # Vær oppmerksom på retning for bakre flate
        back_uvs = [Gf.Vec2f(0, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0)]
        
        UsdGeom.Mesh(back_roof_mesh).CreatePointsAttr(back_vertices)
        UsdGeom.Mesh(back_roof_mesh).CreateFaceVertexIndicesAttr(back_faces)
        UsdGeom.Mesh(back_roof_mesh).CreateFaceVertexCountsAttr([4])
        UsdGeom.Mesh(back_roof_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(back_uvs)
        
        # Opprett gavler
        if width > depth:
            await self._create_gable_end(f"{roof_xform_path}/LeftGable", 
                                       [min_x - overhang, 0, min_z - overhang],
                                       [min_x - overhang, 0, max_z + overhang],
                                       [min_x - overhang, height, mid_z])
            
            await self._create_gable_end(f"{roof_xform_path}/RightGable", 
                                       [max_x + overhang, 0, min_z - overhang],
                                       [max_x + overhang, 0, max_z + overhang],
                                       [max_x + overhang, height, mid_z])
        
        return UsdGeom.Xform(roof_xform_prim)
    
    async def _create_gable_end(self, path: str, bottom_left: List[float], 
                              bottom_right: List[float], top: List[float]) -> UsdGeom.Mesh:
        """Opprett en gavl (trekantet ende) for saltak"""
        gable_prim = self.stage.DefinePrim(path, "Mesh")
        gable_mesh = UsdGeom.Mesh(gable_prim)
        
        # Vertekser
        vertices = [
            Gf.Vec3f(*bottom_left),
            Gf.Vec3f(*bottom_right),
            Gf.Vec3f(*top)
        ]
        
        # Triangulær flate
        faces = [0, 1, 2]
        
        # UV-koordinater
        uvs = [
            Gf.Vec2f(0, 0),
            Gf.Vec2f(1, 0),
            Gf.Vec2f(0.5, 1)
        ]
        
        # Sett mesh-data
        gable_mesh.CreatePointsAttr(vertices)
        gable_mesh.CreateFaceVertexIndicesAttr(faces)
        gable_mesh.CreateFaceVertexCountsAttr([3])
        gable_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        return gable_mesh
    
    async def _create_hip_roof(self, parent_path: Sdf.Path, outline: List[List[float]], 
                             height: float, overhang: float) -> UsdGeom.Xform:
        """Opprett et valmet tak"""
        # Finn sentrum og dimensjoner
        min_x = min(p[0] for p in outline)
        max_x = max(p[0] for p in outline)
        min_z = min(p[1] for p in outline)
        max_z = max(p[1] for p in outline)
        
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        
        # Ekspander omriss med takutstikk
        expanded_outline = self._expand_outline(outline, overhang)
        
        # Sett opp takxform
        roof_xform_path = f"{parent_path}/HipRoof"
        roof_xform_prim = self.stage.DefinePrim(roof_xform_path, "Xform")
        
        # Opprett sentertopp-punkt
        roof_peak = [center_x, height, center_z]
        
        # For hvert linjestykke i omrisset, opprett en takflate
        sides = len(expanded_outline)
        for i in range(sides):
            p1 = expanded_outline[i]
            p2 = expanded_outline[(i + 1) % sides]
            
            # Opprett takflate fra linjestykke til toppen
            slope_path = f"{roof_xform_path}/Slope_{i}"
            await self._create_roof_slope(slope_path, 
                                        [p1[0], 0, p1[1]], 
                                        [p2[0], 0, p2[1]], 
                                        roof_peak)
        
        return UsdGeom.Xform(roof_xform_prim)
    
    async def _create_roof_slope(self, path: str, p1: List[float], p2: List[float], 
                               peak: List[float]) -> UsdGeom.Mesh:
        """Opprett en takflate fra to bunnpunkter til et toppunkt"""
        slope_prim = self.stage.DefinePrim(path, "Mesh")
        slope_mesh = UsdGeom.Mesh(slope_prim)
        
        # Vertekser
        vertices = [
            Gf.Vec3f(*p1),
            Gf.Vec3f(*p2),
            Gf.Vec3f(*peak)
        ]
        
        # Triangulær flate
        faces = [0, 1, 2]
        
        # UV-koordinater
        uvs = [
            Gf.Vec2f(0, 0),
            Gf.Vec2f(1, 0),
            Gf.Vec2f(0.5, 1)
        ]
        
        # Sett mesh-data
        slope_mesh.CreatePointsAttr(vertices)
        slope_mesh.CreateFaceVertexIndicesAttr(faces)
        slope_mesh.CreateFaceVertexCountsAttr([3])
        slope_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        return slope_mesh
    
    def _expand_outline(self, outline: List[List[float]], expansion: float) -> List[List[float]]:
        """Utvider et omriss uniformt i alle retninger"""
        if not outline or len(outline) < 3:
            logger.warning("Cannot expand invalid outline. Using default rectangle.")
            return [[-5, -5], [5, -5], [5, 5], [-5, 5]]
        
        # For enkel implementasjon, finn bounding box og utvid
        min_x = min(p[0] for p in outline)
        max_x = max(p[0] for p in outline)
        min_z = min(p[1] for p in outline)
        max_z = max(p[1] for p in outline)
        
        # Utvid i alle retninger
        expanded = [
            [min_x - expansion, min_z - expansion],
            [max_x + expansion, min_z - expansion],
            [max_x + expansion, max_z + expansion],
            [min_x - expansion, max_z + expansion]
        ]
        
        return expanded
    
    async def _add_structural_elements(self, structural_data: Dict[str, Any]):
        """Legg til strukturelle elementer med fysikksimulering"""
        try:
            logger.info("Adding structural elements")
            
            # Opprett strukturelt hierarki
            structure_prim = self.stage.DefinePrim("/World/Structure", "Xform")
            
            # Legg til bjelker
            beam_tasks = []
            for beam in structural_data.get("beams", []):
                beam_tasks.append(self._create_beam(beam, structure_prim.GetPath()))
            
            await asyncio.gather(*beam_tasks)
            
            # Legg til søyler
            column_tasks = []
            for column in structural_data.get("columns", []):
                column_tasks.append(self._create_column(column, structure_prim.GetPath()))
            
            await asyncio.gather(*column_tasks)
            
            # Legg til trapper
            stair_tasks = []
            for stair in structural_data.get("stairs", []):
                stair_tasks.append(self._create_staircase(stair, structure_prim.GetPath()))
            
            await asyncio.gather(*stair_tasks)
            
            logger.info("Structural elements added successfully")
            
        except Exception as e:
            logger.error(f"Error adding structural elements: {str(e)}")
            raise
    
    async def _create_beam(self, beam_data: Dict[str, Any], parent_path: Sdf.Path) -> UsdGeom.Mesh:
        """Opprett en bjelke med fysikk"""
        # Hent bjelkedata
        start = beam_data.get("start", [0, 0, 0])
        end = beam_data.get("end", [5,
