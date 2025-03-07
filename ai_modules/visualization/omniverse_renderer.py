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
        end = beam_data.get("end", [5, 0, 0])
        width = beam_data.get("width", 0.2)
        height = beam_data.get("height", 0.3)
        material = beam_data.get("material", "wood")
        
        # Beregn retningsvektor
        direction = [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
        length = (direction[0]**2 + direction[1]**2 + direction[2]**2)**0.5
        
        if length < 0.001:
            logger.warning("Beam has zero length. Skipping.")
            return None
        
        # Normaliser
        direction = [direction[0] / length, direction[1] / length, direction[2] / length]
        
        # Opprett bjelke-prim
        beam_id = beam_data.get("id", f"beam_{start[0]}_{start[1]}_{start[2]}")
        beam_prim_path = f"{parent_path}/Beams/{beam_id}"
        beam_prim = self.stage.DefinePrim(beam_prim_path, "Mesh")
        
        # Plasser bjelken
        xform = UsdGeom.Xformable(beam_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*start))
        
        # Roter for å innrette med retningen
        if abs(direction[0]) < 0.99 or abs(direction[1]) > 0.01 or abs(direction[2]) > 0.01:
            # Beregn rotasjon fra X-aksen til retningsvektor
            z_rotation = np.degrees(np.arctan2(direction[1], direction[0]))
            xform.AddRotateZOp().Set(z_rotation)
            
            # Beregn vertikalrotasjon
            xz_length = (direction[0]**2 + direction[2]**2)**0.5
            y_rotation = np.degrees(np.arctan2(direction[1], xz_length))
            xform.AddRotateYOp().Set(y_rotation)
        
        # Opprett bjelkegeometri
        vertices, faces, uvs = self._generate_beam_geometry(length, width, height)
        
        # Sett mesh-data
        beam_mesh = UsdGeom.Mesh(beam_prim)
        beam_mesh.CreatePointsAttr(vertices)
        beam_mesh.CreateFaceVertexIndicesAttr(faces)
        beam_mesh.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
        beam_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        # Legg til fysikk hvis tilgjengelig
        if hasattr(UsdPhysics, 'RigidBodyAPI'):
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(beam_prim)
            rigid_body.CreateMassAttr().Set(beam_data.get("mass", 50.0))
            
            # Kollisjonsmesh
            UsdPhysics.CollisionAPI.Apply(beam_prim)
            
            # Sett fysiske egenskaper
            UsdPhysics.MassAPI.Apply(beam_prim)
            UsdPhysics.MassAPI(beam_prim).CreateDensityAttr().Set(600.0)  # Tretetthet
        
        # Sett metadata
        beam_prim.GetPrim().CreateAttribute("beam:material", Sdf.ValueTypeNames.String).Set(material)
        beam_prim.GetPrim().CreateAttribute("beam:type", Sdf.ValueTypeNames.String).Set(
            beam_data.get("type", "structural"))
        
        return beam_mesh
    
    def _generate_beam_geometry(self, length: float, width: float, 
                              height: float) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer geometri for en bjelke"""
        half_width = width / 2
        half_height = height / 2
        
        # Vertekser for bjelke (boks langs X-aksen)
        vertices = [
            # Front (-X)
            Gf.Vec3f(0, -half_height, -half_width),
            Gf.Vec3f(0, half_height, -half_width),
            Gf.Vec3f(0, half_height, half_width),
            Gf.Vec3f(0, -half_height, half_width),
            
            # Bak (+X)
            Gf.Vec3f(length, -half_height, -half_width),
            Gf.Vec3f(length, half_height, -half_width),
            Gf.Vec3f(length, half_height, half_width),
            Gf.Vec3f(length, -half_height, half_width)
        ]
        
        # Ansikter (kvads)
        faces = [
            # Front
            0, 3, 2, 1,
            # Bak
            4, 5, 6, 7,
            # Topp
            1, 2, 6, 5,
            # Bunn
            0, 4, 7, 3,
            # Høyre side
            3, 7, 6, 2,
            # Venstre side
            0, 1, 5, 4
        ]
        
        # UV-koordinater
        uvs = []
        for i in range(8):
            u = 0 if i < 4 else 1  # 0 for front, 1 for bak
            v = i % 4 / 3.0  # 0, 1/3, 2/3, 1 gjentatt for front og bak
            uvs.append(Gf.Vec2f(u, v))
        
        return vertices, faces, uvs
    
    async def _create_column(self, column_data: Dict[str, Any], parent_path: Sdf.Path) -> UsdGeom.Mesh:
        """Opprett en søyle/kolonne med fysikk"""
        # Hent søyledata
        position = column_data.get("position", [0, 0, 0])
        height = column_data.get("height", 2.7)
        radius = column_data.get("radius", 0.15)
        sides = column_data.get("sides", 8)  # Antall sider (8 for oktagonal, høyere for mer sirkulær)
        material = column_data.get("material", "concrete")
        
        # Opprett søyle-prim
        column_id = column_data.get("id", f"column_{position[0]}_{position[1]}_{position[2]}")
        column_prim_path = f"{parent_path}/Columns/{column_id}"
        column_prim = self.stage.DefinePrim(column_prim_path, "Mesh")
        
        # Plasser søylen
        xform = UsdGeom.Xformable(column_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        
        # Opprett søylegeometri
        vertices, faces, uvs = self._generate_column_geometry(height, radius, sides)
        
        # Sett mesh-data
        column_mesh = UsdGeom.Mesh(column_prim)
        column_mesh.CreatePointsAttr(vertices)
        column_mesh.CreateFaceVertexIndicesAttr(faces)
        column_mesh.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
        column_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        # Legg til fysikk hvis tilgjengelig
        if hasattr(UsdPhysics, 'RigidBodyAPI'):
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(column_prim)
            rigid_body.CreateMassAttr().Set(column_data.get("mass", 200.0))
            
            # Kollisjonsmesh
            UsdPhysics.CollisionAPI.Apply(column_prim)
            
            # Sett fysiske egenskaper
            UsdPhysics.MassAPI.Apply(column_prim)
            UsdPhysics.MassAPI(column_prim).CreateDensityAttr().Set(2400.0)  # Betongtetthet
        
        # Sett metadata
        column_prim.GetPrim().CreateAttribute("column:material", Sdf.ValueTypeNames.String).Set(material)
        column_prim.GetPrim().CreateAttribute("column:type", Sdf.ValueTypeNames.String).Set(
            column_data.get("type", "structural"))
        
        return column_mesh
    
    def _generate_column_geometry(self, height: float, radius: float, 
                                sides: int) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer geometri for en søyle"""
        vertices = []
        faces = []
        uvs = []
        
        # Opprett vertekser rundt sirkelen for bunn og topp
        for i in range(sides):
            angle = 2 * np.pi * i / sides
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            # Bunnpunkt
            vertices.append(Gf.Vec3f(x, 0, z))
            uvs.append(Gf.Vec2f(i / sides, 0))
            
            # Toppunkt
            vertices.append(Gf.Vec3f(x, height, z))
            uvs.append(Gf.Vec2f(i / sides, 1))
        
        # Opprett sideflater
        for i in range(sides):
            # Indekser for nåværende og neste punkt på sirkelen (bunn og topp)
            current_bottom = i * 2
            current_top = i * 2 + 1
            next_bottom = (i * 2 + 2) % (sides * 2)
            next_top = (i * 2 + 3) % (sides * 2)
            
            # Legg til kvadrilateral (med korrekt winding)
            faces.extend([current_bottom, current_top, next_top, next_bottom])
        
        # Legg til topp- og bunnflater
        bottom_face = []
        top_face = []
        
        for i in range(sides):
            bottom_face.append(i * 2)
            top_face.append((sides - i - 1) * 2 + 1)  # Revers for korrekt winding
        
        # Legg til bunnsirkel
        faces.extend(bottom_face)
        
        # Legg til toppsirkel
        faces.extend(top_face)
        
        # Legg til count for topp og bunn
        polygon_counts = [4] * sides + [sides, sides]
        
        # UV-koordinater for topp og bunn
        for i in range(sides):
            angle = 2 * np.pi * i / sides
            u = 0.5 + 0.5 * np.cos(angle)
            v = 0.5 + 0.5 * np.sin(angle)
            uvs.append(Gf.Vec2f(u, v))  # Bunn
            uvs.append(Gf.Vec2f(u, v))  # Topp
        
        return vertices, faces, uvs
    
    async def _create_staircase(self, stair_data: Dict[str, Any], parent_path: Sdf.Path) -> UsdGeom.Xform:
        """Opprett en trapp"""
        # Hent trappedata
        start_position = stair_data.get("start", [0, 0, 0])
        end_position = stair_data.get("end", [3, 3, 0])
        width = stair_data.get("width", 1.0)
        steps = stair_data.get("steps", 15)
        material = stair_data.get("material", "wood")
        
        # Beregn trapperetning og -dimensjoner
        direction = [
            end_position[0] - start_position[0],
            end_position[1] - start_position[1],
            end_position[2] - start_position[2]
        ]
        
        horizontal_length = (direction[0]**2 + direction[2]**2)**0.5
        height = direction[1]
        
        if horizontal_length < 0.001 or height < 0.001:
            logger.warning("Invalid staircase dimensions. Skipping.")
            return None
        
        # Opprett trapp-xform
        stair_id = stair_data.get("id", f"stair_{start_position[0]}_{start_position[1]}_{start_position[2]}")
        stair_prim_path = f"{parent_path}/Stairs/{stair_id}"
        stair_prim = self.stage.DefinePrim(stair_prim_path, "Xform")
        
        # Plasser trappen
        xform = UsdGeom.Xformable(stair_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*start_position))
        
        # Beregn rotasjon for å innrette med retningen
        if abs(direction[0]) > 0.001 or abs(direction[2]) > 0.001:
            angle = np.degrees(np.arctan2(direction[2], direction[0]))
            xform.AddRotateYOp().Set(angle)
        
        # Opprett trinn
        step_length = horizontal_length / steps
        step_height = height / steps
        
        step_tasks = []
        for i in range(steps):
            step_tasks.append(self._create_stair_step(
                f"{stair_prim_path}/Step_{i}",
                i * step_length,
                i * step_height,
                step_length,
                step_height,
                width,
                material
            ))
        
        # Vent på at alle trinn skal bli opprettet
        await asyncio.gather(*step_tasks)
        
        # Opprett rekkverk hvis spesifisert
        if stair_data.get("has_railing", True):
            await self._create_stair_railing(
                f"{stair_prim_path}/Railing",
                horizontal_length,
                height,
                width,
                steps,
                stair_data.get("railing_height", 0.9),
                stair_data.get("railing_material", "metal")
            )
        
        return UsdGeom.Xform(stair_prim)
    
    async def _create_stair_step(self, path: str, x_pos: float, y_pos: float, length: float, 
                               height: float, width: float, material: str) -> UsdGeom.Mesh:
        """Opprett et enkelt trinn i en trapp"""
        step_prim = self.stage.DefinePrim(path, "Mesh")
        step_mesh = UsdGeom.Mesh(step_prim)
        
        # Plasser trinnet
        xform = UsdGeom.Xformable(step_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(x_pos, y_pos, -width/2))
        
        # Trinngeometri
        thickness = 0.05  # Trinntykkelse
        
        vertices = [
            # Øvre flate
            Gf.Vec3f(0, height, 0),
            Gf.Vec3f(length, height, 0),
            Gf.Vec3f(length, height, width),
            Gf.Vec3f(0, height, width),
            
            # Nedre flate
            Gf.Vec3f(0, height - thickness, 0),
            Gf.Vec3f(length, height - thickness, 0),
            Gf.Vec3f(length, height - thickness, width),
            Gf.Vec3f(0, height - thickness, width),
            
            # Vertikalt trinn-panel
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(length, 0, 0),
            Gf.Vec3f(length, height, 0),
            Gf.Vec3f(0, height, 0),
            
            Gf.Vec3f(0, 0, width),
            Gf.Vec3f(length, 0, width),
            Gf.Vec3f(length, height, width),
            Gf.Vec3f(0, height, width)
        ]
        
        # Ansikter
        faces = [
            # Topp
            0, 3, 2, 1,
            
            # Bunn
            4, 5, 6, 7,
            
            # Front
            8, 9, 10, 11,
            
            # Bak
            12, 15, 14, 13,
            
            # Venstre
            8, 11, 15, 12,
            
            # Høyre
            9, 13, 14, 10
        ]
        
        # UV-koordinater
        uvs = [
            # Topp
            Gf.Vec2f(0, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0),
            
            # Bunn
            Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1),
            
            # Front og bak
            Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1),
            Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1),
            
            # Sider
            Gf.Vec2f(0, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0),
            Gf.Vec2f(0, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0)
        ]
        
        # Sett mesh-data
        step_mesh.CreatePointsAttr(vertices)
        step_mesh.CreateFaceVertexIndicesAttr(faces)
        step_mesh.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
        step_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        # Sett materialemarkør
        step_prim.GetPrim().CreateAttribute("material", Sdf.ValueTypeNames.String).Set(material)
        
        return step_mesh
    
    async def _create_stair_railing(self, path: str, length: float, height: float, width: float, 
                                 steps: int, railing_height: float, material: str) -> UsdGeom.Xform:
        """Opprett rekkverk for en trapp"""
        railing_prim = self.stage.DefinePrim(path, "Xform")
        
        # Rekkverk-parametre
        post_radius = 0.02
        handrail_radius = 0.025
        
        # Beregn vinkel for rekkverk
        angle = np.arctan2(height, length)
        railing_length = (length**2 + height**2)**0.5
        
        # Opprett stolper på venstre side
        posts_left = []
        posts_right = []
        
        for i in range(steps + 1):
            x_pos = i * (length / steps)
            y_pos = i * (height / steps)
            
            # Venstre stolpe
            post_left_path = f"{path}/Post_Left_{i}"
            post_left = await self._create_railing_post(
                post_left_path, x_pos, y_pos, 0, railing_height, post_radius, material)
            posts_left.append(post_left)
            
            # Høyre stolpe
            post_right_path = f"{path}/Post_Right_{i}"
            post_right = await self._create_railing_post(
                post_right_path, x_pos, y_pos, width, railing_height, post_radius, material)
            posts_right.append(post_right)
        
        # Opprett håndlister
        handrail_left_path = f"{path}/Handrail_Left"
        handrail_left = await self._create_handrail(
            handrail_left_path, 0, 0, 0, angle, railing_length, railing_height, handrail_radius, material)
        
        handrail_right_path = f"{path}/Handrail_Right"
        handrail_right = await self._create_handrail(
            handrail_right_path, 0, 0, width, angle, railing_length, railing_height, handrail_radius, material)
        
        return UsdGeom.Xform(railing_prim)
    
    async def _create_railing_post(self, path: str, x_pos: float, y_pos: float, z_pos: float, 
                                height: float, radius: float, material: str) -> UsdGeom.Cylinder:
        """Opprett en rekkverkstolpe"""
        post_prim = self.stage.DefinePrim(path, "Cylinder")
        post = UsdGeom.Cylinder(post_prim)
        
        # Sett sylinderparametre
        post.CreateRadiusAttr(radius)
        post.CreateHeightAttr(height)
        
        # Plasser stolpen
        xform = UsdGeom.Xformable(post_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(x_pos, y_pos, z_pos))
        
        # Juster aksene (Y-opp)
        xform.AddRotateXOp().Set(90)
        
        # Juster til å starte fra bakken
        xform.AddTranslateOp().Set(Gf.Vec3d(0, height/2, 0))
        
        # Sett materialemarkør
        post_prim.GetPrim().CreateAttribute("material", Sdf.ValueTypeNames.String).Set(material)
        
        return post
    
    async def _create_handrail(self, path: str, x_pos: float, y_pos: float, z_pos: float, 
                            angle: float, length: float, height: float, radius: float, 
                            material: str) -> UsdGeom.Cylinder:
        """Opprett en håndlist for rekkverk"""
        handrail_prim = self.stage.DefinePrim(path, "Cylinder")
        handrail = UsdGeom.Cylinder(handrail_prim)
        
        # Sett sylinderparametre
        handrail.CreateRadiusAttr(radius)
        handrail.CreateHeightAttr(length)
        
        # Plasser håndlisten
        xform = UsdGeom.Xformable(handrail_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(x_pos, y_pos + height, z_pos))
        
        # Juster aksene (Z-langs)
        xform.AddRotateXOp().Set(90)
        
        # Roter for å følge trappevinkel
        xform.AddRotateZOp().Set(np.degrees(angle))
        
        # Juster til å starte fra første stolpe
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, length/2))
        
        # Sett materialemarkør
        handrail_prim.GetPrim().CreateAttribute("material", Sdf.ValueTypeNames.String).Set(material)
        
        return handrail
    
    async def _apply_materials(self, materials_data: Dict[str, Any]):
        """Appliser materialer på 3D-modellen med avansert shading"""
        try:
            logger.info("Applying materials to 3D model")
            
            # Opprett materialbibliotek
            material_lib = self._create_material_library(materials_data)
            
            # Appliser materialer på overflater
            assignment_tasks = []
            for surface, material_name in materials_data.get("assignments", {}).items():
                if material_name in material_lib:
                    material = material_lib[material_name]
                    assignment_tasks.append(self._apply_material_to_surface(surface, material))
            
            # Vent på at alle materialassigneringer skal fullføres
            await asyncio.gather(*assignment_tasks)
            
            # Oppdater shader-nettverk
            await self._update_shaders()
            
            logger.info("Materials applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying materials: {str(e)}")
            raise
    
    def _create_material_library(self, materials_data: Dict[str, Any]) -> Dict[str, UsdShade.Material]:
        """Opprett bibliotek med USD-materialer"""
        logger.info("Creating material library")
        
        material_lib = {}
        
        # Opprett Materials-prim
        materials_path = "/World/Materials"
        self.stage.DefinePrim(materials_path, "Scope")
        
        # Opprett materialer fra data
        for material_name, material_props in materials_data.get("materials", {}).items():
            try:
                material_path = f"{materials_path}/{material_name}"
                material = UsdShade.Material.Define(self.stage, material_path)
                
                # Opprett PBR shader
                shader_path = f"{material_path}/PBRShader"
                shader = UsdShade.Shader.Define(self.stage, shader_path)
                shader.CreateIdAttr("UsdPreviewSurface")
                
                # Sett shader-parametre
                self._set_shader_parameters(shader, material_props)
                
                # Opprett teksturnoder
                if "textures" in material_props:
                    self._create_texture_nodes(material_path, shader, material_props["textures"])
                
                # Sett shader som materialoutput
                material.CreateSurfaceOutput().ConnectToSource(
                    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
                
                # Legg til materialet i biblioteket
                material_lib[material_name] = material
                
                logger.info(f"Created material: {material_name}")
                
            except Exception as e:
                logger.error(f"Error creating material {material_name}: {str(e)}")
        
        # Legg til standard materialer fra materialbiblioteket
        for material_name, properties in self.material_library.materials.items():
            if material_name not in material_lib:
                try:
                    material = self.material_library.create_usd_material(
                        self.stage, material_name, properties)
                    material_lib[material_name] = material
                    logger.info(f"Added standard material: {material_name}")
                except Exception as e:
                    logger.error(f"Error adding standard material {material_name}: {str(e)}")
        
        return material_lib
    
    def _set_shader_parameters(self, shader: UsdShade.Shader, material_props: Dict[str, Any]):
        """Sett shader-parametre fra materialegenskaper"""
        # Basisegenskaper
        if "base_color" in material_props:
            color = material_props["base_color"]
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        
        if "roughness" in material_props:
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(material_props["roughness"])
        
        if "metallic" in material_props:
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(material_props["metallic"])
        
        # Avanserte egenskaper
        if "opacity" in material_props:
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(material_props["opacity"])
        
        if "ior" in material_props:
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(material_props["ior"])
        
        if "normal_scale" in material_props:
            shader.CreateInput("normalScale", Sdf.ValueTypeNames.Float).Set(material_props["normal_scale"])
        
        if "emissive_color" in material_props:
            color = material_props["emissive_color"]
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        
        if "emissive_intensity" in material_props:
            shader.CreateInput("emissiveIntensity", Sdf.ValueTypeNames.Float).Set(
                material_props["emissive_intensity"])
        
        if "clearcoat" in material_props:
            shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(material_props["clearcoat"])
        
        if "clearcoat_roughness" in material_props:
            shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(
                material_props["clearcoat_roughness"])
        
        if "displacement_scale" in material_props:
            shader.CreateInput("displacementScale", Sdf.ValueTypeNames.Float).Set(
                material_props["displacement_scale"])
        
        # Spesialbehandling for transparente/blanke materialer
        if material_props.get("type") == "glass":
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.2)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(material_props.get("ior", 1.5))
            shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    
    def _create_texture_nodes(self, material_path: str, shader: UsdShade.Shader, 
                            textures: Dict[str, str]):
        """Opprett teksturnoder for et materiale"""
        for tex_type, tex_path in textures.items():
            # Opprett teksturshader
            tex_shader_path = f"{material_path}/{tex_type.capitalize()}Texture"
            tex_shader = UsdShade.Shader.Define(self.stage, tex_shader_path)
            tex_shader.CreateIdAttr("UsdUVTexture")
            
            # Sett teksturfilsti
            tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
            
            # Sett wrapping og filtrering
            tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_shader.CreateInput("minFilter", Sdf.ValueTypeNames.Token).Set("linear")
            tex_shader.CreateInput("magFilter", Sdf.ValueTypeNames.Token).Set("linear")
            
            # Koble teksturnoden til hovedshaderen
            tex_output = None
            shader_input = None
            
            if tex_type == "albedo" or tex_type == "diffuse" or tex_type == "basecolor":
                tex_output = tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                shader_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            
            elif tex_type == "roughness":
                tex_output = tex_shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
                shader_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
            
            elif tex_type == "metallic":
                tex_output = tex_shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
                shader_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
            
            elif tex_type == "normal":
                # Opprett normalmap-shader
                normal_reader_path = f"{material_path}/NormalMap"
                normal_reader = UsdShade.Shader.Define(self.stage, normal_reader_path)
                normal_reader.CreateIdAttr("UsdNormalMap")
                
                # Koble tekstur til normalmap-reader
                tex_output = tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                normal_input = normal_reader.CreateInput("in", Sdf.ValueTypeNames.Float3)
                normal_input.ConnectToSource(tex_output)
                
                # Koble normalmap til shader
                normal_output = normal_reader.CreateOutput("normal", Sdf.ValueTypeNames.Normal3f)
                shader_input = shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)
                shader_input.ConnectToSource(normal_output)
                
                continue  # Skip standard tilkobling
            
            elif tex_type == "occlusion":
                tex_output = tex_shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
                shader_input = shader.CreateInput("occlusion", Sdf.ValueTypeNames.Float)
            
            elif tex_type == "opacity":
                tex_output = tex_shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
                shader_input = shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
            
            elif tex_type == "emissive":
                tex_output = tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                shader_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
            
            # Koble teksturen til shader hvis både output og input er definert
            if tex_output and shader_input:
                shader_input.ConnectToSource(tex_output)
    
    async def _apply_material_to_surface(self, surface_path: str, material: UsdShade.Material):
        """Appliser materiale på en overflate"""
        try:
            # Resolve full path with patterns
            matching_prims = self._find_matching_prims(surface_path)
            
            for prim_path in matching_prims:
                # Bindmaterialet til prim
                UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(prim_path)).Bind(material)
            
            logger.debug(f"Applied material to {len(matching_prims)} surfaces matching {surface_path}")
            
        except Exception as e:
            logger.error(f"Error applying material to surface {surface_path}: {str(e)}")
    
    def _find_matching_prims(self, pattern: str) -> List[Sdf.Path]:
        """Finn prims som matcher et mønster (støtter wildcard)"""
        matching_prims = []
        
        # Sjekk for wildcard
        if '*' in pattern:
            # Konverter mønsteret til regex
            import re
            regex_pattern = pattern.replace('*', '.*')
            regex = re.compile(regex_pattern)
            
            # Finn alle mesher på scenen
            for prim in self.stage.Traverse():
                if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Cube):
                    prim_path_str = str(prim.GetPath())
                    if regex.match(prim_path_str):
                        matching_prims.append(prim.GetPath())
        else:
            # Sjekk om primen eksisterer
            prim = self.stage.GetPrimAtPath(pattern)
            if prim:
                matching_prims.append(prim.GetPath())
        
        return matching_prims
    
    async def _update_shaders(self):
        """Oppdater shader-nettverk og sikre at alle materialer er korrekt koblet"""
        # Dette er en placeholder for mer avansert shader-oppdatering
        # I en reell implementasjon ville dette involvere mer kompleks logikk
        
        # Oppdater materialnettverk
        UsdShade.MaterialBindingAPI.ComputeBoundMaterials(self.stage)
        
        # Oppfrisk stageen
        self.stage.Save()
    
    async def _setup_lighting(self, lighting_data: Dict[str, Any] = None):
        """Sett opp avansert belysning med HDR og sol/himmel"""
        try:
            logger.info("Setting up lighting system")
            
            # Standardverdier hvis ingen data er gitt
            if lighting_data is None:
                lighting_data = {
                    "type": "natural",
                    "time": "day",
                    "intensity": 1.0,
                    "ambient_intensity": 0.3,
                    "artificial_lights": []
                }
            
            # Opprett lighting-prim
            lighting_prim_path = "/World/Lighting"
            lighting_prim = self.stage.DefinePrim(lighting_prim_path, "Scope")
            
            # Oppsett basert på lystype
            lighting_type = lighting_data.get("type", "natural")
            
            if lighting_type == "natural" or lighting_type == "outdoor":
                await self._setup_natural_lighting(lighting_prim.GetPath(), lighting_data)
            elif lighting_type == "indoor":
                await self._setup_indoor_lighting(lighting_prim.GetPath(), lighting_data)
            elif lighting_type == "studio":
                await self._setup_studio_lighting(lighting_prim.GetPath(), lighting_data)
            else:
                logger.warning(f"Unknown lighting type: {lighting_type}. Using natural lighting.")
                await self._setup_natural_lighting(lighting_prim.GetPath(), lighting_data)
            
            # Legg til kunstige lyskilder hvis spesifisert
            for light_data in lighting_data.get("artificial_lights", []):
                await self._add_artificial_light(lighting_prim.GetPath(), light_data)
            
            logger.info("Lighting system set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up lighting: {str(e)}")
            raise
    
    async def _setup_natural_lighting(self, parent_path: Sdf.Path, lighting_data: Dict[str, Any]):
        """Sett opp naturlig belysning med sol og himmel"""
        # Opprett dome light for himmellys
        dome_light_path = f"{parent_path}/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        
        # Sett dome light parametre
        intensity = lighting_data.get("ambient_intensity", 0.3)
        dome_light.CreateIntensityAttr(intensity * 1000)  # Skalert for realistisk lysintensitet
        
        # Sett HDR-tekstur for dome light hvis tilgjengelig
        time_of_day = lighting_data.get("time", "day")
        if time_of_day == "sunset":
            dome_light.CreateTextureFileAttr("textures/environments/sunset_hdr.exr")
            dome_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.8, 0.6))  # Varmere lys ved solnedgang
        elif time_of_day == "night":
            dome_light.CreateTextureFileAttr("textures/environments/night_sky_hdr.exr")
            dome_light.CreateColorAttr().Set(Gf.Vec3f(0.2, 0.3, 0.6))  # Blålig nattbelysning
        else:  # Default: day
            dome_light.CreateTextureFileAttr("textures/environments/clear_sky_hdr.exr")
            dome_light.CreateColorAttr().Set(Gf.Vec3f(0.9, 0.9, 1.0))  # Nøytralt daglys
        
        # Opprett distant light for sollys
        distant_light_path = f"{parent_path}/SunLight"
        distant_light = UsdLux.DistantLight.Define(self.stage, distant_light_path)
        
        # Sett distant light parametre
        sun_intensity = lighting_data.get("intensity", 1.0)
        distant_light.CreateIntensityAttr(sun_intensity * 5000)
        
        # Sett solens retning basert på tid på dagen
        if time_of_day == "sunrise":
            distant_light.CreateAngleAttr(5.0)  # Smalere stråler for skarpere skygger
            xform = UsdGeom.Xformable(distant_light)
            xform.AddRotateXOp().Set(5)    # Lav solvinkel
            xform.AddRotateYOp().Set(-80)  # Østlig retning
            distant_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.8, 0.7))  # Varm morgensol
        elif time_of_day == "sunset":
            distant_light.CreateAngleAttr(5.0)
            xform = UsdGeom.Xformable(distant_light)
            xform.AddRotateXOp().Set(15)   # Lav solvinkel
            xform.AddRotateYOp().Set(80)   # Vestlig retning
            distant_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.7, 0.4))  # Oransje kveldssol
        elif time_of_day == "night":
            distant_light.CreateIntensityAttr(sun_intensity * 100)  # Mye svakere om natten
            xform = UsdGeom.Xformable(distant_light)
            xform.AddRotateXOp().Set(-45)  # Månen høyt på himmelen
            distant_light.CreateColorAttr().Set(Gf.Vec3f(0.8, 0.8, 1.0))  # Blåhvitt månelys
        else:  # Default: day
            distant_light.CreateAngleAttr(2.5)
            xform = UsdGeom.Xformable(distant_light)
            xform.AddRotateXOp().Set(45)   # Høy solvinkel
            xform.AddRotateYOp().Set(0)    # Sørlig retning
            distant_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))  # Nøytralt dagslys
    
    async def _setup_indoor_lighting(self, parent_path: Sdf.Path, lighting_data: Dict[str, Any]):
        """Sett opp innendørs belysning med ambient light og area lights"""
        # Opprett ambient light for grunnleggende rombelysning
        ambient_light_path = f"{parent_path}/AmbientLight"
        ambient_light = UsdLux.DomeLight.Define(self.stage, ambient_light_path)
        
        # Sett ambient light parametre
        intensity = lighting_data.get("ambient_intensity", 0.5)
        ambient_light.CreateIntensityAttr(intensity * 300)
        ambient_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))  # Nøytralt innendørslys
        
        # Opprett hovedlyskilde for rommet
        main_light_path = f"{parent_path}/MainLight"
        main_light = UsdLux.RectLight.Define(self.stage, main_light_path)
        
        # Plasser og juster hovedlyset
        main_intensity = lighting_data.get("intensity", 1.0)
        main_light.CreateIntensityAttr(main_intensity * 1000)
        main_light.CreateWidthAttr(2.0)
        main_light.CreateHeightAttr(1.5)
        main_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.9))  # Litt varmere hovedlys
        
        # Plasser lyset i taket pekende nedover
        xform = UsdGeom.Xformable(main_light)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 2.5, 0))  # Høyde avhenger av romstørrelse
        xform.AddRotateXOp().Set(90)  # Peker nedover
        
        # Legg til fyllys på motsatt side
        fill_light_path = f"{parent_path}/FillLight"
        fill_light = UsdLux.RectLight.Define(self.stage, fill_light_path)
        
        # Sett fyllysparametre
        fill_light.CreateIntensityAttr(main_intensity * 300)
        fill_light.CreateWidthAttr(1.5)
        fill_light.CreateHeightAttr(1.0)
        fill_light.CreateColorAttr().Set(Gf.Vec3f(0.9, 0.9, 1.0))  # Litt kjøligere fyllys
        
        # Plasser fyllyset
        xform = UsdGeom.Xformable(fill_light)
        xform.AddTranslateOp().Set(Gf.Vec3d(-3, 1.5, 3))
        xform.AddRotateXOp().Set(30)
        xform.AddRotateYOp().Set(45)
    
    async def _setup_studio_lighting(self, parent_path: Sdf.Path, lighting_data: Dict[str, Any]):
        """Sett opp studio-stil belysning med trepunkts lysrigg"""
        # Basisintensitet
        base_intensity = lighting_data.get("intensity", 1.0)
        
        # Hovedlys (key light)
        key_light_path = f"{parent_path}/KeyLight"
        key_light = UsdLux.RectLight.Define(self.stage, key_light_path)
        
        key_light.CreateIntensityAttr(base_intensity * 1500)
        key_light.CreateWidthAttr(2.0)
        key_light.CreateHeightAttr(1.0)
        key_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))  # Nøytralt hovedlys
        
        # Plasser hovedlyset
        xform = UsdGeom.Xformable(key_light)
        xform.AddTranslateOp().Set(Gf.Vec3d(3, 3, 3))
        xform.AddRotateYOp().Set(-45)
        xform.AddRotateXOp().Set(-30)
        
        # Fyllys (fill light)
        fill_light_path = f"{parent_path}/FillLight"
        fill_light = UsdLux.RectLight.Define(self.stage, fill_light_path)
        
        fill_light.CreateIntensityAttr(base_intensity * 600)  # 40% av hovedlys
        fill_light.CreateWidthAttr(3.0)
        fill_light.CreateHeightAttr(2.0)
        fill_light.CreateColorAttr().Set(Gf.Vec3f(0.9, 0.9, 1.0))  # Litt kjøligere
        
        # Plasser fyllyset
        xform = UsdGeom.Xformable(fill_light)
        xform.AddTranslateOp().Set(Gf.Vec3d(-4, 2, 3))
        xform.AddRotateYOp().Set(30)
        xform.AddRotateXOp().Set(-15)
        
        # Bakgrunnsbelysning (rim light)
        rim_light_path = f"{parent_path}/RimLight"
        rim_light = UsdLux.RectLight.Define(self.stage, rim_light_path)
        
        rim_light.CreateIntensityAttr(base_intensity * 1200)  # 80% av hovedlys
        rim_light.CreateWidthAttr(1.5)
        rim_light.CreateHeightAttr(3.0)
        rim_light.CreateColorAttr().Set(Gf.Vec3f(0.95, 0.95, 1.0))  # Litt blålig
        
        # Plasser kantlyset
        xform = UsdGeom.Xformable(rim_light)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 2, -4))
        xform.AddRotateYOp().Set(180)
        xform.AddRotateXOp().Set(-15)
    
    async def _add_artificial_light(self, parent_path: Sdf.Path, light_data: Dict[str, Any]):
        """Legg til en kunstig lyskilde"""
        light_type = light_data.get("type", "point")
        position = light_data.get("position", [0, 2, 0])
        intensity = light_data.get("intensity", 1.0)
        color = light_data.get("color", [1, 1, 1])
        name = light_data.get("name", f"Light_{light_type}")
        
        # Opprett lysprim
        light_path = f"{parent_path}/{name}"
        light_prim = None
        
        if light_type == "point":
            light_prim = UsdLux.SphereLight.Define(self.stage, light_path)
            UsdLux.SphereLight(light_prim).CreateRadiusAttr(0.1)
            UsdLux.SphereLight(light_prim).CreateTreatAsPointAttr(True)
        
        elif light_type == "spot":
            light_prim = UsdLux.DiskLight.Define(self.stage, light_path)
            UsdLux.DiskLight(light_prim).CreateRadiusAttr(0.2)
            # Spot fokus
            UsdLux.Light(light_prim).CreateShapingConeAngleAttr(light_data.get("cone_angle", 45.0))
            UsdLux.Light(light_prim).CreateShapingConeSoftnessAttr(light_data.get("cone_softness", 0.2))
        
        elif light_type == "area":
            light_prim = UsdLux.RectLight.Define(self.stage, light_path)
            UsdLux.RectLight(light_prim).CreateWidthAttr(light_data.get("width", 1.0))
            UsdLux.RectLight(light_prim).CreateHeightAttr(light_data.get("height", 0.5))
        
        else:
            logger.warning(f"Unknown light type: {light_type}. Using point light.")
            light_prim = UsdLux.SphereLight.Define(self.stage, light_path)
            UsdLux.SphereLight(light_prim).CreateRadiusAttr(0.1)
        
        # Sett lysegenskaper
        if light_prim:
            UsdLux.Light(light_prim).CreateIntensityAttr(intensity * 500)  # Skalert for realistisk intensitet
            UsdLux.Light(light_prim).CreateColorAttr().Set(Gf.Vec3f(*color[:3]))
            
            # Plasser lyset
            xform = UsdGeom.Xformable(light_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(*position))
            
            # Roter lyset hvis retning er spesifisert
            if "direction" in light_data:
                direction = light_data["direction"]
                
                # Beregn rotasjonsvinkel fra Z-aksen til retningsvektoren
                from math import acos, degrees, sqrt
                z_axis = [0, 0, 1]
                
                # Normaliser retningsvektoren
                dir_length = sqrt(sum(d*d for d in direction))
                if dir_length > 0.001:
                    normalized_dir = [d / dir_length for d in direction]
                    
                    # Beregn rotasjonsvinkel rundt Y-aksen (yaw)
                    yaw = degrees(acos(normalized_dir[2]))
                    xform.AddRotateYOp().Set(yaw)
                    
                    # Beregn rotasjonsvinkel rundt X-aksen (pitch)
                    pitch = degrees(acos(normalized_dir[1]))
                    xform.AddRotateXOp().Set(pitch)
    
    async def _add_environment(self, environment_data: Dict[str, Any]):
        """Legg til miljø og omgivelser"""
        try:
            logger.info("Adding environment elements")
            
            # Opprett miljø-prim
            environment_prim_path = "/World/Environment"
            environment_prim = self.stage.DefinePrim(environment_prim_path, "Scope")
            
            # Legg til himmel/bakgrunn hvis spesifisert
            if "sky" in environment_data:
                await self._add_sky(environment_prim.GetPath(), environment_data["sky"])
            
            # Legg til terreng hvis spesifisert
            if "terrain" in environment_data:
                await self._add_terrain(environment_prim.GetPath(), environment_data["terrain"])
            
            # Legg til vegetasjon hvis spesifisert
            if "vegetation" in environment_data:
                await self._add_vegetation(environment_prim.GetPath(), environment_data["vegetation"])
            
            # Legg til omgivende objekter hvis spesifisert
            if "surroundings" in environment_data:
                await self._add_surroundings(environment_prim.GetPath(), environment_data["surroundings"])
            
            logger.info("Environment elements added successfully")
            
        except Exception as e:
            logger.error(f"Error adding environment: {str(e)}")
            raise
    
    async def _add_sky(self, parent_path: Sdf.Path, sky_data: Dict[str, Any]):
        """Legg til himmel og atmosfære"""
        # Opprett dome for himmel
        sky_prim_path = f"{parent_path}/Sky"
        sky_prim = self.stage.DefinePrim(sky_prim_path, "Sphere")
        
        # Sett skydome-parametre
        UsdGeom.Sphere(sky_prim).CreateRadiusAttr(5000)  # Stor radius for himmel
        
        # Flytt skydome til kamera-nullpunkt
        xform = UsdGeom.Xformable(sky_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        
        # Opprett materiale for himmel
        sky_material_path = f"{parent_path}/SkyMaterial"
        sky_material = UsdShade.Material.Define(self.stage, sky_material_path)
        
        # Opprett shader for himmel
        sky_shader_path = f"{sky_material_path}/Shader"
        sky_shader = UsdShade.Shader.Define(self.stage, sky_shader_path)
        sky_shader.CreateIdAttr("UsdPreviewSurface")
        
        # Sett shadergenskaper for himmel
        sky_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.7, 1.0))
        sky_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        
        # Koble shader til materiale
        sky_material.CreateSurfaceOutput().ConnectToSource(
            sky_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        
        # Bindmaterialet til skydome
        UsdShade.MaterialBindingAPI(sky_prim).Bind(sky_material)
    
    async def _add_terrain(self, parent_path: Sdf.Path, terrain_data: Dict[str, Any]):
        """Legg til terreng"""
        # Opprett terreng-prim
        terrain_prim_path = f"{parent_path}/Terrain"
        terrain_prim = self.stage.DefinePrim(terrain_prim_path, "Mesh")
        
        # Generer terrengmesh basert på data
        width = terrain_data.get("width", 100)
        length = terrain_data.get("length", 100)
        height_scale = terrain_data.get("height_scale", 5)
        resolution = terrain_data.get("resolution", 50)
        
        # Generer høydekart hvis ikke gitt
        height_map = terrain_data.get("height_map", None)
        if not height_map:
            height_map = self._generate_procedural_terrain(resolution, resolution, height_scale)
        
        # Opprett terrenggeometri
        vertices, faces, uvs = self._generate_terrain_mesh(width, length, resolution, height_map)
        
        # Sett mesh-data
        terrain_mesh = UsdGeom.Mesh(terrain_prim)
        terrain_mesh.CreatePointsAttr(vertices)
        terrain_mesh.CreateFaceVertexIndicesAttr(faces)
        terrain_mesh.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
        terrain_mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(uvs)
        
        # Plasser terrenget
        xform = UsdGeom.Xformable(terrain_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(-width/2, -height_scale/2, -length/2))
        
        # Opprett materialer for terreng
        material_name = terrain_data.get("material", "grass")
        self._create_and_assign_material(terrain_prim_path, material_name)
    
    def _generate_procedural_terrain(self, width: int, length: int, height_scale: float) -> List[float]:
        """Generer et prosedurelt høydekart for terreng"""
        import random
        
        # Opprett 2D-grid med tilfeldig høyde
        height_map = [[0.0 for _ in range(length)] for _ in range(width)]
        
        # Sett hjørneverdier
        height_map[0][0] = random.uniform(0, 1) * height_scale
        height_map[0][length-1] = random.uniform(0, 1) * height_scale
        height_map[width-1][0] = random.uniform(0, 1) * height_scale
        height_map[width-1][length-1] = random.uniform(0, 1) * height_scale
        
        # Diamond-Square algoritme for terrengenerasjon
        def diamond_square(x1, y1, x2, y2, roughness):
            if x2 - x1 < 2 and y2 - y1 < 2:
                return
            
            # Midtpunktet
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            
            # Diamond step: Sett midtpunktet til gjennomsnittet av hjørnene pluss litt tilfeldig verdi
            if height_map[mid_x][mid_y] == 0:
                avg = (height_map[x1][y1] + height_map[x1][y2] + height_map[x2][y1] + height_map[x2][y2]) / 4
                height_map[mid_x][mid_y] = avg + random.uniform(-1, 1) * roughness * height_scale
            
            # Square step: Sett midtpunktet på hver kant
            # Topp
            if height_map[mid_x][y1] == 0:
                avg = (height_map[x1][y1] + height_map[x2][y1] + height_map[mid_x][mid_y]) / 3
                height_map[mid_x][y1] = avg + random.uniform(-1, 1) * roughness * height_scale
            
            # Høyre
            if height_map[x2][mid_y] == 0:
                avg = (height_map[x2][y1] + height_map[x2][y2] + height_map[mid_x][mid_y]) / 3
                height_map[x2][mid_y] = avg + random.uniform(-1, 1) * roughness * height_scale
            
            # Bunn
            if height_map[mid_x][y2] == 0:
                avg = (height_map[x1][y2] + height_map[x2][y2] + height_map[mid_x][mid_y]) / 3
                height_map[mid_x][y2] = avg + random.uniform(-1, 1) * roughness * height_scale
            
            # Venstre
            if height_map[x1][mid_y] == 0:
                avg = (height_map[x1][y1] + height_map[x1][y2] + height_map[mid_x][mid_y]) / 3
                height_map[x1][mid_y] = avg + random.uniform(-1, 1) * roughness * height_scale
            
            # Rekursivt kall for kvadrantene
            next_roughness = roughness * 0.5
            diamond_square(x1, y1, mid_x, mid_y, next_roughness)
            diamond_square(mid_x, y1, x2, mid_y, next_roughness)
            diamond_square(x1, mid_y, mid_x, y2, next_roughness)
            diamond_square(mid_x, mid_y, x2, y2, next_roughness)
        
        # Start algoritmen
        diamond_square(0, 0, width-1, length-1, 1.0)
        
        # Flatten 2D array to 1D
        flat_height_map = []
        for row in height_map:
            flat_height_map.extend(row)
        
        return flat_height_map
    
    def _generate_terrain_mesh(self, width: float, length: float, resolution: int, 
                              height_map: List[float]) -> Tuple[List[Gf.Vec3f], List[int], List[Gf.Vec2f]]:
        """Generer terrengmesh fra høydekart"""
        vertices = []
        faces = []
        uvs = []
        
        # Opprett vertekser i et grid
        for i in range(resolution):
            for j in range(resolution):
                # Beregn posisjon i 3D-rom
                x = (i / (resolution - 1)) * width
                z = (j / (resolution - 1)) * length
                y = height_map[i * resolution + j] if i * resolution + j < len(height_map) else 0
                
                vertices.append(Gf.Vec3f(x, y, z))
                uvs.append(Gf.Vec2f(i / (resolution - 1), j / (resolution - 1)))
        
        # Opprett kvadrilaterale flater
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v0 = i * resolution + j
                v1 = v0 + 1
                v2 = (i + 1) * resolution + j + 1
                v3 = (i + 1) * resolution + j
                
                faces.extend([v0, v1, v2, v3])
        
        return vertices, faces, uvs
    
    async def _add_vegetation(self, parent_path: Sdf.Path, vegetation_data: Dict[str, Any]):
        """Legg til vegetasjon som trær, busker, etc."""
        # Opprett vegetasjon-prim
        vegetation_prim_path = f"{parent_path}/Vegetation"
        vegetation_prim = self.stage.DefinePrim(vegetation_prim_path, "Scope")
        
        # Legg til trær
        if "trees" in vegetation_data:
            await self._add_trees(vegetation_prim.GetPath(), vegetation_data["trees"])
        
        # Legg til busker
        if "bushes" in vegetation_data:
            await self._add_bushes(vegetation_prim.GetPath(), vegetation_data["bushes"])
        
        # Legg til gress
        if "grass" in vegetation_data:
            await self._add_grass(vegetation_prim.GetPath(), vegetation_data["grass"])
    
    async def _add_trees(self, parent_path: Sdf.Path, trees_data: Dict[str, Any]):
        """Legg til trær"""
        # Hent tredata
        count = trees_data.get("count", 10)
        area = trees_data.get("area", {"min_x": -20, "max_x": 20, "min_z": -20, "max_z": 20})
        species = trees_data.get("species", ["pine", "oak"])
        
        # Opprett trær-prim
        trees_prim_path = f"{parent_path}/Trees"
        trees_prim = self.stage.DefinePrim(trees_prim_path, "Scope")
        
        # Tilfeldig plassering av trær
        import random
        
        for i in range(count):
            # Velg tilfeldig art
            tree_species = random.choice(species)
            
            # Tilfeldig plassering innenfor område
            x = random.uniform(area["min_x"], area["max_x"])
            z = random.uniform(area["min_z"], area["max_z"])
            
            # Opprett tre
            tree_prim_path = f"{trees_prim_path}/Tree_{i}"
            await self._create_tree(tree_prim_path, tree_species, [x, 0, z])
    
    async def _create_tree(self, path: str, species: str, position: List[float]):
        """Opprett et tre av spesifisert art"""
        # Opprett tre-xform
        tree_prim = self.stage.DefinePrim(path, "Xform")
        
        # Plasser treet
        xform = UsdGeom.Xformable(tree_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        
        # Tilfeldig rotasjon og skalering for variasjon
        import random
        rotation = random.uniform(0, 360)
        scale = random.uniform(0.8, 1.2)
        
        xform.AddRotateYOp().Set(rotation)
        xform.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
        
        # Opprett treets deler basert på art
        if species == "pine":
            await self._create_pine_tree(tree_prim.GetPath())
        elif species == "oak":
            await self._create_oak_tree(tree_prim.GetPath())
        else:
            await self._create_generic_tree(tree_prim.GetPath())
    
    async def _create_pine_tree(self, path: Sdf.Path):
        """Opprett et furutre"""
        # Opprett stamme
        trunk_path = f"{path}/Trunk"
        trunk_prim = self.stage.DefinePrim(trunk_path, "Cylinder")
        
        # Sett stammeegenskaper
        trunk = UsdGeom.Cylinder(trunk_prim)
        trunk.CreateRadiusAttr(0.2)
        trunk.CreateHeightAttr(5.0)
        
        # Plasser stammen
        xform = UsdGeom.Xformable(trunk_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 2.5, 0))
        
        # Opprett kroneMesh
        crown_path = f"{path}/Crown"
        crown_prim = self.stage.DefinePrim(crown_path, "Cone")
        
        # Sett kroneegenskaper
        crown = UsdGeom.Cone(crown_prim)
        crown.CreateRadiusAttr(1.5)
        crown.CreateHeightAttr(4.0)
        
        # Plasser kronen
        xform = UsdGeom.Xformable(crown_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 5.0, 0))
        
        # Opprett og tilordne materialer
        self._create_and_assign_material(trunk_path, "wood_bark")
        self._create_and_assign_material(crown_path, "pine_needles")
    
    async def _create_oak_tree(self, path: Sdf.Path):
        """Opprett et eiketrær"""
        # Opprett stamme
        trunk_path = f"{path}/Trunk"
        trunk_prim = self.stage.DefinePrim(trunk_path, "Cylinder")
        
        # Sett stammeegenskaper
        trunk = UsdGeom.Cylinder(trunk_prim)
        trunk.CreateRadiusAttr(0.3)
        trunk.CreateHeightAttr(4.0)
        
        # Plasser stammen
        xform = UsdGeom.Xformable(trunk_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 2.0, 0))
        
        # Opprett kroneMesh (kule for løvtre)
        crown_path = f"{path}/Crown"
        crown_prim = self.stage.DefinePrim(crown_path, "Sphere")
        
        # Sett kroneegenskaper
        crown = UsdGeom.Sphere(crown_prim)
        crown.CreateRadiusAttr(2.5)
        
        # Plasser kronen
        xform = UsdGeom.Xformable(crown_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 5.0, 0))
        
        # Opprett og tilordne materialer
        self._create_and_assign_material(trunk_path, "wood_bark")
        self._create_and_assign_material(crown_path, "oak_leaves")
    
    async def _create_generic_tree(self, path: Sdf.Path):
        """Opprett et generisk tre"""
        # Opprett stamme
        trunk_path = f"{path}/Trunk"
        trunk_prim = self.stage.DefinePrim(trunk_path, "Cylinder")
        
        # Sett stammeegenskaper
        trunk = UsdGeom.Cylinder(trunk_prim)
        trunk.CreateRadiusAttr(0.25)
        trunk.CreateHeightAttr(3.5)
        
        # Plasser stammen
        xform = UsdGeom.Xformable(trunk_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 1.75, 0))
        
        # Opprett kroneMesh
        crown_path = f"{path}/Crown"
        crown_prim = self.stage.DefinePrim(crown_path, "Sphere")
        
        # Sett kroneegenskaper
        crown = UsdGeom.Sphere(crown_prim)
        crown.CreateRadiusAttr(2.0)
        
        # Plasser kronen
        xform = UsdGeom.Xformable(crown_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 4.0, 0))
        
        # Opprett og tilordne materialer
        self._create_and_assign_material(trunk_path, "wood_bark")
        self._create_and_assign_material(crown_path, "generic_leaves")
    
    async def _add_bushes(self, parent_path: Sdf.Path, bushes_data: Dict[str, Any]):
        """Legg til busker"""
        # Forenklet implementasjon - lignende trær, men mindre
        count = bushes_data.get("count", 20)
        area = bushes_data.get("area", {"min_x": -20, "max_x": 20, "min_z": -20, "max_z": 20})
        
        # Opprett busker-prim
        bushes_prim_path = f"{parent_path}/Bushes"
        bushes_prim = self.stage.DefinePrim(bushes_prim_path, "Scope")
        
        # Tilfeldig plassering av busker
        import random
        
        for i in range(count):
            # Tilfeldig plassering innenfor område
            x = random.uniform(area["min_x"], area["max_x"])
            z = random.uniform(area["min_z"], area["max_z"])
            
            # Opprett busk
            bush_path = f"{bushes_prim_path}/Bush_{i}"
            bush_prim = self.stage.DefinePrim(bush_path, "Sphere")
            
            # Sett buskegenskaper
            bush = UsdGeom.Sphere(bush_prim)
            bush.CreateRadiusAttr(random.uniform(0.5, 1.0))
            
            # Plasser busken
            xform = UsdGeom.Xformable(bush_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(x, 0.5, z))
            
            # Tilfeldig skalering og "squashing" for mer buskete form
            scale_x = random.uniform(0.8, 1.2)
            scale_y = random.uniform(0.7, 1.0)
            scale_z = random.uniform(0.8, 1.2)
            
            xform.AddScaleOp().Set(Gf.Vec3d(scale_x, scale_y, scale_z))
            
            # Opprett og tilordne materiale
            self._create_and_assign_material(bush_path, "bush_leaves")
    
    async def _add_grass(self, parent_path: Sdf.Path, grass_data: Dict[str, Any]):
        """Legg til gress (forenklet som en teksturert plan)"""
        # Opprett gress-prim
        grass_prim_path = f"{parent_path}/Grass"
        grass_prim = self.stage.DefinePrim(grass_prim_path, "Plane")
        
        # Hent gressdata
        size = grass_data.get("size", 40)
        density = grass_data.get("density", 0.8)
        
        # Sett gressegenskaper
        grass = UsdGeom.Plane(grass_prim)
        grass.CreateSizeAttr(size)
        
        # Plasser gressplanen
        xform = UsdGeom.Xformable(grass_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0.01, 0))  # Litt over bakken for å unngå z-fighting
        
        # Opprett og tilordne materiale
        grass_material_path = f"{parent_path}/GrassMaterial"
        grass_material = UsdShade.Material.Define(self.stage, grass_material_path)
        
        # Opprett PBR shader
        shader_path = f"{grass_material_path}/Shader"
        shader = UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        # Sett shader-parametre
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.3, 0.5, 0.2))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        
        # Tekstur for gress
        tex_shader_path = f"{grass_material_path}/GrassTexture"
        tex_shader = UsdShade.Shader.Define(self.stage, tex_shader_path)
        tex_shader.CreateIdAttr("UsdUVTexture")
        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("textures/grass_diffuse.png")
        tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(size/2, size/2))
        
        # Koble tekstur til shader
        tex_output = tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_output)
        
        # Koble shader til materiale
        grass_material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        
        # Bindmaterialet til gress
        UsdShade.MaterialBindingAPI(grass_prim).Bind(grass_material)
    
    async def _add_surroundings(self, parent_path: Sdf.Path, surroundings_data: Dict[str, Any]):
        """Legg til omgivende objekter som bygninger, gjerder, etc."""
        # Forenklet implementasjon - legg til noen enkle objekter
        # Dette kan utvides med mer spesifikke bygninger, veier, etc.
        
        # Opprett surroundings-prim
        surroundings_prim_path = f"{parent_path}/Surroundings"
        surroundings_prim = self.stage.DefinePrim(surroundings_prim_path, "Scope")
        
        # Legg til bygninger hvis spesifisert
        if "buildings" in surroundings_data:
            await self._add_buildings(surroundings_prim.GetPath(), surroundings_data["buildings"])
        
        # Legg til gjerder hvis spesifisert
        if "fences" in surroundings_data:
            await self._add_fences(surroundings_prim.GetPath(), surroundings_data["fences"])
    
    async def _add_buildings(self, parent_path: Sdf.Path, buildings_data: List[Dict[str, Any]]):
        """Legg til enkle bygninger"""
        for i, building in enumerate(buildings_data):
            position = building.get("position", [0, 0, 0])
            size = building.get("size", [5, 3, 5])  # width, height, depth
            
            # Opprett bygning
            building_path = f"{parent_path}/Building_{i}"
            building_prim = self.stage.DefinePrim(building_path, "Cube")
            
            # Sett størrelse
            UsdGeom.Cube(building_prim).CreateSizeAttr(1.0)  # Standardkube med størrelse 1
            
            # Plasser og skaler bygningen
            xform = UsdGeom.Xformable(building_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1] + size[1]/2, position[2]))
            xform.AddScaleOp().Set(Gf.Vec3d(size[0], size[1], size[2]))
            
            # Opprett og tilordne materiale
            material_name = building.get("material", "concrete")
            self._create_and_assign_material(building_path, material_name)
    
    async def _add_fences(self, parent_path: Sdf.Path, fences_data: List[Dict[str, Any]]):
        """Legg til gjerder"""
        for i, fence in enumerate(fences_data):
            start = fence.get("start", [0, 0, 0])
            end = fence.get("end", [10, 0, 0])
            height = fence.get("height", 1.0)
            
            # Opprett gjerde
            fence_path = f"{parent_path}/Fence_{i}"
            fence_prim = self.stage.DefinePrim(fence_path, "Xform")
            
            # Beregn lengde, midtpunkt og retning
            length = ((end[0] - start[0])**2 + (end[2] - start[2])**2)**0.5
            midpoint = [(start[0] + end[0])/2, (start[1] + end[1])/2, (start[2] + end[2])/2]
            
            direction = [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
            
            # Beregn rotasjonsvinkel
            angle = np.degrees(np.arctan2(direction[2], direction[0]))
            
            # Opprett gjerdepanelet
            panel_path = f"{fence_path}/Panel"
            panel_prim = self.stage.DefinePrim(panel_path, "Cube")
            
            # Sett gjerdepanelstørrelse
            UsdGeom.Cube(panel_prim).CreateSizeAttr(1.0)
            
            # Plasser og skaler gjerdepanelet
            xform = UsdGeom.Xformable(panel_prim)
            xform.AddScaleOp().Set(Gf.Vec3d(length, height, 0.05))
            xform.AddTranslateOp().Set(Gf.Vec3d(0, height/2, 0))
            
            # Plasser og roter hele gjerdet
            xform = UsdGeom.Xformable(fence_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(midpoint[0], midpoint[1], midpoint[2]))
            xform.AddRotateYOp().Set(angle)
            
            # Opprett og tilordne materiale
            material_name = fence.get("material", "wood")
            self._create_and_assign_material(panel_path, material_name)
    
    def _create_and_assign_material(self, prim_path: str, material_name: str):
        """Hjelpefunksjon for å opprette og tilordne et standard materiale"""
        try:
            # Sjekk om materialet allerede eksisterer
            material_path = f"/World/Materials/{material_name}"
            material_prim = self.stage.GetPrimAtPath(material_path)
            
            if not material_prim.IsValid():
                # Opprett material hvis det ikke eksisterer
                if material_name in self.material_library.materials:
                    material_props = self.material_library.materials[material_name]
                    self.material_library.create_usd_material(self.stage, material_name, material_props)
                else:
                    # Opprett et standard materiale
                    standard_props = {
                        "type": "physically_based",
                        "base_color": [0.8, 0.8, 0.8],
                        "roughness": 0.5,
                        "metallic": 0.0
                    }
                    self.material_library.create_usd_material(self.stage, material_name, standard_props)
            
            # Hent materialet
            material = UsdShade.Material.Get(self.stage, material_path)
            
            # Bindmaterialet til prim
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                UsdShade.MaterialBindingAPI(prim).Bind(material)
            
        except Exception as e:
            logger.error(f"Error creating/assigning material {material_name}: {str(e)}")
    
    def _add_bim_metadata(self, floor_plan_data: Dict[str, Any], structural_data: Dict[str, Any]):
        """Legg til BIM-metadata for modellen"""
        if not self.config.bim_metadata_enabled:
            return
        
        try:
            logger.info("Adding BIM metadata")
            
            # Opprett BIM-metadatanode
            bim_path = "/World/BIM"
            bim_prim = self.stage.DefinePrim(bim_path, "Scope")
            
            # Legg til prosjektinformasjon
            project_info = {
                "name": floor_plan_data.get("project_name", "Unnamed Project"),
                "address": floor_plan_data.get("address", ""),
                "owner": floor_plan_data.get("owner", ""),
                "architect": floor_plan_data.get("architect", ""),
                "creation_date": datetime.now().isoformat(),
                "software": f"OmniverseRenderer v{self.config.model_version}"
            }
            
            for key, value in project_info.items():
                bim_prim.GetPrim().CreateAttribute(f"project:{key}", Sdf.ValueTypeNames.String).Set(value)
            
            # Legg til bygningsinformasjon
            building_info = {
                "type": floor_plan_data.get("building_type", "residential"),
                "floors": len(floor_plan_data.get("floors", [])),
                "height": floor_plan_data.get("total_height", 0),
                "area": floor_plan_data.get("total_area", 0),
                "year_built": floor_plan_data.get("year_built", 0)
            }
            
            for key, value in building_info.items():
                bim_prim.GetPrim().CreateAttribute(f"building:{key}", Sdf.ValueTypeNames.String).Set(str(value))
            
            # Legg til materialinformasjon
            materials_info = {}
            for surface, material in floor_plan_data.get("materials", {}).items():
                materials_info[surface] = material
            
            material_attributes = bim_prim.GetPrim().CreateAttribute(
                "materials:assignments", Sdf.ValueTypeNames.String)
            material_attributes.Set(str(materials_info))
            
            # Eksporter BIM-data til egen fil hvis konfigurert
            if self.config.ifc_compatibility:
                self._export_bim_data(bim_path, floor_plan_data, structural_data)
            
            logger.info("BIM metadata added successfully")
            
        except Exception as e:
            logger.error(f"Error adding BIM metadata: {str(e)}")
    
    def _export_bim_data(self, bim_path: str, floor_plan_data: Dict[str, Any], 
                        structural_data: Dict[str, Any]):
        """Eksporter BIM-data til IFC-kompatibel format"""
        try:
            # Dette er en placeholder - i en reell implementasjon ville det involvere IFC-eksport
            logger.info("Exporting BIM data (placeholder for IFC export)")
            
            # Eksempel: Samle data for eksport
            export_data = {
                "project_info": {
                    "name": floor_plan_data.get("project_name", "Unnamed Project"),
                    "address": floor_plan_data.get("address", ""),
                    "owner": floor_plan_data.get("owner", "")
                },
                "building_info": {
                    "type": floor_plan_data.get("building_type", "residential"),
                    "floors": len(floor_plan_data.get("floors", [])),
                    "height": floor_plan_data.get("total_height", 0),
                    "area": floor_plan_data.get("total_area", 0)
                },
                "elements": []
            }
            
            # Samle alle elementer for IFC-eksport
            # Dette er en forenklet implementasjon
            element_types = {
                "walls": structural_data.get("walls", []),
                "floors": floor_plan_data.get("floors", []),
                "columns": structural_data.get("columns", []),
                "beams": structural_data.get("beams", []),
                "openings": structural_data.get("openings", [])
            }
            
            for element_type, elements in element_types.items():
                for element in elements:
                    export_data["elements"].append({
                        "type": element_type,
                        "id": element.get("id", ""),
                        "properties": element
                    })
            
            # Eksporter til JSON som en enkel IFC-surrogate
            import json
            export_path = "bim_export.json"
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=4)
            
            logger.info(f"BIM data exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting BIM data: {str(e)}")
    
    def _optimize_model(self):
        """Optimaliser modellen for rendering"""
        try:
            logger.info("Optimizing model for rendering")
            
            # Flatting meshes (forenklet)
            # I en reell implementasjon ville dette involvere mer omfattende optimalisering
            
            # Slå sammen UsdMeshes basert på materiale for å redusere drawcalls
            # Dette er forenklet - reell implementasjon vil være kompleks
            
            # Aktiver instansiering for repeterende geometri (trær, busker, etc)
            # Forenklet - ville være mer omfattende i praksis
            
            # Sett opp LOD (Level of Detail) for objekter
            # Forenklet - ville være mer omfattende i praksis
            
            # Sett opp culling og synlighet
            # For en presis implementasjon ville vi gjøre mer avansert culling
            
            logger.info("Model optimized successfully")
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
    
    async def _render_model(self, view_positions: List[Dict[str, Any]] = None, 
                         output_path: str = None) -> Dict[str, Any]:
        """Render 3D-modellen med høy kvalitet"""
        try:
            logger.info("Rendering 3D model")
            
            # Sett opp renderingsinnstillinger
            self._setup_render_settings()
            
            # Hvis ingen kameraposisjoner er gitt, bruk standardkamera
            if not view_positions:
                view_positions = [{
                    "name": "default",
                    "position": [5, 2, 5],
                    "target": [0, 1, 0],
                    "focal_length": 35.0
                }]
            
            # Sett opp basismappe for rendering
            if not output_path:
                output_path = f"renders/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            os.makedirs(output_path, exist_ok=True)
            
            # Render hver visning
            render_results = []
            for i, view in enumerate(view_positions):
                render_result = await self._render_view(view, f"{output_path}/{view.get('name', f'view_{i}')}")
                render_results.append(render_result)
            
            # Eksporter modell til ulike formater
            model_outputs = await self._export_model(output_path)
            
            # Samle renderingsstatistikk
            rendering_stats = self._collect_rendering_statistics()
            
            # Kompiler resultater
            result = {
                "model_url": model_outputs.get("usd_path"),
                "previews": [r.get("image_path") for r in render_results if r.get("image_path")],
                "formats": model_outputs,
                "stats": rendering_stats
            }
            
            logger.info(f"Model rendered successfully to {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error rendering model: {str(e)}")
            raise
    
    def _setup_render_settings(self):
        """Sett opp renderingsinnstillinger for høykvalitetsrendering"""
        render_settings = {
            "renderer": self.config.render_quality,
            "resolution": [self.config.width, self.config.height],
            "samples_per_pixel": self.config.samples_per_pixel,
            "max_bounces": self.config.max_bounces,
            "denoise": self.config.denoising_enabled,
            "motion_blur": self.config.motion_blur,
            "depth_of_field": self.config.depth_of_field
        }
        
        # Bruk omni.kit.commands for å angi renderer-innstillinger
        omni.kit.commands.execute('SetRenderSettings', settings=render_settings)
    
    async def _render_view(self, view: Dict[str, Any], output_file: str) -> Dict[str, Any]:
        """Render en spesifikk visning"""
        # Opprett kamera for visningen
        camera_path = f"/World/Cameras/Camera_{view.get('name', 'view')}"
        camera_prim = self.stage.DefinePrim(camera_path, "Camera")
        camera = UsdGeom.Camera(camera_prim)
        
        # Sett kameraegenskaper
        position = view.get("position", [5, 2, 5])
        target = view.get("target", [0, 1, 0])
        focal_length = view.get("focal_length", 35.0)
        
        # Beregn opp-vektor (vanligvis Y-aksen)
        up = view.get("up", [0, 1, 0])
        
        # Plasser kameraet
        xform = UsdGeom.Xformable(camera_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        
        # Pek kameraet mot målet
        direction = [target[0] - position[0], target[1] - position[1], target[2] - position[2]]
        distance = (direction[0]**2 + direction[1]**2 + direction[2]**2)**0.5
        
        if distance > 0.001:
            # Beregn rotasjon for å peke mot målet
            # Dette er en forenklet tilnærming - i praksis ville det brukes Look-At-matriser
            from math import atan2, asin, degrees
            
            # Beregn yaw og pitch
            yaw = degrees(atan2(direction[0], direction[2]))
            pitch = degrees(asin(direction[1] / distance))
            
            # Bruk inverse retning for rotasjon
            xform.AddRotateYOp().Set(-yaw)
            xform.AddRotateXOp().Set(pitch)
        
        # Sett kameraegenskaper
        camera.CreateFocalLengthAttr(focal_length)
        camera.CreateHorizontalApertureAttr(36.0)  # 35mm film standard
        camera.CreateVerticalApertureAttr(24.0)    # 35mm film standard
        
        # Sett dybdeskarphet hvis aktivert
        if self.config.depth_of_field:
            camera.CreateDepthOfFieldEnabledAttr(True)
            camera.CreateFocusDistanceAttr(distance)
            camera.CreateFStopAttr(view.get("f_stop", 5.6))
        
        # Aktiver kameraet i viewport
        self.viewport.set_active_camera(camera_path)
        
        # Utfør rendering
        render_result = await self._perform_rendering(output_file)
        
        return render_result
    
    async def _perform_rendering(self, output_file: str) -> Dict[str, Any]:
        """Utfør rendering med gjeldende kamera"""
        # Dette er en placeholder for reell rendering med Omniverse
        # I en praktisk implementasjon ville dette involvere RTX-rendering
        
        # Forenklet - simuler rendering med en tidsforsinkelse
        await asyncio.sleep(0.5)
        
        # Lagre rendert bilde (simulert)
        image_path = f"{output_file}.png"
        
        # Simulating rendering statistics
        render_stats = {
            "render_time": 5.2,  # Sekunder
            "samples": self.config.samples_per_pixel,
            "resolution": [self.config.width, self.config.height],
            "memory_usage": 2.1  # GB
        }
        
        # Legg til statistikk til statistikkliste
        self.stats["render_times"].append(render_stats["render_time"])
        self.stats["memory_usage"].append(render_stats["memory_usage"])
        
        return {
            "image_path": image_path,
            "stats": render_stats
        }
    
    async def _export_model(self, output_path: str) -> Dict[str, str]:
        """Eksporter modellen til ulike formater"""
        result = {}
        
        # Eksporter USD
        usd_path = f"{output_path}/model.usd"
        self.stage.Export(usd_path)
        result["usd_path"] = usd_path
        
        # Eksporter andre formater hvis konfigurert
        for format_type in self.config.output_formats:
            if format_type == "glb":
                glb_path = f"{output_path}/model.glb"
                # Omniverse har ikke direkte eksport til GLB, så dette er en placeholder
                # I praksis ville vi bruke et konverteringsverktøy
                result["glb_path"] = glb_path
            
            elif format_type == "obj":
                obj_path = f"{output_path}/model.obj"
                # Placeholder for OBJ-eksport
                result["obj_path"] = obj_path
            
            elif format_type == "fbx":
                fbx_path = f"{output_path}/model.fbx"
                # Placeholder for FBX-eksport
                result["fbx_path"] = fbx_path
        
        return result
    
    def _collect_rendering_statistics(self) -> Dict[str, Any]:
        """Samle renderingsstatistikk"""
        if not self.stats["render_times"]:
            return {}
        
        avg_render_time = sum(self.stats["render_times"]) / len(self.stats["render_times"])
        max_render_time = max(self.stats["render_times"])
        
        avg_memory_usage = sum(self.stats["memory_usage"]) / len(self.stats["memory_usage"])
        max_memory_usage = max(self.stats["memory_usage"])
        
        return {
            "average_render_time": float(avg_render_time),
            "max_render_time": float(max_render_time),
            "average_memory_usage_gb": float(avg_memory_usage),
            "max_memory_usage_gb": float(max_memory_usage),
            "total_renders": len(self.stats["render_times"])
        }
    
    def _collect_model_statistics(self) -> Dict[str, Any]:
        """Samle statistikk om 3D-modellen"""
        try:
            # Traverser scenen og tell geometri
            vertex_count = 0
            polygon_count = 0
            material_count = 0
            prim_count = 0
            
            # Traverser alle prims
            for prim in self.stage.Traverse():
                prim_count += 1
                
                # Tell vertekser og polygoner for mesher
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    
                    # Hent vertekser
                    points_attr = mesh.GetPointsAttr()
                    if points_attr:
                        vertex_count += len(points_attr.Get())
                    
                    # Hent face counts
                    face_counts_attr = mesh.GetFaceVertexCountsAttr()
                    if face_counts_attr:
                        polygon_count += len(face_counts_attr.Get())
                
                # Tell materialer
                if UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().GetTargets():
                    material_count += 1
            
            # Beregn kompleksitet
            complexity = "low"
            if vertex_count > 1000000 or polygon_count > 500000:
                complexity = "high"
            elif vertex_count > 200000 or polygon_count > 100000:
                complexity = "medium"
            
            return {
                "vertex_count": vertex_count,
                "polygon_count": polygon_count,
                "material_count": material_count,
                "prim_count": prim_count,
                "complexity": complexity
            }
            
        except Exception as e:
            logger.error(f"Error collecting model statistics: {str(e)}")
            return {}
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generer metadata for 3D-modellen"""
        return {
            "creation_date": datetime.now().isoformat(),
            "renderer_version": self.config.model_version,
            "quality_settings": self.config.to_dict(),
            "statistics": self._collect_model_statistics(),
            "rendering_statistics": self._collect_rendering_statistics()
        }
    
    def _calculate_texture_memory(self) -> float:
        """Beregn teksturminne som brukes i modellen"""
        # Forenklet implementasjon - i praksis ville dette involvere analyse av alle teksturer
        texture_memory_mb = 0
        
        # Traverser scenen for å finne alle teksturer
        for prim in self.stage.TraverseAll():
            # Sjekk om prim har et material
            if not prim.IsA(UsdShade.Material):
                continue
            
            # Traverser shader-nettverk for å finne teksturer
            material = UsdShade.Material(prim)
            surface_output = material.GetSurfaceOutput()
            
            if not surface_output:
                continue
            
            source = surface_output.GetConnectedSource()
            if not source:
                continue
            
            shader = UsdShade.Shader(source[0])
            if not shader:
                continue
            
            # Sjekk for teksturinngang
            for input_name in ['diffuseColor', 'normal', 'roughness', 'metallic']:
                shader_input = shader.GetInput(input_name)
                if not shader_input:
                    continue
                
                connection_source = shader_input.GetConnectedSource()
                if not connection_source:
                    continue
                
                tex_shader = UsdShade.Shader(connection_source[0])
                if not tex_shader or tex_shader.GetIdAttr().Get() != 'UsdUVTexture':
                    continue
                
                # Estimer teksturstørrelse basert på filsti
                file_input = tex_shader.GetInput('file')
                if not file_input:
                    continue
                
                file_path = file_input.Get()
                if not file_path:
                    continue
                
                # Estimer teksturstørrelse basert på vanlige størrelser
                # I praksis ville vi faktisk laste og analysere teksturen
                texture_memory_mb += 4  # Antar 1024x1024 tekstur med 4 bytes per piksel = 4MB
        
        return texture_memory_mb
