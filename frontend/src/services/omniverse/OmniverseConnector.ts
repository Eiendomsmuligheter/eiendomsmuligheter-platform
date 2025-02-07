import { OmniPBRMaterial, Omniverse } from '@nvidia/omniverse';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader';

interface OmniverseConfig {
  quality: 'low' | 'medium' | 'high';
  enableRayTracing: boolean;
  textureResolution: number;
  enableShadows?: boolean;
  enableReflections?: boolean;
}

export class OmniverseConnector {
  private omniverse: typeof Omniverse;
  private materials: Map<string, OmniPBRMaterial>;
  private gltfLoader: GLTFLoader;
  private dracoLoader: DRACOLoader;
  private scene: THREE.Scene | null;

  constructor() {
    this.materials = new Map();
    this.scene = null;

    // Initialiser GLTF loader med Draco-komprimering
    this.gltfLoader = new GLTFLoader();
    this.dracoLoader = new DRACOLoader();
    this.dracoLoader.setDecoderPath('/draco/');
    this.gltfLoader.setDRACOLoader(this.dracoLoader);

    // Initialiser Omniverse
    this.initializeOmniverse();
  }

  private async initializeOmniverse() {
    try {
      this.omniverse = await Omniverse.initialize({
        appId: process.env.NVIDIA_APP_ID,
        apiKey: process.env.NVIDIA_API_KEY,
      });

      // Sett opp standard materialer
      await this.setupDefaultMaterials();
    } catch (error) {
      console.error('Failed to initialize Omniverse:', error);
      throw error;
    }
  }

  private async setupDefaultMaterials() {
    // Standard materialer for ulike overflater
    const materialConfigs = {
      wood: {
        baseColor: new THREE.Color(0.7, 0.5, 0.3),
        roughness: 0.7,
        metallic: 0.0,
        normalStrength: 1.0,
      },
      concrete: {
        baseColor: new THREE.Color(0.8, 0.8, 0.8),
        roughness: 0.9,
        metallic: 0.1,
        normalStrength: 0.5,
      },
      glass: {
        baseColor: new THREE.Color(0.9, 0.9, 0.9),
        roughness: 0.1,
        metallic: 0.9,
        transparency: 0.9,
        ior: 1.5,
      },
      metal: {
        baseColor: new THREE.Color(0.8, 0.8, 0.8),
        roughness: 0.4,
        metallic: 1.0,
        normalStrength: 0.5,
      },
    };

    for (const [name, config] of Object.entries(materialConfigs)) {
      const material = new OmniPBRMaterial({
        ...config,
        name,
      });
      this.materials.set(name, material);
    }
  }

  public async loadModel(
    modelUrl: string,
    config: OmniverseConfig
  ): Promise<THREE.Group> {
    try {
      // Last modellen via GLTF først
      const gltf = await this.loadGLTFModel(modelUrl);

      // Konverter til Omniverse-format
      const omniverseModel = await this.convertToOmniverse(gltf, config);

      // Optimaliser modellen
      await this.optimizeModel(omniverseModel, config);

      return omniverseModel;
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  private async loadGLTFModel(url: string): Promise<THREE.Group> {
    return new Promise((resolve, reject) => {
      this.gltfLoader.load(
        url,
        (gltf) => resolve(gltf.scene),
        undefined,
        reject
      );
    });
  }

  private async convertToOmniverse(
    model: THREE.Group,
    config: OmniverseConfig
  ): Promise<THREE.Group> {
    // Konverter materiale til Omniverse PBR
    model.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const material = child.material as THREE.MeshStandardMaterial;
        const omniverseMaterial = this.createOmniverseMaterial(material, config);
        child.material = omniverseMaterial;
      }
    });

    // Optimaliser geometri
    model.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry.computeBoundingBox();
        child.geometry.computeVertexNormals();
        if (config.quality !== 'low') {
          child.geometry.computeTangents();
        }
      }
    });

    return model;
  }

  private createOmniverseMaterial(
    originalMaterial: THREE.MeshStandardMaterial,
    config: OmniverseConfig
  ): OmniPBRMaterial {
    const omniverseMaterial = new OmniPBRMaterial({
      baseColor: originalMaterial.color,
      roughness: originalMaterial.roughness,
      metallic: originalMaterial.metalness,
      normalStrength: 1.0,
      enableRayTracing: config.enableRayTracing,
    });

    // Overfør teksturer
    if (originalMaterial.map) {
      omniverseMaterial.setTexture('baseColor', this.optimizeTexture(
        originalMaterial.map,
        config.textureResolution
      ));
    }
    if (originalMaterial.normalMap) {
      omniverseMaterial.setTexture('normal', this.optimizeTexture(
        originalMaterial.normalMap,
        config.textureResolution
      ));
    }
    if (originalMaterial.roughnessMap) {
      omniverseMaterial.setTexture('roughness', this.optimizeTexture(
        originalMaterial.roughnessMap,
        config.textureResolution
      ));
    }

    return omniverseMaterial;
  }

  private optimizeTexture(
    texture: THREE.Texture,
    resolution: number
  ): THREE.Texture {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    
    // Sett oppløsning
    canvas.width = resolution;
    canvas.height = resolution;
    
    // Tegn original tekstur til canvas med ny oppløsning
    ctx.drawImage(texture.image, 0, 0, resolution, resolution);
    
    // Opprett ny tekstur
    const optimizedTexture = new THREE.Texture(canvas);
    optimizedTexture.needsUpdate = true;
    
    return optimizedTexture;
  }

  private async optimizeModel(
    model: THREE.Group,
    config: OmniverseConfig
  ): Promise<void> {
    // Optimaliser basert på kvalitetsnivå
    if (config.quality === 'low') {
      this.simplifyGeometry(model, 0.5); // 50% reduksjon
      this.reduceTextureResolution(model, 512);
    } else if (config.quality === 'medium') {
      this.simplifyGeometry(model, 0.75); // 25% reduksjon
      this.reduceTextureResolution(model, 1024);
    }

    // Sett opp skygger
    if (config.enableShadows) {
      model.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
    }

    // Sett opp refleksjoner
    if (config.enableReflections) {
      model.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          const material = child.material as OmniPBRMaterial;
          if (material.metallic > 0.5 || material.roughness < 0.5) {
            material.enableSSR = true;
          }
        }
      });
    }
  }

  private simplifyGeometry(model: THREE.Group, factor: number): void {
    // Implementer geometriforenkling
    model.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        // Reduser vertekser
        const geometry = child.geometry;
        const vertices = geometry.attributes.position.array;
        // ... implementer geometriforenkling
      }
    });
  }

  private reduceTextureResolution(model: THREE.Group, maxResolution: number): void {
    model.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const material = child.material as OmniPBRMaterial;
        // Reduser teksturoppløsning for alle teksturkart
        Object.values(material.textures).forEach((texture) => {
          if (texture && (texture.image.width > maxResolution || 
              texture.image.height > maxResolution)) {
            this.optimizeTexture(texture, maxResolution);
          }
        });
      }
    });
  }

  public update(deltaTime: number): void {
    // Oppdater materialer og effekter
    this.materials.forEach((material) => {
      material.update(deltaTime);
    });

    // Oppdater ray-tracing hvis aktivert
    if (this.scene) {
      this.omniverse.updateRayTracing(this.scene);
    }
  }

  public dispose(): void {
    // Rydd opp ressurser
    this.materials.forEach((material) => {
      material.dispose();
    });
    this.materials.clear();
    this.dracoLoader.dispose();
  }
}