/**
 * TerrainVisualizer.ts
 * 
 * Avansert terrengvisualiseringsmodul for Eiendomsmuligheter Platform
 * Håndterer generering og rendering av terreng basert på høydedata med avanserte
 * funksjoner som adaptiv detaljnivå (LOD), teksturblending og erosjonssimulering.
 * 
 * @author Eiendomsmuligheter Platform
 * @version 1.0.0
 */

import * as THREE from 'three';
// @ts-ignore - Ignorer typeproblemene for Three.js-imports
import { SimplexNoise } from 'three/examples/jsm/math/SimplexNoise';
// @ts-ignore
import { MeshSurfaceSampler } from 'three/examples/jsm/math/MeshSurfaceSampler';

/**
 * Interface for terrengkonfigurasjon
 */
export interface TerrainSettings {
  // Basis konfigurasjon
  width: number;
  height: number;
  resolution: number;
  heightScale: number;
  
  // Teksturkontroll
  baseTexture?: string;
  detailTexture?: string;
  slopeTexture?: string;
  rockTexture?: string;
  snowTexture?: string;
  
  // LOD-innstillinger
  useLOD?: boolean;
  lodLevels?: number;
  lodDistance?: number;
  
  // Vegetasjon
  includeVegetation?: boolean;
  vegetationDensity?: number;
  vegetationTypes?: string[];
  
  // Vann
  includeWater?: boolean;
  waterLevel?: number;
  waterColor?: number;
  
  // Avanserte funksjoner
  useErosion?: boolean;
  erosionIterations?: number;
}

/**
 * Grensesnitt for terrengdata
 */
export interface TerrainData {
  heightmap: Float32Array | number[];
  width: number;
  depth: number;
  metadata?: any;
}

/**
 * Klasse for avansert terrengvisualisering
 */
export class TerrainVisualizer {
  private scene: THREE.Scene;
  private terrain: THREE.Mesh | null = null;
  private waterMesh: THREE.Mesh | null = null;
  private vegetation: THREE.Group | null = null;
  
  private heightData: Float32Array | null = null;
  private settings: TerrainSettings;
  private isDisposed: boolean = false;
  
  // Materialer og teksturer
  private terrainMaterial: THREE.Material | null = null;
  private waterMaterial: THREE.Material | null = null;
  private textures: Map<string, THREE.Texture> = new Map();
  
  // LOD-relaterte egenskaper
  private lodMeshes: THREE.Mesh[] = [];
  
  /**
   * Konstruktør
   * 
   * @param scene Three.js scene som terrenget skal legges til
   * @param settings Konfigurasjonsinnstillinger for terrenget
   */
  constructor(scene: THREE.Scene, settings: TerrainSettings) {
    this.scene = scene;
    this.settings = this.applyDefaultSettings(settings);
  }
  
  /**
   * Legg til standardinnstillinger for manglende verdier
   */
  private applyDefaultSettings(settings: TerrainSettings): TerrainSettings {
    return {
      ...settings,
      heightScale: settings.heightScale || 10,
      resolution: settings.resolution || 128,
      useLOD: settings.useLOD !== undefined ? settings.useLOD : true,
      lodLevels: settings.lodLevels || 4,
      lodDistance: settings.lodDistance || 500,
      includeVegetation: settings.includeVegetation !== undefined ? settings.includeVegetation : false,
      vegetationDensity: settings.vegetationDensity || 0.01,
      includeWater: settings.includeWater !== undefined ? settings.includeWater : false,
      waterLevel: settings.waterLevel || 0,
      waterColor: settings.waterColor || 0x1a97f7,
      useErosion: settings.useErosion !== undefined ? settings.useErosion : false,
      erosionIterations: settings.erosionIterations || 5
    };
  }
  
  /**
   * Last inn høydedata fra en URL
   * 
   * @param url URL til PNG høydekartbilde
   * @returns Promise som løses når høydedata er lastet
   */
  public async loadHeightmapFromUrl(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const loader = new THREE.TextureLoader();
      loader.load(
        url,
        (texture) => {
          // Konverter tekstur til høydedata
          this.heightData = this.extractHeightFromTexture(texture);
          resolve();
        },
        undefined,
        (error) => {
          console.error('Feil ved lasting av høydekart:', error);
          reject(error);
        }
      );
    });
  }
  
  /**
   * Last inn terrengdata direkte
   * 
   * @param data Terrengdata med høydekart
   */
  public loadTerrainData(data: TerrainData): void {
    if (Array.isArray(data.heightmap) || data.heightmap instanceof Float32Array) {
      this.heightData = data.heightmap instanceof Float32Array 
        ? data.heightmap 
        : new Float32Array(data.heightmap);
      
      // Oppdater innstillinger basert på data
      this.settings.width = data.width || this.settings.width;
      this.settings.height = data.depth || this.settings.height;
    } else {
      console.error('Ugyldig høydekartdata. Forventer Array eller Float32Array.');
    }
  }
  
  /**
   * Ekstraher høydeverdier fra en tekstur
   */
  private extractHeightFromTexture(texture: THREE.Texture): Float32Array {
    // Opprett en offscreen canvas for å lese pikseldata
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Kunne ikke opprette canvas-kontekst');
    }
    
    // Sett størrelse lik tekstur
    const resolution = this.settings.resolution;
    canvas.width = resolution;
    canvas.height = resolution;
    
    // Tegn teksturen på canvas
    const image = texture.image;
    context.drawImage(image, 0, 0, resolution, resolution);
    
    // Les pikseldata
    const imgData = context.getImageData(0, 0, resolution, resolution);
    const pixels = imgData.data;
    
    // Konverter til float array (bruker kun rødkanal for høyde)
    const heightData = new Float32Array(resolution * resolution);
    for (let i = 0; i < resolution * resolution; i++) {
      // Bruk rødkanalen (hver 4. byte, siden data er RGBA)
      heightData[i] = pixels[i * 4] / 255.0;
    }
    
    return heightData;
  }
  
  /**
   * Generer terrenget basert på innlastede høydedata
   */
  public generateTerrain(): THREE.Mesh | null {
    if (!this.heightData) {
      console.error('Ingen høydedata tilgjengelig. Last inn data først.');
      return null;
    }
    
    // Fjern eksisterende terreng hvis det finnes
    if (this.terrain) {
      this.scene.remove(this.terrain);
      this.terrain.geometry.dispose();
      if (this.terrain.material instanceof THREE.Material) {
        this.terrain.material.dispose();
      }
    }
    
    // Opprett geometri
    const geometry = this.createTerrainGeometry();
    
    // Opprett materiale
    const material = this.createTerrainMaterial();
    
    // Opprett mesh
    this.terrain = new THREE.Mesh(geometry, material);
    this.terrain.castShadow = true;
    this.terrain.receiveShadow = true;
    this.terrain.rotation.x = -Math.PI / 2; // Roter til horisontalplan
    
    // Legg til i scenen
    this.scene.add(this.terrain);
    
    // Legg til vann hvis aktivert
    if (this.settings.includeWater) {
      this.addWater();
    }
    
    // Legg til vegetasjon hvis aktivert
    if (this.settings.includeVegetation) {
      this.addVegetation();
    }
    
    // Opprett LOD-mesh hvis aktivert
    if (this.settings.useLOD) {
      this.createLODMeshes();
    }
    
    return this.terrain;
  }
  
  /**
   * Opprett terrenggeometri basert på høydedata
   */
  private createTerrainGeometry(): THREE.BufferGeometry {
    const resolution = this.settings.resolution;
    const width = this.settings.width;
    const height = this.settings.height;
    const heightScale = this.settings.heightScale;
    
    // Opprett planet geometri
    const geometry = new THREE.PlaneGeometry(
      width, 
      height, 
      resolution - 1, 
      resolution - 1
    );
    
    // Modifiser høydeverdier
    const positions = geometry.attributes.position.array;
    for (let i = 0; i < positions.length / 3; i++) {
      // Z-komponenten er høyden (Y i world space etter rotasjon)
      positions[i * 3 + 2] = (this.heightData as Float32Array)[i] * heightScale;
    }
    
    // Oppdater posisjon og beregn normaler
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
    
    // Beregn biasing for teksturering basert på helning
    this.calculateSlopeAttribute(geometry);
    
    return geometry;
  }
  
  /**
   * Beregn helningsattributt for bruk i teksturblandingen
   */
  private calculateSlopeAttribute(geometry: THREE.BufferGeometry): void {
    const positions = geometry.attributes.position.array;
    const normals = geometry.attributes.normal.array;
    const count = geometry.attributes.position.count;
    
    // Opprett buffer for helningsfaktor (0 = flat, 1 = vertikal)
    const slopeFactors = new Float32Array(count);
    
    for (let i = 0; i < count; i++) {
      // Beregn punktets normal
      const nx = normals[i * 3];
      const ny = normals[i * 3 + 1];
      const nz = normals[i * 3 + 2];
      
      // Beregn helningen ved å sammenligne med oppovervektor (0,1,0)
      // Etter rotasjon vil dette være (0,0,1) i lokale koordinater
      const dotProduct = nx * 0 + ny * 0 + nz * 1;
      
      // Convert dot product to slope factor (0 = vertical, 1 = flat)
      // (dot product gir cosinus til vinkelen mellom normalene)
      slopeFactors[i] = 1.0 - Math.abs(dotProduct);
    }
    
    // Legg til som buffer-attributt
    geometry.setAttribute('slope', new THREE.BufferAttribute(slopeFactors, 1));
  }
  
  /**
   * Opprett terrengmaterialet med støtte for teksturblandinger
   */
  private createTerrainMaterial(): THREE.Material {
    // Last nødvendige teksturer
    const baseTexture = this.loadTexture(this.settings.baseTexture || '/textures/terrain/grass.jpg');
    const detailTexture = this.loadTexture(this.settings.detailTexture || '/textures/terrain/detail.jpg');
    const slopeTexture = this.loadTexture(this.settings.slopeTexture || '/textures/terrain/rock.jpg');
    
    // Juster teksturer
    if (baseTexture) baseTexture.wrapS = baseTexture.wrapT = THREE.RepeatWrapping;
    if (detailTexture) detailTexture.wrapS = detailTexture.wrapT = THREE.RepeatWrapping;
    if (slopeTexture) slopeTexture.wrapS = slopeTexture.wrapT = THREE.RepeatWrapping;
    
    // Definer tekstur-skala
    const textureScale = [this.settings.width / 20, this.settings.height / 20];
    
    // Opprett shader-materiale for teksturblanding basert på helning
    const material = new THREE.ShaderMaterial({
      uniforms: {
        baseTexture: { value: baseTexture },
        detailTexture: { value: detailTexture },
        slopeTexture: { value: slopeTexture },
        textureScale: { value: new THREE.Vector2(textureScale[0], textureScale[1]) },
        slopeThreshold: { value: 0.3 },
        slopeBlend: { value: 0.1 }
      },
      vertexShader: `
        varying vec2 vUv;
        varying float vSlope;
        attribute float slope;
        
        void main() {
          vUv = uv;
          vSlope = slope;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D baseTexture;
        uniform sampler2D detailTexture;
        uniform sampler2D slopeTexture;
        uniform vec2 textureScale;
        uniform float slopeThreshold;
        uniform float slopeBlend;
        
        varying vec2 vUv;
        varying float vSlope;
        
        void main() {
          // Sample teksturer med riktig skala
          vec2 scaledUv = vUv * textureScale;
          vec4 baseColor = texture2D(baseTexture, scaledUv);
          vec4 detailColor = texture2D(detailTexture, scaledUv * 10.0);
          vec4 slopeColor = texture2D(slopeTexture, scaledUv);
          
          // Bland base og detalj
          vec4 blendedBase = baseColor * 0.8 + detailColor * 0.2;
          
          // Beregn blandingsfaktor basert på helning
          float slopeFactor = smoothstep(slopeThreshold - slopeBlend, slopeThreshold + slopeBlend, vSlope);
          
          // Bland mellom base og helning
          gl_FragColor = mix(blendedBase, slopeColor, slopeFactor);
        }
      `,
      lights: true,
      fog: true
    });
    
    // Fikse TypeScript type-issue med extensions
    (material as any).extensions = {
      derivatives: true,
      fragDepth: false,
      drawBuffers: false,
      shaderTextureLOD: false
    };
    
    this.terrainMaterial = material;
    return material;
  }
  
  /**
   * Last tekstur og caché
   */
  private loadTexture(url: string | undefined): THREE.Texture | null {
    if (!url) return null;
    
    // Sjekk cache
    if (this.textures.has(url)) {
      return this.textures.get(url) || null;
    }
    
    // Last tekstur
    const texture = new THREE.TextureLoader().load(url);
    this.textures.set(url, texture);
    return texture;
  }
  
  /**
   * Legg til vann
   */
  private addWater(): void {
    if (!this.settings.includeWater) return;
    
    // Fjern eksisterende vannmesh
    if (this.waterMesh) {
      this.scene.remove(this.waterMesh);
      this.waterMesh.geometry.dispose();
      if (this.waterMesh.material instanceof THREE.Material) {
        this.waterMesh.material.dispose();
      }
    }
    
    // Opprett geometri for vann (enkelt plan)
    const waterGeometry = new THREE.PlaneGeometry(
      this.settings.width * 1.2, // Litt større enn terrenget
      this.settings.height * 1.2
    );
    
    // Opprett vannmateriale med transparens og refleksjon
    const waterMaterial = new THREE.MeshPhysicalMaterial({
      color: this.settings.waterColor || 0x1a97f7,
      transparent: true,
      opacity: 0.8,
      roughness: 0.1,
      metalness: 0.1,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      reflectivity: 1.0,
      side: THREE.DoubleSide
    });
    
    // Opprett vannmesh
    this.waterMesh = new THREE.Mesh(waterGeometry, waterMaterial);
    this.waterMesh.rotation.x = -Math.PI / 2; // Roter til horisontalplan
    this.waterMesh.position.y = this.settings.waterLevel || 0.1; // Litt over bakkenivå
    this.waterMesh.receiveShadow = true;
    
    // Legg til i scenen
    this.scene.add(this.waterMesh);
    this.waterMaterial = waterMaterial;
  }
  
  /**
   * Legg til vegetasjon
   */
  private addVegetation(): void {
    if (!this.settings.includeVegetation || !this.terrain) return;
    
    // Fjern eksisterende vegetasjon
    if (this.vegetation) {
      this.scene.remove(this.vegetation);
      // Dispose av geometrier og materialer...
    }
    
    // Opprett vegetasjonsgruppe
    this.vegetation = new THREE.Group();
    
    // Bruk MeshSurfaceSampler for å plassere vegetasjon naturlig
    const sampler = new MeshSurfaceSampler(this.terrain)
      .setWeightAttribute('slope') // Bruk helning-attributtet
      .build();
    
    // TODO: Last vegetasjonsmodeller og plasser dem på terrenget
    // Dette er en avansert funksjon som krever modellbibliotek
    
    // Legg til i scenen
    this.scene.add(this.vegetation);
  }
  
  /**
   * Opprett LOD-mesh (Level of Detail)
   */
  private createLODMeshes(): void {
    if (!this.settings.useLOD || !this.terrain) return;
    
    // Fjern eksisterende LOD-mesh
    for (const mesh of this.lodMeshes) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      }
    }
    this.lodMeshes = [];
    
    // TODO: Implementer faktisk LOD-systemet
    // Dette involverer å opprette forenklede versjoner av terrenggeometri
  }
  
  /**
   * Appliker erosjonssimulering for mer realistisk terreng
   */
  private applyErosion(): void {
    if (!this.settings.useErosion || !this.heightData) return;
    
    // TODO: Implementer hydraulisk erosjonssimulering
    // Dette er en kompleks algoritme som krever mer arbeid
    // Se eksempler: https://github.com/SebLague/Hydraulic-Erosion
  }
  
  /**
   * Oppdater terrenget (bør kalles fra renderingsløkke)
   * 
   * @param camera Aktiv kamera
   */
  public update(camera: THREE.Camera): void {
    if (this.isDisposed) return;
    
    // Oppdater LOD basert på kameraavstand hvis aktivert
    if (this.settings.useLOD && this.terrain && this.lodMeshes.length > 0) {
      const cameraPosition = camera.position;
      // TODO: Implementer LOD-switching basert på kameradistanse
    }
    
    // Animer vann hvis det er aktivert
    if (this.waterMesh && this.waterMesh.material instanceof THREE.ShaderMaterial) {
      // TODO: Animer vannoverflate
    }
  }
  
  /**
   * Kast ressurser
   */
  public dispose(): void {
    this.isDisposed = true;
    
    // Fjern terreng
    if (this.terrain) {
      this.scene.remove(this.terrain);
      this.terrain.geometry.dispose();
      if (this.terrain.material instanceof THREE.Material) {
        this.terrain.material.dispose();
      }
      this.terrain = null;
    }
    
    // Fjern vann
    if (this.waterMesh) {
      this.scene.remove(this.waterMesh);
      this.waterMesh.geometry.dispose();
      if (this.waterMesh.material instanceof THREE.Material) {
        this.waterMesh.material.dispose();
      }
      this.waterMesh = null;
    }
    
    // Fjern vegetasjon
    if (this.vegetation) {
      this.scene.remove(this.vegetation);
      // Dispose barneressurser...
      this.vegetation = null;
    }
    
    // Fjern LOD-mesh
    for (const mesh of this.lodMeshes) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      }
    }
    this.lodMeshes = [];
    
    // Dispose teksturer
    this.textures.forEach(texture => {
      texture.dispose();
    });
    this.textures.clear();
    
    // Nullstill data
    this.heightData = null;
  }
} 