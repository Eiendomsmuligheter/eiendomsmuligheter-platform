/**
 * TerravisionEngine.ts
 * 
 * En høyytelse 3D-visualiseringsmotor for eiendomsdata
 * Optimalisert for web-plattformer og implementerer avanserte rendering-teknikker
 * med fokus på ytelse og brukervennlighet.
 * 
 * @author Eiendomsmuligheter Platform
 * @version 1.1.0
 */

import * as THREE from 'three';
// @ts-ignore - Ignorer typeproblemene for Three.js-imports
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
// @ts-ignore
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
// @ts-ignore
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader';
// @ts-ignore
import { mergeBufferGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils';

// Egendefinerte interfaces for Three.js utvidelser
interface ExtendedWebGLRenderer extends THREE.WebGLRenderer {
  physicallyCorrectLights?: boolean;
  outputEncoding?: number;
  toneMapping?: number;
  toneMappingExposure?: number;
  preserveDrawingBuffer?: boolean;
  info: {
    memory: {
      geometries: number;
      textures: number;
    };
    render: {
      calls: number;
      triangles: number;
      points: number;
      lines: number;
    };
  };
}

// WebGL-spesifikke interfaces
interface WebGLDebugInfo {
  UNMASKED_VENDOR_WEBGL: number;
  UNMASKED_RENDERER_WEBGL: number;
}

interface WebGLRenderingContext {
  getExtension(name: string): any;
  getParameter(pname: number): any;
  MAX_TEXTURE_SIZE: number;
}

// Konfigurasjon for TerravisionEngine
export interface TerravisionOptions {
  container: HTMLElement;
  width?: number;
  height?: number;
  antialias?: boolean;
  shadows?: boolean;
  highQuality?: boolean;
  backgroundColor?: number;
  environmentMap?: string;
  cameraPosition?: THREE.Vector3;
  cameraTarget?: THREE.Vector3;
  onReady?: () => void;
  onProgress?: (progress: number) => void;
  onError?: (error: Error) => void;
  debugMode?: boolean;
}

// Terrengkonfigurasjon
export interface TerrainOptions {
  heightMap: string;
  texture: string;
  width: number;
  depth: number;
  height?: number;
  resolution?: number;
  segments?: number;
  levels?: number;
  wireframe?: boolean;
}

// Bygningskonfigurasjon
export interface BuildingOptions {
  model?: string;
  position: THREE.Vector3;
  rotation?: THREE.Euler;
  scale?: THREE.Vector3;
  color?: number;
  opacity?: number;
  castShadow?: boolean;
  receiveShadow?: boolean;
  userData?: any;
}

export interface PerformanceStats {
  fps: number;
  drawCalls: number;
  triangles: number;
  textures: number;
  geometries: number;
  memory: number;
  renderTime: number;
}

export interface ModelData {
  model_url: string;
  textures?: string[];
  materials?: any[];
  scale?: number;
  position?: { x: number; y: number; z: number };
  rotation?: { x: number; y: number; z: number };
}

// Hovedmotor
export class TerravisionEngine {
  // THREE.js hovedkomponenter
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: ExtendedWebGLRenderer;
  private controls: OrbitControls;
  
  // Resources og lastere
  private gltfLoader: GLTFLoader;
  private dracoLoader: DRACOLoader;
  private textureLoader: THREE.TextureLoader;
  
  // States og tracking
  private isDisposed: boolean = false;
  private isInitialized: boolean = false;
  private animationFrameId: number | null = null;
  private lastRenderTime: number = 0;
  private frameCount: number = 0;
  private frameStartTime: number = 0;
  private fps: number = 0;
  
  // Modeller og objekter
  private loadedModels: Map<string, THREE.Group> = new Map();
  private terrain: THREE.Mesh | null = null;
  
  // Lys
  private directionalLight: THREE.DirectionalLight | null = null;
  private ambientLight: THREE.AmbientLight | null = null;
  
  // Cache for ressursoptimalisering
  private geometryCache: Map<string, THREE.BufferGeometry> = new Map();
  private textureCache: Map<string, THREE.Texture> = new Map();
  private materialCache: Map<string, THREE.Material> = new Map();
  
  // Callbacks
  private loadingCallback: ((progress: number) => void) | null = null;
  private errorCallback: ((error: Error) => void) | null = null;
  
  /**
   * Konstruktør - setter opp og initialiserer motorkomponenter
   * 
   * @param options Konfigurasjon for motoroppsettet
   */
  constructor(options: TerravisionOptions) {
    // Valider inndata
    if (!options.container) {
      throw new Error('TerravisionEngine: Container element er påkrevd');
    }
    
    // Sett hovedreferanser
    this.container = options.container;
    this.loadingCallback = options.onProgress || null;
    this.errorCallback = options.onError || null;
    
    // Initialiser Three.js-komponenter
    this.initializeScene(options);
    this.initializeCamera();
    this.initializeRenderer(options);
    this.initializeControls();
    this.initializeLights(options);
    this.initializeLoaders();
    
    // Sett opp event listeners
    this.setupEventListeners();
    
    // Start render loop
    this.isInitialized = true;
    this.animate();
    
    console.log('TerravisionEngine: Initialisert');
  }
  
  /**
   * Initialiser 3D-scenen med bakgrunn og grunnleggende oppsett
   */
  private initializeScene(options: TerravisionOptions): void {
    this.scene = new THREE.Scene();
    
    // Sett bakgrunnsfarge
    const backgroundColor = options.backgroundColor !== undefined ? options.backgroundColor : 0xf0f0f0;
    this.scene.background = new THREE.Color(backgroundColor);
    
    // Legg til grunnleggende grid for orientering
    const gridHelper = new THREE.GridHelper(50, 50, 0x888888, 0xcccccc);
    this.scene.add(gridHelper);
  }
  
  /**
   * Initialiser kameraet med riktig aspektforhold og plassering
   */
  private initializeCamera(): void {
    const { width, height } = this.getContainerDimensions();
    const aspectRatio = width / height;
    
    this.camera = new THREE.PerspectiveCamera(
      60, // FOV
      aspectRatio,
      0.1, // near clipping plane
      2000 // far clipping plane
    );
    
    // Plasser kameraet i en god startposisjon
    this.camera.position.set(10, 10, 10);
    this.camera.lookAt(0, 0, 0);
  }
  
  /**
   * Initialiser WebGL-renderer med konfigurasjon for beste ytelse
   */
  private initializeRenderer(options: TerravisionOptions): void {
    // Renderer-konfigurasjon
    const rendererConfig: THREE.WebGLRendererParameters = {
      antialias: options.antialias !== undefined ? options.antialias : true,
      alpha: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: options.preserveDrawingBuffer || false
    };
    
    // Opprett renderer
    this.renderer = new THREE.WebGLRenderer(rendererConfig) as ExtendedWebGLRenderer;
    
    // Konfigurer rendereren for prosjektet
    const { width, height } = this.getContainerDimensions();
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    
    // Konfigurer skygger hvis det er aktivert
    if (options.shadows !== false) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = options.highQuality ? 
        THREE.PCFSoftShadowMap : THREE.BasicShadowMap;
    }
    
    // Konfigurer ytterligere egenskaper for høykvalitetsrendering
    if (options.highQuality) {
      this.renderer.physicallyCorrectLights = true;
      this.renderer.outputEncoding = THREE.sRGBEncoding;
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.0;
    }
    
    // Legg til canvas til container-elementet
    this.container.appendChild(this.renderer.domElement);
  }
  
  /**
   * Initialiser kamerakontroller for navigasjon
   */
  private initializeControls(): void {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    
    // Konfigurer kontrollene for bedre brukeropplevelse
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 500;
    this.controls.maxPolarAngle = Math.PI * 0.85; // Forhindre kamera under bakken
    
    // Utsett første oppdatering
    this.controls.update();
  }
  
  /**
   * Initialiser lyskildene i scenen
   */
  private initializeLights(options: TerravisionOptions): void {
    // Ambient light (basis belysning)
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(this.ambientLight);
    
    // Directional light (hovedlyskilde)
    this.directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.directionalLight.position.set(50, 50, 20);
    this.directionalLight.castShadow = options.shadows !== false;
    
    // Konfigurer skygger
    if (options.shadows !== false) {
      const shadowSize = options.highQuality ? 4096 : 2048;
      
      this.directionalLight.shadow.mapSize.width = shadowSize;
      this.directionalLight.shadow.mapSize.height = shadowSize;
      this.directionalLight.shadow.camera.left = -50;
      this.directionalLight.shadow.camera.right = 50;
      this.directionalLight.shadow.camera.top = 50;
      this.directionalLight.shadow.camera.bottom = -50;
      this.directionalLight.shadow.camera.near = 0.5;
      this.directionalLight.shadow.camera.far = 500;
      this.directionalLight.shadow.bias = -0.0005;
      this.directionalLight.shadow.normalBias = 0.02;
      
      // Forbedrer skyggevariansen
      this.directionalLight.shadow.radius = 2;
      
      // Myk transformasjon for solbevegelse
      this.directionalLight.userData.originalPosition = this.directionalLight.position.clone();
      
      // Legg til en hjelpelys for å forbedre skyggedetaljer
      const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
      fillLight.position.set(-30, 40, -20);
      fillLight.castShadow = false;
      this.scene.add(fillLight);
    }
    
    // Omgivelseslys (hemisphere)
    const hemisphereLight = new THREE.HemisphereLight(0x88bbff, 0x886644, 0.5);
    this.scene.add(hemisphereLight);
    
    this.scene.add(this.directionalLight);
    
    // Legg til bakgrunnsmiljø hvis spesifisert
    if (options.environmentMap) {
      this.loadEnvironmentMap(options.environmentMap);
    }
  }
  
  /**
   * Laster omgivelseskart for realistisk belysning
   */
  private loadEnvironmentMap(path: string): void {
    try {
      const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
      pmremGenerator.compileEquirectangularShader();
      
      // Load environment texture
      new THREE.TextureLoader().load(
        path,
        (texture) => {
          const envMap = pmremGenerator.fromEquirectangular(texture).texture;
          this.scene.environment = envMap;
          
          // Også bruk som bakgrunn hvis høy kvalitet
          if (this.scene.background instanceof THREE.Color) {
            // Behold bakgrunnsfarge, men med miljøkart synlig i refleksjoner
          } else {
            this.scene.background = envMap;
          }
          
          texture.dispose();
          pmremGenerator.dispose();
        },
        undefined,
        (error) => {
          console.warn('Kunne ikke laste miljøkart:', error);
        }
      );
    } catch (error) {
      console.warn('Miljøkart ikke støttet av nettleseren:', error);
    }
  }
  
  /**
   * Oppdaterer solposisjonen basert på tidspunkt på dagen
   */
  public updateSunPosition(hour: number = 12): void {
    if (!this.directionalLight || !this.directionalLight.userData.originalPosition) return;
    
    // Normaliser timer til 0-24
    hour = hour % 24;
    
    // Beregn solposisjon basert på tid på dagen
    const angleRad = (hour - 12) * (Math.PI / 12);
    const distance = 50;
    const height = Math.cos(angleRad) * distance;
    const xPos = Math.sin(angleRad) * distance;
    
    // Animer solposisjonen for smooth overgang
    const targetPosition = new THREE.Vector3(xPos, height, 20);
    
    // Enkel animasjon
    const animateSunPosition = () => {
      if (!this.directionalLight) return;
      
      this.directionalLight.position.lerp(targetPosition, 0.05);
      
      // Fortsett animasjonen hvis vi ikke er nær nok målposisjonen
      if (this.directionalLight.position.distanceTo(targetPosition) > 0.1) {
        requestAnimationFrame(animateSunPosition);
      }
    };
    
    animateSunPosition();
  }
  
  /**
   * Optimalisert modellasting med caching og progressiv detalj
   */
  private loadModelOptimized(url: string, onProgress?: (progress: number) => void): Promise<THREE.Group> {
    // Sjekk om modellen allerede er lastet
    const cacheKey = `model_${url}`;
    if (this.loadedModels.has(cacheKey)) {
      return Promise.resolve(this.loadedModels.get(cacheKey)!.clone());
    }
    
    return new Promise((resolve, reject) => {
      // Konfigurer DRACO-loader med riktige innstillinger
      this.dracoLoader.setDecoderPath('https://www.gstatic.com/draco/v1/decoders/');
      this.dracoLoader.setDecoderConfig({
        type: 'js', // Bruk JavaScript-decoder istedenfor WASM for bedre kompatibilitet
      });
      
      // Last modellen
      this.gltfLoader.load(
        url,
        (gltf) => {
          const model = gltf.scene;
          
          // Optimalisere modellen
          model.traverse((node) => {
            if (node instanceof THREE.Mesh) {
              // Aktiver skygger
              node.castShadow = true;
              node.receiveShadow = true;
              
              // Optimaliser geometri
              if (node.geometry) {
                // Beregn normaler hvis de mangler
                if (!node.geometry.attributes.normal) {
                  node.geometry.computeVertexNormals();
                }
                
                // Komprimer buffergeometrier for bedre ytelse
                const geometry = node.geometry as THREE.BufferGeometry;
                
                // Optimaliser UV koordinater
                if (geometry.attributes.uv && geometry.attributes.uv.count > 1000) {
                  geometry.setAttribute(
                    'uv',
                    new THREE.BufferAttribute(
                      geometry.attributes.uv.array,
                      2
                    )
                  );
                }
                
                // Legg til i cache for gjenbruk
                this.geometryCache.set(node.uuid, geometry);
              }
              
              // Optimaliser materialer
              if (node.material) {
                // Konverter til standard-materiale hvis det er basis-materiale
                if (node.material instanceof THREE.MeshBasicMaterial && !node.material.map) {
                  const stdMaterial = new THREE.MeshStandardMaterial({
                    color: node.material.color,
                    roughness: 0.7,
                    metalness: 0.1
                  });
                  node.material = stdMaterial;
                }
                
                // Legg til i cache for gjenbruk
                this.materialCache.set(node.uuid, node.material);
              }
            }
          });
          
          // Legg til i cache
          this.loadedModels.set(cacheKey, model.clone());
          
          resolve(model);
        },
        (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const percent = (progressEvent.loaded / progressEvent.total) * 100;
            onProgress(percent);
          }
        },
        reject
      );
    });
  }
  
  /**
   * Lastemetode for 3D-modeller
   * 
   * @param modelData Data for modellen som skal lastes
   * @param id Valgfri ID for senere referanse
   * @returns Promise som løses med den lastede modellen
   */
  public async loadModel(modelData: ModelData, id?: string): Promise<THREE.Group> {
    if (this.isDisposed) {
      throw new Error('TerravisionEngine er allerede avviklet');
    }
    
    return new Promise((resolve, reject) => {
      // Validering av URL
      if (!modelData.model_url) {
        const error = new Error('Mangler model_url i modelData');
        if (this.errorCallback) this.errorCallback(error);
        reject(error);
        return;
      }
      
      // Last modellen
      this.loadModelOptimized(modelData.model_url, this.loadingCallback).then((model) => {
        try {
          // Skaler og plasser modellen
          if (modelData.scale) {
            model.scale.set(modelData.scale.x, modelData.scale.y, modelData.scale.z);
          }
          
          if (modelData.position) {
            model.position.set(
              modelData.position.x || 0,
              modelData.position.y || 0,
              modelData.position.z || 0
            );
          }
          
          if (modelData.rotation) {
            model.rotation.set(
              modelData.rotation.x || 0,
              modelData.rotation.y || 0,
              modelData.rotation.z || 0
            );
          }
          
          // Legg til i scenen
          this.scene.add(model);
          
          // Lagre referanse hvis ID er gitt
          if (id) {
            this.loadedModels.set(id, model);
          }
          
          console.log(`TerravisionEngine: Modell lastet - ${modelData.model_url}`);
          resolve(model);
        } catch (error: any) {
          console.error('TerravisionEngine: Feil ved prosessering av modell', error);
          if (this.errorCallback) this.errorCallback(error);
          reject(error);
        }
      }).catch((error) => {
        console.error('TerravisionEngine: Feil ved lasting av modell', error);
        if (this.errorCallback) this.errorCallback(error);
        reject(error);
      });
    });
  }
  
  /**
   * Genererer terreng basert på høydedata
   * 
   * @param heightData Høydedata som Float32Array eller tall-array
   * @param options Terrengkonfigurasjon
   * @returns Det genererte terreng-meshet
   */
  public generateTerrain(
    heightData: Float32Array | number[], 
    options: TerrainOptions = {}
  ): THREE.Mesh {
    // Standardverdier hvis ikke angitt
    const size = options.size || 100;
    const segments = options.segments || 128;
    const heightScale = options.heightScale || 10;
    
    // Opprett geometri
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
    
    // Sett høydeverdier
    const positionAttr = geometry.attributes.position;
    for (let i = 0; i < positionAttr.count; i++) {
      // Z-komponenten er høyden (Y i world space etter rotasjon)
      if (i < heightData.length) {
        const height = heightData[i] * heightScale;
        positionAttr.setZ(i, height);
      }
    }
    
    // Oppdater og beregn normaler for riktig belysning
    geometry.computeVertexNormals();
    
    // Opprett material
    let material: THREE.Material;
    if (options.material) {
      material = options.material;
    } else {
      // Standard materiale
      material = new THREE.MeshStandardMaterial({
        color: 0x408040,
        metalness: 0.1,
        roughness: 0.9,
        flatShading: false
      });
    }
    
    // Opprett og plasser terreng-mesh
    const terrain = new THREE.Mesh(geometry, material);
    terrain.rotation.x = -Math.PI / 2; // Roter for å vise "opp"
    terrain.receiveShadow = true;
    
    // Legg til i scenen og lagre referanse
    this.scene.add(terrain);
    this.terrain = terrain;
    
    return terrain;
  }
  
  /**
   * Genererer en enkel bygning
   * 
   * @param width Bredde på bygningen
   * @param depth Dybde på bygningen
   * @param height Høyde på bygningen
   * @param options Bygningsalternativer
   * @returns Den genererte bygningen som Group
   */
  public generateBuilding(
    width: number = 10,
    depth: number = 10,
    height: number = 6,
    options: BuildingOptions = {}
  ): THREE.Group {
    const building = new THREE.Group();
    
    // Innstillinger
    const detailLevel = options.detailLevel || 'medium';
    const showInterior = options.showInterior || false;
    const castShadows = options.castShadows !== false;
    const receiveShadows = options.receiveShadows !== false;
    
    // Beregn detaljenivå basert på innstillinger
    let segmentsW, segmentsD;
    switch (detailLevel) {
      case 'low':
        segmentsW = segmentsD = 1;
        break;
      case 'medium':
        segmentsW = Math.max(1, Math.floor(width / 2));
        segmentsD = Math.max(1, Math.floor(depth / 2));
        break;
      case 'high':
        segmentsW = Math.max(2, Math.floor(width));
        segmentsD = Math.max(2, Math.floor(depth));
        break;
    }
    
    // Hovedkroppsgeometri
    const bodyGeom = new THREE.BoxGeometry(width, height, depth, segmentsW, 2, segmentsD);
    
    // Materiale
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: 0xeeeeee,
      metalness: 0.2,
      roughness: 0.7,
      wireframe: options.wireframe || false
    });
    
    // Opprett kropp-mesh
    const body = new THREE.Mesh(bodyGeom, bodyMaterial);
    body.position.y = height / 2;
    body.castShadow = castShadows;
    body.receiveShadow = receiveShadows;
    
    building.add(body);
    
    // Legg til tak (enkel utgave)
    const roofHeight = height * 0.3;
    const roofGeom = new THREE.ConeGeometry(
      Math.sqrt(width * width + depth * depth) / 2, 
      roofHeight, 
      4
    );
    
    const roofMaterial = new THREE.MeshStandardMaterial({
      color: 0xC92C2C,
      metalness: 0.1,
      roughness: 0.8,
      wireframe: options.wireframe || false
    });
    
    const roof = new THREE.Mesh(roofGeom, roofMaterial);
    roof.position.y = height + roofHeight / 2;
    roof.rotation.y = Math.PI / 4; // 45 grader rotasjon
    roof.castShadow = castShadows;
    roof.receiveShadow = receiveShadows;
    
    building.add(roof);
    
    // Legg til bygningen i scenen
    this.scene.add(building);
    
    return building;
  }
  
  /**
   * Tar et øyeblikksbilde av gjeldende visning 
   * 
   * @param width Bredde på bildet (standard: gjeldende canvas-bredde)
   * @param height Høyde på bildet (standard: gjeldende canvas-høyde)
   * @returns Data-URL for bildet
   */
  public takeScreenshot(width?: number, height?: number): string {
    // Hvis preserveDrawingBuffer ikke er aktivert, må vi gjengi en ekstra gang
    if (!this.renderer.preserveDrawingBuffer) {
      console.warn('TerravisionEngine: For optimale skjermbilder, initialiseres med preserveDrawingBuffer: true');
      this.renderer.render(this.scene, this.camera);
    }
    
    // Hent dimensjonene
    const { width: canvasWidth, height: canvasHeight } = this.getContainerDimensions();
    const targetWidth = width || canvasWidth;
    const targetHeight = height || canvasHeight;
    
    // Hvis dimensjonene er forskjellige fra canvas, må vi tilpasse
    if (targetWidth !== canvasWidth || targetHeight !== canvasHeight) {
      // Lagre gjeldende innstillinger
      const oldAspect = this.camera.aspect;
      const oldSize = { width: canvasWidth, height: canvasHeight };
      
      // Tilpass kamera og renderer
      this.camera.aspect = targetWidth / targetHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(targetWidth, targetHeight);
      
      // Gjengi scenen
      this.renderer.render(this.scene, this.camera);
      
      // Hent bildet
      const dataURL = this.renderer.domElement.toDataURL('image/png');
      
      // Gjenopprett originale innstillinger
      this.camera.aspect = oldAspect;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(oldSize.width, oldSize.height);
      this.renderer.render(this.scene, this.camera);
      
      return dataURL;
    } else {
      // Dimensjonene matcher, simpelthen hent bildet
      return this.renderer.domElement.toDataURL('image/png');
    }
  }
  
  /**
   * Endre tidspunkt på dagen som påvirker lyssetting
   * 
   * @param hour Time (0-24)
   */
  public setTimeOfDay(hour: number): void {
    if (!this.directionalLight || !this.ambientLight) return;
    
    // Begrens time til gyldig område
    hour = Math.max(0, Math.min(23.99, hour));
    
    // Beregn solens posisjon basert på tid
    const theta = ((hour - 12) / 12) * Math.PI; // -PI til PI
    const phi = Math.PI * 0.4; // Høyde over horisonten
    
    // Regn ut posisjon i sfæriske koordinater
    const distance = 100;
    const x = distance * Math.cos(theta) * Math.cos(phi);
    const y = distance * Math.sin(phi);
    const z = distance * Math.sin(theta) * Math.cos(phi);
    
    this.directionalLight.position.set(x, y, z);
    
    // Juster styrke basert på tid på dagen
    let intensity = 0;
    if (hour >= 6 && hour <= 18) {
      // Dag
      intensity = 1.0 - Math.abs((hour - 12) / 12) * 0.5;
    } else {
      // Natt
      intensity = 0.1;
    }
    
    this.directionalLight.intensity = intensity;
    
    // Juster ambient lys basert på tid
    if (hour >= 6 && hour <= 18) {
      this.ambientLight.intensity = 0.4;
    } else {
      // Natt - mørkere ambient
      this.ambientLight.intensity = 0.1;
    }
  }
  
  /**
   * Animasjon (render loop)
   */
  private animate = (): void => {
    if (this.isDisposed) return;
    
    this.animationFrameId = requestAnimationFrame(this.animate);
    this.frameStartTime = performance.now();
    
    // Oppdater kontroller (for dempet bevegelse)
    this.controls?.update();
    
    // Utfør rendering
    this.renderer.render(this.scene, this.camera);
    
    // Oppdater FPS-beregninger
    this.frameCount++;
    const now = performance.now();
    
    // Beregn FPS én gang i sekundet
    if (now - this.lastRenderTime >= 1000) {
      this.fps = (this.frameCount * 1000) / (now - this.lastRenderTime);
      this.frameCount = 0;
      this.lastRenderTime = now;
    }
  };
  
  /**
   * Henter dimensjonene til container-elementet
   */
  private getContainerDimensions(): { width: number; height: number } {
    return {
      width: this.container.clientWidth || window.innerWidth,
      height: this.container.clientHeight || window.innerHeight
    };
  }
  
  /**
   * Henter gjeldende ytelsestatistikker
   */
  public getPerformanceStats(): PerformanceStats {
    const renderTime = performance.now() - this.frameStartTime;
    
    const geometries = this.renderer.info.memory.geometries;
    const textures = this.renderer.info.memory.textures;
    const triangles = this.renderer.info.render.triangles;
    const drawCalls = this.renderer.info.render.calls;
    
    // Beregn minnebruk (grovt estimat)
    const vertexMemory = triangles * 3 * 4 * 8; // 3 vertices per triangle, 4 floats per vertex, 8 bytes per float
    const textureMemory = textures * 1024 * 1024 * 4; // Anta gjennomsnittlig 1024x1024 tekstur med 4 kanaler
    const totalMemory = (vertexMemory + textureMemory) / (1024 * 1024); // MB
    
    return {
      fps: this.fps,
      drawCalls,
      triangles,
      textures,
      geometries,
      memory: totalMemory,
      renderTime
    };
  }
  
  /**
   * Rydder opp og frigir ressurser
   */
  public dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    
    // Stopp animasjonsloop
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Fjern event listeners
    window.removeEventListener('resize', this.onWindowResize.bind(this));
    
    // Rydd opp kontroller
    if (this.controls) {
      this.controls.dispose();
    }
    
    // Rydd opp lastere
    this.dracoLoader.dispose();
    
    // Rydd opp scenen
    this.disposeSceneObjects();
    
    // Fjern canvas fra DOM
    if (this.renderer && this.container.contains(this.renderer.domElement)) {
      this.container.removeChild(this.renderer.domElement);
    }
    
    // Frigjør WebGL-kontekst
    this.renderer.dispose();
    
    console.log('TerravisionEngine: Ressurser frigitt');
  }
  
  /**
   * Fjerner og frigir alle objekter i scenen
   */
  private disposeSceneObjects(): void {
    // Rydd opp lastede modeller
    this.loadedModels.forEach((model) => {
      this.disposeObject3D(model);
      this.scene.remove(model);
    });
    this.loadedModels.clear();
    
    // Rydd opp terreng
    if (this.terrain) {
      this.disposeObject3D(this.terrain);
      this.scene.remove(this.terrain);
      this.terrain = null;
    }
    
    // Frigjør geometri-cache
    this.geometryCache.forEach((geometry) => {
      geometry.dispose();
    });
    this.geometryCache.clear();
    
    // Frigjør tekstur-cache
    this.textureCache.forEach((texture) => {
      texture.dispose();
    });
    this.textureCache.clear();
    
    // Frigjør material-cache
    this.materialCache.forEach((material) => {
      material.dispose();
    });
    this.materialCache.clear();
  }
  
  /**
   * Rekursivt rydder opp ressurser assosiert med et Three.js-objekt
   */
  private disposeObject3D(object: THREE.Object3D): void {
    object.traverse((obj) => {
      // Fjern fra scenen
      this.scene.remove(obj);
      
      // Rydd opp Mesh-ressurser
      if (obj instanceof THREE.Mesh) {
        if (obj.geometry) {
          obj.geometry.dispose();
        }
        
        if (obj.material) {
          this.disposeMaterial(obj.material);
        }
      }
    });
  }
  
  /**
   * Frigir ressurser tilknyttet materialer
   */
  private disposeMaterial(material: THREE.Material | THREE.Material[]): void {
    if (Array.isArray(material)) {
      material.forEach(mat => this.disposeMaterial(mat));
      return;
    }
    
    material.dispose();
    
    // Frigjør teksturer
    const stdMat = material as any;
    
    if (stdMat.map) stdMat.map.dispose();
    if (stdMat.normalMap) stdMat.normalMap.dispose();
    if (stdMat.specularMap) stdMat.specularMap.dispose();
    if (stdMat.emissiveMap) stdMat.emissiveMap.dispose();
    if (stdMat.alphaMap) stdMat.alphaMap.dispose();
    if (stdMat.aoMap) stdMat.aoMap.dispose();
    if (stdMat.displacementMap) stdMat.displacementMap.dispose();
    if (stdMat.metalnessMap) stdMat.metalnessMap.dispose();
    if (stdMat.roughnessMap) stdMat.roughnessMap.dispose();
  }
}

/**
 * Hjelper for deteksjon av GPU-kapabiliteter
 */
export function detectGPUCapabilities(): { 
  renderer: string; 
  vendor: string; 
  performance: 'high'|'medium'|'low'|'unknown';
  maxTextureSize: number;
} {
  try {
    // Opprett et midlertidig canvas for WebGL-testing
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || 
              canvas.getContext('webgl') || 
              canvas.getContext('experimental-webgl');
              
    if (!gl) {
      return { 
        renderer: 'unknown', 
        vendor: 'unknown', 
        performance: 'unknown',
        maxTextureSize: 0
      };
    }
    
    // Prøv å få renderer info
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    let vendor = 'unknown';
    let renderer = 'unknown';
    
    if (debugInfo) {
      vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
      renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    }
    
    // Hent maks teksturstørrelse
    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    
    // Estimer ytelsesnivå basert på GPU-info
    let performance: 'high'|'medium'|'low'|'unknown' = 'unknown';
    
    if (renderer.includes('NVIDIA') || renderer.includes('RTX')) {
      performance = 'high';
    } else if (renderer.includes('AMD') || renderer.includes('Radeon')) {
      performance = 'high';
    } else if (renderer.includes('Intel') && !renderer.includes('HD Graphics')) {
      performance = 'medium';
    } else if (renderer.includes('Intel') || renderer.includes('HD Graphics')) {
      performance = 'medium';
    } else if (renderer.includes('Mobile') || renderer.includes('Apple')) {
      performance = 'medium';
    } else if (renderer.includes('Mesa') || renderer.includes('llvmpipe')) {
      performance = 'low';
    }
    
    return { renderer, vendor, performance, maxTextureSize };
  } catch (e) {
    console.error('Feil ved deteksjon av GPU-kapabiliteter:', e);
    return { 
      renderer: 'unknown', 
      vendor: 'unknown', 
      performance: 'unknown',
      maxTextureSize: 0
    };
  }
} 