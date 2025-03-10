import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { TerravisionEngine, TerravisionOptions, BuildingOptions, TerrainOptions } from '../../frontend/components/TerravisionEngine';

// Lager en mock for three.js siden vi ikke vil laste det faktiske biblioteket i testene
vi.mock('three', () => {
  return {
    Scene: vi.fn().mockImplementation(() => ({
      add: vi.fn(),
      children: [],
      remove: vi.fn()
    })),
    WebGLRenderer: vi.fn().mockImplementation(() => ({
      setSize: vi.fn(),
      setClearColor: vi.fn(),
      render: vi.fn(),
      domElement: document.createElement('canvas')
    })),
    PerspectiveCamera: vi.fn().mockImplementation(() => ({
      position: { set: vi.fn() },
      lookAt: vi.fn()
    })),
    Vector3: vi.fn().mockImplementation(() => ({
      set: vi.fn()
    })),
    BoxGeometry: vi.fn(),
    MeshBasicMaterial: vi.fn(),
    Mesh: vi.fn().mockImplementation(() => ({
      position: { set: vi.fn() },
      scale: { set: vi.fn() },
      rotation: { set: vi.fn() }
    })),
    DirectionalLight: vi.fn().mockImplementation(() => ({
      position: { set: vi.fn() }
    })),
    AmbientLight: vi.fn(),
    Object3D: vi.fn().mockImplementation(() => ({
      position: { set: vi.fn() },
      children: [],
      add: vi.fn()
    })),
    Group: vi.fn().mockImplementation(() => ({
      add: vi.fn(),
      children: []
    })),
    Color: vi.fn()
  };
});

describe('TerravisionEngine', () => {
  let container: HTMLDivElement;
  let terravision: TerravisionEngine;
  let defaultOptions: TerravisionOptions;

  beforeEach(() => {
    // Oppretter en container for enginen
    container = document.createElement('div');
    container.id = 'terravision-container';
    document.body.appendChild(container);

    // Standardopsjoner for tester
    defaultOptions = {
      container: '#terravision-container',
      width: 800,
      height: 600,
      backgroundColor: '#f0f0f0',
      antialias: true,
      renderQuality: 'high',
      enableShadows: false
    };

    // Oppretter instansen av TerravisionEngine
    terravision = new TerravisionEngine(defaultOptions);
  });

  afterEach(() => {
    // Rydder opp i DOM
    if (container && container.parentNode) {
      container.parentNode.removeChild(container);
    }
    
    // Resetter alle mocks
    vi.clearAllMocks();
  });

  it('skal opprette en instans av TerravisionEngine', () => {
    expect(terravision).toBeInstanceOf(TerravisionEngine);
  });

  it('skal initialisere motoren med riktige opsjoner', () => {
    expect(terravision.getOptions()).toEqual(defaultOptions);
  });

  it('skal kunne endre renderkvalitet', () => {
    terravision.setRenderQuality('medium');
    expect(terravision.getOptions().renderQuality).toBe('medium');
  });

  it('skal kunne legge til bygninger', () => {
    const buildingOptions: BuildingOptions = {
      position: { x: 10, y: 0, z: 10 },
      dimensions: { width: 20, height: 15, depth: 20 },
      color: '#ff0000',
      name: 'Testbygg'
    };
    
    const buildingId = terravision.addBuilding(buildingOptions);
    const buildings = terravision.getAllBuildings();
    
    expect(buildingId).toBeDefined();
    expect(buildings).toHaveLength(1);
    expect(buildings[0].name).toBe('Testbygg');
  });

  it('skal kunne fjerne bygninger', () => {
    const buildingOptions: BuildingOptions = {
      position: { x: 10, y: 0, z: 10 },
      dimensions: { width: 20, height: 15, depth: 20 },
      color: '#ff0000',
      name: 'Testbygg'
    };
    
    const buildingId = terravision.addBuilding(buildingOptions);
    expect(terravision.getAllBuildings()).toHaveLength(1);
    
    terravision.removeBuilding(buildingId);
    expect(terravision.getAllBuildings()).toHaveLength(0);
  });

  it('skal kunne legge til terreng', () => {
    const terrainOptions: TerrainOptions = {
      width: 1000,
      depth: 1000,
      resolution: 100,
      heightMap: 'dummy-heightmap.png',
      texture: 'dummy-texture.png'
    };
    
    const terrainId = terravision.addTerrain(terrainOptions);
    expect(terrainId).toBeDefined();
    expect(terravision.hasActiveTerrain()).toBe(true);
  });

  it('skal kunne oppdatere kamera', () => {
    const cameraPosition = { x: 100, y: 50, z: 100 };
    const lookAt = { x: 0, y: 0, z: 0 };
    
    terravision.updateCamera(cameraPosition, lookAt);
    // Vi kan ikke direkte sjekke internals, men vi kan sjekke at funksjonen ikke gir feil
    expect(() => terravision.updateCamera(cameraPosition, lookAt)).not.toThrow();
  });

  it('skal kunne rendere scenen', () => {
    const renderSpy = vi.spyOn(terravision, 'render');
    
    terravision.render();
    
    expect(renderSpy).toHaveBeenCalledTimes(1);
  });

  it('skal håndtere vindustørrelsesendringer', () => {
    const newWidth = 1024;
    const newHeight = 768;
    
    terravision.resize(newWidth, newHeight);
    
    expect(terravision.getOptions().width).toBe(newWidth);
    expect(terravision.getOptions().height).toBe(newHeight);
  });

  it('skal eksportere scenen til JSON', () => {
    // Legg til noen objekter først
    terravision.addBuilding({
      position: { x: 10, y: 0, z: 10 },
      dimensions: { width: 20, height: 15, depth: 20 },
      color: '#ff0000',
      name: 'Testbygg 1'
    });
    
    terravision.addBuilding({
      position: { x: -20, y: 0, z: 30 },
      dimensions: { width: 15, height: 10, depth: 15 },
      color: '#00ff00',
      name: 'Testbygg 2'
    });
    
    const sceneData = terravision.exportScene();
    
    expect(sceneData).toBeDefined();
    expect(sceneData.buildings).toHaveLength(2);
    expect(sceneData.version).toBeDefined();
  });

  it('skal importere scenen fra JSON', () => {
    const sceneData = {
      version: '1.0',
      buildings: [
        {
          id: '1',
          name: 'Importert bygg',
          position: { x: 10, y: 0, z: 10 },
          dimensions: { width: 20, height: 15, depth: 20 },
          color: '#ff0000'
        }
      ],
      terrain: null,
      cameraPosition: { x: 100, y: 50, z: 100 }
    };
    
    terravision.importScene(sceneData);
    
    const buildings = terravision.getAllBuildings();
    expect(buildings).toHaveLength(1);
    expect(buildings[0].name).toBe('Importert bygg');
  });
}); 