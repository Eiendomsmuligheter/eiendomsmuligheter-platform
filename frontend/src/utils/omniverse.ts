import { OmniPlatform, OmniViewer } from '@nvidia/omniverse-platform';

interface OmniverseInitOptions {
  container: HTMLElement;
  modelUrl: string;
  initialCamera: {
    position: [number, number, number];
    target: [number, number, number];
  };
}

export const initializeNVIDIAOmniverse = async ({
  container,
  modelUrl,
  initialCamera,
}: OmniverseInitOptions) => {
  // Initialize NVIDIA Omniverse Platform
  const platform = new OmniPlatform({
    applicationId: process.env.NVIDIA_APP_ID,
    apiKey: process.env.NVIDIA_API_KEY,
  });

  await platform.initialize();

  // Create viewer instance
  const viewer = new OmniViewer(container, {
    platform,
    rendererOptions: {
      antialias: true,
      shadows: true,
      physicallyBasedRendering: true,
    },
  });

  // Load the 3D model
  await viewer.loadModel(modelUrl);

  // Set initial camera position
  viewer.setCamera({
    position: initialCamera.position,
    target: initialCamera.target,
  });

  // Add viewer methods
  return {
    dispose: () => {
      viewer.dispose();
      platform.shutdown();
    },
    setLayerVisibility: (layerId: string, visible: boolean) => {
      viewer.setLayerVisibility(layerId, visible);
    },
    resetCamera: (camera: typeof initialCamera) => {
      viewer.setCamera({
        position: camera.position,
        target: camera.target,
      });
    },
    zoomIn: () => {
      viewer.zoom(1.2);
    },
    zoomOut: () => {
      viewer.zoom(0.8);
    },
    setViewMode: (mode: '3d' | 'floorplan' | 'facade') => {
      switch (mode) {
        case '3d':
          viewer.set3DView();
          break;
        case 'floorplan':
          viewer.setTopView();
          break;
        case 'facade':
          viewer.setFrontView();
          break;
      }
    },
    takeScreenshot: async (): Promise<string> => {
      return viewer.takeScreenshot();
    },
    getMeasurements: () => {
      return viewer.getMeasurements();
    },
    enableMeasurementTool: () => {
      viewer.enableMeasurementTool();
    },
    disableMeasurementTool: () => {
      viewer.disableMeasurementTool();
    },
    setQuality: (quality: 'low' | 'medium' | 'high' | 'ultra') => {
      viewer.setQuality(quality);
    },
    setShadows: (enabled: boolean) => {
      viewer.setShadows(enabled);
    },
    setBackgroundColor: (color: string) => {
      viewer.setBackgroundColor(color);
    },
  };
};

export const prepareModelForOmniverse = async (modelData: ArrayBuffer) => {
  // Convert and optimize the model for Omniverse
  const platform = new OmniPlatform({
    applicationId: process.env.NVIDIA_APP_ID,
    apiKey: process.env.NVIDIA_API_KEY,
  });

  await platform.initialize();

  const optimizedModel = await platform.optimizeModel(modelData, {
    format: 'usd',
    quality: 'high',
    compress: true,
    generateLODs: true,
  });

  platform.shutdown();

  return optimizedModel;
};