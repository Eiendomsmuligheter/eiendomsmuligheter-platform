import React, { useEffect, useRef, useState } from 'react';
import { Box, Grid, Paper } from '@mui/material';
import { PropertyViewerProvider } from '../../contexts/PropertyViewerContext';
import { LayerManager } from './LayerManager';
import { initializeNVIDIAOmniverse } from '../../utils/omniverse';
import { LayerVisibility } from '../../types/building';
import ModelControls from './ModelControls';

interface PropertyViewerProps {
  modelUrl: string;
  initialCamera?: {
    position: [number, number, number];
    target: [number, number, number];
  };
}

export const PropertyViewer: React.FC<PropertyViewerProps> = ({
  modelUrl,
  initialCamera = {
    position: [10, 10, 10],
    target: [0, 0, 0],
  },
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [viewerInstance, setViewerInstance] = useState<any>(null);

  useEffect(() => {
    if (containerRef.current) {
      const initViewer = async () => {
        try {
          const instance = await initializeNVIDIAOmniverse({
            container: containerRef.current!,
            modelUrl,
            initialCamera,
          });
          setViewerInstance(instance);
          setIsLoading(false);
        } catch (error) {
          console.error('Failed to initialize NVIDIA Omniverse:', error);
          setIsLoading(false);
        }
      };

      initViewer();
    }

    return () => {
      if (viewerInstance) {
        viewerInstance.dispose();
      }
    };
  }, [modelUrl]);

  const handleLayerChange = (layers: LayerVisibility) => {
    if (viewerInstance) {
      Object.entries(layers).forEach(([layerId, visible]) => {
        viewerInstance.setLayerVisibility(layerId, visible);
      });
    }
  };

  return (
    <PropertyViewerProvider>
      <Grid container spacing={2}>
        <Grid item xs={9}>
          <Paper 
            elevation={3}
            sx={{
              height: '80vh',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <Box
              ref={containerRef}
              sx={{
                width: '100%',
                height: '100%',
                position: 'relative',
              }}
            >
              {isLoading && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                  }}
                >
                  Laster inn 3D-modell...
                </Box>
              )}
            </Box>
            <Box
              sx={{
                position: 'absolute',
                bottom: 16,
                left: '50%',
                transform: 'translateX(-50%)',
              }}
            >
              <ModelControls
                onReset={() => {
                  if (viewerInstance) {
                    viewerInstance.resetCamera(initialCamera);
                  }
                }}
                onZoomIn={() => {
                  if (viewerInstance) {
                    viewerInstance.zoomIn();
                  }
                }}
                onZoomOut={() => {
                  if (viewerInstance) {
                    viewerInstance.zoomOut();
                  }
                }}
              />
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={3}>
          <Paper
            elevation={3}
            sx={{
              height: '80vh',
              overflow: 'auto',
              p: 2,
            }}
          >
            <LayerManager onLayerChange={handleLayerChange} />
          </Paper>
        </Grid>
      </Grid>
    </PropertyViewerProvider>
  );
};

export default PropertyViewer;