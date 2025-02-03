import React, { useEffect, useRef } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { OmniverseSDK } from '@nvidia/omniverse-sdk';

interface OmniverseViewerProps {
  modelData: {
    model_url: string;
    textures: string[];
    materials: any[];
  };
}

export const OmniverseViewer: React.FC<OmniverseViewerProps> = ({ modelData }) => {
  const viewerRef = useRef<HTMLDivElement>(null);
  const omniverseRef = useRef<any>(null);

  useEffect(() => {
    if (!viewerRef.current || !modelData) return;

    const initializeOmniverse = async () => {
      try {
        // Initialiser NVIDIA Omniverse SDK
        omniverseRef.current = await OmniverseSDK.initialize({
          container: viewerRef.current,
          credentials: {
            appId: process.env.NEXT_PUBLIC_OMNIVERSE_APP_ID,
            appSecret: process.env.NEXT_PUBLIC_OMNIVERSE_APP_SECRET
          }
        });

        // Last 3D-modellen
        await omniverseRef.current.loadModel({
          url: modelData.model_url,
          textures: modelData.textures,
          materials: modelData.materials
        });

        // Sett opp kameraposisjon og belysning
        await omniverseRef.current.setupScene({
          camera: {
            position: { x: 0, y: 5, z: 10 },
            target: { x: 0, y: 0, z: 0 }
          },
          lighting: {
            ambient: 0.5,
            directional: {
              intensity: 0.8,
              position: { x: 1, y: 1, z: 1 }
            }
          }
        });

        // Aktiver interaktive kontroller
        await omniverseRef.current.enableControls({
          orbit: true,
          pan: true,
          zoom: true
        });

      } catch (error) {
        console.error('Error initializing Omniverse:', error);
      }
    };

    initializeOmniverse();

    return () => {
      // Cleanup når komponenten unmountes
      if (omniverseRef.current) {
        omniverseRef.current.dispose();
      }
    };
  }, [modelData]);

  const handleViewModeChange = async (mode: '3d' | 'floorplan' | 'facade') => {
    if (!omniverseRef.current) return;

    try {
      await omniverseRef.current.setViewMode(mode);
    } catch (error) {
      console.error('Error changing view mode:', error);
    }
  };

  if (!modelData) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '600px', position: 'relative' }}>
      <Box
        ref={viewerRef}
        sx={{
          width: '100%',
          height: '100%',
          bgcolor: '#f5f5f5',
          borderRadius: 1,
          overflow: 'hidden'
        }}
      />

      {/* Kontrollpanel */}
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          bgcolor: 'rgba(255, 255, 255, 0.9)',
          p: 2,
          borderRadius: 1,
          boxShadow: 1
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          Visningsmodus
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <button onClick={() => handleViewModeChange('3d')}>3D</button>
          <button onClick={() => handleViewModeChange('floorplan')}>Plantegning</button>
          <button onClick={() => handleViewModeChange('facade')}>Fasade</button>
        </Box>
      </Box>
    </Box>
  );
};