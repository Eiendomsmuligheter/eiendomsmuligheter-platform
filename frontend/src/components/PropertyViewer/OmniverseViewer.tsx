import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import { Omniverse, OmniversePhysX, OmniverseRTX } from '@nvidia/omniverse-js';

interface OmniverseViewerProps {
  modelData: any;
  onSceneLoaded?: () => void;
}

const OmniverseViewer: React.FC<OmniverseViewerProps> = ({ modelData, onSceneLoaded }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const omniverseRef = useRef<Omniverse | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize Omniverse
    const omniverse = new Omniverse({
      container: containerRef.current,
      apiKey: process.env.REACT_APP_NVIDIA_API_KEY,
      settings: {
        rtx: true,
        physx: true,
        quality: 'high'
      }
    });

    // Add PhysX for physics simulation
    const physics = new OmniversePhysX();
    omniverse.addExtension(physics);

    // Add RTX for real-time raytracing
    const rtx = new OmniverseRTX();
    omniverse.addExtension(rtx);

    // Load the 3D model
    omniverse.loadScene(modelData).then(() => {
      if (onSceneLoaded) {
        onSceneLoaded();
      }
    });

    omniverseRef.current = omniverse;

    return () => {
      if (omniverseRef.current) {
        omniverseRef.current.dispose();
      }
    };
  }, [modelData, onSceneLoaded]);

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        height: '600px',
        backgroundColor: '#000',
        borderRadius: 1,
        overflow: 'hidden'
      }}
    />
  );
};

export default OmniverseViewer;