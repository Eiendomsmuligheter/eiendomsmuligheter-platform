import React, { useEffect, useRef, useState } from 'react';
import { Box, Paper, Typography, CircularProgress, Button } from '@mui/material';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface PropertyViewerProps {
  modelUrl?: string;
  initialData?: any;
  onAnalysisComplete?: (result: any) => void;
}

const PropertyViewer: React.FC<PropertyViewerProps> = ({
  modelUrl,
  initialData,
  onAnalysisComplete
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<any>(initialData);

  // Camera settings
  const cameraSettings = {
    position: [10, 10, 10],
    fov: 75,
    near: 0.1,
    far: 1000
  };

  // Scene setup
  const Scene: React.FC = () => {
    const { scene } = useThree();

    useEffect(() => {
      // Set up scene
      scene.background = new THREE.Color(0xf0f0f0);

      // Add ambient light
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      scene.add(ambientLight);

      // Add directional light
      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(5, 5, 5);
      scene.add(directionalLight);

      return () => {
        scene.remove(ambientLight);
        scene.remove(directionalLight);
      };
    }, [scene]);

    return null;
  };

  // Load 3D model
  useEffect(() => {
    if (modelUrl) {
      loadModel(modelUrl);
    }
  }, [modelUrl]);

  const loadModel = async (url: string) => {
    try {
      setLoading(true);
      // TODO: Implement model loading logic
      setLoading(false);
    } catch (err) {
      setError('Failed to load 3D model');
      setLoading(false);
    }
  };

  // Analysis functions
  const analyzeDevelopmentPotential = async () => {
    try {
      setLoading(true);
      // TODO: Implement development potential analysis
      const result = await fetch('/api/analyze/potential', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ modelUrl })
      });
      
      const data = await result.json();
      setAnalysisData(data);
      onAnalysisComplete?.(data);
      setLoading(false);
    } catch (err) {
      setError('Analysis failed');
      setLoading(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>
        Eiendomsvisualisering
      </Typography>
      
      {loading && (
        <Box display="flex" justifyContent="center" alignItems="center" flex={1}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Box display="flex" justifyContent="center" alignItems="center" flex={1}>
          <Typography color="error">{error}</Typography>
        </Box>
      )}

      {!loading && !error && (
        <>
          <Box ref={containerRef} sx={{ flex: 1, minHeight: 400 }}>
            <Canvas>
              <Scene />
              <PerspectiveCamera makeDefault {...cameraSettings} />
              <OrbitControls enableDamping dampingFactor={0.05} />
            </Canvas>
          </Box>

          <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
            <Button 
              variant="contained" 
              color="primary"
              onClick={analyzeDevelopmentPotential}
            >
              Analyser utviklingspotensial
            </Button>
          </Box>

          {analysisData && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">Analyseresultater:</Typography>
              {/* TODO: Implement analysis results visualization */}
            </Box>
          )}
        </>
      )}
    </Paper>
  );
};

export default PropertyViewer;