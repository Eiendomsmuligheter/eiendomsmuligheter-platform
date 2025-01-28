<<<<<<< HEAD
import React, { useEffect, useRef } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Paper } from '@material-ui/core';
import * as THREE from 'three';
import { OmniverseConnector } from '../services/omniverseService';

const useStyles = makeStyles((theme) => ({
  viewerContainer: {
    width: '100%',
    height: '600px',
    position: 'relative',
  },
  canvas: {
    width: '100%',
    height: '100%',
  },
}));

interface PropertyViewerProps {
  property: {
    modelData?: any;
    floorPlan?: any;
    dimensions?: any;
  };
}

export const PropertyViewer: React.FC<PropertyViewerProps> = ({ property }) => {
  const classes = useStyles();
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const omniverseRef = useRef<OmniverseConnector | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize Three.js
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    cameraRef.current = camera;
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    rendererRef.current = renderer;
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    containerRef.current.appendChild(renderer.domElement);

    // Initialize Omniverse connector
    omniverseRef.current = new OmniverseConnector();

    // Set up lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    scene.add(directionalLight);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      if (renderer && scene && camera) {
        renderer.render(scene, camera);
      }
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !camera || !renderer) return;

      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current && renderer) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Update 3D model when property data changes
  useEffect(() => {
    if (!property.modelData || !sceneRef.current || !omniverseRef.current) return;

    // Clear existing model
    while (sceneRef.current.children.length > 0) {
      sceneRef.current.remove(sceneRef.current.children[0]);
    }

    // Load new model using Omniverse
    omniverseRef.current.loadModel(property.modelData).then((model) => {
      if (sceneRef.current && model) {
        sceneRef.current.add(model);
      }
    });
  }, [property.modelData]);

  return (
    <Paper className={classes.viewerContainer} ref={containerRef}>
      {/* Three.js canvas will be inserted here */}
=======
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
>>>>>>> 05b417208bb8af307dcc4b59d05bb20e32529392
    </Paper>
  );
};

export default PropertyViewer;