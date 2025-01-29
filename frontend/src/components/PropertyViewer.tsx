import React, { useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import * as THREE from 'three';
import { OmniverseSDK } from '@nvidia/omniverse';

interface PropertyViewerProps {
  property: {
    modelData?: any;
    address: string;
    images?: string[];
  };
}

export const PropertyViewer: React.FC<PropertyViewerProps> = ({ property }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const omniverseRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize Three.js
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ antialias: true });

    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    containerRef.current.appendChild(renderer.domElement);

    // Initialize NVIDIA Omniverse
    const initializeOmniverse = async () => {
      try {
        omniverseRef.current = await OmniverseSDK.initialize({
          appId: 'eiendomsmuligheter-platform',
          version: '1.0.0'
        });

        // Load property model if available
        if (property.modelData) {
          await loadPropertyModel(property.modelData);
        }
      } catch (error) {
        console.error('Failed to initialize Omniverse:', error);
      }
    };

    initializeOmniverse();

    // Set up scene
    camera.position.z = 5;
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    scene.add(directionalLight);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };

    animate();

    // Store refs
    rendererRef.current = renderer;
    sceneRef.current = scene;
    cameraRef.current = camera;

    // Cleanup
    return () => {
      if (containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;

      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;

      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const loadPropertyModel = async (modelData: any) => {
    if (!omniverseRef.current || !sceneRef.current) return;

    try {
      // Load model using Omniverse SDK
      const model = await omniverseRef.current.loadModel(modelData);
      sceneRef.current.add(model);
    } catch (error) {
      console.error('Failed to load property model:', error);
    }
  };

  return (
    <Box sx={{ width: '100%', height: '600px', position: 'relative' }}>
      <Box ref={containerRef} sx={{ width: '100%', height: '100%' }} />
      <Typography
        variant="subtitle1"
        sx={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          padding: 1,
          borderRadius: 1
        }}
      >
        {property.address}
      </Typography>
    </Box>
  );
};