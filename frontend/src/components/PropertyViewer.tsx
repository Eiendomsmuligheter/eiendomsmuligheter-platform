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
    </Paper>
  );
};

export default PropertyViewer;