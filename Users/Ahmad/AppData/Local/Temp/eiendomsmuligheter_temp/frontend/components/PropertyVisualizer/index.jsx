import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';

const VisualizerContainer = styled(motion.div)`
  width: 100%;
  height: 80vh;
  position: relative;
  background: linear-gradient(180deg, #1a1a1a 0%, #2a2a2a 100%);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
`;

const ControlPanel = styled.div`
  position: absolute;
  right: 20px;
  top: 20px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  padding: 20px;
  border-radius: 15px;
  color: white;
  z-index: 100;
`;

const Button = styled.button`
  background: rgba(0, 114, 229, 0.8);
  border: none;
  color: white;
  padding: 10px 20px;
  border-radius: 8px;
  margin: 5px;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    background: rgba(0, 114, 229, 1);
    transform: translateY(-2px);
  }
`;

const InfoOverlay = styled(motion.div)`
  position: absolute;
  left: 20px;
  bottom: 20px;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(10px);
  padding: 20px;
  border-radius: 15px;
  color: white;
  max-width: 300px;
`;

const PropertyVisualizer = ({ propertyData }) => {
  const containerRef = useRef();
  const sceneRef = useRef();
  const rendererRef = useRef();
  const cameraRef = useRef();

  useEffect(() => {
    // Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x1a1a1a);

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    cameraRef.current = camera;
    camera.position.z = 5;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    rendererRef.current = renderer;
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    containerRef.current.appendChild(renderer.domElement);

    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Add property model (example with basic geometry)
    const geometry = new THREE.BoxGeometry(2, 2, 2);
    const material = new THREE.MeshPhongMaterial({
      color: 0x00ff00,
      transparent: true,
      opacity: 0.8,
    });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      renderer.dispose();
      containerRef.current?.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <VisualizerContainer
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
    >
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      
      <ControlPanel>
        <Button onClick={() => console.log('Toggle Floor Plan')}>
          Vis Plantegning
        </Button>
        <Button onClick={() => console.log('Toggle 3D Model')}>
          3D Modell
        </Button>
        <Button onClick={() => console.log('Toggle Analysis')}>
          Vis Analyse
        </Button>
      </ControlPanel>

      <InfoOverlay
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h3>Eiendomsdetaljer</h3>
        <p>BRA: {propertyData?.bra || '150'} m²</p>
        <p>Tomteareal: {propertyData?.landArea || '500'} m²</p>
        <p>Byggeår: {propertyData?.buildYear || '1985'}</p>
      </InfoOverlay>
    </VisualizerContainer>
  );
};

export default PropertyVisualizer;