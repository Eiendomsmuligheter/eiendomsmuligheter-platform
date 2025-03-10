import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, CircularProgress, Button, ButtonGroup } from '@mui/material';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

interface ThreeViewerProps {
  modelData: {
    model_url: string;
    textures?: string[];
    materials?: any[];
  };
}

type ViewMode = '3d' | 'floorplan' | 'facade';

export const OmniverseViewer: React.FC<ThreeViewerProps> = ({ modelData }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('3d');
  const [error, setError] = useState<string | null>(null);

  // Initialisering av Three.js-scene
  useEffect(() => {
    if (!containerRef.current || !modelData) return;
    
    // Opprydding fra tidligere render
    if (rendererRef.current && containerRef.current.contains(rendererRef.current.domElement)) {
      containerRef.current.removeChild(rendererRef.current.domElement);
    }
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    try {
      // Oppsett av scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf5f5f5);
      sceneRef.current = scene;
      
      // Oppsett av kamera
      const container = containerRef.current;
      const width = container.clientWidth;
      const height = container.clientHeight;
      const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
      camera.position.set(0, 5, 10);
      cameraRef.current = camera;
      
      // Oppsett av renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.outputEncoding = THREE.sRGBEncoding;
      container.appendChild(renderer.domElement);
      rendererRef.current = renderer;
      
      // Legg til lys
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      scene.add(ambientLight);
      
      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(1, 1, 1).normalize();
      scene.add(directionalLight);
      
      // Oppsett av kontroller
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controlsRef.current = controls;
      
      // Last inn modell
      if (modelData.model_url) {
        loadModel(modelData.model_url);
      } else {
        // Fallback: Vis en enkel kube hvis ingen modell er tilgjengelig
        showPlaceholder();
        setLoading(false);
      }
      
      // Resizing
      const handleResize = () => {
        if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
        
        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;
        
        cameraRef.current.aspect = width / height;
        cameraRef.current.updateProjectionMatrix();
        
        rendererRef.current.setSize(width, height);
      };
      
      window.addEventListener('resize', handleResize);
      
      // Animate loop
      const animate = () => {
        animationFrameRef.current = requestAnimationFrame(animate);
        
        if (controlsRef.current) {
          controlsRef.current.update();
        }
        
        if (rendererRef.current && sceneRef.current && cameraRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current);
        }
      };
      
      animate();
      
      // Cleanup
      return () => {
        window.removeEventListener('resize', handleResize);
        
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        
        if (controlsRef.current) {
          controlsRef.current.dispose();
        }
        
        if (rendererRef.current) {
          rendererRef.current.dispose();
          
          if (containerRef.current && containerRef.current.contains(rendererRef.current.domElement)) {
            containerRef.current.removeChild(rendererRef.current.domElement);
          }
        }
      };
    } catch (err) {
      console.error('Error initializing Three.js:', err);
      setError('Kunne ikke laste 3D-visning');
      setLoading(false);
    }
  }, [modelData]);
  
  // Håndtere endring av visningsmodus
  useEffect(() => {
    if (!sceneRef.current || !cameraRef.current) return;
    
    switch (viewMode) {
      case '3d':
        cameraRef.current.position.set(0, 5, 10);
        break;
      case 'floorplan':
        cameraRef.current.position.set(0, 20, 0);
        cameraRef.current.lookAt(0, 0, 0);
        break;
      case 'facade':
        cameraRef.current.position.set(15, 5, 0);
        cameraRef.current.lookAt(0, 5, 0);
        break;
    }
    
    if (controlsRef.current) {
      controlsRef.current.update();
    }
  }, [viewMode]);
  
  // Last inn 3D-modellen
  const loadModel = (url: string) => {
    if (!sceneRef.current) return;
    
    const loader = new GLTFLoader();
    
    loader.load(
      url,
      (gltf) => {
        if (sceneRef.current) {
          // Fjern eventuelle tidligere modeller
          while (sceneRef.current.children.length > 0) {
            const child = sceneRef.current.children[0];
            if (child instanceof THREE.Light) {
              // Behold lysene
              sceneRef.current.children.shift();
              continue;
            }
            sceneRef.current.remove(child);
          }
          
          // Legg til den nye modellen
          sceneRef.current.add(gltf.scene);
          
          // Sentrer modellen
          const box = new THREE.Box3().setFromObject(gltf.scene);
          const center = box.getCenter(new THREE.Vector3());
          gltf.scene.position.x = -center.x;
          gltf.scene.position.y = -center.y;
          gltf.scene.position.z = -center.z;
          
          setLoading(false);
        }
      },
      (progress) => {
        // Håndter lasting
        console.log('Loading progress:', (progress.loaded / progress.total) * 100, '%');
      },
      (error) => {
        console.error('Error loading 3D model:', error);
        setError('Kunne ikke laste 3D-modellen');
        showPlaceholder();
        setLoading(false);
      }
    );
  };
  
  // Vis en placeholder hvis modellen ikke kan lastes
  const showPlaceholder = () => {
    if (!sceneRef.current) return;
    
    // Opprett en enkel kube som placeholder
    const geometry = new THREE.BoxGeometry(5, 5, 5);
    const material = new THREE.MeshStandardMaterial({ color: 0x999999 });
    const cube = new THREE.Mesh(geometry, material);
    sceneRef.current.add(cube);
    
    // Legg til en enkel grid
    const gridHelper = new THREE.GridHelper(20, 20);
    sceneRef.current.add(gridHelper);
  };
  
  if (error) {
    return (
      <Box sx={{ 
        width: '100%', 
        height: '600px', 
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center', 
        bgcolor: '#f0f0f0',
        p: 4
      }}>
        <Typography variant="h6" color="error" gutterBottom>
          {error}
        </Typography>
        <Typography variant="body1">
          Prøv å laste siden på nytt eller kontakt support hvis problemet vedvarer.
        </Typography>
      </Box>
    );
  }
  
  return (
    <Box sx={{ width: '100%', height: '600px', position: 'relative' }}>
      {loading && (
        <Box sx={{ 
          position: 'absolute', 
          top: 0, 
          left: 0, 
          width: '100%', 
          height: '100%',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          bgcolor: 'rgba(255,255,255,0.7)',
          zIndex: 10
        }}>
          <CircularProgress />
        </Box>
      )}
      
      <Box
        ref={containerRef}
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
          boxShadow: 1,
          zIndex: 5
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          Visningsmodus
        </Typography>
        <ButtonGroup size="small">
          <Button 
            variant={viewMode === '3d' ? 'contained' : 'outlined'}
            onClick={() => setViewMode('3d')}
          >
            3D
          </Button>
          <Button 
            variant={viewMode === 'floorplan' ? 'contained' : 'outlined'}
            onClick={() => setViewMode('floorplan')}
          >
            Plantegning
          </Button>
          <Button 
            variant={viewMode === 'facade' ? 'contained' : 'outlined'}
            onClick={() => setViewMode('facade')}
          >
            Fasade
          </Button>
        </ButtonGroup>
      </Box>
    </Box>
  );
};