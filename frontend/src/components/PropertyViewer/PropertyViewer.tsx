import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei';
import { Box, Typography, Paper, Tabs, Tab, IconButton } from '@mui/material';
import { styled } from '@mui/material/styles';
import { FaRuler, FaHome, FaTree, FaSun } from 'react-icons/fa';
import { PropertyData, Visualization } from '../../types';

// Stiliserte komponenter
const ViewerContainer = styled(Paper)(({ theme }) => ({
  height: 'calc(100vh - 100px)',
  position: 'relative',
  overflow: 'hidden',
  borderRadius: theme.shape.borderRadius,
}));

const ControlsContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  right: theme.spacing(2),
  zIndex: 1000,
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
}));

interface PropertyViewerProps {
  propertyData: PropertyData;
  visualization: Visualization;
}

// Scene komponenter
const Scene: React.FC<{ visualization: Visualization }> = ({ visualization }) => {
  const { camera, scene } = useThree();
  
  useEffect(() => {
    // Initialiser scenen med visualiseringsdata
    if (visualization.model) {
      // TODO: Last inn 3D-modell
    }
  }, [visualization]);

  useFrame(() => {
    // Oppdater scenen per frame
  });

  return null;
};

const PropertyViewer: React.FC<PropertyViewerProps> = ({ propertyData, visualization }) => {
  const [viewMode, setViewMode] = useState<'3d' | 'plan' | 'facade'>('3d');
  const [showMeasurements, setShowMeasurements] = useState(false);
  const [timeOfDay, setTimeOfDay] = useState<'day' | 'night'>('day');
  const [showLandscape, setShowLandscape] = useState(true);

  const handleViewModeChange = (event: React.SyntheticEvent, newValue: '3d' | 'plan' | 'facade') => {
    setViewMode(newValue);
  };

  return (
    <ViewerContainer>
      <Tabs
        value={viewMode}
        onChange={handleViewModeChange}
        centered
        sx={{ borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab label="3D Visning" value="3d" />
        <Tab label="Plantegning" value="plan" />
        <Tab label="Fasade" value="facade" />
      </Tabs>

      <ControlsContainer>
        <IconButton
          onClick={() => setShowMeasurements(!showMeasurements)}
          color={showMeasurements ? 'primary' : 'default'}
        >
          <FaRuler />
        </IconButton>
        <IconButton
          onClick={() => setShowLandscape(!showLandscape)}
          color={showLandscape ? 'primary' : 'default'}
        >
          <FaTree />
        </IconButton>
        <IconButton
          onClick={() => setTimeOfDay(timeOfDay === 'day' ? 'night' : 'day')}
        >
          <FaSun />
        </IconButton>
      </ControlsContainer>

      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[10, 10, 10]} />
        <OrbitControls enableDamping dampingFactor={0.05} />
        
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        
        <Scene visualization={visualization} />
        
        <Environment preset={timeOfDay === 'day' ? 'sunset' : 'night'} />
      </Canvas>

      {/* Informasjonspanel */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          bgcolor: 'rgba(255,255,255,0.9)',
          p: 2,
        }}
      >
        <Typography variant="h6">
          {propertyData.address}
        </Typography>
        <Typography variant="body2">
          BRA: {propertyData.floorArea}m² | Tomteareal: {propertyData.lotSize}m²
        </Typography>
      </Box>
    </ViewerContainer>
  );
};

export default PropertyViewer;