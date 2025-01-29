import React from 'react';
import {
  Box,
  ButtonGroup,
  Button,
  Slider,
  Typography,
  ToggleButtonGroup,
  ToggleButton,
  IconButton,
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  ThreeDRotation,
  ViewQuilt,
  Refresh,
  Brightness4,
  Brightness7,
} from '@mui/icons-material';

interface ModelControlsProps {
  onZoomChange: (zoom: number) => void;
  onRotationChange: (rotation: { x: number; y: number; z: number }) => void;
  onViewModeChange: (mode: '3d' | 'plan' | 'facade') => void;
  onReset: () => void;
  onLightingChange: (intensity: number) => void;
  currentMode: '3d' | 'plan' | 'facade';
  currentZoom: number;
  currentRotation: { x: number; y: number; z: number };
  currentLighting: number;
}

export const ModelControls: React.FC<ModelControlsProps> = ({
  onZoomChange,
  onRotationChange,
  onViewModeChange,
  onReset,
  onLightingChange,
  currentMode,
  currentZoom,
  currentRotation,
  currentLighting,
}) => {
  const handleZoomChange = (event: Event, newValue: number | number[]) => {
    onZoomChange(newValue as number);
  };

  const handleRotationChange = (axis: 'x' | 'y' | 'z', value: number) => {
    onRotationChange({
      ...currentRotation,
      [axis]: value,
    });
  };

  return (
    <Box sx={{ p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
      {/* Visningsmoduser */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Visningsmodus
        </Typography>
        <ToggleButtonGroup
          value={currentMode}
          exclusive
          onChange={(e, value) => value && onViewModeChange(value)}
          size="small"
        >
          <ToggleButton value="3d">
            <ThreeDRotation /> 3D
          </ToggleButton>
          <ToggleButton value="plan">
            <ViewQuilt /> Plantegning
          </ToggleButton>
          <ToggleButton value="facade">
            <ViewQuilt /> Fasade
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Zoom-kontroller */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Zoom
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton size="small" onClick={() => onZoomChange(currentZoom - 10)}>
            <ZoomOut />
          </IconButton>
          <Slider
            value={currentZoom}
            onChange={handleZoomChange}
            min={10}
            max={200}
            step={1}
          />
          <IconButton size="small" onClick={() => onZoomChange(currentZoom + 10)}>
            <ZoomIn />
          </IconButton>
        </Box>
      </Box>

      {/* Rotasjonskontroller */}
      {currentMode === '3d' && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Rotasjon
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box>
              <Typography variant="caption">X-akse</Typography>
              <Slider
                value={currentRotation.x}
                onChange={(e, value) => handleRotationChange('x', value as number)}
                min={0}
                max={360}
                step={1}
              />
            </Box>
            <Box>
              <Typography variant="caption">Y-akse</Typography>
              <Slider
                value={currentRotation.y}
                onChange={(e, value) => handleRotationChange('y', value as number)}
                min={0}
                max={360}
                step={1}
              />
            </Box>
            <Box>
              <Typography variant="caption">Z-akse</Typography>
              <Slider
                value={currentRotation.z}
                onChange={(e, value) => handleRotationChange('z', value as number)}
                min={0}
                max={360}
                step={1}
              />
            </Box>
          </Box>
        </Box>
      )}

      {/* Belysningskontroller */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Belysning
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Brightness4 />
          <Slider
            value={currentLighting}
            onChange={(e, value) => onLightingChange(value as number)}
            min={0}
            max={100}
            step={1}
          />
          <Brightness7 />
        </Box>
      </Box>

      {/* Reset-knapp */}
      <Button
        variant="outlined"
        startIcon={<Refresh />}
        onClick={onReset}
        fullWidth
      >
        Tilbakestill visning
      </Button>
    </Box>
  );
};