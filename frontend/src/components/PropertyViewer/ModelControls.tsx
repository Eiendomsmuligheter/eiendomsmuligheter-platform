import React from 'react';
import { Box, IconButton, Tooltip } from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  RestartAlt,
  ViewInAr,
  AspectRatio,
  ThreeDRotation,
} from '@mui/icons-material';

interface ModelControlsProps {
  onReset: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

const ModelControls: React.FC<ModelControlsProps> = ({
  onReset,
  onZoomIn,
  onZoomOut,
}) => {
  return (
    <Box
      sx={{
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        borderRadius: 2,
        padding: 1,
        display: 'flex',
        gap: 1,
      }}
    >
      <Tooltip title="3D-visning">
        <IconButton onClick={() => {}}>
          <ViewInAr />
        </IconButton>
      </Tooltip>
      <Tooltip title="Plantegning">
        <IconButton onClick={() => {}}>
          <AspectRatio />
        </IconButton>
      </Tooltip>
      <Tooltip title="Roter">
        <IconButton onClick={() => {}}>
          <ThreeDRotation />
        </IconButton>
      </Tooltip>
      <Tooltip title="Zoom inn">
        <IconButton onClick={onZoomIn}>
          <ZoomIn />
        </IconButton>
      </Tooltip>
      <Tooltip title="Zoom ut">
        <IconButton onClick={onZoomOut}>
          <ZoomOut />
        </IconButton>
      </Tooltip>
      <Tooltip title="Tilbakestill visning">
        <IconButton onClick={onReset}>
          <RestartAlt />
        </IconButton>
      </Tooltip>
    </Box>
  );
};

export default ModelControls;