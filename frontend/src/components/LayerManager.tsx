import React from 'react';
import {
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Switch,
  Collapse,
  Paper,
  Typography,
  IconButton,
  Box,
  Slider,
  Tooltip,
} from '@mui/material';
import {
  Layers,
  ExpandMore,
  ExpandLess,
  Visibility,
  VisibilityOff,
  Opacity,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

interface Layer {
  id: string;
  name: string;
  type: 'structural' | 'electrical' | 'plumbing' | 'hvac' | 'architectural';
  visible: boolean;
  opacity: number;
  sublayers?: Layer[];
}

interface LayerManagerProps {
  layers: Layer[];
  onLayerChange: (layerId: string, changes: Partial<Layer>) => void;
  onLayerToggle: (layerId: string) => void;
}

const StyledPaper = styled(Paper)(({ theme }) => ({
  maxWidth: 400,
  margin: theme.spacing(2),
  maxHeight: '80vh',
  overflow: 'auto',
}));

const LayerItem: React.FC<{
  layer: Layer;
  onLayerChange: (layerId: string, changes: Partial<Layer>) => void;
  onLayerToggle: (layerId: string) => void;
  depth?: number;
}> = ({ layer, onLayerChange, onLayerToggle, depth = 0 }) => {
  const [expanded, setExpanded] = React.useState(false);

  const handleOpacityChange = (event: Event, newValue: number | number[]) => {
    onLayerChange(layer.id, { opacity: newValue as number });
  };

  const handleVisibilityToggle = () => {
    onLayerChange(layer.id, { visible: !layer.visible });
  };

  return (
    <>
      <ListItem
        style={{ paddingLeft: depth * 16 }}
        secondaryAction={
          <Switch
            edge="end"
            checked={layer.visible}
            onChange={handleVisibilityToggle}
          />
        }
      >
        <ListItemIcon>
          {layer.visible ? <Visibility /> : <VisibilityOff />}
        </ListItemIcon>
        <ListItemText
          primary={layer.name}
          secondary={
            <Box sx={{ width: 120, ml: 2 }}>
              <Slider
                value={layer.opacity}
                min={0}
                max={1}
                step={0.1}
                onChange={handleOpacityChange}
                disabled={!layer.visible}
              />
            </Box>
          }
        />
        {layer.sublayers && (
          <IconButton onClick={() => setExpanded(!expanded)}>
            {expanded ? <ExpandLess /> : <ExpandMore />}
          </IconButton>
        )}
      </ListItem>
      {layer.sublayers && (
        <Collapse in={expanded}>
          <List disablePadding>
            {layer.sublayers.map((sublayer) => (
              <LayerItem
                key={sublayer.id}
                layer={sublayer}
                onLayerChange={onLayerChange}
                onLayerToggle={onLayerToggle}
                depth={depth + 1}
              />
            ))}
          </List>
        </Collapse>
      )}
    </>
  );
};

export const LayerManager: React.FC<LayerManagerProps> = ({
  layers,
  onLayerChange,
  onLayerToggle,
}) => {
  return (
    <StyledPaper>
      <Box p={2}>
        <Typography variant="h6" gutterBottom>
          Lag-styring
        </Typography>
        <List>
          {layers.map((layer) => (
            <LayerItem
              key={layer.id}
              layer={layer}
              onLayerChange={onLayerChange}
              onLayerToggle={onLayerToggle}
            />
          ))}
        </List>
      </Box>
    </StyledPaper>
  );
};

export default LayerManager;