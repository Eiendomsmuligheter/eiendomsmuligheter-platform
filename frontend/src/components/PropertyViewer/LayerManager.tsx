import React, { useState } from 'react';
import { LayerVisibility, BuildingLayer } from '../../types/building';
import { Box, List, ListItem, Switch, Typography, Tooltip } from '@mui/material';
import { usePropertyViewerContext } from '../../contexts/PropertyViewerContext';

interface LayerManagerProps {
  onLayerChange: (layers: LayerVisibility) => void;
}

const defaultLayers: BuildingLayer[] = [
  {
    id: 'structure',
    name: 'Bygningsstruktur',
    description: 'Vegger, tak og gulv',
    visible: true,
  },
  {
    id: 'plumbing',
    name: 'Rørleggerarbeid',
    description: 'Vann, avløp og ventilasjon',
    visible: true,
  },
  {
    id: 'electrical',
    name: 'Elektrisk',
    description: 'Strømkabler og elektriske installasjoner',
    visible: true,
  },
  {
    id: 'interior',
    name: 'Interiør',
    description: 'Møbler og innredning',
    visible: true,
  },
  {
    id: 'measurements',
    name: 'Måleverdier',
    description: 'Dimensjoner og avstander',
    visible: true,
  },
  {
    id: 'regulations',
    name: 'Reguleringer',
    description: 'Byggegrenser og forskriftskrav',
    visible: true,
  },
];

export const LayerManager: React.FC<LayerManagerProps> = ({ onLayerChange }) => {
  const [layers, setLayers] = useState<BuildingLayer[]>(defaultLayers);
  const { updateLayerVisibility } = usePropertyViewerContext();

  const handleLayerToggle = (layerId: string) => {
    const updatedLayers = layers.map(layer => {
      if (layer.id === layerId) {
        return { ...layer, visible: !layer.visible };
      }
      return layer;
    });
    setLayers(updatedLayers);

    // Oppdater lag-synlighet i kontekst
    const visibilityState = updatedLayers.reduce((acc, layer) => {
      acc[layer.id] = layer.visible;
      return acc;
    }, {} as LayerVisibility);

    onLayerChange(visibilityState);
    updateLayerVisibility(visibilityState);
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 360, bgcolor: 'background.paper' }}>
      <List>
        {layers.map((layer) => (
          <ListItem
            key={layer.id}
            secondaryAction={
              <Switch
                edge="end"
                onChange={() => handleLayerToggle(layer.id)}
                checked={layer.visible}
              />
            }
          >
            <Tooltip title={layer.description} placement="left">
              <Typography variant="body1">{layer.name}</Typography>
            </Tooltip>
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default LayerManager;