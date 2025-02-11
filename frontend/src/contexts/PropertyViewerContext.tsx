import React, { createContext, useContext, useState } from 'react';
import { LayerVisibility } from '../types/building';

interface PropertyViewerContextType {
  layerVisibility: LayerVisibility;
  updateLayerVisibility: (layers: LayerVisibility) => void;
  viewMode: '3d' | 'floorplan' | 'facade';
  setViewMode: (mode: '3d' | 'floorplan' | 'facade') => void;
}

const PropertyViewerContext = createContext<PropertyViewerContextType | undefined>(
  undefined
);

export const usePropertyViewerContext = () => {
  const context = useContext(PropertyViewerContext);
  if (!context) {
    throw new Error(
      'usePropertyViewerContext must be used within a PropertyViewerProvider'
    );
  }
  return context;
};

export const PropertyViewerProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [layerVisibility, setLayerVisibility] = useState<LayerVisibility>({
    structure: true,
    plumbing: true,
    electrical: true,
    interior: true,
    measurements: true,
    regulations: true,
  });

  const [viewMode, setViewMode] = useState<'3d' | 'floorplan' | 'facade'>('3d');

  const updateLayerVisibility = (layers: LayerVisibility) => {
    setLayerVisibility(layers);
  };

  return (
    <PropertyViewerContext.Provider
      value={{
        layerVisibility,
        updateLayerVisibility,
        viewMode,
        setViewMode,
      }}
    >
      {children}
    </PropertyViewerContext.Provider>
  );
};

export default PropertyViewerContext;