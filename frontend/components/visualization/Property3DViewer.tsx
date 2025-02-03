import React, { useEffect, useRef } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { OmniverseClient } from '../../services/omniverseService';
import { BuildingData } from '../../types/property';
import styles from '../../styles/Property3DViewer.module.css';

interface Property3DViewerProps {
  buildingData: BuildingData;
  className?: string;
}

export const Property3DViewer: React.FC<Property3DViewerProps> = ({
  buildingData,
  className
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const initializeViewer = async () => {
      try {
        if (!containerRef.current) return;
        
        // Initialize NVIDIA Omniverse client
        const omniverse = new OmniverseClient();
        
        // Create viewer instance
        viewerRef.current = await omniverse.createViewer({
          container: containerRef.current,
          settings: {
            quality: 'high',
            shadows: true,
            lighting: 'physical',
            background: 'environment'
          }
        });
        
        // Load environment
        await viewerRef.current.loadEnvironment('daylight');
        
        // Generate and load 3D model from building data
        const model = await generateBuildingModel(buildingData);
        await viewerRef.current.loadModel(model);
        
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to initialize 3D viewer:', err);
        setError('Kunne ikke laste 3D-visning');
        setIsLoading(false);
      }
    };
    
    initializeViewer();
    
    return () => {
      // Cleanup viewer when component unmounts
      if (viewerRef.current) {
        viewerRef.current.dispose();
      }
    };
  }, []);
  
  // Update model when building data changes
  useEffect(() => {
    const updateModel = async () => {
      if (!viewerRef.current) return;
      
      try {
        setIsLoading(true);
        
        // Generate new model from updated building data
        const model = await generateBuildingModel(buildingData);
        
        // Update existing model
        await viewerRef.current.updateModel(model);
        
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to update 3D model:', err);
        setError('Kunne ikke oppdatere 3D-modell');
        setIsLoading(false);
      }
    };
    
    updateModel();
  }, [buildingData]);
  
  const generateBuildingModel = async (data: BuildingData) => {
    // Convert building data to USD format for Omniverse
    const modelGenerator = new BuildingModelGenerator();
    
    // Generate base structure
    modelGenerator.addFoundation(data.foundation);
    
    // Add walls
    data.walls.forEach(wall => {
      modelGenerator.addWall(wall);
    });
    
    // Add floors and ceilings
    data.floors.forEach(floor => {
      modelGenerator.addFloor(floor);
    });
    
    // Add roof
    modelGenerator.addRoof(data.roof);
    
    // Add windows and doors
    data.openings.forEach(opening => {
      if (opening.type === 'window') {
        modelGenerator.addWindow(opening);
      } else {
        modelGenerator.addDoor(opening);
      }
    });
    
    // Add interior walls and features
    data.interiorFeatures.forEach(feature => {
      modelGenerator.addInteriorFeature(feature);
    });
    
    // Apply materials
    await modelGenerator.applyMaterials(data.materials);
    
    // Generate USD file
    return await modelGenerator.generateUSD();
  };
  
  return (
    <Box 
      ref={containerRef}
      className={`${styles.container} ${className}`}
    >
      {isLoading && (
        <Box className={styles.loading}>
          <CircularProgress />
          <p>Laster 3D-visning...</p>
        </Box>
      )}
      
      {error && (
        <Box className={styles.error}>
          <p>{error}</p>
        </Box>
      )}
      
      {!isLoading && !error && (
        <Box className={styles.controls}>
          <button onClick={() => viewerRef.current?.resetCamera()}>
            Tilbakestill visning
          </button>
          <button onClick={() => viewerRef.current?.toggleWireframe()}>
            Vis/Skjul trådmodell
          </button>
          <button onClick={() => viewerRef.current?.toggleMeasurementMode()}>
            Måleverktøy
          </button>
          <select 
            onChange={(e) => viewerRef.current?.setViewMode(e.target.value)}
            defaultValue="3d"
          >
            <option value="3d">3D</option>
            <option value="floor-plan">Plantegning</option>
            <option value="section">Snitt</option>
            <option value="elevation">Fasade</option>
          </select>
        </Box>
      )}
    </Box>
  );
};