import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  MapPin, 
  Maximize, 
  Minimize, 
  Layers, 
  Sun, 
  Moon, 
  Compass, 
  Grid, 
  Home, 
  Mountain,
  Download
} from 'lucide-react';

interface VisualizerProps {
  propertyId: string;
  address: string;
  terrainUrl?: string;
  modelUrl?: string;
  textureUrl?: string;
}

export const PropertyVisualizer: React.FC<VisualizerProps> = ({
  propertyId,
  address,
  terrainUrl,
  modelUrl,
  textureUrl
}) => {
  const [activeView, setActiveView] = useState<'terrain' | 'building' | 'combined'>('terrain');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isDayMode, setIsDayMode] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showCompass, setShowCompass] = useState(true);
  
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };
  
  const toggleDayNight = () => {
    setIsDayMode(!isDayMode);
  };
  
  const toggleGrid = () => {
    setShowGrid(!showGrid);
  };
  
  const toggleCompass = () => {
    setShowCompass(!showCompass);
  };
  
  return (
    <div className={`bg-black ${isFullscreen ? 'fixed inset-0 z-50' : 'relative'}`}>
      {/* Toolbar */}
      <div className="absolute top-4 left-4 z-10 flex flex-col space-y-2">
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="bg-black/70 text-white p-2 rounded-lg backdrop-blur-sm border border-white/10 hover:bg-black/90"
          onClick={() => setActiveView('terrain')}
        >
          <Mountain size={20} className={activeView === 'terrain' ? 'text-blue-400' : 'text-white'} />
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="bg-black/70 text-white p-2 rounded-lg backdrop-blur-sm border border-white/10 hover:bg-black/90"
          onClick={() => setActiveView('building')}
        >
          <Home size={20} className={activeView === 'building' ? 'text-blue-400' : 'text-white'} />
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="bg-black/70 text-white p-2 rounded-lg backdrop-blur-sm border border-white/10 hover:bg-black/90"
          onClick={() => setActiveView('combined')}
        >
          <Layers size={20} className={activeView === 'combined' ? 'text-blue-400' : 'text-white'} />
        </motion.button>
      </div>
      
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex flex-col space-y-2">
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="bg-black/70 text-white p-2 rounded-lg backdrop-blur-sm border border-white/10 hover:bg-black/90"
          onClick={toggleFullscreen}
        >
          {isFullscreen ? <Minimize size={20} /> : <Maximize size={20} />}
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="bg-black/70 text-white p-2 rounded-lg backdrop-blur-sm border border-white/10 hover:bg-black/90"
          onClick={toggleDayNight}
        >
          {isDayMode ? <Sun size={20} /> : <Moon size={20} />}
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="bg-black/70 text-white p-2 rounded-lg backdrop-blur-sm border border-white/10 hover:bg-black/90"
          onClick={toggleGrid}
        >
          <Grid size={20} className={showGrid ? 'text-blue-400' : 'text-white'} />
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="bg-black/70 text-white p-2 rounded-lg backdrop-blur-sm border border-white/10 hover:bg-black/90"
          onClick={toggleCompass}
        >
          <Compass size={20} className={showCompass ? 'text-blue-400' : 'text-white'} />
        </motion.button>
      </div>
      
      {/* Visualization area */}
      <div className={`relative ${isFullscreen ? 'h-screen' : 'aspect-[16/9] max-h-[80vh]'} w-full bg-gradient-to-b from-gray-900 to-black overflow-hidden`}>
        {/* Fallback visualizations until 3D is implemented */}
        {activeView === 'terrain' && (
          <div className="absolute inset-0 flex items-center justify-center">
            {textureUrl ? (
              <div className="relative w-full h-full">
                <img 
                  src={textureUrl} 
                  alt="Terrengvisualisering" 
                  className="absolute inset-0 w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent pointer-events-none"></div>
              </div>
            ) : (
              <div className="text-white text-center p-8">
                <Mountain size={64} className="mx-auto mb-4 text-blue-400 opacity-50" />
                <p>Terrengvisualisering ikke tilgjengelig</p>
              </div>
            )}
          </div>
        )}
        
        {activeView === 'building' && (
          <div className="absolute inset-0 flex items-center justify-center">
            {modelUrl ? (
              <div className="text-center">
                <div className="w-32 h-32 relative mx-auto mb-4">
                  <div className="absolute inset-0 bg-blue-500/20 rounded-xl animate-pulse"></div>
                  <Home size={64} className="absolute inset-0 m-auto text-blue-400" />
                </div>
                <p className="text-white">3D bygningsmodell laster...</p>
                <p className="text-gray-400 text-sm mt-2">Modell: {modelUrl.split('/').pop()}</p>
              </div>
            ) : (
              <div className="text-white text-center p-8">
                <Home size={64} className="mx-auto mb-4 text-blue-400 opacity-50" />
                <p>3D bygningsmodell ikke tilgjengelig</p>
              </div>
            )}
          </div>
        )}
        
        {activeView === 'combined' && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-32 h-32 relative mx-auto mb-4">
                <div className="absolute inset-0 bg-purple-500/20 rounded-xl animate-pulse"></div>
                <Layers size={64} className="absolute inset-0 m-auto text-purple-400" />
              </div>
              <p className="text-white">Kombinert visualisering</p>
              <p className="text-gray-400 text-sm mt-2">Terreng + 3D-modell</p>
            </div>
          </div>
        )}
        
        {/* Compass */}
        {showCompass && (
          <div className="absolute bottom-8 right-8 z-10">
            <div className="bg-black/70 rounded-full w-16 h-16 flex items-center justify-center backdrop-blur-sm border border-white/10">
              <Compass size={32} className="text-blue-400" />
            </div>
          </div>
        )}
        
        {/* Grid overlay */}
        {showGrid && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="w-full h-full bg-[linear-gradient(to_right,rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:50px_50px]"></div>
          </div>
        )}
        
        {/* Location pin */}
        <div className="absolute bottom-8 left-8 z-10 flex items-center space-x-3 bg-black/70 px-4 py-2 rounded-lg backdrop-blur-sm border border-white/10">
          <MapPin size={20} className="text-blue-400" />
          <div>
            <p className="text-white text-sm">{address}</p>
            <p className="text-gray-400 text-xs">ID: {propertyId}</p>
          </div>
        </div>
      </div>
      
      {/* Download buttons */}
      {!isFullscreen && (
        <div className="mt-4 flex space-x-4">
          {terrainUrl && (
            <a 
              href={terrainUrl} 
              download={`terrain_${propertyId}.png`}
              className="flex items-center space-x-2 text-blue-400 hover:text-blue-300 text-sm"
            >
              <Download size={16} />
              <span>Last ned terrengdata</span>
            </a>
          )}
          
          {modelUrl && (
            <a 
              href={modelUrl} 
              download={`model_${propertyId}.glb`}
              className="flex items-center space-x-2 text-blue-400 hover:text-blue-300 text-sm"
            >
              <Download size={16} />
              <span>Last ned 3D-modell</span>
            </a>
          )}
        </div>
      )}
    </div>
  );
}; 