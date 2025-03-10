import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { Building, Ruler, Maximize, TrendingUp, Zap, Download, Map, Cube } from 'lucide-react';
import { motion } from 'framer-motion';
import { ChatAssistant } from '../../components/ChatAssistant';
import { PropertyVisualizer } from '../../components/PropertyVisualizer';

export default function PropertyPage() {
  const router = useRouter();
  const { id } = router.query;

  const [propertyData, setPropertyData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!id) return;

    // Hent eiendomsdata fra API når ID er tilgjengelig
    const fetchPropertyData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/property/summary/${id}`);
        
        if (!response.ok) {
          throw new Error(`Kunne ikke hente eiendomsdata for ID: ${id}`);
        }
        
        const data = await response.json();
        setPropertyData(data);
        setLoading(false);
      } catch (error) {
        console.error('Feil ved henting av eiendomsdata:', error);
        setError(error.message);
        setLoading(false);
      }
    };

    fetchPropertyData();
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-white text-center">
          <div className="animate-spin h-12 w-12 border-t-2 border-b-2 border-blue-500 rounded-full mx-auto mb-4"></div>
          <p className="text-xl">Laster eiendomsdata...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-white text-center p-8 bg-red-900/20 rounded-xl border border-red-500/30 max-w-md">
          <h2 className="text-2xl font-bold mb-4">Noe gikk galt</h2>
          <p className="mb-4">{error}</p>
          <button 
            onClick={() => router.push('/')}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            Gå tilbake til hjemmesiden
          </button>
        </div>
      </div>
    );
  }

  // Fallback til dummy-data hvis vi ikke har data fra API
  const property = propertyData || {
    property_id: id,
    address: "Moreneveien 37, 3058 Solbergmoen",
    municipality_name: "Drammen",
    lot_size: 650.0,
    current_utilization: 0.25,
    building_height: 7.5,
    floor_area_ratio: 0.5,
    zoning_category: "Bolig",
    analysis: {
      max_buildable_area: 325.0,
      max_height: 9.0,
      max_units: 3,
      roi_estimate: 0.15,
      energy_class: "C",
      recommendations: [
        "Bygge rekkehus for optimal utnyttelse av tomten",
        "Vurdere solcellepaneler på taket",
        "Søk om dispensasjon for økt byggehøyde"
      ]
    },
    visualization: {
      has_3d_model: true,
      has_terrain: true,
      terrain_url: `/api/static/heightmaps/heightmap_${id}.png`,
      model_url: `/api/static/models/building_${id}.glb`,
      texture_url: `/api/static/textures/texture_${id}.jpg`
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-blue-950 text-white">
      {/* Header */}
      <header className="py-6 border-b border-white/10 bg-black/50 backdrop-blur-lg sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
            Eiendomsmuligheter
          </h1>
          <nav className="flex gap-6">
            <a href="/" className="text-gray-300 hover:text-white transition-colors">Hjem</a>
            <a href="/search" className="text-gray-300 hover:text-white transition-colors">Søk</a>
            <a href="/dashboard" className="text-gray-300 hover:text-white transition-colors">Dashboard</a>
            <a href="/pricing" className="text-gray-300 hover:text-white transition-colors">Abonnementer</a>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Property Header */}
        <div className="mb-12">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
            <div>
              <h1 className="text-4xl font-bold mb-2">{property.address}</h1>
              <p className="text-gray-400">{property.municipality_name} | Eiendoms-ID: {property.property_id}</p>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold flex items-center gap-2"
            >
              <Download className="w-5 h-5" />
              Last ned rapport
            </motion.button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white/5 rounded-lg p-4 border border-white/10 flex items-center gap-4">
              <Ruler className="w-10 h-10 text-blue-400" />
              <div>
                <p className="text-sm text-gray-400">Tomtestørrelse</p>
                <p className="text-xl font-semibold">{property.lot_size} m²</p>
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-4 border border-white/10 flex items-center gap-4">
              <Maximize className="w-10 h-10 text-blue-400" />
              <div>
                <p className="text-sm text-gray-400">Maks byggbart areal</p>
                <p className="text-xl font-semibold">{property.analysis.max_buildable_area} m²</p>
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-4 border border-white/10 flex items-center gap-4">
              <TrendingUp className="w-10 h-10 text-blue-400" />
              <div>
                <p className="text-sm text-gray-400">Estimert ROI</p>
                <p className="text-xl font-semibold">{(property.analysis.roi_estimate * 100).toFixed(1)}%</p>
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-4 border border-white/10 flex items-center gap-4">
              <Zap className="w-10 h-10 text-blue-400" />
              <div>
                <p className="text-sm text-gray-400">Energiklasse</p>
                <p className="text-xl font-semibold">{property.analysis.energy_class}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Analysis */}
          <div className="lg:col-span-1 space-y-8">
            {/* Property Analysis */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 overflow-hidden"
            >
              <div className="p-6 border-b border-white/10">
                <h2 className="text-2xl font-bold">Eiendomsanalyse</h2>
              </div>
              <div className="p-6 space-y-4">
                <div>
                  <p className="text-gray-400 mb-1">Reguleringstype</p>
                  <p className="font-semibold">{property.zoning_category}</p>
                </div>
                <div>
                  <p className="text-gray-400 mb-1">Utnyttelsesgrad</p>
                  <p className="font-semibold">{(property.current_utilization * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-gray-400 mb-1">Maksimal byggehøyde</p>
                  <p className="font-semibold">{property.analysis.max_height} meter</p>
                </div>
                <div>
                  <p className="text-gray-400 mb-1">Maksimalt antall enheter</p>
                  <p className="font-semibold">{property.analysis.max_units}</p>
                </div>
                <div>
                  <p className="text-gray-400 mb-1">BRA-faktor</p>
                  <p className="font-semibold">{property.floor_area_ratio}</p>
                </div>
              </div>
            </motion.div>

            {/* Recommendations */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 overflow-hidden"
            >
              <div className="p-6 border-b border-white/10">
                <h2 className="text-2xl font-bold">Anbefalinger</h2>
              </div>
              <div className="p-6">
                <ul className="space-y-3">
                  {property.analysis.recommendations.map((recommendation, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <div className="bg-blue-500/20 p-1 rounded-full mt-0.5">
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
                      </div>
                      <p>{recommendation}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          </div>

          {/* Right Column - Visualization */}
          <div className="lg:col-span-2 space-y-8">
            {/* Property Visualizer */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 overflow-hidden p-4"
            >
              <PropertyVisualizer 
                propertyId={property.property_id}
                address={property.address}
                terrainUrl={property.visualization.terrain_url}
                modelUrl={property.visualization.model_url}
                textureUrl={property.visualization.texture_url}
              />
            </motion.div>
          </div>
        </div>
      </main>

      {/* Chat Assistant */}
      <ChatAssistant />
    </div>
  );
} 