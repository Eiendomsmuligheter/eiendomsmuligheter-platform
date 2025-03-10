import React from 'react';
import { Home, Zap } from 'lucide-react';
import { LegalDisclaimer } from '../common/LegalDisclaimer';
import { motion } from 'framer-motion';

export const RentalListingHeader = () => {
  return (
    <div className="mb-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Leie av bolig</h1>
          <p className="text-gray-400 max-w-2xl">
            Finn din drømmebolig til leie - eller legg ut din egen eiendom for utleie. 
            Vår plattform kobler sammen utleiere og leietakere, uten mellomledd eller 
            skjulte kostnader.
          </p>
        </div>
        
        <motion.button 
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="mt-4 md:mt-0 px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium flex items-center gap-2 hover:opacity-90 transition-opacity"
        >
          <Home size={18} />
          Legg ut bolig for utleie
        </motion.button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white/5 border border-white/10 rounded-lg p-4 flex items-center">
          <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center mr-3">
            <Zap size={20} className="text-blue-400" />
          </div>
          <div>
            <h3 className="font-medium text-white text-sm">Direkte kontakt</h3>
            <p className="text-gray-400 text-xs">Direkte kommunikasjon med utleiere</p>
          </div>
        </div>
        
        <div className="bg-white/5 border border-white/10 rounded-lg p-4 flex items-center">
          <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center mr-3">
            <Zap size={20} className="text-green-400" />
          </div>
          <div>
            <h3 className="font-medium text-white text-sm">Verifiserte boliger</h3>
            <p className="text-gray-400 text-xs">Alle boliger gjennomgår en kvalitetssjekk</p>
          </div>
        </div>
        
        <div className="bg-white/5 border border-white/10 rounded-lg p-4 flex items-center">
          <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center mr-3">
            <Zap size={20} className="text-purple-400" />
          </div>
          <div>
            <h3 className="font-medium text-white text-sm">Digitale leiekontrakter</h3>
            <p className="text-gray-400 text-xs">Enkelt å signere og administrere</p>
          </div>
        </div>
      </div>
      
      <LegalDisclaimer type="rental" compact={true} />
    </div>
  );
}; 