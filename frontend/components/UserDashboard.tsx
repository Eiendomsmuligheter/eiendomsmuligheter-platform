import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Home, 
  PieChart, 
  Settings, 
  Bell, 
  FileText, 
  TrendingUp,
  Map,
  Users,
  HelpCircle,
  Search,
  Plus,
  Filter,
  ArrowUpRight
} from 'lucide-react';

// Dashboard sidebar navikasjon
const Sidebar = ({ activeTab, setActiveTab }) => {
  const menuItems = [
    { icon: <Home size={20} />, label: 'Oversikt', id: 'overview' },
    { icon: <Map size={20} />, label: 'Mine eiendommer', id: 'properties' },
    { icon: <PieChart size={20} />, label: 'Analyser', id: 'analytics' },
    { icon: <TrendingUp size={20} />, label: 'Investeringer', id: 'investments' },
    { icon: <FileText size={20} />, label: 'Rapporter', id: 'reports' },
    { icon: <Bell size={20} />, label: 'Varsler', id: 'notifications' },
    { icon: <Users size={20} />, label: 'Team', id: 'team' },
    { icon: <Settings size={20} />, label: 'Innstillinger', id: 'settings' },
    { icon: <HelpCircle size={20} />, label: 'Hjelp & støtte', id: 'support' },
  ];

  return (
    <div className="bg-gradient-to-b from-gray-900 to-black w-64 p-4 border-r border-white/10 h-full flex flex-col">
      <div className="mb-8">
        <h2 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
          Eiendomsmuligheter
        </h2>
      </div>
      
      <nav className="flex-1">
        <ul className="space-y-2">
          {menuItems.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  activeTab === item.id
                    ? 'bg-blue-600/20 text-blue-400'
                    : 'text-gray-400 hover:bg-white/5 hover:text-white'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
                {activeTab === item.id && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-400"></div>
                )}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      
      <div className="mt-auto pt-6 border-t border-white/10">
        <div className="flex items-center space-x-3 px-4 py-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <span className="text-white font-medium">AR</span>
          </div>
          <div>
            <p className="text-white text-sm font-medium">Ahmad Rezae</p>
            <p className="text-gray-400 text-xs">Professional Plan</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// Property card for displaying properties
const PropertyCard = ({ property }) => {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 overflow-hidden"
    >
      <div 
        className="h-36 bg-cover bg-center"
        style={{ backgroundImage: `url(${property.image})` }}
      >
        <div className="w-full h-full bg-gradient-to-t from-black to-transparent p-4 flex flex-col justify-end">
          <div className="bg-blue-600/80 text-white text-xs font-medium px-2 py-1 rounded w-fit">
            {property.status}
          </div>
        </div>
      </div>
      <div className="p-4">
        <h3 className="text-white font-medium mb-1">{property.address}</h3>
        <p className="text-gray-400 text-sm mb-3">{property.municipality}</p>
        
        <div className="flex justify-between text-sm mb-4">
          <div>
            <p className="text-gray-400">Tomt</p>
            <p className="text-white">{property.lotSize} m²</p>
          </div>
          <div>
            <p className="text-gray-400">Byggepotensial</p>
            <p className="text-white">{property.buildingPotential} m²</p>
          </div>
          <div>
            <p className="text-gray-400">ROI</p>
            <p className="text-white">{property.roi}%</p>
          </div>
        </div>
        
        <div className="flex justify-between">
          <button className="text-blue-400 text-sm flex items-center">
            <ArrowUpRight size={16} className="mr-1" />
            Se detaljer
          </button>
          <div className="text-gray-400 text-xs flex items-center">
            Oppdatert {property.lastUpdated}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Stat card component
const StatCard = ({ title, value, icon, change, changeType }) => {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 p-4"
    >
      <div className="flex justify-between items-start mb-3">
        <p className="text-gray-400">{title}</p>
        <div className="p-2 rounded-lg bg-blue-500/20">
          {icon}
        </div>
      </div>
      <div className="flex items-baseline">
        <p className="text-2xl font-semibold text-white">{value}</p>
        {change && (
          <div className={`ml-2 flex items-center text-xs font-medium ${
            changeType === 'positive' ? 'text-green-400' : 'text-red-400'
          }`}>
            {changeType === 'positive' ? '↑' : '↓'} {change}
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Dashboard content
const DashboardContent = ({ activeTab }) => {
  // Eksempeldata for egenskapene
  const properties = [
    {
      id: 1,
      address: 'Moreneveien 37, 3058 Solbergmoen',
      municipality: 'Drammen',
      status: 'Analysert',
      lotSize: 650,
      buildingPotential: 325,
      roi: 15.2,
      lastUpdated: 'i dag',
      image: 'https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?ixlib=rb-1.2.1&auto=format&fit=crop&w=2850&q=80'
    },
    {
      id: 2,
      address: 'Storgata 45, 0182 Oslo',
      municipality: 'Oslo',
      status: 'Ny mulighet',
      lotSize: 420,
      buildingPotential: 210,
      roi: 12.8,
      lastUpdated: '3 dager siden',
      image: 'https://images.unsplash.com/photo-1580587771525-78b9dba3b914?ixlib=rb-1.2.1&auto=format&fit=crop&w=2734&q=80'
    },
    {
      id: 3,
      address: 'Bergveien 12, 5003 Bergen',
      municipality: 'Bergen',
      status: 'Under utvikling',
      lotSize: 780,
      buildingPotential: 390,
      roi: 18.5,
      lastUpdated: '1 uke siden',
      image: 'https://images.unsplash.com/photo-1568605114967-8130f3a36994?ixlib=rb-1.2.1&auto=format&fit=crop&w=2850&q=80'
    },
  ];

  if (activeTab === 'overview') {
    return (
      <div className="space-y-8">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-white">Velkommen, Ahmad</h1>
          <div className="flex space-x-3">
            <button className="bg-white/5 hover:bg-white/10 p-2 rounded-lg text-white">
              <Bell size={20} />
            </button>
            <button className="bg-white/5 hover:bg-white/10 p-2 rounded-lg text-white">
              <Settings size={20} />
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard 
            title="Totalt analyserte eiendommer" 
            value="12" 
            icon={<Home size={20} className="text-blue-400" />} 
            change="3" 
            changeType="positive" 
          />
          <StatCard 
            title="Gjennomsnittlig ROI" 
            value="14.7%" 
            icon={<TrendingUp size={20} className="text-blue-400" />} 
            change="2.3%" 
            changeType="positive" 
          />
          <StatCard 
            title="Totalt byggepotensial" 
            value="4,250 m²" 
            icon={<PieChart size={20} className="text-blue-400" />} 
            change="850 m²" 
            changeType="positive" 
          />
          <StatCard 
            title="Aktive varsler" 
            value="3" 
            icon={<Bell size={20} className="text-blue-400" />} 
          />
        </div>
        
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold text-white">Nylige eiendommer</h2>
            <button className="text-blue-400 text-sm flex items-center">
              Se alle eiendommer
              <ArrowUpRight size={16} className="ml-1" />
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {properties.map(property => (
              <PropertyCard key={property.id} property={property} />
            ))}
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-white">Aktivitetslogg</h2>
            <button className="text-gray-400 text-sm">Filtrer</button>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-start space-x-4">
              <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center mt-1 flex-shrink-0">
                <Map size={16} className="text-green-400" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <p className="text-white font-medium">Ny eiendom analysert</p>
                  <span className="text-gray-400 text-sm">i dag, 10:23</span>
                </div>
                <p className="text-gray-400">Du har fullført analyse av Moreneveien 37</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-4">
              <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center mt-1 flex-shrink-0">
                <Bell size={16} className="text-blue-400" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <p className="text-white font-medium">Nytt reguleringsvarsel</p>
                  <span className="text-gray-400 text-sm">i går, 15:47</span>
                </div>
                <p className="text-gray-400">Ny reguleringsplan for området rundt Storgata 45</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-4">
              <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center mt-1 flex-shrink-0">
                <FileText size={16} className="text-purple-400" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <p className="text-white font-medium">Rapport generert</p>
                  <span className="text-gray-400 text-sm">3 dager siden</span>
                </div>
                <p className="text-gray-400">Fullstendig analyserapport for Bergveien 12</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  if (activeTab === 'properties') {
    return (
      <div className="space-y-8">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-white">Mine eiendommer</h1>
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-4 py-2 rounded-lg flex items-center"
          >
            <Plus size={18} className="mr-2" />
            Legg til eiendom
          </motion.button>
        </div>
        
        <div className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 p-4 flex flex-wrap gap-4">
          <div className="relative flex-grow">
            <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input 
              type="text" 
              placeholder="Søk etter adresse, eiendomsnummer..." 
              className="w-full bg-white/5 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <button className="bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg px-4 py-2 text-white flex items-center">
            <Filter size={18} className="mr-2" />
            Filter
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {properties.map(property => (
            <PropertyCard key={property.id} property={property} />
          ))}
          
          <motion.div
            whileHover={{ y: -5 }}
            className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-dashed border-white/20 flex items-center justify-center h-80"
          >
            <div className="text-center p-6">
              <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4">
                <Plus size={24} className="text-blue-400" />
              </div>
              <h3 className="text-white font-medium mb-2">Legg til ny eiendom</h3>
              <p className="text-gray-400 text-sm mb-4">Analyser en ny eiendom for å avdekke utviklingspotensialet</p>
              <button className="bg-white/10 hover:bg-white/20 text-white px-4 py-2 rounded-lg">
                Legg til eiendom
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }
  
  // Fallback content for other tabs
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-white mb-2">
          {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
        </h2>
        <p className="text-gray-400">Denne funksjonen er under utvikling.</p>
      </div>
    </div>
  );
};

export const UserDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  
  return (
    <div className="bg-black min-h-screen flex">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="flex-1 p-8 overflow-auto">
        <DashboardContent activeTab={activeTab} />
      </div>
    </div>
  );
}; 