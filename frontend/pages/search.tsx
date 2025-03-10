import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Search, 
  MapPin, 
  Filter, 
  X, 
  ChevronDown, 
  Home,
  Building,
  Clock,
  ArrowUpRight
} from 'lucide-react';

// Property card for search results
const PropertyCard = ({ property }) => {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 overflow-hidden"
    >
      <div 
        className="h-48 bg-cover bg-center"
        style={{ backgroundImage: `url(${property.image})` }}
      >
        <div className="w-full h-full bg-gradient-to-t from-black to-transparent p-4 flex flex-col justify-end">
          <div className="bg-blue-600/80 text-white text-xs font-medium px-2 py-1 rounded w-fit">
            {property.type}
          </div>
        </div>
      </div>
      <div className="p-4">
        <h3 className="text-xl font-medium text-white mb-1">{property.address}</h3>
        <div className="flex items-center text-gray-400 text-sm mb-4">
          <MapPin size={14} className="mr-1" />
          <span>{property.municipality}</span>
        </div>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-gray-400 text-xs">Tomtestørrelse</p>
            <p className="text-white">{property.lotSize} m²</p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Byggepotensial</p>
            <p className="text-white">{property.buildingPotential} m²</p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Utnyttelsesgrad</p>
            <p className="text-white">{property.utilization}%</p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">ROI</p>
            <p className="text-white">{property.roi}%</p>
          </div>
        </div>
        
        <div className="flex justify-between items-center">
          <div className="flex items-center text-gray-400 text-xs">
            <Clock size={14} className="mr-1" />
            <span>Lagt til {property.addedDate}</span>
          </div>
          <a 
            href={`/property/${property.id}`}
            className="flex items-center text-blue-400 text-sm font-medium"
          >
            Se detaljer
            <ArrowUpRight size={16} className="ml-1" />
          </a>
        </div>
      </div>
    </motion.div>
  );
};

// Filter dropdown
const FilterDropdown = ({ title, options, selectedValues, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white"
      >
        <span>{title}</span>
        <ChevronDown size={16} />
      </button>
      
      {isOpen && (
        <div className="absolute z-10 mt-2 w-full bg-gray-900 border border-white/10 rounded-lg shadow-xl p-2 max-h-60 overflow-y-auto">
          {options.map((option) => (
            <div key={option.value} className="flex items-center p-2 hover:bg-white/5 rounded">
              <input
                type="checkbox"
                id={`option-${option.value}`}
                checked={selectedValues.includes(option.value)}
                onChange={() => {
                  const newValues = selectedValues.includes(option.value)
                    ? selectedValues.filter(v => v !== option.value)
                    : [...selectedValues, option.value];
                  onChange(newValues);
                }}
                className="mr-2"
              />
              <label htmlFor={`option-${option.value}`} className="text-white cursor-pointer">
                {option.label}
              </label>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Range slider component
const RangeSlider = ({ title, min, max, step, value, onChange }) => {
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <label className="text-white">{title}</label>
        <span className="text-gray-400 text-sm">{value.min} - {value.max}</span>
      </div>
      <div className="flex space-x-2">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value.min}
          onChange={(e) => onChange({ ...value, min: parseInt(e.target.value) })}
          className="w-full"
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value.max}
          onChange={(e) => onChange({ ...value, max: parseInt(e.target.value) })}
          className="w-full"
        />
      </div>
    </div>
  );
};

export default function SearchPage() {
  // State for search and filters
  const [searchTerm, setSearchTerm] = useState('');
  const [propertyTypes, setPropertyTypes] = useState([]);
  const [municipalities, setMunicipalities] = useState([]);
  const [lotSizeRange, setLotSizeRange] = useState({ min: 0, max: 2000 });
  const [roiRange, setRoiRange] = useState({ min: 0, max: 30 });
  const [showFilters, setShowFilters] = useState(false);
  
  // Example property data
  const properties = [
    {
      id: '12345',
      address: 'Moreneveien 37, 3058 Solbergmoen',
      municipality: 'Drammen',
      type: 'Boligtomt',
      lotSize: 650,
      buildingPotential: 325,
      utilization: 50,
      roi: 15.2,
      addedDate: '2 dager siden',
      image: 'https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?ixlib=rb-1.2.1&auto=format&fit=crop&w=2850&q=80'
    },
    {
      id: '12346',
      address: 'Storgata 45, 0182 Oslo',
      municipality: 'Oslo',
      type: 'Næringstomt',
      lotSize: 1200,
      buildingPotential: 960,
      utilization: 80,
      roi: 18.7,
      addedDate: '1 uke siden',
      image: 'https://images.unsplash.com/photo-1580587771525-78b9dba3b914?ixlib=rb-1.2.1&auto=format&fit=crop&w=2734&q=80'
    },
    {
      id: '12347',
      address: 'Bergveien 12, 5003 Bergen',
      municipality: 'Bergen',
      type: 'Boligtomt',
      lotSize: 780,
      buildingPotential: 390,
      utilization: 50,
      roi: 14.3,
      addedDate: '3 uker siden',
      image: 'https://images.unsplash.com/photo-1568605114967-8130f3a36994?ixlib=rb-1.2.1&auto=format&fit=crop&w=2850&q=80'
    },
    {
      id: '12348',
      address: 'Sjøgata 78, 7010 Trondheim',
      municipality: 'Trondheim',
      type: 'Næringstomt',
      lotSize: 1500,
      buildingPotential: 1200,
      utilization: 80,
      roi: 16.5,
      addedDate: '1 måned siden',
      image: 'https://images.unsplash.com/photo-1564501049412-61c2a3083791?ixlib=rb-1.2.1&auto=format&fit=crop&w=2089&q=80'
    },
    {
      id: '12349',
      address: 'Havnegata 5, 4306 Sandnes',
      municipality: 'Sandnes',
      type: 'Kombinert',
      lotSize: 950,
      buildingPotential: 710,
      utilization: 75,
      roi: 17.8,
      addedDate: '2 måneder siden',
      image: 'https://images.unsplash.com/photo-1582407947304-fd86f028f716?ixlib=rb-1.2.1&auto=format&fit=crop&w=2296&q=80'
    },
    {
      id: '12350',
      address: 'Fjordveien 23, 3050 Mjøndalen',
      municipality: 'Drammen',
      type: 'Boligtomt',
      lotSize: 520,
      buildingPotential: 260,
      utilization: 50,
      roi: 13.4,
      addedDate: '3 måneder siden',
      image: 'https://images.unsplash.com/photo-1598228723793-52759bba239c?ixlib=rb-1.2.1&auto=format&fit=crop&w=2378&q=80'
    },
  ];
  
  // Filter options
  const propertyTypeOptions = [
    { label: 'Boligtomt', value: 'Boligtomt' },
    { label: 'Næringstomt', value: 'Næringstomt' },
    { label: 'Kombinert', value: 'Kombinert' },
    { label: 'Fritidstomt', value: 'Fritidstomt' },
    { label: 'Landbrukseiendom', value: 'Landbrukseiendom' },
  ];
  
  const municipalityOptions = [
    { label: 'Oslo', value: 'Oslo' },
    { label: 'Bergen', value: 'Bergen' },
    { label: 'Trondheim', value: 'Trondheim' },
    { label: 'Stavanger', value: 'Stavanger' },
    { label: 'Drammen', value: 'Drammen' },
    { label: 'Sandnes', value: 'Sandnes' },
    { label: 'Kristiansand', value: 'Kristiansand' },
    { label: 'Tromsø', value: 'Tromsø' },
  ];
  
  // Filter properties based on search and filters
  const filteredProperties = properties.filter(property => {
    // Search term
    if (searchTerm && !property.address.toLowerCase().includes(searchTerm.toLowerCase()) && 
        !property.municipality.toLowerCase().includes(searchTerm.toLowerCase())) {
      return false;
    }
    
    // Property types
    if (propertyTypes.length > 0 && !propertyTypes.includes(property.type)) {
      return false;
    }
    
    // Municipalities
    if (municipalities.length > 0 && !municipalities.includes(property.municipality)) {
      return false;
    }
    
    // Lot size range
    if (property.lotSize < lotSizeRange.min || property.lotSize > lotSizeRange.max) {
      return false;
    }
    
    // ROI range
    if (property.roi < roiRange.min || property.roi > roiRange.max) {
      return false;
    }
    
    return true;
  });
  
  // Reset all filters
  const resetFilters = () => {
    setSearchTerm('');
    setPropertyTypes([]);
    setMunicipalities([]);
    setLotSizeRange({ min: 0, max: 2000 });
    setRoiRange({ min: 0, max: 30 });
  };

  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <header className="bg-gradient-to-r from-gray-900 to-black py-6 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
              Eiendomsmuligheter
            </h1>
            <nav className="flex gap-6">
              <a href="/" className="text-gray-300 hover:text-white transition-colors">Hjem</a>
              <a href="/search" className="text-blue-400 hover:text-blue-300 transition-colors">Søk</a>
              <a href="/dashboard" className="text-gray-300 hover:text-white transition-colors">Dashboard</a>
              <a href="/pricing" className="text-gray-300 hover:text-white transition-colors">Abonnementer</a>
            </nav>
          </div>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">Finn eiendom med potensial</h1>
          <p className="text-xl text-gray-400">Søk blant tusenvis av eiendommer og avdekk skjulte muligheter.</p>
        </div>
        
        {/* Search and filters */}
        <div className="mb-12">
          <div className="flex flex-col lg:flex-row gap-4 mb-4">
            <div className="flex-grow relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Søk etter adresse, kommune, gnr/bnr..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full py-3 pl-12 pr-4 rounded-lg bg-white/5 backdrop-blur-lg border border-white/20 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              />
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-6 py-3 rounded-lg bg-white/10 hover:bg-white/15 text-white flex items-center justify-center gap-2 transition-colors"
            >
              <Filter size={20} />
              Filtrer
              <div className={`w-2 h-2 rounded-full ${
                propertyTypes.length > 0 || municipalities.length > 0 ? 'bg-blue-400' : 'bg-transparent'
              }`}></div>
            </button>
          </div>
          
          {showFilters && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="bg-gradient-to-br from-gray-900 to-black border border-white/10 rounded-lg p-6 mb-6"
            >
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-semibold text-white">Filtrer eiendommer</h3>
                <button
                  onClick={resetFilters}
                  className="text-gray-400 hover:text-white flex items-center gap-1 transition-colors"
                >
                  <X size={16} />
                  Nullstill
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <FilterDropdown
                  title="Eiendomstype"
                  options={propertyTypeOptions}
                  selectedValues={propertyTypes}
                  onChange={setPropertyTypes}
                />
                
                <FilterDropdown
                  title="Kommune"
                  options={municipalityOptions}
                  selectedValues={municipalities}
                  onChange={setMunicipalities}
                />
                
                <div>
                  <RangeSlider
                    title="Tomtestørrelse (m²)"
                    min={0}
                    max={2000}
                    step={50}
                    value={lotSizeRange}
                    onChange={setLotSizeRange}
                  />
                </div>
                
                <div>
                  <RangeSlider
                    title="ROI (%)"
                    min={0}
                    max={30}
                    step={1}
                    value={roiRange}
                    onChange={setRoiRange}
                  />
                </div>
              </div>
            </motion.div>
          )}
          
          {/* Search summary */}
          <div className="flex justify-between items-center">
            <p className="text-gray-400">
              Viser <span className="text-white font-medium">{filteredProperties.length}</span> eiendommer
            </p>
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Sorter etter:</span>
              <select className="bg-white/5 text-white border border-white/10 rounded-lg px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="added">Nyeste først</option>
                <option value="roi">Høyest ROI</option>
                <option value="lotSize">Største tomter</option>
                <option value="buildingPotential">Størst byggepotensial</option>
              </select>
            </div>
          </div>
        </div>
        
        {/* Results grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filteredProperties.map(property => (
            <PropertyCard key={property.id} property={property} />
          ))}
        </div>
        
        {/* No results */}
        {filteredProperties.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <Building size={64} className="text-gray-700 mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Ingen eiendommer funnet</h3>
            <p className="text-gray-400 max-w-md">
              Prøv å justere søkekriteriene dine eller fjern noen filtre for å se flere resultater.
            </p>
            <button
              onClick={resetFilters}
              className="mt-6 px-6 py-2 bg-white/10 hover:bg-white/15 text-white rounded-lg transition-colors"
            >
              Nullstill alle filtre
            </button>
          </div>
        )}
      </main>
    </div>
  );
} 