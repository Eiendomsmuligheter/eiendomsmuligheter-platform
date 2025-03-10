import React, { useState, useEffect } from 'react';
import { Home, Filter, SlidersHorizontal, Search, MapPin, X } from 'lucide-react';
import { RentalCard } from './RentalCard';
import { RentalListingHeader } from './RentalListingHeader';
import { motion, AnimatePresence } from 'framer-motion';

// Avanserte filtreringsredskaper
const RentalFilters = ({ filters, setFilters, totalProperties }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeFilters, setActiveFilters] = useState(0);
  
  // Beregn antall aktive filtre
  useEffect(() => {
    let count = 0;
    if (filters.location) count++;
    if (filters.type) count++;
    if (filters.maxPrice) count++;
    if (filters.minBedrooms) count++;
    if (filters.petFriendly) count++;
    if (filters.furnished) count++;
    if (filters.balcony) count++;
    if (filters.parking) count++;
    setActiveFilters(count);
  }, [filters]);
  
  // Tilbakestill filtre
  const resetFilters = () => {
    setFilters({
      ...filters,
      location: '',
      type: '',
      maxPrice: '',
      minBedrooms: '',
      petFriendly: false,
      furnished: false,
      balcony: false,
      parking: false,
    });
  };
  
  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-6 mb-8 backdrop-blur-sm">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <Filter className="text-blue-400 mr-3" size={20} />
          <h2 className="text-xl font-semibold text-white">Filtrer leieobjekter</h2>
          {activeFilters > 0 && (
            <div className="ml-3 bg-blue-500 text-white text-xs font-medium h-5 min-w-5 rounded-full flex items-center justify-center px-1.5">
              {activeFilters}
            </div>
          )}
        </div>
        
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-gray-400 hover:text-white flex items-center text-sm transition-colors"
        >
          <SlidersHorizontal size={16} className="mr-1" />
          {isExpanded ? 'Skjul filtre' : 'Vis alle filtre'}
        </button>
      </div>
      
      <div className="relative mb-6">
        <input
          type="text"
          placeholder="Søk etter sted, område eller adresse..."
          className="w-full p-3 pl-10 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
          value={filters.searchTerm || ''}
          onChange={(e) => setFilters({...filters, searchTerm: e.target.value})}
        />
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
        {filters.searchTerm && (
          <button 
            onClick={() => setFilters({...filters, searchTerm: ''})}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
          >
            <X size={16} />
          </button>
        )}
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block text-gray-400 mb-2 text-sm">Område</label>
          <select 
            className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
            value={filters.location}
            onChange={(e) => setFilters({...filters, location: e.target.value})}
          >
            <option value="">Alle områder</option>
            <option value="Oslo">Oslo</option>
            <option value="Bergen">Bergen</option>
            <option value="Trondheim">Trondheim</option>
            <option value="Stavanger">Stavanger</option>
          </select>
        </div>
        
        <div>
          <label className="block text-gray-400 mb-2 text-sm">Type</label>
          <select 
            className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
            value={filters.type}
            onChange={(e) => setFilters({...filters, type: e.target.value})}
          >
            <option value="">Alle typer</option>
            <option value="Leilighet">Leilighet</option>
            <option value="Hus">Hus</option>
            <option value="Rekkehus">Rekkehus</option>
            <option value="Hybel">Hybel</option>
          </select>
        </div>
        
        <div>
          <label className="block text-gray-400 mb-2 text-sm">Maks pris (kr/mnd)</label>
          <input 
            type="number" 
            className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
            value={filters.maxPrice}
            onChange={(e) => setFilters({...filters, maxPrice: e.target.value})}
            placeholder="Maks pris"
          />
        </div>
        
        <div>
          <label className="block text-gray-400 mb-2 text-sm">Min. soverom</label>
          <input 
            type="number" 
            className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
            value={filters.minBedrooms}
            onChange={(e) => setFilters({...filters, minBedrooms: e.target.value})}
            placeholder="Min. antall soverom"
            min="0"
          />
        </div>
      </div>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="flex items-center">
                <input 
                  type="checkbox"
                  id="petFriendly"
                  checked={filters.petFriendly}
                  onChange={(e) => setFilters({...filters, petFriendly: e.target.checked})}
                  className="mr-2"
                />
                <label htmlFor="petFriendly" className="text-gray-300 text-sm">Dyrevennlig</label>
              </div>
              
              <div className="flex items-center">
                <input 
                  type="checkbox"
                  id="furnished"
                  checked={filters.furnished}
                  onChange={(e) => setFilters({...filters, furnished: e.target.checked})}
                  className="mr-2"
                />
                <label htmlFor="furnished" className="text-gray-300 text-sm">Møblert</label>
              </div>
              
              <div className="flex items-center">
                <input 
                  type="checkbox"
                  id="balcony"
                  checked={filters.balcony}
                  onChange={(e) => setFilters({...filters, balcony: e.target.checked})}
                  className="mr-2"
                />
                <label htmlFor="balcony" className="text-gray-300 text-sm">Balkong/terrasse</label>
              </div>
              
              <div className="flex items-center">
                <input 
                  type="checkbox"
                  id="parking"
                  checked={filters.parking}
                  onChange={(e) => setFilters({...filters, parking: e.target.checked})}
                  className="mr-2"
                />
                <label htmlFor="parking" className="text-gray-300 text-sm">Parkering</label>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Minimum areal (m²)</label>
                <input 
                  type="number" 
                  className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  value={filters.minSize || ''}
                  onChange={(e) => setFilters({...filters, minSize: e.target.value})}
                  placeholder="Min. areal"
                />
              </div>
              
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Tilgjengelig fra</label>
                <input 
                  type="date" 
                  className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  value={filters.availableFrom || ''}
                  onChange={(e) => setFilters({...filters, availableFrom: e.target.value})}
                />
              </div>
              
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Min. antall bad</label>
                <input 
                  type="number" 
                  className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  value={filters.minBathrooms || ''}
                  onChange={(e) => setFilters({...filters, minBathrooms: e.target.value})}
                  placeholder="Min. antall bad"
                  min="0"
                />
              </div>
              
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Kun verifiserte</label>
                <select 
                  className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  value={filters.verified || ''}
                  onChange={(e) => setFilters({...filters, verified: e.target.value})}
                >
                  <option value="">Alle</option>
                  <option value="true">Kun verifiserte</option>
                </select>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <div className="flex flex-col md:flex-row md:justify-between md:items-center space-y-4 md:space-y-0">
        <button 
          onClick={resetFilters}
          className={`px-4 py-2 rounded-lg text-white text-sm font-medium transition-colors ${
            activeFilters > 0 
              ? 'bg-white/10 hover:bg-white/15' 
              : 'bg-white/5 text-gray-500 cursor-not-allowed'
          }`}
          disabled={activeFilters === 0}
        >
          Tilbakestill filtre
        </button>
        
        <div className="flex items-center justify-between md:justify-end flex-1 md:flex-none md:ml-4">
          <span className="text-gray-400 text-sm mr-3 flex items-center whitespace-nowrap">
            <MapPin size={14} className="mr-1" />
            Viser {totalProperties} boliger
          </span>
          
          <div className="flex items-center">
            <label className="text-gray-300 text-sm mr-2 hidden md:inline">Sorter etter:</label>
            <select 
              className="p-2 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none text-sm"
              value={filters.sortBy}
              onChange={(e) => setFilters({...filters, sortBy: e.target.value})}
            >
              <option value="price_asc">Laveste pris</option>
              <option value="price_desc">Høyeste pris</option>
              <option value="date_desc">Nyeste først</option>
              <option value="size_desc">Størst areal</option>
              <option value="rating_desc">Høyest rangering</option>
              <option value="popularity">Mest populære</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
};

// Forbedret RentalListing komponent
export const RentalListing = () => {
  const [isLoading, setIsLoading] = useState(true);
  
  // Demo-data for leieobjekter med utvidede egenskaper
  const demoProperties = [
    {
      id: 1,
      title: "Moderne leilighet i sentrum",
      location: "Oslo, Grünerløkka",
      price: 15000,
      bedrooms: 2,
      bathrooms: 1,
      size: 65,
      type: "Leilighet",
      imageUrl: "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
      availableFrom: "01.08.2023",
      minimumRental: 12,
      petFriendly: true,
      furnished: true,
      balcony: true,
      parking: false,
      verified: true,
      rating: 4.8,
      views: 356,
      featured: true
    },
    {
      id: 2,
      title: "Romslig enebolig med hage",
      location: "Bergen, Fana",
      price: 22000,
      bedrooms: 4,
      bathrooms: 2,
      size: 120,
      type: "Hus",
      imageUrl: "https://images.unsplash.com/photo-1583608205776-bfd35f0d9f83?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
      availableFrom: "15.07.2023",
      minimumRental: 12,
      petFriendly: true,
      furnished: false,
      balcony: false,
      parking: true,
      verified: true,
      rating: 4.5,
      views: 210
    },
    {
      id: 3,
      title: "Koselig hybel nær universitetet",
      location: "Trondheim, Gløshaugen",
      price: 7500,
      bedrooms: 1,
      bathrooms: 1,
      size: 30,
      type: "Hybel",
      imageUrl: "https://images.unsplash.com/photo-1554995207-c18c203602cb?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
      availableFrom: "01.08.2023",
      minimumRental: 6,
      petFriendly: false,
      furnished: true,
      balcony: false,
      parking: false,
      rating: 3.9,
      views: 189
    },
    {
      id: 4,
      title: "Moderne rekkehus med garasje",
      location: "Stavanger, Hundvåg",
      price: 18000,
      bedrooms: 3,
      bathrooms: 2,
      size: 95,
      type: "Rekkehus",
      imageUrl: "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
      availableFrom: "01.09.2023",
      minimumRental: 12,
      petFriendly: true,
      furnished: false,
      balcony: true,
      parking: true,
      verified: true,
      views: 145
    },
    {
      id: 5,
      title: "Lys og trivelig leilighet",
      location: "Oslo, Majorstuen",
      price: 13500,
      bedrooms: 2,
      bathrooms: 1,
      size: 58,
      type: "Leilighet",
      imageUrl: "https://images.unsplash.com/photo-1493809842364-78817add7ffb?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
      availableFrom: "15.08.2023",
      minimumRental: 12,
      petFriendly: false,
      furnished: true,
      balcony: true,
      parking: false,
      rating: 4.2,
      views: 278
    },
    {
      id: 6,
      title: "Stort rekkehus med hage",
      location: "Bergen, Åsane",
      price: 19500,
      bedrooms: 3,
      bathrooms: 2,
      size: 110,
      type: "Rekkehus",
      imageUrl: "https://images.unsplash.com/photo-1574362848149-11496d93a7c7?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
      availableFrom: "01.08.2023",
      minimumRental: 12,
      petFriendly: true,
      furnished: false,
      balcony: true,
      parking: true,
      featured: true,
      views: 312
    }
  ];
  
  const [filters, setFilters] = useState({
    searchTerm: '',
    location: '',
    type: '',
    maxPrice: '',
    minBedrooms: '',
    petFriendly: false,
    furnished: false,
    balcony: false,
    parking: false,
    sortBy: 'price_asc',
    minSize: '',
    availableFrom: '',
    minBathrooms: '',
    verified: ''
  });
  
  // Simuler lasting
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Filtrer og sorter leieobjekter
  const filteredProperties = demoProperties
    .filter(property => {
      // Søkeord filter
      if (filters.searchTerm && !property.title.toLowerCase().includes(filters.searchTerm.toLowerCase()) && 
          !property.location.toLowerCase().includes(filters.searchTerm.toLowerCase())) {
        return false;
      }
      
      // Lokasjon filter
      if (filters.location && !property.location.includes(filters.location)) {
        return false;
      }
      
      // Type filter
      if (filters.type && property.type !== filters.type) {
        return false;
      }
      
      // Pris filter
      if (filters.maxPrice && property.price > parseInt(filters.maxPrice)) {
        return false;
      }
      
      // Soverom filter
      if (filters.minBedrooms && property.bedrooms < parseInt(filters.minBedrooms)) {
        return false;
      }
      
      // Ekstra filtre
      if (filters.petFriendly && !property.petFriendly) return false;
      if (filters.furnished && !property.furnished) return false;
      if (filters.balcony && !property.balcony) return false;
      if (filters.parking && !property.parking) return false;
      
      // Avanserte filtre
      if (filters.minSize && property.size < parseInt(filters.minSize)) return false;
      if (filters.minBathrooms && property.bathrooms < parseInt(filters.minBathrooms)) return false;
      if (filters.verified === 'true' && !property.verified) return false;
      
      return true;
    })
    .sort((a, b) => {
      switch (filters.sortBy) {
        case 'price_asc':
          return a.price - b.price;
        case 'price_desc':
          return b.price - a.price;
        case 'size_desc':
          return b.size - a.size;
        case 'date_desc':
          // For demo, vi sorterer basert på ID som en proxy for dato
          return b.id - a.id;
        case 'rating_desc':
          return (b.rating || 0) - (a.rating || 0);
        case 'popularity':
          return (b.views || 0) - (a.views || 0);
        default:
          return 0;
      }
    });
  
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-12">
        <div className="flex flex-col items-center justify-center py-20">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-xl font-medium text-white">Laster inn leieobjekter...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container mx-auto px-4 py-12">
      <RentalListingHeader />
      
      <RentalFilters 
        filters={filters} 
        setFilters={setFilters} 
        totalProperties={filteredProperties.length}
      />
      
      {filteredProperties.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredProperties.map((property, index) => (
            <RentalCard key={property.id} property={property} index={index} />
          ))}
        </div>
      ) : (
        <div className="bg-white/5 border border-white/10 rounded-xl p-12 text-center">
          <Home size={48} className="text-gray-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Ingen leieobjekter funnet</h2>
          <p className="text-gray-400 mb-6">
            Vi kunne ikke finne noen leieobjekter som passer dine kriterier. Prøv å justere filtrene dine.
          </p>
          <button 
            onClick={() => setFilters({
              ...filters,
              location: '',
              type: '',
              maxPrice: '',
              minBedrooms: '',
              petFriendly: false,
              furnished: false,
              balcony: false,
              parking: false,
              minSize: '',
              availableFrom: '',
              minBathrooms: '',
              verified: ''
            })}
            className="px-6 py-3 rounded-lg bg-white/10 hover:bg-white/15 text-white font-medium transition-colors"
          >
            Tilbakestill filtre
          </button>
        </div>
      )}
    </div>
  );
}; 