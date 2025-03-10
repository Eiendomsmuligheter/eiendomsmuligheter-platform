import React, { useState } from 'react';
import { Home, MapPin, BedDouble, Bath, Square, Heart, Share2, Calendar, Clock, Shield, Star, Eye } from 'lucide-react';
import { motion } from 'framer-motion';

interface RentalCardProps {
  property: {
    id: number;
    title: string;
    location: string;
    price: number;
    bedrooms: number;
    bathrooms: number;
    size: number;
    type: string;
    imageUrl: string;
    availableFrom: string;
    minimumRental: number;
    petFriendly?: boolean;
    furnished?: boolean;
    balcony?: boolean;
    parking?: boolean;
    verified?: boolean;
    rating?: number;
    views?: number;
    featured?: boolean;
  };
  index?: number;
}

export const RentalCard = ({ property, index = 0 }: RentalCardProps) => {
  const [isFavorite, setIsFavorite] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  
  // Animation variants
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.5,
        delay: index * 0.1,
        ease: "easeOut"
      }
    }
  };
  
  const handleShare = (e) => {
    e.stopPropagation();
    // Ville implementere deling via navigator.share API i en faktisk implementasjon
    alert(`Deler "${property.title}"`);
  };
  
  const handleFavoriteToggle = (e) => {
    e.stopPropagation();
    setIsFavorite(!isFavorite);
  };
  
  const handleCardClick = () => {
    // Ville navigere til detaljvisning i en faktisk implementasjon
    window.location.href = `/rentals/${property.id}`;
  };
  
  const formatPrice = (price) => {
    return new Intl.NumberFormat('no-NO').format(price);
  };
  
  return (
    <motion.div 
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      whileHover={{ y: -5 }}
      onClick={handleCardClick}
      className={`bg-white/5 border border-white/10 rounded-xl overflow-hidden hover:border-white/20 transition-all ${property.featured ? 'ring-2 ring-blue-500/50' : ''}`}
    >
      <div className="relative">
        {!imageLoaded && (
          <div className="h-48 w-full bg-gray-800 animate-pulse"></div>
        )}
        <img 
          src={property.imageUrl} 
          alt={property.title} 
          className={`h-48 w-full object-cover transition-opacity duration-300 ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
          onLoad={() => setImageLoaded(true)}
        />
        
        {property.featured && (
          <div className="absolute top-0 left-0 w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white text-xs font-medium py-1 px-3">
            Utvalgt bolig
          </div>
        )}
        
        <div className="absolute top-3 left-3 px-2 py-1 rounded-md bg-blue-500 text-xs font-medium text-white">
          {property.type}
        </div>
        
        <div className="absolute bottom-3 left-3 flex items-center">
          {property.verified && (
            <div className="flex items-center bg-green-900/70 text-green-300 text-xs font-medium px-2 py-1 rounded-md backdrop-blur-sm mr-2">
              <Shield size={12} className="mr-1" />
              Verifisert
            </div>
          )}
          
          {property.rating && (
            <div className="flex items-center bg-yellow-900/70 text-yellow-300 text-xs font-medium px-2 py-1 rounded-md backdrop-blur-sm">
              <Star size={12} className="mr-1" />
              {property.rating.toFixed(1)}
            </div>
          )}
        </div>
        
        <button 
          className={`absolute top-3 right-3 w-8 h-8 rounded-full flex items-center justify-center ${
            isFavorite ? 'bg-red-500 text-white' : 'bg-white/20 backdrop-blur-md text-white'
          }`}
          onClick={handleFavoriteToggle}
        >
          <Heart size={16} fill={isFavorite ? 'white' : 'none'} />
        </button>
      </div>
      
      <div className="p-4">
        <div className="flex items-start justify-between">
          <h3 className="text-lg font-semibold text-white">{property.title}</h3>
          <div className="text-lg font-semibold text-white">{formatPrice(property.price)} kr/mnd</div>
        </div>
        
        <div className="flex items-center text-gray-400 mt-1 mb-3">
          <MapPin size={14} className="mr-1" />
          <span className="text-sm">{property.location}</span>
        </div>
        
        <div className="grid grid-cols-3 gap-2 mb-4">
          <div className="flex flex-col items-center p-2 bg-white/5 rounded-lg">
            <BedDouble size={16} className="text-blue-400 mb-1" />
            <span className="text-sm text-gray-300">{property.bedrooms} sov</span>
          </div>
          <div className="flex flex-col items-center p-2 bg-white/5 rounded-lg">
            <Bath size={16} className="text-blue-400 mb-1" />
            <span className="text-sm text-gray-300">{property.bathrooms} bad</span>
          </div>
          <div className="flex flex-col items-center p-2 bg-white/5 rounded-lg">
            <Square size={16} className="text-blue-400 mb-1" />
            <span className="text-sm text-gray-300">{property.size} m²</span>
          </div>
        </div>
        
        {/* Egenskaper som tags */}
        <div className="flex flex-wrap gap-2 mb-4">
          {property.petFriendly && (
            <span className="bg-white/5 text-gray-300 text-xs px-2 py-1 rounded-md">
              Dyrevennlig
            </span>
          )}
          {property.furnished && (
            <span className="bg-white/5 text-gray-300 text-xs px-2 py-1 rounded-md">
              Møblert
            </span>
          )}
          {property.balcony && (
            <span className="bg-white/5 text-gray-300 text-xs px-2 py-1 rounded-md">
              Balkong
            </span>
          )}
          {property.parking && (
            <span className="bg-white/5 text-gray-300 text-xs px-2 py-1 rounded-md">
              Parkering
            </span>
          )}
        </div>
        
        <div className="flex items-center justify-between text-sm text-gray-400 mb-4">
          <div className="flex items-center">
            <Calendar size={14} className="mr-1" />
            <span>Tilgjengelig: {property.availableFrom}</span>
          </div>
          <div className="flex items-center">
            <Clock size={14} className="mr-1" />
            <span>Min. {property.minimumRental} mnd</span>
          </div>
        </div>
        
        <div className="flex gap-2 items-center">
          <button className="flex-1 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white text-sm font-medium hover:opacity-90 transition-opacity">
            Se detaljer
          </button>
          <button 
            onClick={handleShare}
            className="w-10 h-10 flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/15 transition-colors"
          >
            <Share2 size={18} className="text-white" />
          </button>
        </div>
        
        {property.views && (
          <div className="flex items-center justify-center text-xs text-gray-500 mt-3">
            <Eye size={12} className="mr-1" />
            <span>{property.views} visninger siste uke</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}; 