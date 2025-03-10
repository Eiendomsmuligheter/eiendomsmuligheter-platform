import React from 'react';
import { motion } from 'framer-motion';
import { Check, X } from 'lucide-react';

const PricingTier = ({ 
  name, 
  price, 
  description, 
  features, 
  isPopular, 
  ctaText 
}) => {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className={`rounded-2xl overflow-hidden ${
        isPopular 
          ? 'bg-gradient-to-br from-blue-600 to-purple-700 border-0 shadow-lg shadow-blue-500/20' 
          : 'bg-gradient-to-br from-gray-900 to-black border border-white/10'
      }`}
    >
      {isPopular && (
        <div className="bg-white/20 py-1.5 text-center text-sm font-medium text-white">
          Mest populær
        </div>
      )}
      <div className="p-8">
        <h3 className="text-xl font-semibold text-white mb-1">{name}</h3>
        <p className="text-sm text-gray-300 mb-5">{description}</p>
        
        <div className="mb-5">
          <span className="text-4xl font-bold text-white">{price}</span>
          <span className="text-gray-300 ml-2">/mnd</span>
        </div>
        
        <ul className="space-y-3 mb-8">
          {features.map((feature, index) => (
            <li key={index} className="flex items-start gap-3">
              {feature.included ? (
                <Check className="h-5 w-5 text-green-400 mt-0.5 flex-shrink-0" />
              ) : (
                <X className="h-5 w-5 text-gray-500 mt-0.5 flex-shrink-0" />
              )}
              <span className={feature.included ? 'text-white' : 'text-gray-500'}>
                {feature.text}
              </span>
            </li>
          ))}
        </ul>
        
        <motion.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          className={`w-full py-3 rounded-lg font-medium transition-colors ${
            isPopular 
              ? 'bg-white text-purple-700 hover:bg-gray-100' 
              : 'bg-white/10 text-white hover:bg-white/20'
          }`}
        >
          {ctaText}
        </motion.button>
      </div>
    </motion.div>
  );
};

export const Pricing = () => {
  const tiers = [
    {
      name: "Starter",
      price: "499 kr",
      description: "For nye eiendomsutviklere og små prosjekter",
      isPopular: false,
      ctaText: "Kom i gang",
      features: [
        { text: "5 eiendomsanalyser per måned", included: true },
        { text: "Grunnleggende reguleringsanalyse", included: true },
        { text: "2D visualisering", included: true },
        { text: "ROI-beregning", included: true },
        { text: "3D visualisering", included: false },
        { text: "Bygningspotensialanalyse", included: false },
        { text: "Reguleringsvarsler", included: false },
        { text: "Eksport av rapporter", included: false },
        { text: "API-tilgang", included: false },
      ]
    },
    {
      name: "Professional",
      price: "999 kr",
      description: "For profesjonelle utviklere og prosjektselskaper",
      isPopular: true,
      ctaText: "Velg Professional",
      features: [
        { text: "20 eiendomsanalyser per måned", included: true },
        { text: "Avansert reguleringsanalyse", included: true },
        { text: "2D visualisering", included: true },
        { text: "ROI-beregning", included: true },
        { text: "3D visualisering", included: true },
        { text: "Bygningspotensialanalyse", included: true },
        { text: "Reguleringsvarsler", included: true },
        { text: "Eksport av rapporter", included: true },
        { text: "API-tilgang", included: false },
      ]
    },
    {
      name: "Enterprise",
      price: "2499 kr",
      description: "For store eiendomsutviklingsselskaper og investorer",
      isPopular: false,
      ctaText: "Kontakt oss",
      features: [
        { text: "Ubegrenset antall analyser", included: true },
        { text: "Premium reguleringsanalyse", included: true },
        { text: "2D visualisering", included: true },
        { text: "Avansert ROI-beregning", included: true },
        { text: "Fotorealistisk 3D visualisering", included: true },
        { text: "Bygningspotensialanalyse", included: true },
        { text: "Sanntids reguleringsvarsler", included: true },
        { text: "Eksport av rapporter", included: true },
        { text: "Ubegrenset API-tilgang", included: true },
      ]
    }
  ];
  
  return (
    <div className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600 mb-4"
          >
            Abonnementer for alle behov
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            viewport={{ once: true }}
            className="text-xl text-gray-400 max-w-3xl mx-auto"
          >
            Velg abonnementet som passer dine behov, og få tilgang til verdens beste eiendomsanalyse og visualisering.
          </motion.p>
        </div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-3 gap-8"
        >
          {tiers.map((tier, index) => (
            <PricingTier
              key={index}
              name={tier.name}
              price={tier.price}
              description={tier.description}
              features={tier.features}
              isPopular={tier.isPopular}
              ctaText={tier.ctaText}
            />
          ))}
        </motion.div>
      </div>
    </div>
  );
}; 