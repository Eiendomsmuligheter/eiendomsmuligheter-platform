import React from 'react';
import { motion } from 'framer-motion';
import { Building, LineChart, Shield, Eye, Award, Cpu } from 'lucide-react';

const FeatureCard = ({ icon, title, description }) => {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="bg-gradient-to-br from-gray-900 to-black p-6 rounded-xl border border-white/10 backdrop-blur-lg hover:border-blue-500/50 transition-all"
    >
      <div className="bg-blue-500/20 p-3 w-fit rounded-lg mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </motion.div>
  );
};

export const Features = () => {
  const features = [
    {
      icon: <Building className="w-6 h-6 text-blue-400" />,
      title: "Byggepotensialeanalyse",
      description: "Avdekk maksimalt byggepotensiale for enhver eiendom basert på lokale reguleringsbestemmelser."
    },
    {
      icon: <LineChart className="w-6 h-6 text-blue-400" />,
      title: "ROI-beregning",
      description: "Få nøyaktige estimater for avkastning på investeringen med våre avanserte økonomiske modeller."
    },
    {
      icon: <Shield className="w-6 h-6 text-blue-400" />,
      title: "Lovmessig Samsvar",
      description: "Automatisk sjekk mot gjeldende reguleringsplaner og byggeforskrifter."
    },
    {
      icon: <Eye className="w-6 h-6 text-blue-400" />,
      title: "3D Visualisering",
      description: "Se fremtidige byggeprosjekter med fotorealistiske 3D-modeller og terrengrendring."
    },
    {
      icon: <Award className="w-6 h-6 text-blue-400" />,
      title: "Bærekraftsanalyse",
      description: "Vurder miljøpåvirkning og energieffektivitet for mer bærekraftige prosjekter."
    },
    {
      icon: <Cpu className="w-6 h-6 text-blue-400" />,
      title: "AI-drevet Innsikt",
      description: "Utnytt kraften i kunstig intelligens for å identifisere muligheter som andre overser."
    }
  ];

  return (
    <div className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600 mb-4">
            Avanserte Funksjoner for Eiendomsutvikling
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Vår AI-plattform gir deg alle verktøyene du trenger for å analysere, visualisere og maksimere potensialet i enhver eiendom.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <FeatureCard
              key={index}
              icon={feature.icon}
              title={feature.title}
              description={feature.description}
            />
          ))}
        </div>
      </div>
    </div>
  );
}; 