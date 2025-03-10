import React, { useState, useEffect } from 'react';
import { Building, Hammer, Home, ShieldCheck, ChevronRight, Users, CreditCard, Check, ShoppingBag, TrendingUp, BarChart3, DollarSign, Globe, Lock } from 'lucide-react';
import { BankRegistration } from './BankRegistration';
import { ContractorRegistration } from './ContractorRegistration';
import { LegalDisclaimer } from '../common/LegalDisclaimer';
import { motion, AnimatePresence } from 'framer-motion';

// Kort-komponent for partnertyper med forbedret animasjon
const PartnerTypeCard = ({ title, description, icon: Icon, stats, onClick, active }) => {
  return (
    <motion.div 
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`p-6 rounded-xl border cursor-pointer transition-all ${
        active ? 'border-blue-500 bg-blue-900/20' : 'border-white/10 bg-white/5 hover:bg-white/10'
      }`}
      onClick={onClick}
    >
      <div className="flex items-center mb-4">
        <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
          <Icon className="w-6 h-6 text-white" />
        </div>
        <h3 className="text-xl font-semibold text-white ml-4">{title}</h3>
      </div>
      <p className="text-gray-300 mb-4">{description}</p>
      
      {/* Statistikk for kortene */}
      {stats && (
        <div className="grid grid-cols-2 gap-2 mb-4">
          {stats.map((stat, index) => (
            <div key={index} className="bg-white/5 rounded-lg p-2 text-center">
              <p className="text-sm text-gray-400">{stat.label}</p>
              <p className="text-lg font-semibold text-white">{stat.value}</p>
            </div>
          ))}
        </div>
      )}
      
      <div className="flex justify-between items-center">
        <div className="text-sm font-medium text-blue-400">Les mer</div>
        <ChevronRight className="w-5 h-5 text-blue-400" />
      </div>
    </motion.div>
  );
};

// Forbedret statistikk-kort med trendindikator
const StatsCard = ({ title, value, icon: Icon, color, trend = null }) => {
  return (
    <motion.div 
      whileHover={{ y: -5 }}
      className="p-6 rounded-xl border border-white/10 bg-white/5"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-300">{title}</h3>
        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${color}`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
      </div>
      <div className="flex justify-between items-end">
        <p className="text-3xl font-bold text-white">{value}</p>
        {trend && (
          <div className={`flex items-center ${trend.type === 'up' ? 'text-green-400' : 'text-red-400'}`}>
            {trend.type === 'up' ? (
              <TrendingUp size={16} className="mr-1" />
            ) : (
              <TrendingUp size={16} className="mr-1 transform rotate-180" />
            )}
            <span className="text-sm font-medium">{trend.value}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Komponent for inntektskilder og fordeling
const RevenueSources = ({ data }) => {
  const total = data.reduce((acc, item) => acc + item.value, 0);
  
  return (
    <div className="p-6 bg-white/5 border border-white/10 rounded-xl">
      <h3 className="text-xl font-semibold text-white mb-6">Inntektskilder</h3>
      <div className="space-y-4">
        {data.map((item, index) => {
          const percentage = Math.round((item.value / total) * 100);
          return (
            <div key={index}>
              <div className="flex justify-between items-center mb-1">
                <span className="text-gray-300 flex items-center">
                  <item.icon size={16} className="mr-2 text-gray-400" />
                  {item.label}
                </span>
                <span className="text-white font-medium">{item.value.toLocaleString()} NOK</span>
              </div>
              <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full ${item.color}`}
                  style={{ width: `${percentage}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>{percentage}% av total</span>
                <span>{item.growth > 0 ? `+${item.growth}%` : `${item.growth}%`} mot forrige måned</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// AI-drevne anbefalinger for plattformforbedringer
const AIRecommendations = () => {
  return (
    <div className="p-6 bg-white/5 border border-white/10 rounded-xl">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-500 to-indigo-600 flex items-center justify-center mr-3">
          <BarChart3 size={20} className="text-white" />
        </div>
        <h3 className="text-xl font-semibold text-white">AI-drevne anbefalinger</h3>
      </div>
      <div className="space-y-4">
        <div className="p-4 bg-purple-900/20 border border-purple-500/20 rounded-lg">
          <h4 className="font-medium text-white mb-2">Utvid bank-partnerprogrammet</h4>
          <p className="text-gray-300 text-sm">
            Basert på brukerdata ser vi at 72% av eiendomssøk resulterer i finansieringsspørsmål. 
            Ved å legge til 3-5 nye bankpartnere kan du øke konverteringer med opptil 28%.
          </p>
        </div>
        <div className="p-4 bg-blue-900/20 border border-blue-500/20 rounded-lg">
          <h4 className="font-medium text-white mb-2">Legg til prissammenligning for håndverkere</h4>
          <p className="text-gray-300 text-sm">
            Brukere som sammenligner håndverkertjenester har 3.2x høyere konverteringsrate. 
            Implementering av prissammenligning kan øke partnerrevenuen med 40-55%.
          </p>
        </div>
        <div className="p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
          <h4 className="font-medium text-white mb-2">Automatiser partnergodkjenning</h4>
          <p className="text-gray-300 text-sm">
            Nåværende godkjenningsprosess tar 3.4 dager i gjennomsnitt. Implementering av automatisert 
            verifisering kan redusere dette til 2 timer og øke partnertilfredshet med 64%.
          </p>
        </div>
      </div>
    </div>
  );
};

export const PartnerDashboard = () => {
  const [selectedPartnerType, setSelectedPartnerType] = useState('');
  const [viewStats, setViewStats] = useState(false);
  const [disclaimerType, setDisclaimerType] = useState('general');
  const [isLoading, setIsLoading] = useState(true);

  // Simuler lasteprosess
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  // Når partner-typen endres, oppdaterer vi ansvarsfraskrivelsestypen
  useEffect(() => {
    switch (selectedPartnerType) {
      case 'bank':
        setDisclaimerType('financial');
        break;
      case 'contractor':
        setDisclaimerType('contractor');
        break;
      case 'insurance':
        setDisclaimerType('insurance');
        break;
      case 'realEstate':
        setDisclaimerType('rental');
        break;
      default:
        setDisclaimerType('general');
    }
  }, [selectedPartnerType]);

  // Statistikk-data (placeholder)
  const statsData = {
    banks: 18,
    contractors: 132,
    insuranceCompanies: 9,
    retailers: 27,
    totalPartners: 235,
    monthlyRevenue: '1.2M NOK',
    activeLeads: 478,
    userGrowth: 32
  };

  // Data for inntektskilder
  const revenueSourcesData = [
    { 
      label: 'Bankpartnere', 
      value: 520000, 
      icon: Building, 
      color: 'bg-blue-500', 
      growth: 12 
    },
    { 
      label: 'Håndverkere', 
      value: 350000, 
      icon: Hammer, 
      color: 'bg-orange-500', 
      growth: 8 
    },
    { 
      label: 'Forsikring', 
      value: 180000, 
      icon: ShieldCheck, 
      color: 'bg-green-500', 
      growth: 15 
    },
    { 
      label: 'Materialleverandører', 
      value: 150000, 
      icon: ShoppingBag, 
      color: 'bg-purple-500', 
      growth: -3 
    },
  ];

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-xl font-medium text-white">Laster inn partner-dashboard...</p>
        </div>
      );
    }

    // Vis statistikk i stedet for registreringsskjemaer
    if (viewStats) {
      return (
        <div>
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold text-white">Partneroversikt</h2>
            <button 
              onClick={() => setViewStats(false)}
              className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm font-medium transition-colors"
            >
              Tilbake til registrering
            </button>
          </div>
          
          <LegalDisclaimer type="general" compact={true} />
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <StatsCard 
              title="Totalt antall partnere" 
              value={statsData.totalPartners} 
              icon={Users} 
              color="bg-blue-500"
              trend={{ type: 'up', value: '12%' }}
            />
            <StatsCard 
              title="Månedlige inntekter" 
              value={statsData.monthlyRevenue} 
              icon={CreditCard} 
              color="bg-green-500"
              trend={{ type: 'up', value: '8%' }}
            />
            <StatsCard 
              title="Aktive leads" 
              value={statsData.activeLeads} 
              icon={ChevronRight} 
              color="bg-purple-500"
              trend={{ type: 'up', value: '24%' }}
            />
            <StatsCard 
              title="Brukervekst" 
              value={`${statsData.userGrowth}%`} 
              icon={TrendingUp} 
              color="bg-indigo-500"
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <RevenueSources data={revenueSourcesData} />
            <AIRecommendations />
          </div>
          
          <h3 className="text-xl font-semibold text-white mb-4">Partnertyper</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <StatsCard 
              title="Banker" 
              value={statsData.banks} 
              icon={Building} 
              color="bg-blue-500"
            />
            <StatsCard 
              title="Håndverkere" 
              value={statsData.contractors} 
              icon={Hammer} 
              color="bg-orange-500"
            />
            <StatsCard 
              title="Forsikringsselskaper" 
              value={statsData.insuranceCompanies} 
              icon={ShieldCheck} 
              color="bg-green-500"
            />
            <StatsCard 
              title="Materialleverandører" 
              value={statsData.retailers} 
              icon={ShoppingBag} 
              color="bg-purple-500"
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="p-6 bg-white/5 border border-white/10 rounded-xl col-span-1">
              <h3 className="text-xl font-semibold text-white mb-4">Geografisk fordeling</h3>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="flex flex-col items-center p-4 bg-white/5 rounded-lg">
                  <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center mb-2">
                    <Globe className="w-6 h-6 text-blue-400" />
                  </div>
                  <p className="text-sm text-gray-400">Topp region</p>
                  <p className="text-lg font-semibold text-white">Oslo</p>
                  <p className="text-xs text-gray-500">32% av alle partnere</p>
                </div>
                <div className="flex flex-col items-center p-4 bg-white/5 rounded-lg">
                  <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center mb-2">
                    <TrendingUp className="w-6 h-6 text-green-400" />
                  </div>
                  <p className="text-sm text-gray-400">Sterkest vekst</p>
                  <p className="text-lg font-semibold text-white">Trondheim</p>
                  <p className="text-xs text-gray-500">+47% siste kvartal</p>
                </div>
              </div>
              <div className="h-48 bg-white/5 rounded-lg p-4 flex items-center justify-center">
                <p className="text-gray-400">Interaktivt kart kommer snart</p>
              </div>
            </div>
            
            <div className="p-6 bg-white/5 border border-white/10 rounded-xl col-span-1">
              <h3 className="text-xl font-semibold text-white mb-4">Kommende partnere</h3>
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="pb-3 text-gray-400 font-medium">Firma</th>
                    <th className="pb-3 text-gray-400 font-medium">Type</th>
                    <th className="pb-3 text-gray-400 font-medium">Status</th>
                    <th className="pb-3 text-gray-400 font-medium">Dato</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-white/5">
                    <td className="py-4 text-white">XYZ Eiendomsinvestering AS</td>
                    <td className="py-4 text-gray-300">Bank</td>
                    <td className="py-4">
                      <span className="px-2 py-1 rounded-full bg-yellow-500/20 text-yellow-300 text-xs font-medium">
                        Venter på godkjenning
                      </span>
                    </td>
                    <td className="py-4 text-gray-300">15. juni 2023</td>
                  </tr>
                  <tr className="border-b border-white/5">
                    <td className="py-4 text-white">Norske Rørleggere AS</td>
                    <td className="py-4 text-gray-300">Håndverker</td>
                    <td className="py-4">
                      <span className="px-2 py-1 rounded-full bg-green-500/20 text-green-300 text-xs font-medium">
                        Godkjent
                      </span>
                    </td>
                    <td className="py-4 text-gray-300">12. juni 2023</td>
                  </tr>
                  <tr>
                    <td className="py-4 text-white">Trygg Fremtid Forsikring</td>
                    <td className="py-4 text-gray-300">Forsikring</td>
                    <td className="py-4">
                      <span className="px-2 py-1 rounded-full bg-red-500/20 text-red-300 text-xs font-medium">
                        Mer info nødvendig
                      </span>
                    </td>
                    <td className="py-4 text-gray-300">10. juni 2023</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          
          <div className="p-6 bg-gradient-to-r from-blue-900/20 to-purple-900/20 border border-blue-500/20 rounded-xl">
            <div className="flex items-center mb-4">
              <Lock className="w-6 h-6 text-blue-400 mr-3" />
              <h3 className="text-xl font-semibold text-white">Premium-funksjoner</h3>
            </div>
            <p className="text-gray-300 mb-4">
              Få tilgang til avanserte analyser, automatiserte partnergodkjenninger, og AI-drevne 
              markedsføringsverktøy med vår premium-pakke for plattformeiere.
            </p>
            <button className="px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium">
              Utforsk premium-funksjoner
            </button>
          </div>
        </div>
      );
    }

    // Vis det valgte registreringsskjemaet
    switch (selectedPartnerType) {
      case 'bank':
        return (
          <>
            <LegalDisclaimer type="financial" />
            <BankRegistration />
          </>
        );
      case 'contractor':
        return (
          <>
            <LegalDisclaimer type="contractor" />
            <ContractorRegistration />
          </>
        );
      case 'insurance':
        return (
          <>
            <LegalDisclaimer type="insurance" />
            <div className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 p-8 max-w-4xl mx-auto">
              <div className="flex items-center mb-8">
                <ShieldCheck className="w-8 h-8 text-blue-400 mr-3" />
                <h1 className="text-3xl font-bold text-white">Bli forsikringspartner</h1>
              </div>
              <div className="p-8 text-center">
                <p className="text-xl text-gray-300 mb-6">
                  Registrering for forsikringsselskaper kommer snart!
                </p>
                <button 
                  onClick={() => setSelectedPartnerType('')}
                  className="px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium"
                >
                  Gå tilbake
                </button>
              </div>
            </div>
          </>
        );
      case 'realEstate':
        return (
          <>
            <LegalDisclaimer type="rental" />
            <div className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 p-8 max-w-4xl mx-auto">
              <div className="flex items-center mb-8">
                <Home className="w-8 h-8 text-blue-400 mr-3" />
                <h1 className="text-3xl font-bold text-white">Bli eiendomsmegler-partner</h1>
              </div>
              <div className="p-8 text-center">
                <p className="text-xl text-gray-300 mb-6">
                  Registrering for eiendomsmeglere kommer snart!
                </p>
                <button 
                  onClick={() => setSelectedPartnerType('')}
                  className="px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium"
                >
                  Gå tilbake
                </button>
              </div>
            </div>
          </>
        );
      case 'retail':
        return (
          <>
            <LegalDisclaimer type="general" />
            <div className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 p-8 max-w-4xl mx-auto">
              <div className="flex items-center mb-8">
                <ShoppingBag className="w-8 h-8 text-blue-400 mr-3" />
                <h1 className="text-3xl font-bold text-white">Bli materialleverandør-partner</h1>
              </div>
              <div className="p-8 text-center">
                <p className="text-xl text-gray-300 mb-6">
                  Registrering for materialleverandører kommer snart!
                </p>
                <button 
                  onClick={() => setSelectedPartnerType('')}
                  className="px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium"
                >
                  Gå tilbake
                </button>
              </div>
            </div>
          </>
        );
      default:
        // Vis oversikt over partnertyper man kan registrere seg som
        return (
          <div>
            <div className="flex justify-between items-center mb-8">
              <h1 className="text-3xl font-bold text-white">Bli partner med Eiendomsmuligheter</h1>
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setViewStats(true)}
                className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm font-medium transition-colors"
              >
                Vis statistikk
              </motion.button>
            </div>
            
            <LegalDisclaimer type="general" />
            
            <p className="text-xl text-gray-300 mb-8">
              Velg partnertype og registrer din bedrift for å nå potensielle kunder gjennom Eiendomsmuligheter.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
              <PartnerTypeCard 
                title="Bank / Finansinstitusjon" 
                description="Tilby boliglån og finansieringsløsninger til eiendomskjøpere og renoveringsprosjekter."
                icon={Building}
                stats={[
                  { label: 'Aktive partnere', value: '18' },
                  { label: 'Gjennomsnittlig ROI', value: '438%' }
                ]}
                onClick={() => setSelectedPartnerType('bank')}
                active={selectedPartnerType === 'bank'}
              />
              
              <PartnerTypeCard 
                title="Håndverker" 
                description="Presenter dine tjenester og få oppdrag innen renovering, nybygg og vedlikehold av eiendommer."
                icon={Hammer}
                stats={[
                  { label: 'Aktive partnere', value: '132' },
                  { label: 'Leads per måned', value: '850+' }
                ]}
                onClick={() => setSelectedPartnerType('contractor')}
                active={selectedPartnerType === 'contractor'}
              />
              
              <PartnerTypeCard 
                title="Forsikringsselskap" 
                description="Tilby eiendomsforsikring, innboforsikring og byggforsikring til eiendomseiere."
                icon={ShieldCheck}
                stats={[
                  { label: 'Aktive partnere', value: '9' },
                  { label: 'Konverteringsrate', value: '18.4%' }
                ]}
                onClick={() => setSelectedPartnerType('insurance')}
                active={selectedPartnerType === 'insurance'}
              />
              
              <PartnerTypeCard 
                title="Eiendomsmegler" 
                description="Få tilgang til potensielle selgere og kjøpere gjennom vårt nettverk."
                icon={Home}
                stats={[
                  { label: 'Aktive partnere', value: '24' },
                  { label: 'Salg per partner', value: '5.2/mnd' }
                ]}
                onClick={() => setSelectedPartnerType('realEstate')}
                active={selectedPartnerType === 'realEstate'}
              />
              
              <PartnerTypeCard 
                title="Materialleverandør" 
                description="Selg byggematerialer, interiør og annet utstyr direkte til eiendomsutviklere."
                icon={ShoppingBag}
                stats={[
                  { label: 'Aktive partnere', value: '27' },
                  { label: 'Ordre per måned', value: '430+' }
                ]}
                onClick={() => setSelectedPartnerType('retail')}
                active={selectedPartnerType === 'retail'}
              />
              
              <div className="p-6 rounded-xl border border-dashed border-white/20 bg-white/5 flex flex-col items-center justify-center text-center">
                <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center mb-4">
                  <DollarSign className="w-6 h-6 text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">Annen partner?</h3>
                <p className="text-gray-400 mb-4">
                  Er du interessert i å bli partner, men passer ikke i kategoriene over?
                </p>
                <button className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm font-medium transition-colors">
                  Kontakt oss
                </button>
              </div>
            </div>
            
            <div className="p-6 bg-white/5 border border-white/10 rounded-xl">
              <h2 className="text-2xl font-bold text-white mb-4">Fordeler ved å bli partner</h2>
              <ul className="space-y-4">
                <motion.li 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 }}
                  className="flex items-start"
                >
                  <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center mt-1 mr-3">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Nå potensielle kunder</h3>
                    <p className="text-gray-300">
                      Få tilgang til kvalifiserte leads som er aktive i eiendomsmarkedet.
                    </p>
                  </div>
                </motion.li>
                <motion.li 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                  className="flex items-start"
                >
                  <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center mt-1 mr-3">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Øk din synlighet</h3>
                    <p className="text-gray-300">
                      Bli presentert for nye eiendomskjøpere, selgere og renovatører.
                    </p>
                  </div>
                </motion.li>
                <motion.li 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                  className="flex items-start"
                >
                  <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center mt-1 mr-3">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Vekst og innsikt</h3>
                    <p className="text-gray-300">
                      Få tilgang til markedsdata og trender for å optimalisere din forretning.
                    </p>
                  </div>
                </motion.li>
                <motion.li 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                  className="flex items-start"
                >
                  <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center mt-1 mr-3">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Markedsføringsverktøy</h3>
                    <p className="text-gray-300">
                      Bruk våre AI-drevne markedsføringsverktøy for å målrette dine tilbud basert på kundedata.
                    </p>
                  </div>
                </motion.li>
              </ul>
            </div>
          </div>
        );
    }
  };

  return (
    <AnimatePresence>
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="container mx-auto px-4 py-12"
      >
        {renderContent()}
      </motion.div>
    </AnimatePresence>
  );
}; 