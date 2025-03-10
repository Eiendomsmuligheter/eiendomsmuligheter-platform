import React, { useState } from 'react';
import { Check, X, CreditCard, Building, Hammer, Home, User, ShieldCheck, Zap, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { LegalDisclaimer } from '../common/LegalDisclaimer';

// Typer brukergrupper
type UserType = 'personal' | 'professional' | 'partner';

// Abonnementsplankomponent
export const SubscriptionPlans = () => {
  const [selectedUserType, setSelectedUserType] = useState<UserType>('personal');
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'annually'>('monthly');
  
  // Rabatt for årlig betaling
  const annualDiscount = 0.20; // 20% rabatt
  
  // Planer for privatpersoner
  const personalPlans = [
    {
      name: 'Gratis',
      price: 0,
      features: [
        { text: 'Grunnleggende eiendomssøk', included: true },
        { text: 'Enkle karttjenester', included: true },
        { text: 'Begrenset AI-analyse', included: true },
        { text: 'Begrenset til 3 eiendomsanalyser per måned', included: true },
        { text: '3 favoritter', included: true },
        { text: 'Ingen tilgang til avanserte verktøy', included: false },
        { text: 'Ingen eksport av rapporter', included: false },
      ],
      ctaText: 'Kom i gang',
      popular: false,
    },
    {
      name: 'Standard',
      price: billingPeriod === 'monthly' ? 149 : Math.round(149 * 12 * (1 - annualDiscount)),
      features: [
        { text: 'Alt i Gratis-planen', included: true },
        { text: 'Ubegrenset antall eiendomsanalyser', included: true },
        { text: 'Full tilgang til kart og reguleringer', included: true },
        { text: 'Byggepotensialanalyse', included: true },
        { text: 'Ubegrenset antall favoritter', included: true },
        { text: 'Eksport av rapporter', included: true },
        { text: 'Prioritert kundeservice', included: false },
      ],
      ctaText: 'Velg Standard',
      popular: true,
    },
    {
      name: 'Premium',
      price: billingPeriod === 'monthly' ? 349 : Math.round(349 * 12 * (1 - annualDiscount)),
      features: [
        { text: 'Alt i Standard-planen', included: true },
        { text: 'Detaljert AI-analyse', included: true },
        { text: 'Lønnsomhetsberegning', included: true },
        { text: 'Finansieringssimulering', included: true },
        { text: 'Prioriterte tilbud fra partnere', included: true },
        { text: 'Eksport av avanserte rapporter', included: true },
        { text: 'Prioritert 24/7 kundeservice', included: true },
      ],
      ctaText: 'Velg Premium',
      popular: false,
    }
  ];
  
  // Planer for profesjonelle (meglere, arkitekter, etc.)
  const professionalPlans = [
    {
      name: 'Startpakke',
      price: billingPeriod === 'monthly' ? 599 : Math.round(599 * 12 * (1 - annualDiscount)),
      features: [
        { text: 'Grunnleggende eiendomsanalyse', included: true },
        { text: 'Opptil 20 eiendomsanalyser per måned', included: true },
        { text: 'Enkel eksport til PDF', included: true },
        { text: 'Begrenset API-tilgang', included: true },
        { text: '1 brukerkonto', included: true },
        { text: 'Deling med kunder', included: false },
        { text: 'Prioritert kundeservice', included: false },
      ],
      ctaText: 'Prøv Startpakken',
      popular: false,
    },
    {
      name: 'Business',
      price: billingPeriod === 'monthly' ? 1499 : Math.round(1499 * 12 * (1 - annualDiscount)),
      features: [
        { text: 'Ubegrenset antall eiendomsanalyser', included: true },
        { text: 'Avansert AI-analyse', included: true },
        { text: 'Kundeportal og deling', included: true },
        { text: 'Full API-tilgang', included: true },
        { text: 'Opptil 5 brukerkontoer', included: true },
        { text: 'Prioritert kundeservice', included: true },
        { text: 'White-label rapporter', included: false },
      ],
      ctaText: 'Velg Business',
      popular: true,
    },
    {
      name: 'Enterprise',
      price: billingPeriod === 'monthly' ? 2999 : Math.round(2999 * 12 * (1 - annualDiscount)),
      features: [
        { text: 'Alt i Business-planen', included: true },
        { text: 'Ubegrenset antall brukere', included: true },
        { text: 'White-label løsning', included: true },
        { text: 'Tilpassede rapporter', included: true },
        { text: 'Dedikert kundekontakt', included: true },
        { text: 'Integrasjon med CRM-systemer', included: true },
        { text: 'Prioritet på nye funksjoner', included: true },
      ],
      ctaText: 'Kontakt salg',
      popular: false,
    }
  ];
  
  // Planer for samarbeidspartnere (banker, håndverkere, etc.)
  const partnerPlans = [
    {
      name: 'Regional',
      price: billingPeriod === 'monthly' ? 4999 : Math.round(4999 * 12 * (1 - annualDiscount)),
      features: [
        { text: 'Partnerprofil i én region', included: true },
        { text: 'Leads i valgt region', included: true },
        { text: 'Inntil 25 leads per måned', included: true },
        { text: 'Standard plassering i søk', included: true },
        { text: 'Månedlig statistikk', included: true },
        { text: 'Partnermerke til eget nettsted', included: true },
        { text: 'Rådgivning for konvertering', included: false },
      ],
      ctaText: 'Bli regional partner',
      popular: false,
      icon: Building,
    },
    {
      name: 'Nasjonal',
      price: billingPeriod === 'monthly' ? 12999 : Math.round(12999 * 12 * (1 - annualDiscount)),
      features: [
        { text: 'Partnerprofil i hele Norge', included: true },
        { text: 'Ubegrenset antall leads', included: true },
        { text: 'Fremhevet plassering i søk', included: true },
        { text: 'Avansert statistikk og analyse', included: true },
        { text: 'Partnermerke til eget nettsted', included: true },
        { text: 'Dedikert partnerrådgiver', included: true },
        { text: 'Integrasjon med eget CRM-system', included: true },
      ],
      ctaText: 'Bli nasjonal partner',
      popular: true,
      icon: Building,
    },
    {
      name: 'Nordisk',
      price: billingPeriod === 'monthly' ? 29999 : Math.round(29999 * 12 * (1 - annualDiscount)),
      features: [
        { text: 'Partnerprofil i alle nordiske land', included: true },
        { text: 'Ubegrenset antall leads', included: true },
        { text: 'Toppplassering i søk', included: true },
        { text: 'Fullstendig statistikk og analyse', included: true },
        { text: 'Partnermerke til eget nettsted', included: true },
        { text: 'Dedikert strategisk rådgiver', included: true },
        { text: 'API-tilgang for full integrasjon', included: true },
      ],
      ctaText: 'Bli nordisk partner',
      popular: false,
      icon: Building,
    }
  ];
  
  // Velg aktive planer basert på brukertype
  const getActivePlans = () => {
    switch (selectedUserType) {
      case 'personal':
        return personalPlans;
      case 'professional':
        return professionalPlans;
      case 'partner':
        return partnerPlans;
      default:
        return personalPlans;
    }
  };
  
  // Kalkuler besparelse ved årlig betaling
  const calculateSaving = (monthlyPrice) => {
    const annualPrice = monthlyPrice * 12 * (1 - annualDiscount);
    return Math.round(monthlyPrice * 12 - annualPrice);
  };
  
  // Animasjon for planene
  const planContainerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const planVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };
  
  return (
    <div className="py-12 px-4 md:px-0">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">Velg riktig abonnement for dine behov</h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Vi tilbyr skreddersydde abonnementer for privatpersoner, profesjonelle og samarbeidspartnere.
            Finn den pakken som passer deg best.
          </p>
        </div>
        
        {/* Brukertypevalg */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          <button
            onClick={() => setSelectedUserType('personal')}
            className={`px-6 py-3 rounded-lg font-medium flex items-center gap-2 ${
              selectedUserType === 'personal'
                ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white' 
                : 'bg-white/10 text-gray-300 hover:bg-white/15'
            }`}
          >
            <User size={18} />
            Privatperson
          </button>
          <button
            onClick={() => setSelectedUserType('professional')}
            className={`px-6 py-3 rounded-lg font-medium flex items-center gap-2 ${
              selectedUserType === 'professional'
                ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white' 
                : 'bg-white/10 text-gray-300 hover:bg-white/15'
            }`}
          >
            <Home size={18} />
            Profesjonell
          </button>
          <button
            onClick={() => setSelectedUserType('partner')}
            className={`px-6 py-3 rounded-lg font-medium flex items-center gap-2 ${
              selectedUserType === 'partner'
                ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white' 
                : 'bg-white/10 text-gray-300 hover:bg-white/15'
            }`}
          >
            <Building size={18} />
            Partner
          </button>
        </div>
        
        {/* Valg av faktureringsperiode */}
        <div className="flex justify-center mb-12">
          <div className="bg-white/5 border border-white/10 rounded-full p-1 flex">
            <button
              onClick={() => setBillingPeriod('monthly')}
              className={`px-5 py-2 rounded-full text-sm font-medium ${
                billingPeriod === 'monthly'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                  : 'text-gray-300'
              }`}
            >
              Månedlig
            </button>
            <button
              onClick={() => setBillingPeriod('annually')}
              className={`px-5 py-2 rounded-full text-sm font-medium flex items-center ${
                billingPeriod === 'annually'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                  : 'text-gray-300'
              }`}
            >
              Årlig <span className="ml-1 text-xs bg-green-500 text-white px-2 py-0.5 rounded-full">Spar 20%</span>
            </button>
          </div>
        </div>
        
        <LegalDisclaimer type="general" compact={true} />
        
        {/* Abonnementsplaner */}
        <motion.div 
          variants={planContainerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-3 gap-8"
        >
          {getActivePlans().map((plan, index) => (
            <motion.div 
              key={plan.name}
              variants={planVariants}
              className={`bg-white/5 border rounded-2xl overflow-hidden ${
                plan.popular 
                  ? 'border-blue-500 relative transform md:scale-105 z-10' 
                  : 'border-white/10'
              }`}
            >
              {plan.popular && (
                <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white text-center py-1 text-sm font-medium">
                  Mest populær
                </div>
              )}
              
              <div className="p-6">
                <div className="flex items-center mb-4">
                  {selectedUserType === 'partner' && plan.icon && (
                    <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500/20 to-purple-600/20 flex items-center justify-center mr-3">
                      <plan.icon size={18} className="text-blue-400" />
                    </div>
                  )}
                  <h3 className="text-2xl font-bold text-white">{plan.name}</h3>
                </div>
                
                <div className="mb-6">
                  <div className="flex items-end">
                    <span className="text-4xl font-bold text-white">{plan.price === 0 ? 'Gratis' : `${plan.price} kr`}</span>
                    {plan.price > 0 && (
                      <span className="text-gray-400 ml-2 pb-1">/{billingPeriod === 'monthly' ? 'mnd' : 'år'}</span>
                    )}
                  </div>
                  {billingPeriod === 'annually' && plan.price > 0 && (
                    <p className="text-green-400 text-sm mt-1">
                      Spar {calculateSaving(plan.price / 12 / (1 - annualDiscount))} kr per år
                    </p>
                  )}
                </div>
                
                <ul className="space-y-3 mb-8">
                  {plan.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-start">
                      {feature.included ? (
                        <Check size={18} className="text-green-400 mt-0.5 mr-2 flex-shrink-0" />
                      ) : (
                        <X size={18} className="text-gray-600 mt-0.5 mr-2 flex-shrink-0" />
                      )}
                      <span className={feature.included ? 'text-gray-300' : 'text-gray-500'}>
                        {feature.text}
                      </span>
                    </li>
                  ))}
                </ul>
                
                <button
                  className={`w-full py-3 px-4 rounded-lg font-medium flex items-center justify-center gap-2 ${
                    plan.popular
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white' 
                      : 'bg-white/10 text-white hover:bg-white/15'
                  }`}
                >
                  {plan.ctaText}
                  <ArrowRight size={18} />
                </button>
              </div>
            </motion.div>
          ))}
        </motion.div>
        
        {/* Enterprise-seksjon */}
        <div className="mt-20 bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-500/20 rounded-xl p-8">
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="mb-6 md:mb-0">
              <h3 className="text-2xl font-bold text-white mb-2">Trenger du en skreddersydd løsning?</h3>
              <p className="text-gray-300 max-w-2xl">
                Vi tilbyr tilpassede bedriftsløsninger for større organisasjoner. Kontakt oss for et
                personlig tilbud basert på dine unike behov.
              </p>
            </div>
            <button className="px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium flex items-center gap-2 whitespace-nowrap">
              <Zap size={18} />
              Kontakt salgsavdelingen
            </button>
          </div>
        </div>
        
        {/* Vanlige spørsmål */}
        <div className="mt-20">
          <h2 className="text-3xl font-bold text-white text-center mb-12">Vanlige spørsmål</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-2">Hvordan funker betalingen?</h3>
              <p className="text-gray-300">
                Vi tilbyr sikker betaling via kort eller faktura. Ved månedlig abonnement belastes du automatisk 
                hver måned, mens ved årlig abonnement belastes du én gang i året. Alle priser er oppgitt 
                inkludert mva.
              </p>
            </div>
            
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-2">Kan jeg bytte eller avbestille abonnement?</h3>
              <p className="text-gray-300">
                Ja, du kan oppgradere, nedgradere eller avbestille abonnementet ditt når som helst. 
                Endringer trer i kraft ved starten av neste fakturaperiode. Ingen bindingstid.
              </p>
            </div>
            
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-2">Får jeg refusjon hvis jeg avbestiller?</h3>
              <p className="text-gray-300">
                For årlige abonnementer tilbyr vi en forholdsmessig refusjon hvis du avbestiller før 
                abonnementsperioden er over. Månedlige abonnementer refunderes ikke, men tjenesten vil 
                fortsatt være aktiv ut perioden.
              </p>
            </div>
            
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-2">Hva skjer med mine data hvis jeg avslutter?</h3>
              <p className="text-gray-300">
                Dine personlige data og analyser er tilgjengelige i 30 dager etter at abonnementet ditt 
                utløper. Etter dette vil dataene bli arkivert, men kan gjenopprettes hvis du reaktiverer 
                abonnementet innen 90 dager.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 