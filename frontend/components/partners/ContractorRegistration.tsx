import React, { useState } from 'react';
import { Hammer, FileText, Check, ChevronRight } from 'lucide-react';

// Gjenbrukbar stepper-komponent
const Stepper = ({ steps, currentStep }) => {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <div key={index} className="flex items-center">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${
              index < currentStep 
                ? 'bg-blue-500 border-blue-500 text-white' 
                : index === currentStep 
                  ? 'border-blue-500 text-blue-500' 
                  : 'border-white/20 text-gray-400'
            }`}>
              {index < currentStep ? <Check size={18} /> : index + 1}
            </div>
            {index < steps.length - 1 && (
              <div className={`h-1 w-16 sm:w-32 ${
                index < currentStep ? 'bg-blue-500' : 'bg-white/10'
              }`}></div>
            )}
          </div>
        ))}
      </div>
      <div className="flex justify-between mt-2">
        {steps.map((step, index) => (
          <div key={index} className={`text-sm ${
            index === currentStep ? 'text-blue-400' : 'text-gray-400'
          }`}>
            {step}
          </div>
        ))}
      </div>
    </div>
  );
};

export const ContractorRegistration = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState({
    company: {
      name: '',
      orgNumber: '',
      address: '',
      postalCode: '',
      city: '',
      website: '',
      logo: null
    },
    contact: {
      firstName: '',
      lastName: '',
      email: '',
      phone: '',
      position: ''
    },
    services: {
      categories: [],
      description: '',
      priceRange: 'medium',
      projectMinSize: '',
      operatingAreas: []
    },
    subscription: {
      plan: 'regional',
      duration: '6',
      paymentMethod: 'invoice'
    },
    terms: {
      acceptTerms: false,
      acceptPrivacyPolicy: false,
      acceptPartnerAgreement: false
    }
  });

  const steps = [
    "Firmainfo",
    "Kontaktperson",
    "Tjenester",
    "Abonnement",
    "Avtalevilkår"
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = () => {
    console.log('Registrering sendt:', formData);
    alert('Takk for registreringen! Vi vil kontakte deg snart for å bekrefte partnerskapet.');
  };

  const handleChange = (section, field, value) => {
    setFormData({
      ...formData,
      [section]: {
        ...formData[section],
        [field]: value
      }
    });
  };

  // Kontrollere kategorivalg med sjekkbokser
  const handleCategoryChange = (category) => {
    const updatedCategories = [...formData.services.categories];
    if (updatedCategories.includes(category)) {
      const index = updatedCategories.indexOf(category);
      updatedCategories.splice(index, 1);
    } else {
      updatedCategories.push(category);
    }
    handleChange('services', 'categories', updatedCategories);
  };

  // Områder håndverkeren opererer i
  const handleAreaChange = (area) => {
    const updatedAreas = [...formData.services.operatingAreas];
    if (updatedAreas.includes(area)) {
      const index = updatedAreas.indexOf(area);
      updatedAreas.splice(index, 1);
    } else {
      updatedAreas.push(area);
    }
    handleChange('services', 'operatingAreas', updatedAreas);
  };

  const renderStep = () => {
    switch (currentStep) {
      case 0: // Firmainfo
        return (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-white mb-6">Firmainformasjon</h2>
            
            <div>
              <label className="block text-gray-400 mb-2">Firmanavn</label>
              <input
                type="text"
                value={formData.company.name}
                onChange={(e) => handleChange('company', 'name', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="F.eks. Hansen & Sønn AS"
              />
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Organisasjonsnummer</label>
              <input
                type="text"
                value={formData.company.orgNumber}
                onChange={(e) => handleChange('company', 'orgNumber', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="9 siffer"
              />
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Adresse</label>
              <input
                type="text"
                value={formData.company.address}
                onChange={(e) => handleChange('company', 'address', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="Gateadresse"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-400 mb-2">Postnummer</label>
                <input
                  type="text"
                  value={formData.company.postalCode}
                  onChange={(e) => handleChange('company', 'postalCode', e.target.value)}
                  className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  placeholder="Postnr."
                />
              </div>
              <div>
                <label className="block text-gray-400 mb-2">Sted</label>
                <input
                  type="text"
                  value={formData.company.city}
                  onChange={(e) => handleChange('company', 'city', e.target.value)}
                  className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  placeholder="Sted"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Nettside</label>
              <input
                type="text"
                value={formData.company.website}
                onChange={(e) => handleChange('company', 'website', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="https://"
              />
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Last opp logo</label>
              <div className="border-2 border-dashed border-white/10 rounded-lg p-8 text-center hover:border-blue-500/50 transition-colors cursor-pointer">
                <Hammer className="w-12 h-12 text-blue-500 mx-auto mb-4" />
                <p className="text-gray-400">Klikk for å laste opp logo (PNG, JPG, SVG)</p>
                <p className="text-xs text-gray-500 mt-2">Anbefalt størrelse: minst 200x200px</p>
              </div>
            </div>
          </div>
        );
      
      case 1: // Kontaktperson
        return (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-white mb-6">Kontaktperson</h2>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-400 mb-2">Fornavn</label>
                <input
                  type="text"
                  value={formData.contact.firstName}
                  onChange={(e) => handleChange('contact', 'firstName', e.target.value)}
                  className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  placeholder="Fornavn"
                />
              </div>
              <div>
                <label className="block text-gray-400 mb-2">Etternavn</label>
                <input
                  type="text"
                  value={formData.contact.lastName}
                  onChange={(e) => handleChange('contact', 'lastName', e.target.value)}
                  className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  placeholder="Etternavn"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">E-post</label>
              <input
                type="email"
                value={formData.contact.email}
                onChange={(e) => handleChange('contact', 'email', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="E-post"
              />
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Telefon</label>
              <input
                type="tel"
                value={formData.contact.phone}
                onChange={(e) => handleChange('contact', 'phone', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="Telefon"
              />
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Stilling</label>
              <input
                type="text"
                value={formData.contact.position}
                onChange={(e) => handleChange('contact', 'position', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="Stilling"
              />
            </div>
          </div>
        );
      
      case 2: // Tjenester
        return (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-white mb-6">Tjenester</h2>
            
            <div>
              <label className="block text-gray-400 mb-2">Tjenestekategorier</label>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {['Snekring', 'Rørlegger', 'Elektriker', 'Maler', 'Murer', 'Taktekker', 'Kjøkkenmontering', 'Baderomsrenovering', 'Graving', 'Hagearbeid'].map((category) => (
                  <label key={category} className="flex items-center bg-white/5 p-3 rounded-lg hover:bg-white/10 cursor-pointer transition-colors">
                    <input
                      type="checkbox"
                      checked={formData.services.categories.includes(category)}
                      onChange={() => handleCategoryChange(category)}
                      className="mr-2"
                    />
                    <span className="text-gray-300">{category}</span>
                  </label>
                ))}
              </div>
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Beskrivelse av tjenester</label>
              <textarea
                value={formData.services.description}
                onChange={(e) => handleChange('services', 'description', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="Beskriv dine tjenester, spesialiteter og erfaring..."
                rows={4}
              />
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Prisnivå</label>
              <div className="grid grid-cols-3 gap-3 mb-4">
                {['low', 'medium', 'high'].map((level) => (
                  <label
                    key={level}
                    className={`flex flex-col items-center bg-white/5 p-3 rounded-lg cursor-pointer transition-colors ${formData.services.priceRange === level ? 'border border-blue-500 bg-blue-900/20' : 'hover:bg-white/10'}`}
                  >
                    <input
                      type="radio"
                      checked={formData.services.priceRange === level}
                      onChange={() => handleChange('services', 'priceRange', level)}
                      className="sr-only"
                    />
                    <span className="text-white font-medium">
                      {level === 'low' ? 'Budsjett' : level === 'medium' ? 'Standard' : 'Premium'}
                    </span>
                    <span className="text-gray-400 text-sm">
                      {level === 'low' ? 'Lavere priser' : level === 'medium' ? 'Markedspriser' : 'Høy kvalitet'}
                    </span>
                  </label>
                ))}
              </div>
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Minste prosjektstørrelse (kr)</label>
              <input
                type="text"
                value={formData.services.projectMinSize}
                onChange={(e) => handleChange('services', 'projectMinSize', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="F.eks. 10000"
              />
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Områder du opererer i</label>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {['Oslo', 'Bergen', 'Trondheim', 'Stavanger', 'Kristiansand', 'Tromsø', 'Drammen', 'Fredrikstad'].map((area) => (
                  <label key={area} className="flex items-center bg-white/5 p-3 rounded-lg hover:bg-white/10 cursor-pointer transition-colors">
                    <input
                      type="checkbox"
                      checked={formData.services.operatingAreas.includes(area)}
                      onChange={() => handleAreaChange(area)}
                      className="mr-2"
                    />
                    <span className="text-gray-300">{area}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );
      
      case 3: // Abonnement
        return (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-white mb-6">Abonnement</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              <div 
                className={`p-6 rounded-xl border cursor-pointer transition-all ${
                  formData.subscription.plan === 'regional' 
                    ? 'border-blue-500 bg-blue-900/20' 
                    : 'border-white/10 bg-white/5 hover:bg-white/10'
                }`}
                onClick={() => handleChange('subscription', 'plan', 'regional')}
              >
                <h3 className="text-xl font-semibold text-white mb-2">Regional</h3>
                <p className="text-3xl font-bold text-white mb-4">5.000 kr</p>
                <p className="text-sm text-gray-400 mb-4">per måned</p>
                <ul className="space-y-2">
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Synlighet i ett fylke</span>
                  </li>
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Inntil 5 leads per måned</span>
                  </li>
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Standard plassering</span>
                  </li>
                </ul>
              </div>
              
              <div 
                className={`p-6 rounded-xl border cursor-pointer transition-all ${
                  formData.subscription.plan === 'national' 
                    ? 'border-blue-500 bg-blue-900/20' 
                    : 'border-white/10 bg-white/5 hover:bg-white/10'
                }`}
                onClick={() => handleChange('subscription', 'plan', 'national')}
              >
                <h3 className="text-xl font-semibold text-white mb-2">Nasjonal</h3>
                <p className="text-3xl font-bold text-white mb-4">12.000 kr</p>
                <p className="text-sm text-gray-400 mb-4">per måned</p>
                <ul className="space-y-2">
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Synlighet i hele Norge</span>
                  </li>
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Ubegrenset antall leads</span>
                  </li>
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Fremhevet plassering</span>
                  </li>
                </ul>
              </div>
              
              <div 
                className={`p-6 rounded-xl border cursor-pointer transition-all ${
                  formData.subscription.plan === 'premium' 
                    ? 'border-blue-500 bg-blue-900/20' 
                    : 'border-white/10 bg-white/5 hover:bg-white/10'
                }`}
                onClick={() => handleChange('subscription', 'plan', 'premium')}
              >
                <h3 className="text-xl font-semibold text-white mb-2">Premium</h3>
                <p className="text-3xl font-bold text-white mb-4">25.000 kr</p>
                <p className="text-sm text-gray-400 mb-4">per måned</p>
                <ul className="space-y-2">
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Synlighet i hele Norge</span>
                  </li>
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Ubegrenset antall leads</span>
                  </li>
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Topp plassering i søk</span>
                  </li>
                  <li className="flex items-start">
                    <Check size={18} className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-300">Dedikert kundestøtte</span>
                  </li>
                </ul>
              </div>
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Abonnementsperiode</label>
              <select
                value={formData.subscription.duration}
                onChange={(e) => handleChange('subscription', 'duration', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="6">6 måneder (ingen rabatt)</option>
                <option value="12">12 måneder (10% rabatt)</option>
                <option value="24">24 måneder (20% rabatt)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-gray-400 mb-2">Betalingsmetode</label>
              <select
                value={formData.subscription.paymentMethod}
                onChange={(e) => handleChange('subscription', 'paymentMethod', e.target.value)}
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="invoice">Faktura</option>
                <option value="card">Kredittkort</option>
                <option value="directDebit">AvtaleGiro</option>
              </select>
            </div>
          </div>
        );
      
      case 4: // Avtalevilkår
        return (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-white mb-6">Avtalevilkår</h2>
            
            <div className="p-6 bg-white/5 border border-white/10 rounded-xl mb-8">
              <h3 className="text-lg font-semibold text-white mb-4">Samarbeidsavtale for håndverkerpartnere</h3>
              
              <div className="h-64 overflow-y-auto bg-black/30 p-4 rounded-lg mb-4 text-gray-300 text-sm">
                <h4 className="font-semibold mb-2">1. Avtalens parter og formål</h4>
                <p className="mb-4">
                  Denne avtalen inngås mellom Eiendomsmuligheter AS (heretter "Plattformen") og Håndverkeren. 
                  Formålet med avtalen er å regulere vilkårene for Håndverkerens deltakelse og presentasjon av 
                  tjenester på Plattformen.
                </p>
                
                <h4 className="font-semibold mb-2">2. Plattformens ansvar</h4>
                <p className="mb-4">
                  Plattformen stiller til rådighet en digital løsning for presentasjon av Håndverkerens tjenester. 
                  Plattformen har intet ansvar for kvaliteten på Håndverkerens arbeid eller tjenester. Plattformen 
                  garanterer ikke for antall henvendelser, leads eller kunder.
                </p>
                
                <h4 className="font-semibold mb-2">3. Håndverkerens forpliktelser</h4>
                <p className="mb-4">
                  Håndverkeren er ansvarlig for at all informasjon som presenteres er korrekt og oppdatert. 
                  Håndverkeren forplikter seg til å overholde alle lover og forskrifter knyttet til håndverksvirksomhet, 
                  inkludert HMS og faglige standarder. Håndverkeren skal selv foreta all vurdering av prosjekter og 
                  beslutte hvorvidt et oppdrag skal påtas.
                </p>
                
                <h4 className="font-semibold mb-2">4. Ansvarsfraskrivelse</h4>
                <p className="mb-4">
                  Plattformen er utelukkende en formidlingstjeneste. Plattformen har intet ansvar for 
                  kvaliteten på Håndverkerens arbeid, tjenester eller kundeservice. Plattformen påtar seg intet 
                  ansvar for tap, direkte eller indirekte, som måtte oppstå som følge av bruk av tjenesten.
                </p>
                
                <h4 className="font-semibold mb-2">5. Varighet og oppsigelse</h4>
                <p className="mb-4">
                  Avtalen løper for den periode som er valgt ved registrering. Avtalen kan sies opp av 
                  begge parter med 30 dagers varsel. Ved vesentlig mislighold kan avtalen heves med 
                  umiddelbar virkning.
                </p>
              </div>
              
              <label className="flex items-start mb-4">
                <input
                  type="checkbox"
                  checked={formData.terms.acceptPartnerAgreement}
                  onChange={(e) => handleChange('terms', 'acceptPartnerAgreement', e.target.checked)}
                  className="mt-1 mr-2"
                />
                <span className="text-gray-300 text-sm">
                  Jeg har lest og aksepterer samarbeidsavtalen for håndverkerpartnere
                </span>
              </label>
            </div>
            
            <label className="flex items-start mb-4">
              <input
                type="checkbox"
                checked={formData.terms.acceptTerms}
                onChange={(e) => handleChange('terms', 'acceptTerms', e.target.checked)}
                className="mt-1 mr-2"
              />
              <span className="text-gray-300 text-sm">
                Jeg aksepterer generelle <a href="#" className="text-blue-400 underline">brukervilkår</a> for Eiendomsmuligheter
              </span>
            </label>
            
            <label className="flex items-start mb-4">
              <input
                type="checkbox"
                checked={formData.terms.acceptPrivacyPolicy}
                onChange={(e) => handleChange('terms', 'acceptPrivacyPolicy', e.target.checked)}
                className="mt-1 mr-2"
              />
              <span className="text-gray-300 text-sm">
                Jeg aksepterer <a href="#" className="text-blue-400 underline">personvernerklæringen</a> for Eiendomsmuligheter
              </span>
            </label>
            
            <div className="p-4 bg-blue-900/20 rounded-lg border border-blue-500/30 text-sm text-gray-300">
              <p>
                <strong>Viktig juridisk informasjon:</strong> Eiendomsmuligheter AS er utelukkende en formidlingsplattform 
                som muliggjør kontakt mellom håndverkere og potensielle kunder. Eiendomsmuligheter AS 
                gir ingen garantier for kvaliteten på arbeidet som utføres av håndverkerne. Alle avtaler om utførelse 
                av håndverkertjenester inngås direkte mellom håndverkeren og kunden. Eiendomsmuligheter AS er ikke part 
                i slike avtaler, og påtar seg intet ansvar for kvaliteten, lovligheten eller utførelsen av tjenester 
                som tilbys gjennom plattformen.
              </p>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-gradient-to-br from-gray-900 to-black rounded-xl border border-white/10 p-8 max-w-4xl mx-auto">
      <div className="flex items-center mb-8">
        <Hammer className="w-8 h-8 text-blue-400 mr-3" />
        <h1 className="text-3xl font-bold text-white">Bli håndverkerpartner</h1>
      </div>
      
      <Stepper steps={steps} currentStep={currentStep} />
      
      <div className="mt-8">
        {renderStep()}
      </div>
      
      <div className="flex justify-between mt-8">
        {currentStep > 0 ? (
          <button
            onClick={handlePrevious}
            className="px-6 py-3 rounded-lg bg-white/10 hover:bg-white/15 text-white font-medium transition-colors"
          >
            Tilbake
          </button>
        ) : (
          <div></div>
        )}
        
        {currentStep < steps.length - 1 ? (
          <button
            onClick={handleNext}
            className="px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium flex items-center gap-2"
          >
            Neste steg
            <ChevronRight size={18} />
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!formData.terms.acceptTerms || !formData.terms.acceptPrivacyPolicy || !formData.terms.acceptPartnerAgreement}
            className={`px-6 py-3 rounded-lg text-white font-medium flex items-center gap-2 ${
              formData.terms.acceptTerms && formData.terms.acceptPrivacyPolicy && formData.terms.acceptPartnerAgreement
                ? 'bg-gradient-to-r from-blue-500 to-purple-600'
                : 'bg-gray-700 cursor-not-allowed'
            }`}
          >
            Send registrering
            <FileText size={18} />
          </button>
        )}
      </div>
    </div>
  );
}; 