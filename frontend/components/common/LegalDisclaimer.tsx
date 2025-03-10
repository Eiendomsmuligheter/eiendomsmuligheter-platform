import React, { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';

interface LegalDisclaimerProps {
  type?: 'financial' | 'contractor' | 'insurance' | 'rental' | 'general';
  compact?: boolean;
}

export const LegalDisclaimer = ({ type = 'general', compact = false }: LegalDisclaimerProps) => {
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen && compact) return null;

  // Base ansvarsfraskrivelse som gjelder for alle tjenester
  const baseDisclaimer = `
    Eiendomsmuligheter AS ("Plattformen") er utelukkende en informasjons- og formidlingsplattform. 
    Plattformen påtar seg intet ansvar for riktigheten, fullstendigheten eller nøyaktigheten av informasjonen 
    som presenteres eller formidles. All bruk av Plattformen skjer på brukerens eget ansvar.
    
    Plattformen er ikke part i noen avtaler som måtte inngås mellom brukere og tredjeparter, herunder 
    finansinstitusjoner, håndverkere, forsikringsselskaper eller utleiere. Plattformen har intet ansvar for 
    kvaliteten, lovligheten eller leveransen av produkter eller tjenester som tilbys gjennom Plattformen.
    
    I den maksimale utstrekning gjeldende rett tillater det, fraskriver Plattformen seg ethvert ansvar for 
    direkte, indirekte, tilfeldige, spesielle, følgeskader eller straffeerstatning, herunder, men ikke begrenset 
    til, tap av fortjeneste, goodwill, bruk, data eller andre immaterielle tap.
  `;

  // Spesifikke ansvarsfraskrivelser basert på tjenestetype
  const specificDisclaimers = {
    financial: `
      Plattformen gir ingen finansiell rådgivning og er ikke en finansiell rådgiver. Alle finansielle produkter 
      som presenteres på Plattformen tilbys av tredjeparter. Plattformen har ingen kontroll over og påtar seg 
      intet ansvar for vilkårene for slike produkter. Brukere oppfordres til å gjøre sin egen undersøkelse og 
      vurdering før de inngår noen avtale med en tredjepart.
      
      Kredittbeslutninger, lånevilkår, rentesatser og godkjenninger bestemmes utelukkende av den enkelte 
      finansinstitusjon basert på deres egne kriterier. Plattformen garanterer ikke for tilgjengeligheten eller 
      innvilgelsen av noe finansielt produkt og er ikke ansvarlig for eventuelle avslag eller betingelser.
    `,
    contractor: `
      Plattformen fungerer kun som en formidler mellom brukere og håndverkere. Plattformen utfører ingen 
      bakgrunnssjekk eller verifisering av håndverkernes kvalifikasjoner, kompetanse, erfaring eller overholdelse 
      av gjeldende lover og forskrifter.
      
      Det er brukerens eget ansvar å verifisere og sikre at håndverkeren har nødvendige tillatelser, forsikringer, 
      sertifiseringer og kompetanse før det inngås avtale. Plattformen er ikke ansvarlig for kvaliteten på arbeidet 
      utført, overholdelse av tidsfrister, skader på eiendom eller person, eller andre forhold knyttet til 
      håndverkertjenestene.
    `,
    insurance: `
      Forsikringsprodukter som presenteres på Plattformen tilbys av tredjeparter. Plattformen er ikke et 
      forsikringsselskap eller forsikringsmegler og gir ingen forsikringsrådgivning. Alle beslutninger om 
      dekning, premier, vilkår og betingelser tas utelukkende av det respektive forsikringsselskapet.
      
      Plattformen garanterer ikke for utbetaling av erstatning eller behandling av krav under noen omstendighet. 
      Brukeren er selv ansvarlig for å lese og forstå vilkårene i enhver forsikringsavtale før den inngås.
    `,
    rental: `
      Plattformen fungerer kun som en formidler mellom utleiere og leietakere. Plattformen utfører ingen 
      verifisering av eiendommene som annonseres, deres tilstand, lovlighet eller egnethet for utleie. 
      Plattformen verifiserer heller ikke leietakernes kredittverdighet, bakgrunn eller evne til å oppfylle 
      leieavtalen.
      
      Det er brukerens eget ansvar å inspisere eiendommen, verifisere opplysninger og sikre at alle aspekter 
      ved leieavtalen er i samsvar med gjeldende lover og forskrifter. Plattformen er ikke ansvarlig for 
      eventuelle tvister, skader eller andre forhold som måtte oppstå i forbindelse med leieforholdet.
    `,
    general: ''
  };

  const fullDisclaimer = `${baseDisclaimer}${specificDisclaimers[type]}`;

  if (compact) {
    return (
      <div className="bg-gray-900/80 border border-yellow-600/30 rounded-lg p-4 text-sm mb-6">
        <div className="flex items-start">
          <AlertTriangle className="text-yellow-500 mr-3 mt-1 flex-shrink-0" size={18} />
          <div className="flex-1">
            <p className="text-gray-300">
              <strong>Ansvarsfraskrivelse:</strong> Eiendomsmuligheter AS er en formidlingsplattform. 
              Vi gir ingen garantier og tar intet ansvar for nøyaktigheten av informasjon eller tjenester 
              fra tredjeparter. All bruk skjer på eget ansvar.
            </p>
            <button 
              onClick={() => setIsOpen(false)}
              className="text-yellow-500 hover:text-yellow-400 text-xs mt-2 flex items-center"
            >
              <X size={12} className="mr-1" />
              Skjul
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/80 border border-yellow-600/30 rounded-lg p-6 text-sm mb-8">
      <div className="flex items-start mb-4">
        <AlertTriangle className="text-yellow-500 mr-3 mt-1 flex-shrink-0" size={24} />
        <h3 className="text-lg font-semibold text-white">Juridisk ansvarsfraskrivelse</h3>
      </div>
      
      <div className="text-gray-300 space-y-4 mb-4">
        {fullDisclaimer.split('\n\n').map((paragraph, index) => (
          <p key={index}>{paragraph.trim()}</p>
        ))}
      </div>
      
      <p className="text-yellow-500 text-xs">
        Ved å fortsette å bruke denne tjenesten bekrefter du at du har lest, forstått og akseptert 
        disse ansvarsfraskrivelsene og vilkårene.
      </p>
    </div>
  );
}; 