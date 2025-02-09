import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Container,
  Grid,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  CircularProgress
} from '@mui/material';
import { loadStripe } from '@stripe/stripe-js';
import { CheckCircle, CheckCircleOutline } from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { styled } from '@mui/material/styles';

// Stilede komponenter
const CategoryButton = styled(Button)(({ theme }) => ({
  borderRadius: '30px',
  padding: '10px 24px',
  transition: 'all 0.3s ease',
  fontWeight: 500,
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
  },
  '&.Mui-selected': {
    background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.primary.light} 90%)`,
    boxShadow: '0 3px 10px rgba(0,0,0,0.2)',
  },
}));

const PricingCard = styled(Card)<{ recommended?: boolean }>(({ theme, recommended }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'all 0.3s ease-in-out',
  background: recommended 
    ? `linear-gradient(135deg, ${theme.palette.background.paper} 0%, ${theme.palette.background.default} 100%)`
    : theme.palette.background.paper,
  borderRadius: '20px',
  overflow: 'hidden',
  boxShadow: recommended 
    ? '0 8px 32px rgba(0,0,0,0.15)'
    : '0 4px 12px rgba(0,0,0,0.05)',
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 12px 48px rgba(0,0,0,0.2)',
  },
}));

const PricingCardHeader = styled(CardHeader)(({ theme }) => ({
  background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
  color: theme.palette.primary.contrastText,
  padding: '2rem 1.5rem',
  position: 'relative',
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: '4px',
    background: `linear-gradient(90deg, ${theme.palette.secondary.main} 0%, ${theme.palette.primary.light} 100%)`,
  },
}));

const FeatureList = styled(List)(({ theme }) => ({
  flexGrow: 1,
  padding: '1.5rem 1rem',
  '& .MuiListItem-root': {
    padding: '0.8rem 0',
    '&:not(:last-child)': {
      borderBottom: `1px solid ${theme.palette.divider}`,
    },
  },
  '& .MuiListItemIcon-root': {
    minWidth: '40px',
  },
  '& .MuiListItemText-primary': {
    fontSize: '0.95rem',
    fontWeight: 500,
  },
}));

const PriceTag = styled(Typography)(({ theme }) => ({
  fontSize: '2.5rem',
  fontWeight: 700,
  color: theme.palette.primary.main,
  marginBottom: '1rem',
  display: 'flex',
  alignItems: 'baseline',
  justifyContent: 'center',
  '& .currency': {
    fontSize: '1.2rem',
    marginRight: '0.5rem',
    opacity: 0.8,
  },
  '& .period': {
    fontSize: '1rem',
    marginLeft: '0.5rem',
    opacity: 0.7,
  },
}));

const PopularBadge = styled(Typography)(({ theme }) => ({
  position: 'absolute',
  top: '1rem',
  right: '1rem',
  background: theme.palette.secondary.main,
  color: theme.palette.secondary.contrastText,
  padding: '0.4rem 1rem',
  borderRadius: '20px',
  fontSize: '0.8rem',
  fontWeight: 600,
  boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
  zIndex: 1,
}));

interface PricingPlan {
  id: string;
  name: string;
  price: number;
  features: string[];
  recommended?: boolean;
}

const pricingPlans: PricingPlan[] = [
  // Analyserapport
  {
    id: 'analysis_basic',
    name: 'Basic Analyserapport',
    price: 1499,
    features: [
      'Automatisk plantegningsanalyse',
      'Enkel ROI-kalkulator',
      'PDF-rapport',
      'Grunnleggende reguleringssjekk'
    ]
  },
  {
    id: 'analysis_pro',
    name: 'Pro Analyserapport',
    price: 2999,
    recommended: true,
    features: [
      'Alt i Basic',
      'Detaljert potensialanalyse',
      '3D-visualisering',
      'Omfattende reguleringssjekk',
      'Energianalyse',
      'Utleiepotensialanalyse'
    ]
  },
  {
    id: 'analysis_enterprise',
    name: 'Enterprise Analyserapport',
    price: 4999,
    features: [
      'Alt i Professional',
      'API-tilgang',
      'Dedikert støtte',
      'Ubegrenset analyser',
      'Prioritert behandling',
      'Månedlige konsultasjoner'
    ]
  },
  
  // Tegningspakke
  {
    id: 'drawing_basic',
    name: 'Basic Tegningspakke',
    price: 2999,
    features: [
      '2D plantegninger',
      'Enkle fasadetegninger',
      'Situasjonsplan',
      'PDF-format'
    ]
  },
  {
    id: 'drawing_pro',
    name: 'Pro Tegningspakke',
    price: 4999,
    features: [
      'Alt i Basic',
      '3D-modellering',
      'Detaljerte fasadetegninger',
      'Tekniske spesifikasjoner',
      'Revit/AutoCAD-filer'
    ]
  },
  {
    id: 'drawing_enterprise',
    name: 'Enterprise Tegningspakke',
    price: 7999,
    features: [
      'Alt i Professional',
      'BIM-modeller',
      'VR-visualisering',
      'Ubegrensede revisjoner',
      'Komplett byggesøknadspakke'
    ]
  },
  
  // Energirådgivning
  {
    id: 'energy_basic',
    name: 'Basic Energirådgivning',
    price: 1999,
    features: [
      'Energimerking',
      'Enkel energianalyse',
      'Grunnleggende tiltaksforslag',
      'Enova-støtteberegning'
    ]
  },
  {
    id: 'energy_pro',
    name: 'Pro Energirådgivning',
    price: 3999,
    features: [
      'Alt i Basic',
      'Detaljert energianalyse',
      'Termografering',
      'Komplett tiltaksplan',
      'Lønnsomhetsberegninger'
    ]
  },
  {
    id: 'energy_enterprise',
    name: 'Enterprise Energirådgivning',
    price: 6999,
    features: [
      'Alt i Professional',
      'Bygningssimulering',
      'Klimaregnskap',
      'Søknadsassistanse Enova',
      'Årlig oppfølging'
    ]
  },
  
  // Byggesøknad
  {
    id: 'building_basic',
    name: 'Basic Byggesøknad',
    price: 4999,
    features: [
      'Søknadsskjemaer',
      'Enkle tegninger',
      'Nabovarsel',
      'Digital innsending'
    ]
  },
  {
    id: 'building_pro',
    name: 'Pro Byggesøknad',
    price: 7999,
    features: [
      'Alt i Basic',
      'Komplett tegningssett',
      'Teknisk beskrivelse',
      'Ansvarserklæringer',
      'Dokumenthåndtering'
    ]
  },
  {
    id: 'building_enterprise',
    name: 'Enterprise Byggesøknad',
    price: 12999,
    features: [
      'Alt i Professional',
      'Prosjektledelse',
      'Dispensasjonssøknader',
      'Møter med kommune',
      'Oppfølging til ferdigattest'
    ]
  },
  
  // Komplett pakke
  {
    id: 'complete_basic',
    name: 'Basic Totalpakke',
    price: 9999,
    features: [
      'Eiendomsanalyse Basic',
      'Tegningspakke Basic',
      'Energirådgivning Basic',
      'Byggesøknad Basic',
      '10% rabatt på totalpris'
    ]
  },
  {
    id: 'complete_pro',
    name: 'Pro Totalpakke',
    price: 16999,
    recommended: true,
    features: [
      'Eiendomsanalyse Professional',
      'Tegningspakke Professional',
      'Energirådgivning Professional',
      'Byggesøknad Professional',
      '15% rabatt på totalpris'
    ]
  },
  {
    id: 'complete_enterprise',
    name: 'Enterprise Totalpakke',
    price: 27999,
    features: [
      'Eiendomsanalyse Enterprise',
      'Tegningspakke Enterprise',
      'Energirådgivning Enterprise',
      'Byggesøknad Enterprise',
      '20% rabatt på totalpris',
      'Dedikert prosjektleder'
    ]
  }
];

export const PaymentFlow: React.FC = () => {
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Håndter valg av abonnementsplan
  const handlePlanSelect = (planId: string) => {
    setSelectedPlan(planId);
  };
  
  // Start checkout-prosessen
  const handleCheckout = async () => {
    if (!selectedPlan) return;
    
    try {
      setLoading(true);
      setError(null);
      
      // Hent checkout-sesjon fra backend
      const response = await fetch('/api/create-checkout-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          planId: selectedPlan,
          successUrl: `${window.location.origin}/payment-success`,
          cancelUrl: `${window.location.origin}/pricing`
        })
      });
      
      if (!response.ok) {
        throw new Error('Kunne ikke opprette checkout-sesjon');
      }
      
      const { sessionId, publicKey } = await response.json();
      
      // Initialiser Stripe og redirect til checkout
      const stripe = await loadStripe(publicKey);
      if (!stripe) {
        throw new Error('Kunne ikke laste Stripe');
      }
      
      const { error } = await stripe.redirectToCheckout({
        sessionId
      });
      
      if (error) {
        throw error;
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'En feil oppstod');
    } finally {
      setLoading(false);
    }
  };
  
  // Animasjonsvariabler
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };
  
  // Grupperer planer etter type
const planTypes = {
  analysis: pricingPlans.filter(plan => plan.id.startsWith('analysis_')),
  drawing: pricingPlans.filter(plan => plan.id.startsWith('drawing_')),
  energy: pricingPlans.filter(plan => plan.id.startsWith('energy_')),
  building: pricingPlans.filter(plan => plan.id.startsWith('building_')),
  complete: pricingPlans.filter(plan => plan.id.startsWith('complete_'))
};

const planTypeNames = {
  analysis: 'Analyserapporter',
  drawing: 'Tegningspakker',
  energy: 'Energirådgivning',
  building: 'Byggesøknader',
  complete: 'Komplette Pakker'
};

const [selectedType, setSelectedType] = useState('complete');

return (
    <Container maxWidth="xl">
      <Box sx={{ my: 6, textAlign: 'center' }}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography variant="h2" component="h1" gutterBottom 
            sx={{ 
              fontWeight: 700,
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
            Velg Din Løsning
          </Typography>
          <Typography variant="h5" color="textSecondary" paragraph sx={{ maxWidth: '800px', margin: '0 auto', mb: 4 }}>
            Få tilgang til markedets mest avanserte verktøy for eiendomsutvikling og optimalisering
          </Typography>
        </motion.div>

        {/* Kategorivelger */}
        <Box 
          sx={{ 
            mb: 6, 
            display: 'flex', 
            flexWrap: 'wrap',
            justifyContent: 'center', 
            gap: 2,
            '& > *': { minWidth: '180px' }
          }}
        >
          {Object.entries(planTypeNames).map(([type, name], index) => (
            <motion.div
              key={type}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <CategoryButton
                selected={selectedType === type}
                onClick={() => setSelectedType(type)}
                variant={selectedType === type ? 'contained' : 'outlined'}
                fullWidth
              >
                {name}
              </CategoryButton>
            </motion.div>
          ))}
        </Box>
      </Box>
      
      {/* Prisplaner */}
      <Grid container spacing={4} sx={{ mb: 6 }}>
        <AnimatePresence mode="wait">
          {planTypes[selectedType as keyof typeof planTypes].map((plan, index) => (
            <Grid item xs={12} md={selectedType === 'complete' ? 4 : 4} key={plan.id}>
              <motion.div
                variants={cardVariants}
                initial="hidden"
                animate="visible"
                exit="hidden"
                transition={{ duration: 0.4, delay: index * 0.1 }}
              >
                <PricingCard recommended={plan.recommended}>
                  {plan.recommended && (
                    <PopularBadge>
                      Mest Populær
                    </PopularBadge>
                  )}
                  
                  <PricingCardHeader
                    title={
                      <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
                        {plan.name}
                      </Typography>
                    }
                    subheader={
                      <PriceTag>
                        <span className="currency">NOK</span>
                        {plan.price.toLocaleString('nb-NO')}
                      </PriceTag>
                    }
                  />
                  
                  <CardContent sx={{ 
                    height: '100%', 
                    display: 'flex', 
                    flexDirection: 'column',
                    p: 0 
                  }}>
                    <FeatureList>
                      {plan.features.map((feature, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <CheckCircle 
                              sx={{ 
                                color: plan.recommended ? 'secondary.main' : 'primary.main',
                                fontSize: '1.2rem' 
                              }} 
                            />
                          </ListItemIcon>
                          <ListItemText 
                            primary={feature}
                            primaryTypographyProps={{
                              sx: { 
                                fontSize: '0.95rem',
                                fontWeight: 500
                              }
                            }}
                          />
                        </ListItem>
                      ))}
                    </FeatureList>
                    
                    <Box sx={{ p: 3, mt: 'auto' }}>
                      <Button
                        fullWidth
                        variant={plan.recommended ? "contained" : "outlined"}
                        color={plan.recommended ? "secondary" : "primary"}
                        size="large"
                        onClick={() => handlePlanSelect(plan.id)}
                        sx={{
                          py: 1.8,
                          fontSize: '1.1rem',
                          fontWeight: 600,
                          borderRadius: '30px',
                          transition: 'all 0.3s',
                          background: plan.recommended 
                            ? 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)'
                            : 'transparent',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: '0 8px 25px rgba(33, 150, 243, 0.25)'
                          }
                        }}
                        disabled={loading}
                      >
                        {selectedPlan === plan.id ? (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <CheckCircleOutline sx={{ mr: 1 }} />
                            Valgt Plan
                          </Box>
                        ) : (
                          'Velg Denne Pakken'
                        )}
                      </Button>
                    </Box>
                  </CardContent>
                </PricingCard>
              </motion.div>
            </Grid>
          ))}
        </AnimatePresence>
      </Grid>
      
      {selectedPlan && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.3 }}
        >
          <Box
            sx={{
              position: 'fixed',
              bottom: 0,
              left: 0,
              right: 0,
              backgroundColor: 'background.paper',
              boxShadow: '0 -4px 20px rgba(0,0,0,0.1)',
              p: 3,
              zIndex: 1000,
              borderTop: '1px solid',
              borderColor: 'divider'
            }}
          >
            <Container maxWidth="lg">
              <Grid container spacing={3} alignItems="center">
                <Grid item xs={12} md={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box sx={{ mr: 3 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {pricingPlans.find(p => p.id === selectedPlan)?.name}
                      </Typography>
                      <Typography variant="subtitle1" color="primary" sx={{ fontWeight: 700 }}>
                        NOK {pricingPlans.find(p => p.id === selectedPlan)?.price.toLocaleString('nb-NO')}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={6} sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                  <Button
                    variant="outlined"
                    color="primary"
                    size="large"
                    onClick={() => setSelectedPlan(null)}
                    disabled={loading}
                    sx={{
                      px: 4,
                      borderRadius: '30px',
                      fontSize: '1rem'
                    }}
                  >
                    Avbryt
                  </Button>
                  
                  <Button
                    variant="contained"
                    color="primary"
                    size="large"
                    onClick={handleCheckout}
                    disabled={loading}
                    sx={{
                      px: 4,
                      borderRadius: '30px',
                      fontSize: '1rem',
                      background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                      boxShadow: '0 3px 15px rgba(33, 150, 243, 0.3)',
                      '&:hover': {
                        boxShadow: '0 5px 20px rgba(33, 150, 243, 0.4)'
                      }
                    }}
                  >
                    {loading ? (
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <CircularProgress size={20} color="inherit" sx={{ mr: 1 }} />
                        Behandler...
                      </Box>
                    ) : (
                      'Fortsett til Betaling'
                    )}
                  </Button>
                </Grid>
              </Grid>
              
              {error && (
                <Box sx={{ mt: 2, textAlign: 'center' }}>
                  <Typography 
                    color="error"
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 1
                    }}
                  >
                    <ErrorOutline />
                    {error}
                  </Typography>
                </Box>
              )}
            </Container>
          </Box>
        </motion.div>
      )}
    </Container>
  );
};

export default PaymentFlow;