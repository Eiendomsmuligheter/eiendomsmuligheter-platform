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
const PricingCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'scale(1.02)',
  },
}));

const PricingCardHeader = styled(CardHeader)(({ theme }) => ({
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.primary.contrastText,
}));

const FeatureList = styled(List)(({ theme }) => ({
  flexGrow: 1,
}));

interface PricingPlan {
  id: string;
  name: string;
  price: number;
  features: string[];
  recommended?: boolean;
}

const pricingPlans: PricingPlan[] = [
  {
    id: 'basic',
    name: 'Basic',
    price: 999,
    features: [
      'Grunnleggende eiendomsanalyse',
      '3D-visualisering',
      'Energianalyse',
      '1 analyse per måned',
      'E-poststøtte'
    ]
  },
  {
    id: 'pro',
    name: 'Professional',
    price: 2999,
    recommended: true,
    features: [
      'Alt i Basic',
      'Avansert utviklingsanalyse',
      'Automatisk byggesøknad',
      'Direkte kommuneintegrasjon',
      '10 analyser per måned',
      'Prioritert støtte'
    ]
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: 9999,
    features: [
      'Alt i Professional',
      'Ubegrenset analyser',
      'Dedikert supportteam',
      'API-tilgang',
      'Tilpassede rapporter',
      '24/7 telefonsupport'
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
  
  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Velg Abonnement
        </Typography>
        <Typography variant="h6" color="textSecondary" paragraph>
          Få tilgang til verdens beste eiendomsanalyse-plattform
        </Typography>
      </Box>
      
      <Grid container spacing={4} sx={{ mb: 4 }}>
        <AnimatePresence>
          {pricingPlans.map((plan) => (
            <Grid item xs={12} md={4} key={plan.id}>
              <motion.div
                variants={cardVariants}
                initial="hidden"
                animate="visible"
                exit="hidden"
                transition={{ duration: 0.3 }}
              >
                <PricingCard
                  raised={plan.recommended}
                  sx={{
                    border: plan.recommended ? 2 : 0,
                    borderColor: 'primary.main'
                  }}
                >
                  <PricingCardHeader
                    title={plan.name}
                    subheader={`${plan.price} NOK/mnd`}
                    titleTypographyProps={{ align: 'center' }}
                    subheaderTypographyProps={{ align: 'center' }}
                    action={plan.recommended && (
                      <Typography
                        variant="caption"
                        sx={{
                          backgroundColor: 'secondary.main',
                          color: 'secondary.contrastText',
                          px: 2,
                          py: 0.5,
                          borderRadius: 1
                        }}
                      >
                        Anbefalt
                      </Typography>
                    )}
                  />
                  
                  <CardContent>
                    <FeatureList>
                      {plan.features.map((feature, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <CheckCircle color="primary" />
                          </ListItemIcon>
                          <ListItemText primary={feature} />
                        </ListItem>
                      ))}
                    </FeatureList>
                    
                    <Button
                      fullWidth
                      variant={plan.recommended ? "contained" : "outlined"}
                      color="primary"
                      size="large"
                      onClick={() => handlePlanSelect(plan.id)}
                      sx={{ mt: 2 }}
                      disabled={loading}
                    >
                      {selectedPlan === plan.id ? (
                        <>
                          Valgt
                          <CheckCircleOutline sx={{ ml: 1 }} />
                        </>
                      ) : (
                        'Velg Plan'
                      )}
                    </Button>
                  </CardContent>
                </PricingCard>
              </motion.div>
            </Grid>
          ))}
        </AnimatePresence>
      </Grid>
      
      {selectedPlan && (
        <Box sx={{ textAlign: 'center', my: 4 }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            onClick={handleCheckout}
            disabled={loading}
            sx={{ minWidth: 200 }}
          >
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : (
              'Fortsett til Betaling'
            )}
          </Button>
          
          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              {error}
            </Typography>
          )}
        </Box>
      )}
    </Container>
  );
};

export default PaymentFlow;