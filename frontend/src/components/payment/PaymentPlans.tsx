import React from 'react';
import { Box, Card, Typography, Button, Grid, Container } from '@mui/material';
import { useStripe } from '@stripe/stripe-react-js';
import { useAuth } from '../../hooks/useAuth';

interface PricingPlan {
  id: string;
  name: string;
  price: number;
  features: string[];
  stripePriceId: string;
}

const pricingPlans: PricingPlan[] = [
  {
    id: 'basic',
    name: 'Basis Analyse',
    price: 499,
    stripePriceId: 'price_basic_monthly',
    features: [
      'Grunnleggende eiendomsanalyse',
      'Plantegningsanalyse',
      'Reguleringsplan-sjekk',
      'PDF-rapport'
    ]
  },
  {
    id: 'pro',
    name: 'Pro Analyse',
    price: 999,
    stripePriceId: 'price_pro_monthly',
    features: [
      'Alt i Basis pakken',
      '3D-visualisering',
      'Utleiepotensial-analyse',
      'BIM-modell eksport',
      'Energianalyse',
      'Automatisk byggesøknad'
    ]
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: 2499,
    stripePriceId: 'price_enterprise_monthly',
    features: [
      'Alt i Pro pakken',
      'Dedikert støtte',
      'API-tilgang',
      'Ubegrenset analyse',
      'NVIDIA Omniverse integrasjon',
      'Automatisert dokumentgenerering',
      'Prioritert behandling'
    ]
  }
];

export const PaymentPlans: React.FC = () => {
  const stripe = useStripe();
  const { user } = useAuth();

  const handleSubscribe = async (stripePriceId: string) => {
    try {
      const response = await fetch('/api/create-checkout-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          priceId: stripePriceId,
          customerEmail: user?.email,
        }),
      });

      const session = await response.json();
      
      if (stripe) {
        const { error } = await stripe.redirectToCheckout({
          sessionId: session.id,
        });
        
        if (error) {
          console.error('Error:', error);
        }
      }
    } catch (err) {
      console.error('Error:', err);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 8 }}>
      <Typography variant="h2" align="center" gutterBottom>
        Velg Analysepakke
      </Typography>
      <Typography variant="h5" align="center" color="textSecondary" paragraph>
        Få tilgang til verdens mest avanserte eiendomsanalyse
      </Typography>
      <Grid container spacing={4} alignItems="flex-start" sx={{ mt: 4 }}>
        {pricingPlans.map((plan) => (
          <Grid item key={plan.id} xs={12} sm={6} md={4}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                p: 3,
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'scale(1.02)',
                  boxShadow: 6,
                },
              }}
            >
              <Typography variant="h4" component="h2" gutterBottom>
                {plan.name}
              </Typography>
              <Typography variant="h3" color="primary" gutterBottom>
                {plan.price} kr
                <Typography variant="subtitle1" component="span">
                  /mnd
                </Typography>
              </Typography>
              <Box sx={{ flexGrow: 1 }}>
                {plan.features.map((feature) => (
                  <Typography
                    key={feature}
                    variant="body1"
                    sx={{ py: 1 }}
                    component="li"
                  >
                    {feature}
                  </Typography>
                ))}
              </Box>
              <Button
                variant="contained"
                color="primary"
                size="large"
                fullWidth
                sx={{ mt: 4 }}
                onClick={() => handleSubscribe(plan.stripePriceId)}
              >
                Velg {plan.name}
              </Button>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};