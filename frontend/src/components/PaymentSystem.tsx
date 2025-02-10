import React, { useState, useEffect } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import {
  Elements,
  PaymentElement,
  useStripe,
  useElements
} from '@stripe/react-stripe-js';
import { Box, Button, Typography, CircularProgress, Alert } from '@mui/material';
import { styled } from '@mui/material/styles';

// Styled komponenter
const PaymentContainer = styled(Box)(({ theme }) => ({
  maxWidth: 800,
  margin: '0 auto',
  padding: theme.spacing(4),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3]
}));

const PaymentForm = styled('form')(({ theme }) => ({
  marginTop: theme.spacing(3),
  '& .StripeElement': {
    border: `1px solid ${theme.palette.divider}`,
    padding: theme.spacing(2),
    borderRadius: theme.shape.borderRadius,
    backgroundColor: theme.palette.background.default
  }
}));

// Stripe Promise
const stripePromise = loadStripe(process.env.REACT_APP_STRIPE_PUBLIC_KEY!);

// Hovedkomponent
const PaymentSystem: React.FC = () => {
  const [clientSecret, setClientSecret] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Hent betalingsintent fra backend
    const fetchPaymentIntent = async () => {
      try {
        const response = await fetch('/api/create-payment-intent', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            packageId: 'pro',
            customerEmail: 'kunde@example.com'
          })
        });

        if (!response.ok) {
          throw new Error('Kunne ikke hente betalingsdetaljer');
        }

        const data = await response.json();
        setClientSecret(data.clientSecret);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'En feil oppstod');
      } finally {
        setLoading(false);
      }
    };

    fetchPaymentIntent();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <PaymentContainer>
      <Typography variant="h4" gutterBottom>
        Betaling
      </Typography>
      {clientSecret && (
        <Elements stripe={stripePromise} options={{ clientSecret }}>
          <CheckoutForm />
        </Elements>
      )}
    </PaymentContainer>
  );
};

// Betalingsskjema
const CheckoutForm: React.FC = () => {
  const stripe = useStripe();
  const elements = useElements();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!stripe || !elements) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const { error: submitError } = await stripe.confirmPayment({
        elements,
        confirmParams: {
          return_url: `${window.location.origin}/payment-success`,
        },
      });

      if (submitError) {
        throw new Error(submitError.message);
      }

      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'En feil oppstod');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <Alert severity="success" sx={{ mt: 2 }}>
        Betalingen var vellykket! Du vil bli videresendt til kvitteringssiden.
      </Alert>
    );
  }

  return (
    <PaymentForm onSubmit={handleSubmit}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Box mb={3}>
        <PaymentElement />
      </Box>

      <Button
        variant="contained"
        color="primary"
        type="submit"
        disabled={loading || !stripe}
        fullWidth
        size="large"
      >
        {loading ? <CircularProgress size={24} /> : 'Betal nå'}
      </Button>
    </PaymentForm>
  );
};

// Prisvisning
const PricingDisplay: React.FC<{ selectedPlan: string }> = ({ selectedPlan }) => {
  const plans = {
    basic: {
      name: 'Basic',
      price: '999',
      features: [
        'Grunnleggende eiendomsanalyse',
        '3D-visualisering',
        'Energianalyse'
      ]
    },
    pro: {
      name: 'Pro',
      price: '2999',
      features: [
        'Alt i Basic',
        'Avansert utviklingsanalyse',
        'Automatisk byggesøknad',
        'Direkte kommuneintegrasjon'
      ]
    },
    enterprise: {
      name: 'Enterprise',
      price: '9999',
      features: [
        'Alt i Pro',
        'Ubegrenset analyser',
        'Dedikert support',
        'API-tilgang'
      ]
    }
  };

  const plan = plans[selectedPlan as keyof typeof plans];

  return (
    <Box mb={4}>
      <Typography variant="h5" gutterBottom>
        {plan.name} Plan
      </Typography>
      <Typography variant="h4" color="primary" gutterBottom>
        NOK {plan.price}
      </Typography>
      <Box mt={2}>
        {plan.features.map((feature, index) => (
          <Typography key={index} variant="body1" color="text.secondary">
            ✓ {feature}
          </Typography>
        ))}
      </Box>
    </Box>
  );
};

export default PaymentSystem;