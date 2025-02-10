import React, { useState, useEffect } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import {
  Elements,
  PaymentElement,
  useStripe,
  useElements,
} from '@stripe/react-stripe-js';
import styled from 'styled-components';

// Styled components
const PaymentContainer = styled.div`
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const PlanCard = styled.div<{ selected: boolean }>`
  padding: 1.5rem;
  border: 2px solid ${props => props.selected ? '#0066cc' : '#e0e0e0'};
  border-radius: 6px;
  margin-bottom: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
`;

const Button = styled.button`
  background: #0066cc;
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.3s ease;

  &:hover {
    background: #0052a3;
  }

  &:disabled {
    background: #cccccc;
    cursor: not-allowed;
  }
`;

const ErrorMessage = styled.div`
  color: #dc3545;
  margin: 1rem 0;
  padding: 1rem;
  background: #fff3f3;
  border-radius: 4px;
`;

// Prisplaner
const plans = [
  {
    id: 'basic',
    name: 'Basic',
    price: 999,
    features: [
      'Grunnleggende eiendomsanalyse',
      '3D-visualisering',
      'Energianalyse'
    ]
  },
  {
    id: 'pro',
    name: 'Pro',
    price: 2999,
    features: [
      'Alt i Basic',
      'Avansert utviklingsanalyse',
      'Automatisk byggesøknad',
      'Direkte kommuneintegrasjon'
    ]
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: 9999,
    features: [
      'Alt i Pro',
      'Ubegrenset analyser',
      'Dedikert support',
      'API-tilgang'
    ]
  }
];

// Stripe promise
const stripePromise = loadStripe(process.env.REACT_APP_STRIPE_PUBLIC_KEY!);

// Checkout Form Component
const CheckoutForm = () => {
  const stripe = useStripe();
  const elements = useElements();
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState(plans[0]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!stripe || !elements) {
      return;
    }

    setProcessing(true);

    try {
      // Opprett betalingssesjon
      const response = await fetch('/api/create-checkout-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          planId: selectedPlan.id,
          customerEmail: localStorage.getItem('userEmail'),
          successUrl: `${window.location.origin}/success`,
          cancelUrl: `${window.location.origin}/cancel`,
        }),
      });

      const session = await response.json();

      if (session.error) {
        setError(session.error.message);
        setProcessing(false);
        return;
      }

      // Redirect til Stripe Checkout
      const { error: stripeError } = await stripe.redirectToCheckout({
        sessionId: session.sessionId,
      });

      if (stripeError) {
        setError(stripeError.message);
      }
    } catch (err) {
      setError('Det oppstod en feil ved behandling av betalingen.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        {plans.map(plan => (
          <PlanCard
            key={plan.id}
            selected={selectedPlan.id === plan.id}
            onClick={() => setSelectedPlan(plan)}
          >
            <h3>{plan.name}</h3>
            <p>{plan.price} NOK/mnd</p>
            <ul>
              {plan.features.map(feature => (
                <li key={feature}>{feature}</li>
              ))}
            </ul>
          </PlanCard>
        ))}
      </div>

      <PaymentElement />

      {error && <ErrorMessage>{error}</ErrorMessage>}

      <Button type="submit" disabled={!stripe || processing}>
        {processing ? 'Behandler...' : `Betal ${selectedPlan.price} NOK`}
      </Button>
    </form>
  );
};

// Main Payment Component
const PaymentComponent: React.FC = () => {
  const [clientSecret, setClientSecret] = useState('');

  useEffect(() => {
    // Hent betalingsintensjons client secret fra backend
    const fetchClientSecret = async () => {
      try {
        const response = await fetch('/api/create-payment-intent', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            planId: plans[0].id,
          }),
        });
        const data = await response.json();
        setClientSecret(data.clientSecret);
      } catch (err) {
        console.error('Feil ved henting av client secret:', err);
      }
    };

    fetchClientSecret();
  }, []);

  return (
    <PaymentContainer>
      <h2>Velg abonnement</h2>
      {clientSecret && (
        <Elements
          stripe={stripePromise}
          options={{
            clientSecret,
            appearance: {
              theme: 'stripe',
              variables: {
                colorPrimary: '#0066cc',
              },
            },
          }}
        >
          <CheckoutForm />
        </Elements>
      )}
    </PaymentContainer>
  );
};

export default PaymentComponent;