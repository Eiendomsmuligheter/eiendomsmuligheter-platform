import React, { useState, useEffect } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import {
  Elements,
  CardElement,
  useStripe,
  useElements,
} from '@stripe/stripe-js/pure';
import axios from 'axios';
import styled from 'styled-components';

// Styled components
const PlanContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
`;

const PlanCard = styled.div<{ featured?: boolean }>`
  background: ${props => props.featured ? 'linear-gradient(135deg, #00b4db 0%, #0083b0 100%)' : 'rgba(255, 255, 255, 0.05)'};
  border-radius: 20px;
  padding: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  transition: transform 0.3s ease;
  border: 1px solid rgba(255, 255, 255, 0.1);
  
  &:hover {
    transform: translateY(-5px);
  }
`;

const PlanTitle = styled.h2`
  color: #ffffff;
  margin-bottom: 1rem;
  font-size: 1.8rem;
`;

const PlanPrice = styled.div`
  font-size: 2.5rem;
  font-weight: bold;
  color: #4dd0e1;
  margin: 1rem 0;
  
  span {
    font-size: 1rem;
    opacity: 0.8;
  }
`;

const FeatureList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 2rem 0;
  text-align: left;
  
  li {
    margin: 0.8rem 0;
    display: flex;
    align-items: center;
    
    &:before {
      content: "✓";
      color: #4dd0e1;
      margin-right: 0.5rem;
    }
  }
`;

const SubscribeButton = styled.button<{ primary?: boolean }>`
  background: ${props => props.primary ? '#4dd0e1' : 'transparent'};
  color: white;
  border: ${props => props.primary ? 'none' : '2px solid #4dd0e1'};
  padding: 1rem 2rem;
  border-radius: 30px;
  font-size: 1.1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(77, 208, 225, 0.3);
  }
`;

const CardContainer = styled.div`
  background: rgba(255, 255, 255, 0.05);
  padding: 1rem;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin: 1rem 0;
`;

interface Plan {
  id: string;
  name: string;
  price: number;
  currency: string;
  interval: string;
  features: string[];
}

interface PaymentPlansProps {
  onSubscriptionComplete: (subscriptionId: string) => void;
}

const PaymentPlans: React.FC<PaymentPlansProps> = ({ onSubscriptionComplete }) => {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchPlans = async () => {
      try {
        const response = await axios.get('/api/plans');
        setPlans(response.data);
      } catch (err) {
        setError('Kunne ikke laste abonnementsplaner');
        console.error('Error fetching plans:', err);
      }
    };
    
    fetchPlans();
  }, []);

  const handleSubscribe = async (plan: Plan) => {
    setSelectedPlan(plan);
  };

  return (
    <div>
      <h1>Velg abonnementsplan</h1>
      
      {error && (
        <div style={{ color: 'red', margin: '1rem 0' }}>
          {error}
        </div>
      )}
      
      <PlanContainer>
        {plans.map((plan) => (
          <PlanCard key={plan.id} featured={plan.name === 'Professional'}>
            <PlanTitle>{plan.name}</PlanTitle>
            <PlanPrice>
              {plan.price} {plan.currency}
              <span>/{plan.interval}</span>
            </PlanPrice>
            
            <FeatureList>
              {plan.features.map((feature, index) => (
                <li key={index}>{feature}</li>
              ))}
            </FeatureList>
            
            <SubscribeButton
              primary={plan.name === 'Professional'}
              onClick={() => handleSubscribe(plan)}
              disabled={loading}
            >
              {loading ? 'Behandler...' : 'Velg plan'}
            </SubscribeButton>
          </PlanCard>
        ))}
      </PlanContainer>

      {selectedPlan && (
        <PaymentForm
          plan={selectedPlan}
          onSubscriptionComplete={onSubscriptionComplete}
          onCancel={() => setSelectedPlan(null)}
        />
      )}
    </div>
  );
};

interface PaymentFormProps {
  plan: Plan;
  onSubscriptionComplete: (subscriptionId: string) => void;
  onCancel: () => void;
}

const PaymentForm: React.FC<PaymentFormProps> = ({
  plan,
  onSubscriptionComplete,
  onCancel,
}) => {
  const stripe = useStripe();
  const elements = useElements();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (!stripe || !elements) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Create payment method
      const { error: stripeError, paymentMethod } = await stripe.createPaymentMethod({
        type: 'card',
        card: elements.getElement(CardElement)!,
      });

      if (stripeError) {
        throw new Error(stripeError.message);
      }

      // Create subscription
      const response = await axios.post('/api/create-subscription', {
        paymentMethodId: paymentMethod.id,
        planId: plan.id,
      });

      const { subscriptionId, clientSecret } = response.data;

      // Confirm payment if required
      if (clientSecret) {
        const { error: confirmError } = await stripe.confirmCardPayment(clientSecret);
        if (confirmError) {
          throw new Error(confirmError.message);
        }
      }

      onSubscriptionComplete(subscriptionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'En feil oppstod');
      console.error('Payment error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h3>Betalingsinformasjon</h3>
      
      <CardContainer>
        <CardElement
          options={{
            style: {
              base: {
                fontSize: '16px',
                color: '#ffffff',
                '::placeholder': {
                  color: '#aab7c4',
                },
              },
            },
          }}
        />
      </CardContainer>

      {error && (
        <div style={{ color: 'red', margin: '1rem 0' }}>
          {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
        <SubscribeButton type="submit" primary disabled={loading}>
          {loading ? 'Behandler...' : 'Fullfør betaling'}
        </SubscribeButton>
        
        <SubscribeButton type="button" onClick={onCancel} disabled={loading}>
          Avbryt
        </SubscribeButton>
      </div>
    </form>
  );
};

// Wrap with Stripe Elements
const stripePromise = loadStripe(process.env.REACT_APP_STRIPE_PUBLIC_KEY!);

export const PaymentPlansWrapper: React.FC<PaymentPlansProps> = (props) => (
  <Elements stripe={stripePromise}>
    <PaymentPlans {...props} />
  </Elements>
);

export default PaymentPlansWrapper;