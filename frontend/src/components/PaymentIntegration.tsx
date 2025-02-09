import React, { useState, useEffect } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import {
  Elements,
  CardElement,
  useStripe,
  useElements,
} from '@stripe/react-stripe-js';
import axios from 'axios';

// Laster Stripe med vår publishable key
const stripePromise = loadStripe(process.env.REACT_APP_STRIPE_PUBLISHABLE_KEY || '');

const CheckoutForm = ({ onSuccess, price, productId }) => {
  const stripe = useStripe();
  const elements = useElements();
  const [error, setError] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [succeeded, setSucceeded] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setProcessing(true);

    if (!stripe || !elements) {
      return;
    }

    try {
      // Hent betalingshensikt fra backend
      const { data: { clientSecret } } = await axios.post('/api/create-payment-intent', {
        amount: price,
        productId,
      });

      // Bekreft betalingen
      const payload = await stripe.confirmCardPayment(clientSecret, {
        payment_method: {
          card: elements.getElement(CardElement),
        },
      });

      if (payload.error) {
        setError(`Betalingsfeil: ${payload.error.message}`);
        setProcessing(false);
      } else {
        setError(null);
        setSucceeded(true);
        setProcessing(false);
        onSuccess(payload);
      }
    } catch (err) {
      setError('Det oppstod en feil ved behandling av betalingen.');
      setProcessing(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="payment-form">
      <div className="form-row">
        <CardElement
          options={{
            style: {
              base: {
                fontSize: '16px',
                color: '#424770',
                '::placeholder': {
                  color: '#aab7c4',
                },
              },
              invalid: {
                color: '#9e2146',
              },
            },
          }}
        />
      </div>
      {error && <div className="error-message">{error}</div>}
      <button
        type="submit"
        disabled={!stripe || processing || succeeded}
        className="submit-button"
      >
        {processing ? 'Behandler...' : 'Betal nå'}
      </button>
      {succeeded && (
        <div className="success-message">
          Betalingen var vellykket! Analyserer eiendom...
        </div>
      )}
    </form>
  );
};

const PaymentIntegration = ({ price, productId, onPaymentSuccess }) => {
  return (
    <div className="payment-container">
      <Elements stripe={stripePromise}>
        <CheckoutForm
          price={price}
          productId={productId}
          onSuccess={onPaymentSuccess}
        />
      </Elements>
    </div>
  );
};

export default PaymentIntegration;