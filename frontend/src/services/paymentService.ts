import axios from 'axios';

export interface PaymentIntent {
  clientSecret: string;
  paymentIntentId: string;
}

export interface PaymentConfig {
  amount: number;
  currency: string;
  paymentMethodTypes: string[];
}

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const createPaymentIntent = async (config: PaymentConfig): Promise<PaymentIntent> => {
  try {
    const response = await axios.post(`${API_URL}/api/payments/create-intent`, config);
    return response.data;
  } catch (error) {
    throw new Error('Kunne ikke opprette betalingsintent');
  }
};

export const confirmPayment = async (paymentIntentId: string): Promise<boolean> => {
  try {
    const response = await axios.post(`${API_URL}/api/payments/confirm`, {
      paymentIntentId,
    });
    return response.data.success;
  } catch (error) {
    throw new Error('Kunne ikke bekrefte betaling');
  }
};

export const getSubscriptionPlans = async () => {
  try {
    const response = await axios.get(`${API_URL}/api/payments/plans`);
    return response.data;
  } catch (error) {
    throw new Error('Kunne ikke hente abonnementsplaner');
  }
};