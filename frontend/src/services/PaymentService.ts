import { loadStripe } from '@stripe/stripe-js';
import axios from 'axios';

const stripePromise = loadStripe(process.env.REACT_APP_STRIPE_PUBLIC_KEY!);

export class PaymentService {
  private static instance: PaymentService;
  private baseUrl: string;

  private constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api';
  }

  public static getInstance(): PaymentService {
    if (!PaymentService.instance) {
      PaymentService.instance = new PaymentService();
    }
    return PaymentService.instance;
  }

  async createCheckoutSession(propertyAnalysisId: string, selectedPlan: string) {
    try {
      const response = await axios.post(`${this.baseUrl}/payments/create-checkout-session`, {
        propertyAnalysisId,
        selectedPlan
      });

      const { sessionId } = response.data;
      const stripe = await stripePromise;

      if (!stripe) {
        throw new Error('Stripe kunne ikke initialiseres');
      }

      const { error } = await stripe.redirectToCheckout({ sessionId });
      
      if (error) {
        throw new Error(error.message);
      }
    } catch (error: any) {
      console.error('Betalingsfeil:', error);
      throw new Error(error.response?.data?.message || 'Kunne ikke prosessere betalingen');
    }
  }

  async getSubscriptionStatus(userId: string) {
    try {
      const response = await axios.get(`${this.baseUrl}/payments/subscription-status/${userId}`);
      return response.data;
    } catch (error: any) {
      console.error('Feil ved henting av abonnementsstatus:', error);
      throw new Error(error.response?.data?.message || 'Kunne ikke hente abonnementsstatus');
    }
  }

  async cancelSubscription(subscriptionId: string) {
    try {
      const response = await axios.post(`${this.baseUrl}/payments/cancel-subscription`, {
        subscriptionId
      });
      return response.data;
    } catch (error: any) {
      console.error('Feil ved kansellering av abonnement:', error);
      throw new Error(error.response?.data?.message || 'Kunne ikke kansellere abonnementet');
    }
  }

  async createCustomPortalSession(customerId: string) {
    try {
      const response = await axios.post(`${this.baseUrl}/payments/create-portal-session`, {
        customerId
      });
      return response.data;
    } catch (error: any) {
      console.error('Feil ved opprettelse av kundeportal:', error);
      throw new Error(error.response?.data?.message || 'Kunne ikke opprette kundeportal');
    }
  }
}