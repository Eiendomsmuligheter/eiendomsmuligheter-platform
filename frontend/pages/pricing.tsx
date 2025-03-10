import React from 'react';
import Head from 'next/head';
import { Layout } from '../components/common/Layout';
import { SubscriptionPlans } from '../components/subscription/SubscriptionPlans';

const PricingPage = () => {
  return (
    <>
      <Head>
        <title>Priser og abonnementer | Eiendomsmuligheter</title>
        <meta name="description" content="Finn det abonnementet som passer deg best for å få mest mulig ut av Eiendomsmuligheter. Vi tilbyr abonnementer for privatpersoner, profesjonelle og samarbeidspartnere." />
      </Head>
      
      <Layout>
        <SubscriptionPlans />
      </Layout>
    </>
  );
};

export default PricingPage; 