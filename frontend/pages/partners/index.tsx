import React from 'react';
import Head from 'next/head';
import { Layout } from '../../components/common/Layout';
import { PartnerDashboard } from '../../components/partners/PartnerDashboard';

const PartnersPage = () => {
  return (
    <>
      <Head>
        <title>Partnere | Eiendomsmuligheter</title>
        <meta name="description" content="Bli partner med Eiendomsmuligheter og få tilgang til nye kunder innen eiendomsmarkedet" />
      </Head>
      
      <Layout>
        <PartnerDashboard />
      </Layout>
    </>
  );
};

export default PartnersPage; 