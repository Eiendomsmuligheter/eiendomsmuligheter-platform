import React from 'react';
import Head from 'next/head';
import { Layout } from '../../components/common/Layout';
import { RentalListing } from '../../components/rentals/RentalListing';

const RentalsPage = () => {
  return (
    <>
      <Head>
        <title>Leieobjekter | Eiendomsmuligheter</title>
        <meta name="description" content="Finn din nye leiebolig med Eiendomsmuligheter. Vi har et stort utvalg av leiligheter, hus, rekkehus og hybler til leie." />
      </Head>
      
      <Layout>
        <RentalListing />
      </Layout>
    </>
  );
};

export default RentalsPage; 