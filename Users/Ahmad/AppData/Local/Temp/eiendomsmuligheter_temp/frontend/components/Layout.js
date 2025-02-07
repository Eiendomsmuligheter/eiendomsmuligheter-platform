import Head from 'next/head';
import Navbar from './Navbar';
import Footer from './Footer';

export default function Layout({ children }) {
  return (
    <>
      <Head>
        <title>EiendomsMuligheter.no - Finn utleiepotensial i din eiendom</title>
        <meta name="description" content="Bruk vår AI-drevne teknologi for å analysere din eiendom og finn skjulte utleiemuligheter" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-grow">
          {children}
        </main>
        <Footer />
      </div>
    </>
  );
}