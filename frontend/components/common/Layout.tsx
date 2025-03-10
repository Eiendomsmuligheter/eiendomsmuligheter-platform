import React from 'react';
import { Navbar } from './Navbar';
import Link from 'next/link';

const Footer = () => {
  return (
    <footer className="border-t border-white/10 py-12 bg-black/50">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-xl font-bold text-white mb-4">Eiendomsmuligheter</h3>
            <p className="text-gray-400">
              Vi hjelper deg å finne og utnytte eiendomsmuligheter med avansert AI-analyse.
            </p>
          </div>
          <div>
            <h4 className="text-lg font-medium text-white mb-4">Tjenester</h4>
            <ul className="space-y-2 text-gray-400">
              <li><Link href="/analyse" className="hover:text-white transition-colors">Eiendomsanalyse</Link></li>
              <li><Link href="/kart" className="hover:text-white transition-colors">Muligheter på kart</Link></li>
              <li><Link href="/rentals" className="hover:text-white transition-colors">Leieobjekter</Link></li>
              <li><Link href="/partners" className="hover:text-white transition-colors">Bli partner</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="text-lg font-medium text-white mb-4">Selskapet</h4>
            <ul className="space-y-2 text-gray-400">
              <li><Link href="/om-oss" className="hover:text-white transition-colors">Om oss</Link></li>
              <li><Link href="/kontakt" className="hover:text-white transition-colors">Kontakt</Link></li>
              <li><Link href="/karriere" className="hover:text-white transition-colors">Karriere</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="text-lg font-medium text-white mb-4">Juridisk</h4>
            <ul className="space-y-2 text-gray-400">
              <li><Link href="/personvern" className="hover:text-white transition-colors">Personvernerklæring</Link></li>
              <li><Link href="/vilkar" className="hover:text-white transition-colors">Brukervilkår</Link></li>
              <li><Link href="/cookies" className="hover:text-white transition-colors">Cookies</Link></li>
            </ul>
          </div>
        </div>
        <div className="border-t border-white/10 mt-12 pt-8 text-center text-gray-500">
          <p>© {new Date().getFullYear()} Eiendomsmuligheter AS. Alle rettigheter reservert.</p>
        </div>
      </div>
    </footer>
  );
};

export const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
      <Navbar />
      <main>
        {children}
      </main>
      <Footer />
    </div>
  );
}; 