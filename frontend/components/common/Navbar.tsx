import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { Menu, X, ChevronDown, CreditCard } from 'lucide-react';

export const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [servicesMenuOpen, setServicesMenuOpen] = useState(false);
  const router = useRouter();

  const isActive = (path) => {
    return router.pathname === path;
  };

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  const toggleServicesMenu = () => {
    setServicesMenuOpen(!servicesMenuOpen);
  };

  return (
    <header className="border-b border-white/10 bg-black/50 backdrop-blur-lg sticky top-0 z-10">
      <div className="container mx-auto px-4 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <Link href="/" className="text-2xl font-bold text-white">
              Eiendomsmuligheter
            </Link>
            
            {/* Desktop Navigation */}
            <nav className="hidden md:flex space-x-6 ml-10">
              <Link href="/" className={`transition-colors ${isActive('/') ? 'text-blue-400' : 'text-gray-400 hover:text-white'}`}>
                Hjem
              </Link>
              
              <div className="relative">
                <button 
                  className={`flex items-center transition-colors ${
                    isActive('/analyse') || isActive('/kart') || isActive('/rentals') 
                      ? 'text-blue-400' 
                      : 'text-gray-400 hover:text-white'
                  }`}
                  onClick={toggleServicesMenu}
                >
                  Tjenester
                  <ChevronDown size={16} className="ml-1" />
                </button>
                
                {/* Services Dropdown */}
                {servicesMenuOpen && (
                  <div className="absolute left-0 mt-2 w-56 bg-gray-900 border border-white/10 rounded-lg shadow-lg overflow-hidden z-20">
                    <div className="py-2">
                      <Link 
                        href="/analyse" 
                        className={`block px-4 py-2 text-sm ${isActive('/analyse') ? 'bg-blue-900/20 text-blue-400' : 'text-gray-300 hover:bg-white/5'}`}
                      >
                        Eiendomsanalyse
                      </Link>
                      <Link 
                        href="/kart" 
                        className={`block px-4 py-2 text-sm ${isActive('/kart') ? 'bg-blue-900/20 text-blue-400' : 'text-gray-300 hover:bg-white/5'}`}
                      >
                        Muligheter på kart
                      </Link>
                      <Link 
                        href="/rentals" 
                        className={`block px-4 py-2 text-sm ${isActive('/rentals') ? 'bg-blue-900/20 text-blue-400' : 'text-gray-300 hover:bg-white/5'}`}
                      >
                        Leieobjekter
                      </Link>
                    </div>
                  </div>
                )}
              </div>
              
              <Link 
                href="/partners" 
                className={`transition-colors ${isActive('/partners') ? 'text-blue-400' : 'text-gray-400 hover:text-white'}`}
              >
                Partnere
              </Link>
              
              <Link 
                href="/pricing" 
                className={`transition-colors flex items-center ${isActive('/pricing') ? 'text-blue-400' : 'text-gray-400 hover:text-white'}`}
              >
                <CreditCard size={16} className="mr-1" />
                Priser
              </Link>
              
              <Link 
                href="/om-oss" 
                className={`transition-colors ${isActive('/om-oss') ? 'text-blue-400' : 'text-gray-400 hover:text-white'}`}
              >
                Om oss
              </Link>
            </nav>
          </div>
          
          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
            <button 
              onClick={toggleMobileMenu}
              className="text-gray-400 hover:text-white focus:outline-none"
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
          
          {/* Auth Buttons */}
          <div className="hidden md:flex items-center space-x-4">
            <Link 
              href="/login" 
              className="px-4 py-2 rounded-lg border border-white/10 hover:bg-white/5 transition-colors text-sm font-medium text-white"
            >
              Logg inn
            </Link>
            <Link 
              href="/registrer" 
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transition-colors text-sm font-medium text-white"
            >
              Registrer deg
            </Link>
          </div>
        </div>
        
        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden mt-4 pb-4">
            <nav className="flex flex-col space-y-4">
              <Link 
                href="/" 
                className={`py-2 ${isActive('/') ? 'text-blue-400' : 'text-gray-300'}`}
                onClick={toggleMobileMenu}
              >
                Hjem
              </Link>
              
              <button 
                className={`flex justify-between items-center py-2 ${
                  isActive('/analyse') || isActive('/kart') || isActive('/rentals') 
                    ? 'text-blue-400' 
                    : 'text-gray-300'
                }`}
                onClick={() => setServicesMenuOpen(!servicesMenuOpen)}
              >
                Tjenester
                <ChevronDown size={16} className={`transition-transform ${servicesMenuOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {servicesMenuOpen && (
                <div className="pl-4 border-l border-white/10 space-y-2 my-2">
                  <Link 
                    href="/analyse" 
                    className={`block py-2 text-sm ${isActive('/analyse') ? 'text-blue-400' : 'text-gray-300'}`}
                    onClick={toggleMobileMenu}
                  >
                    Eiendomsanalyse
                  </Link>
                  <Link 
                    href="/kart" 
                    className={`block py-2 text-sm ${isActive('/kart') ? 'text-blue-400' : 'text-gray-300'}`}
                    onClick={toggleMobileMenu}
                  >
                    Muligheter på kart
                  </Link>
                  <Link 
                    href="/rentals" 
                    className={`block py-2 text-sm ${isActive('/rentals') ? 'text-blue-400' : 'text-gray-300'}`}
                    onClick={toggleMobileMenu}
                  >
                    Leieobjekter
                  </Link>
                </div>
              )}
              
              <Link 
                href="/partners" 
                className={`py-2 ${isActive('/partners') ? 'text-blue-400' : 'text-gray-300'}`}
                onClick={toggleMobileMenu}
              >
                Partnere
              </Link>
              
              <Link 
                href="/pricing" 
                className={`py-2 flex items-center ${isActive('/pricing') ? 'text-blue-400' : 'text-gray-300'}`}
                onClick={toggleMobileMenu}
              >
                <CreditCard size={16} className="mr-2" />
                Priser
              </Link>
              
              <Link 
                href="/om-oss" 
                className={`py-2 ${isActive('/om-oss') ? 'text-blue-400' : 'text-gray-300'}`}
                onClick={toggleMobileMenu}
              >
                Om oss
              </Link>
              
              <div className="flex space-x-4 pt-2">
                <Link 
                  href="/login" 
                  className="flex-1 px-4 py-2 rounded-lg border border-white/10 text-center text-sm font-medium text-white"
                  onClick={toggleMobileMenu}
                >
                  Logg inn
                </Link>
                <Link 
                  href="/registrer" 
                  className="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-center text-sm font-medium text-white"
                  onClick={toggleMobileMenu}
                >
                  Registrer deg
                </Link>
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}; 