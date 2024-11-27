import { useState } from 'react';
import Link from 'next/link';
import { useAuth } from '../contexts/AuthContext';

export default function Navbar() {
  const { user, logout } = useAuth();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <nav className="bg-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between h-16">
          <div className="flex-shrink-0 flex items-center">
            <Link href="/">
              <a className="text-xl font-bold text-blue-600">EiendomsMuligheter.no</a>
            </Link>
          </div>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-8">
            <Link href="/#hvordan">
              <a className="text-gray-600 hover:text-blue-600">Hvordan det fungerer</a>
            </Link>
            <Link href="/#priser">
              <a className="text-gray-600 hover:text-blue-600">Priser</a>
            </Link>
            
            {user ? (
              <>
                <Link href="/dashboard">
                  <a className="text-gray-600 hover:text-blue-600">Dashboard</a>
                </Link>
                <button
                  onClick={logout}
                  className="text-gray-600 hover:text-blue-600"
                >
                  Logg ut
                </button>
              </>
            ) : (
              <>
                <Link href="/login">
                  <a className="text-gray-600 hover:text-blue-600">Logg inn</a>
                </Link>
                <Link href="/register">
                  <a className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
                    Registrer deg
                  </a>
                </Link>
              </>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
            >
              <span className="sr-only">Åpne meny</span>
              {/* Hamburger icon */}
              <svg
                className={`${isMenuOpen ? 'hidden' : 'block'} h-6 w-6`}
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
              {/* Close icon */}
              <svg
                className={`${isMenuOpen ? 'block' : 'hidden'} h-6 w-6`}
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div className={`${isMenuOpen ? 'block' : 'hidden'} md:hidden`}>
        <div className="px-2 pt-2 pb-3 space-y-1">
          <Link href="/#hvordan">
            <a className="block px-3 py-2 text-gray-600 hover:text-blue-600">
              Hvordan det fungerer
            </a>
          </Link>
          <Link href="/#priser">
            <a className="block px-3 py-2 text-gray-600 hover:text-blue-600">
              Priser
            </a>
          </Link>
          
          {user ? (
            <>
              <Link href="/dashboard">
                <a className="block px-3 py-2 text-gray-600 hover:text-blue-600">
                  Dashboard
                </a>
              </Link>
              <button
                onClick={logout}
                className="block w-full text-left px-3 py-2 text-gray-600 hover:text-blue-600"
              >
                Logg ut
              </button>
            </>
          ) : (
            <>
              <Link href="/login">
                <a className="block px-3 py-2 text-gray-600 hover:text-blue-600">
                  Logg inn
                </a>
              </Link>
              <Link href="/register">
                <a className="block px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                  Registrer deg
                </a>
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}