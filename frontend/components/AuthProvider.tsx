import React, { createContext, useContext, useEffect, useState } from 'react';
import { AuthService } from '../services/AuthService';

interface AuthContextType {
  isAuthenticated: boolean;
  user: any;
  loading: boolean;
  error: string | null;
  login: () => Promise<void>;
  logout: () => Promise<void>;
  getToken: () => Promise<string>;
}

const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  user: null,
  loading: true,
  error: null,
  login: async () => {},
  logout: async () => {},
  getToken: async () => '',
});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const authService = AuthService.getInstance();
  
  useEffect(() => {
    const initAuth = async () => {
      try {
        // Håndter redirect callback
        if (window.location.search.includes('code=')) {
          await authService.handleRedirectCallback();
          window.history.replaceState({}, document.title, window.location.pathname);
        }
        
        // Sjekk autentisering og hent bruker
        const authenticated = await authService.isAuthenticated();
        setIsAuthenticated(authenticated);
        
        if (authenticated) {
          const userData = await authService.getUser();
          setUser(userData);
        }
      } catch (err) {
        setError('Failed to initialize authentication');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    initAuth();
  }, []);
  
  const login = async () => {
    try {
      await authService.login();
    } catch (err) {
      setError('Failed to log in');
      console.error(err);
    }
  };
  
  const logout = async () => {
    try {
      await authService.logout();
      setIsAuthenticated(false);
      setUser(null);
    } catch (err) {
      setError('Failed to log out');
      console.error(err);
    }
  };
  
  const getToken = async () => {
    try {
      return await authService.getToken();
    } catch (err) {
      setError('Failed to get token');
      console.error(err);
      return '';
    }
  };
  
  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        user,
        loading,
        error,
        login,
        logout,
        getToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};