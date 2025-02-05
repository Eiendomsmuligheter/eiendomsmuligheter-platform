import React, { createContext, useContext, useState, useEffect } from 'react';
import { Auth0Provider, useAuth0 } from '@auth0/auth0-react';
import { User } from '../types/user';
import { CircularProgress, Container } from '@mui/material';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: Error | null;
  loginWithRedirect: () => Promise<void>;
  logout: () => void;
  getAccessToken: () => Promise<string>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <Auth0Provider
      domain="eiendomsmuligheter.eu.auth0.com"
      clientId="YOUR_CLIENT_ID"
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience: "https://api.eiendomsmuligheter.no",
        scope: "openid profile email"
      }}
    >
      <Auth0ContextProvider>
        {children}
      </Auth0ContextProvider>
    </Auth0Provider>
  );
};

const Auth0ContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const {
    user: auth0User,
    isAuthenticated,
    isLoading,
    error,
    loginWithRedirect,
    logout: auth0Logout,
    getAccessTokenSilently
  } = useAuth0();

  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (auth0User && isAuthenticated) {
      // Transform Auth0 user to our User type
      const transformedUser: User = {
        id: auth0User.sub!,
        email: auth0User.email!,
        name: auth0User.name || auth0User.email!,
        picture: auth0User.picture,
        subscription: null, // Will be fetched from our backend
        roles: auth0User['https://eiendomsmuligheter.no/roles'] || []
      };
      setUser(transformedUser);

      // Fetch additional user data from our backend
      fetchUserData(transformedUser.id).then(userData => {
        setUser(prev => ({ ...prev!, ...userData }));
      });
    } else {
      setUser(null);
    }
  }, [auth0User, isAuthenticated]);

  const fetchUserData = async (userId: string) => {
    try {
      const token = await getAccessTokenSilently();
      const response = await fetch(`/api/users/${userId}`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      if (!response.ok) throw new Error('Failed to fetch user data');
      return await response.json();
    } catch (error) {
      console.error('Error fetching user data:', error);
      return {};
    }
  };

  const logout = () => {
    auth0Logout({ 
      logoutParams: {
        returnTo: window.location.origin 
      }
    });
  };

  const getAccessToken = () => getAccessTokenSilently();

  if (isLoading) {
    return (
      <Container sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Container>
    );
  }

  const value = {
    user,
    isAuthenticated,
    isLoading,
    error,
    loginWithRedirect,
    logout,
    getAccessToken
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Protected Route Component
export const ProtectedRoute: React.FC<{ 
  children: React.ReactNode,
  requiredRoles?: string[] 
}> = ({ children, requiredRoles = [] }) => {
  const { isAuthenticated, isLoading, user, loginWithRedirect } = useAuth();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      loginWithRedirect();
    }
  }, [isLoading, isAuthenticated, loginWithRedirect]);

  if (isLoading) {
    return (
      <Container sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (!isAuthenticated || !user) {
    return null;
  }

  if (requiredRoles.length > 0 && !requiredRoles.some(role => user.roles.includes(role))) {
    return (
      <Container>
        <h1>Ingen tilgang</h1>
        <p>Du har ikke tilstrekkelige rettigheter til å se denne siden.</p>
      </Container>
    );
  }

  return <>{children}</>;
};