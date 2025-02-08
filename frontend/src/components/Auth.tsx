import React, { useEffect, useState, createContext, useContext } from 'react';
import { Auth0Provider, useAuth0 } from '@auth0/auth0-react';
import styled from 'styled-components';
import axios from 'axios';

// Styled components
const AuthContainer = styled.div`
  padding: 2rem;
  max-width: 400px;
  margin: 0 auto;
`;

const AuthButton = styled.button`
  background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%);
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 30px;
  font-size: 1.1rem;
  cursor: pointer;
  width: 100%;
  margin: 1rem 0;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 180, 219, 0.3);
  }

  &:disabled {
    background: #cccccc;
    cursor: not-allowed;
  }
`;

const UserProfile = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 15px;
  padding: 2rem;
  margin-top: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const UserAvatar = styled.img`
  width: 100px;
  height: 100px;
  border-radius: 50%;
  margin: 0 auto 1rem;
  display: block;
`;

const ErrorMessage = styled.div`
  color: #ff4757;
  background: rgba(255, 71, 87, 0.1);
  padding: 1rem;
  border-radius: 10px;
  margin: 1rem 0;
  text-align: center;
`;

// Create auth context
interface AuthContextType {
  isAuthenticated: boolean;
  user: any;
  loading: boolean;
  error: string | null;
  login: () => void;
  logout: () => void;
  getAccessToken: () => Promise<string>;
}

const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  user: null,
  loading: true,
  error: null,
  login: () => {},
  logout: () => {},
  getAccessToken: async () => '',
});

// Auth provider component
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const {
    isAuthenticated,
    user,
    loginWithRedirect,
    logout,
    getAccessTokenSilently,
    isLoading,
    error
  } = useAuth0();

  // Set up axios interceptor for adding auth token
  useEffect(() => {
    const interceptor = axios.interceptors.request.use(async (config) => {
      if (isAuthenticated) {
        const token = await getAccessTokenSilently();
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    return () => {
      axios.interceptors.request.eject(interceptor);
    };
  }, [isAuthenticated, getAccessTokenSilently]);

  const value = {
    isAuthenticated,
    user,
    loading: isLoading,
    error: error?.message || null,
    login: loginWithRedirect,
    logout: () => logout({ returnTo: window.location.origin }),
    getAccessToken: getAccessTokenSilently,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Hook for using auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Login component
export const Login: React.FC = () => {
  const { login, error } = useAuth();

  return (
    <AuthContainer>
      <h2>Logg inn for å fortsette</h2>
      {error && <ErrorMessage>{error}</ErrorMessage>}
      <AuthButton onClick={login}>
        Logg inn / Registrer deg
      </AuthButton>
    </AuthContainer>
  );
};

// User profile component
export const UserProfileComponent: React.FC = () => {
  const { user, logout, loading } = useAuth();
  const [userMetadata, setUserMetadata] = useState<any>(null);

  useEffect(() => {
    const fetchUserMetadata = async () => {
      try {
        const response = await axios.get('/api/auth/profile');
        setUserMetadata(response.data);
      } catch (error) {
        console.error('Error fetching user metadata:', error);
      }
    };

    if (user?.sub) {
      fetchUserMetadata();
    }
  }, [user?.sub]);

  if (loading) {
    return <div>Laster...</div>;
  }

  return (
    <UserProfile>
      {user?.picture && <UserAvatar src={user.picture} alt="Profile" />}
      <h3>{user?.name}</h3>
      <p>{user?.email}</p>
      {userMetadata && (
        <div>
          <p>Selskap: {userMetadata.company || 'Ikke angitt'}</p>
          <p>Medlemskap: {userMetadata.subscription_type || 'Basic'}</p>
        </div>
      )}
      <AuthButton onClick={logout}>Logg ut</AuthButton>
    </UserProfile>
  );
};

// Auth0 configuration wrapper
export const Auth0Wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <Auth0Provider
      domain={process.env.REACT_APP_AUTH0_DOMAIN!}
      clientId={process.env.REACT_APP_AUTH0_CLIENT_ID!}
      redirectUri={window.location.origin}
      audience={process.env.REACT_APP_AUTH0_AUDIENCE}
      scope="openid profile email"
    >
      <AuthProvider>{children}</AuthProvider>
    </Auth0Provider>
  );
};

// Protected route wrapper
export const ProtectedRoute: React.FC<{
  children: React.ReactNode;
  requiredRoles?: string[];
}> = ({ children, requiredRoles }) => {
  const { isAuthenticated, user, loading } = useAuth();
  const [hasAccess, setHasAccess] = useState(false);

  useEffect(() => {
    const checkAccess = async () => {
      if (!isAuthenticated || !user) {
        setHasAccess(false);
        return;
      }

      if (!requiredRoles) {
        setHasAccess(true);
        return;
      }

      try {
        const response = await axios.get('/api/auth/permissions');
        const userPermissions = response.data.permissions;
        
        const hasRequiredRole = requiredRoles.some(role => 
          userPermissions.includes(`access:${role}`) || userPermissions.includes('*')
        );
        
        setHasAccess(hasRequiredRole);
      } catch (error) {
        console.error('Error checking permissions:', error);
        setHasAccess(false);
      }
    };

    checkAccess();
  }, [isAuthenticated, user, requiredRoles]);

  if (loading) {
    return <div>Laster...</div>;
  }

  if (!isAuthenticated) {
    return <Login />;
  }

  if (!hasAccess) {
    return (
      <ErrorMessage>
        Du har ikke tilgang til denne siden. 
        Oppgrader medlemskapet ditt for å få tilgang.
      </ErrorMessage>
    );
  }

  return <>{children}</>;
};

export default Auth0Wrapper;