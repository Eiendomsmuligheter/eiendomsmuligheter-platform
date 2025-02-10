import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import styled from 'styled-components';

// Styled components
const AuthContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
`;

const Button = styled.button`
  background: #0066cc;
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.3s ease;
  margin: 0.5rem;

  &:hover {
    background: #0052a3;
  }
`;

const UserProfile = styled.div`
  background: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-top: 2rem;
  width: 100%;
  max-width: 600px;
`;

const Avatar = styled.img`
  width: 100px;
  height: 100px;
  border-radius: 50%;
  margin-bottom: 1rem;
`;

const LoadingSpinner = styled.div`
  border: 4px solid #f3f3f3;
  border-top: 4px solid #0066cc;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const AuthComponent: React.FC = () => {
  const {
    isLoading,
    isAuthenticated,
    error,
    user,
    loginWithRedirect,
    logout,
    getAccessTokenSilently
  } = useAuth0();

  const handleLogin = () => {
    loginWithRedirect({
      appState: {
        returnTo: window.location.pathname
      }
    });
  };

  const handleLogout = () => {
    logout({
      returnTo: window.location.origin
    });
  };

  // Funksjon for å hente brukerens token
  const getToken = async () => {
    try {
      const token = await getAccessTokenSilently({
        audience: process.env.REACT_APP_AUTH0_AUDIENCE,
        scope: "read:current_user update:current_user_metadata"
      });
      return token;
    } catch (error) {
      console.error('Feil ved henting av token:', error);
      return null;
    }
  };

  if (isLoading) {
    return (
      <AuthContainer>
        <LoadingSpinner />
      </AuthContainer>
    );
  }

  if (error) {
    return (
      <AuthContainer>
        <div>Autentiseringsfeil: {error.message}</div>
      </AuthContainer>
    );
  }

  return (
    <AuthContainer>
      {!isAuthenticated ? (
        <Button onClick={handleLogin}>
          Logg inn
        </Button>
      ) : (
        <>
          <UserProfile>
            {user?.picture && (
              <Avatar src={user.picture} alt={user.name || 'Profilbilde'} />
            )}
            <h2>Velkommen, {user?.name}!</h2>
            <p>E-post: {user?.email}</p>
            {user?.sub && <p>Bruker-ID: {user.sub}</p>}
            {user?.email_verified && (
              <p>✅ E-post er verifisert</p>
            )}
          </UserProfile>
          <Button onClick={handleLogout}>
            Logg ut
          </Button>
        </>
      )}
    </AuthContainer>
  );
};

export default AuthComponent;