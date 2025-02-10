interface Auth0Config {
  domain: string;
  clientId: string;
  audience: string;
  scope: string;
}

export const auth0Config: Auth0Config = {
  domain: process.env.REACT_APP_AUTH0_DOMAIN || '',
  clientId: process.env.REACT_APP_AUTH0_CLIENT_ID || '',
  audience: process.env.REACT_APP_AUTH0_AUDIENCE || '',
  scope: 'read:current_user update:current_user_metadata'
};

export const getAuth0Config = (): Auth0Config => {
  // Valider konfigurasjon
  if (!auth0Config.domain || !auth0Config.clientId) {
    throw new Error('Auth0 konfigurasjon mangler! Sjekk miljøvariabler.');
  }
  return auth0Config;
};