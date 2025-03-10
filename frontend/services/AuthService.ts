import { Auth0Client, Auth0ClientOptions, RedirectLoginOptions, LogoutOptions, GetTokenSilentlyOptions, User } from '@auth0/auth0-spa-js';

export interface AuthUser extends User {
  email?: string;
  email_verified?: boolean;
  name?: string;
  nickname?: string;
  picture?: string;
  sub?: string;
  updated_at?: string;
}

export interface LoginOptions extends RedirectLoginOptions {
  screen_hint?: 'signup' | 'login';
}

class AuthError extends Error {
  constructor(message: string, public originalError?: Error) {
    super(message);
    this.name = 'AuthError';
    
    // Behold original stack trace hvis tilgjengelig
    if (originalError && originalError.stack) {
      this.stack = originalError.stack;
    }
  }
}

/**
 * AuthService - Håndterer autentisering mot Auth0
 */
export class AuthService {
  private static instance: AuthService;
  private auth0: Auth0Client;
  private initialized: boolean = false;
  private initPromise: Promise<void> | null = null;
  
  /**
   * Privat konstruktør for singleton-mønster
   */
  private constructor() {
    const domain = process.env.NEXT_PUBLIC_AUTH0_DOMAIN;
    const clientId = process.env.NEXT_PUBLIC_AUTH0_CLIENT_ID;
    const audience = process.env.NEXT_PUBLIC_AUTH0_AUDIENCE;
    
    if (!domain || !clientId) {
      throw new AuthError('Auth0 konfigurasjonen mangler. Sjekk miljøvariablene.');
    }
    
    const options: Auth0ClientOptions = {
      domain,
      clientId,
      authorizationParams: {
        redirect_uri: typeof window !== 'undefined' ? window.location.origin : '',
        audience
      },
      cacheLocation: 'localstorage',
      useRefreshTokens: true
    };
    
    this.auth0 = new Auth0Client(options);
    this.initializeClient();
  }
  
  /**
   * Initialiserer Auth0-klienten
   */
  private initializeClient(): void {
    if (this.initialized || this.initPromise) return;
    
    // Kjør initialiseringen bare én gang
    this.initPromise = (async () => {
      try {
        // Håndter callback hvis vi kommer fra en innloggingsredireksjon
        if (typeof window !== 'undefined' && 
            window.location.search.includes('code=') && 
            window.location.search.includes('state=')) {
          await this.auth0.handleRedirectCallback();
          
          // Fjern URL-parametere
          const redirectUrl = window.location.pathname;
          window.history.replaceState({}, document.title, redirectUrl);
        }
        
        this.initialized = true;
      } catch (error) {
        this.initialized = false;
        console.error('Feil ved initialisering av Auth0:', error);
      } finally {
        this.initPromise = null;
      }
    })();
  }
  
  /**
   * Returnerer en singleton-instans av AuthService
   */
  public static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }
  
  /**
   * Venter på initialisering av Auth0-klienten
   */
  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    if (this.initPromise) await this.initPromise;
    if (!this.initialized) {
      throw new AuthError('Auth0-klienten kunne ikke initialiseres.');
    }
  }
  
  /**
   * Omdirigerer brukeren til Auth0-innloggingssiden
   */
  async login(options?: LoginOptions): Promise<void> {
    try {
      await this.ensureInitialized();
      
      // Forbered innloggingsalternativer
      const loginOptions: RedirectLoginOptions = {
        authorizationParams: {
          redirect_uri: typeof window !== 'undefined' ? window.location.origin : '',
          ...options?.authorizationParams
        },
        ...options
      };
      
      await this.auth0.loginWithRedirect(loginOptions);
    } catch (error: any) {
      console.error('Feil ved innlogging:', error);
      throw new AuthError('Kunne ikke starte innloggingsprosessen.', error);
    }
  }
  
  /**
   * Logger ut brukeren
   */
  async logout(options?: LogoutOptions): Promise<void> {
    try {
      await this.ensureInitialized();
      
      // Forbered utloggingsalternativer
      const logoutOptions: LogoutOptions = {
        logoutParams: {
          returnTo: typeof window !== 'undefined' ? window.location.origin : '',
          ...options?.logoutParams
        },
        ...options
      };
      
      await this.auth0.logout(logoutOptions);
      
      // Fjern eventuelle lokale tokens
      if (typeof localStorage !== 'undefined') {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
      }
    } catch (error: any) {
      console.error('Feil ved utlogging:', error);
      throw new AuthError('Kunne ikke logge ut.', error);
    }
  }
  
  /**
   * Håndterer redirect-callback etter innlogging
   */
  async handleRedirectCallback(): Promise<void> {
    try {
      await this.auth0.handleRedirectCallback();
      
      // Lagre token lokalt for bruk i API-kall
      const token = await this.auth0.getTokenSilently();
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('auth_token', token);
      }
    } catch (error: any) {
      console.error('Feil ved håndtering av redirect:', error);
      throw new AuthError('Kunne ikke fullføre innloggingen.', error);
    }
  }
  
  /**
   * Sjekker om brukeren er autentisert
   */
  async isAuthenticated(): Promise<boolean> {
    try {
      await this.ensureInitialized();
      return await this.auth0.isAuthenticated();
    } catch (error: any) {
      console.error('Feil ved sjekk av autentisering:', error);
      return false;
    }
  }
  
  /**
   * Henter informasjon om innlogget bruker
   */
  async getUser<T = AuthUser>(): Promise<T | null> {
    try {
      await this.ensureInitialized();
      
      if (!(await this.isAuthenticated())) {
        return null;
      }
      
      return await this.auth0.getUser<T>();
    } catch (error: any) {
      console.error('Feil ved henting av brukerdata:', error);
      throw new AuthError('Kunne ikke hente brukerinfo.', error);
    }
  }
  
  /**
   * Henter en access token for API-kall
   */
  async getToken(options?: GetTokenSilentlyOptions): Promise<string> {
    try {
      await this.ensureInitialized();
      
      const token = await this.auth0.getTokenSilently(options);
      
      // Lagre token lokalt for bruk i API-kall
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('auth_token', token);
      }
      
      return token;
    } catch (error: any) {
      console.error('Feil ved henting av token:', error);
      
      // Hvis vi får 'login_required', omdiriger til innlogging
      if (error.error === 'login_required') {
        await this.login();
        throw new AuthError('Innlogging kreves.', error);
      }
      
      throw new AuthError('Kunne ikke hente autentiseringstoken.', error);
    }
  }
  
  /**
   * Sjekker om brukeren har en spesifikk tillatelse
   */
  async hasPermission(permission: string): Promise<boolean> {
    try {
      const user = await this.getUser<AuthUser & { permissions?: string[] }>();
      if (!user || !user.permissions) return false;
      
      return user.permissions.includes(permission);
    } catch (error) {
      console.error('Feil ved sjekk av tillatelser:', error);
      return false;
    }
  }
}

export default AuthService;