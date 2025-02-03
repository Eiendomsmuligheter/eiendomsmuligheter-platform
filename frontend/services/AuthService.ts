import { Auth0Client } from '@auth0/auth0-spa-js';

export class AuthService {
  private static instance: AuthService;
  private auth0: Auth0Client;
  
  private constructor() {
    this.auth0 = new Auth0Client({
      domain: process.env.NEXT_PUBLIC_AUTH0_DOMAIN!,
      client_id: process.env.NEXT_PUBLIC_AUTH0_CLIENT_ID!,
      redirect_uri: window.location.origin,
      audience: process.env.NEXT_PUBLIC_AUTH0_AUDIENCE,
      cacheLocation: 'localstorage'
    });
  }
  
  static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }
  
  async login(): Promise<void> {
    try {
      await this.auth0.loginWithRedirect({
        redirect_uri: window.location.origin
      });
    } catch (error) {
      console.error('Error logging in:', error);
      throw error;
    }
  }
  
  async logout(): Promise<void> {
    try {
      await this.auth0.logout({
        returnTo: window.location.origin
      });
    } catch (error) {
      console.error('Error logging out:', error);
      throw error;
    }
  }
  
  async handleRedirectCallback(): Promise<void> {
    try {
      await this.auth0.handleRedirectCallback();
    } catch (error) {
      console.error('Error handling redirect:', error);
      throw error;
    }
  }
  
  async isAuthenticated(): Promise<boolean> {
    try {
      return await this.auth0.isAuthenticated();
    } catch (error) {
      console.error('Error checking authentication:', error);
      return false;
    }
  }
  
  async getUser(): Promise<any> {
    try {
      return await this.auth0.getUser();
    } catch (error) {
      console.error('Error getting user:', error);
      throw error;
    }
  }
  
  async getToken(): Promise<string> {
    try {
      return await this.auth0.getTokenSilently();
    } catch (error) {
      console.error('Error getting token:', error);
      throw error;
    }
  }
}