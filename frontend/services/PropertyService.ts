import axios, { AxiosError, AxiosResponse, AxiosInstance, AxiosRequestConfig } from 'axios';

// Bruk en miljøvariabel eller standard URL
const API_BASE_URL = typeof window !== 'undefined' 
  ? (window as any).__ENV?.API_URL || 'http://localhost:8000/api'
  : 'http://localhost:8000/api';

// Cache-innstillinger
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutter i millisekunder
const MAX_CACHE_ENTRIES = 100; // Begrens antall cache-oppføringer
const CACHE_PREFIX = 'eiendom_';

// API-timeout (ms)
const API_TIMEOUT = 30000; // 30 sekunder

// Type definisjoner for klarere kode
export interface PropertyData {
  property_id?: string;
  address: string;
  municipality_id?: string;
  zoning_category?: string;
  lot_size: number;
  current_utilization: number;
  building_height: number;
  floor_area_ratio: number;
  images?: string[];
  additional_data?: Record<string, any>;
}

export interface RegulationRule {
  id: string;
  rule_type: string;
  value: any;
  description: string;
  unit?: string;
  category?: string;
}

export interface BuildingPotential {
  max_buildable_area: number;
  max_height: number;
  max_units: number;
  optimal_configuration: string;
  constraints?: string[];
  recommendations?: string[];
}

export interface EnergyProfile {
  energy_class: string;
  heating_demand: number;
  cooling_demand: number;
  primary_energy_source: string;
  recommendations?: string[];
}

export interface PropertyAnalysisResult {
  property_id: string;
  address: string;
  regulations: RegulationRule[];
  building_potential: BuildingPotential;
  energy_profile?: EnergyProfile;
  roi_estimate?: number;
  risk_assessment?: Record<string, any>;
  recommendations?: string[];
}

export interface Municipality {
  id: string;
  name: string;
  supportLevel: string;
}

export interface MunicipalityContact {
  municipalityId: string;
  name: string;
  department: string;
  email: string;
  phone?: string;
  role: string;
}

interface ApiErrorResponse {
  message?: string;
  error?: string;
  detail?: string;
  status?: number;
}

// Cache-klasse med automatic expiry
class ApiCache {
  private static cache: Map<string, { data: any; timestamp: number }> = new Map();
  private static cacheKeys: string[] = [];

  static get<T>(key: string): T | null {
    const cacheKey = `${CACHE_PREFIX}${key}`;
    const cached = this.cache.get(cacheKey);
    
    if (!cached) return null;
    
    // Sjekk om cachen er utløpt
    if (Date.now() - cached.timestamp > CACHE_DURATION) {
      this.cache.delete(cacheKey);
      return null;
    }
    
    return cached.data as T;
  }
  
  static set<T>(key: string, data: T): void {
    const cacheKey = `${CACHE_PREFIX}${key}`;
    
    // Begrens cache-størrelse ved å fjerne eldste oppføringer
    if (this.cacheKeys.length >= MAX_CACHE_ENTRIES) {
      const oldestKey = this.cacheKeys.shift();
      if (oldestKey) this.cache.delete(oldestKey);
    }
    
    // Legg til ny oppføring
    this.cache.set(cacheKey, { data, timestamp: Date.now() });
    this.cacheKeys.push(cacheKey);
  }
  
  static invalidate(keyPattern?: string): void {
    if (!keyPattern) {
      // Fjern all cache
      this.cache.clear();
      this.cacheKeys = [];
      return;
    }
    
    // Fjern oppføringer som matcher mønsteret
    const pattern = `${CACHE_PREFIX}${keyPattern}`;
    this.cacheKeys = this.cacheKeys.filter(key => {
      if (key.includes(pattern)) {
        this.cache.delete(key);
        return false;
      }
      return true;
    });
  }
}

/**
 * PropertyService - Tjeneste for å håndtere eiendomsdata og analyser
 */
export class PropertyService {
  private apiClient: AxiosInstance;
  private static instance: PropertyService;

  /**
   * Privat konstruktør (Singleton-mønster)
   */
  private constructor() {
    this.apiClient = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // Request interceptor
    this.apiClient.interceptors.request.use(
      (config) => {
        // Hent token fra localStorage
        const token = typeof localStorage !== 'undefined' ? localStorage.getItem('auth_token') : null;
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        console.error('Request error:', error);
        return Promise.reject(error);
      }
    );
    
    // Response interceptor
    this.apiClient.interceptors.response.use(
      (response) => response,
      async (error) => {
        // Håndter 401 Unauthorized
        if (error.response?.status === 401) {
          // Hvis vi har en refresh token, forsøk å fornye
          const refreshToken = typeof localStorage !== 'undefined' ? localStorage.getItem('refresh_token') : null;
          
          if (refreshToken) {
            try {
              // Implementer token refresh her
              // const refreshResponse = await this.refreshAccessToken(refreshToken);
              // localStorage.setItem('auth_token', refreshResponse.data.token);
              // Gjenta den opprinnelige forespørselen
              // return this.apiClient(error.config);
            } catch (refreshError) {
              // Refresh feilet, logg ut brukeren
              if (typeof localStorage !== 'undefined') {
                localStorage.removeItem('auth_token');
                localStorage.removeItem('refresh_token');
              }
              // Omdiriger til innloggingssiden hvis vi er i en nettleser
              if (typeof window !== 'undefined') {
                window.location.href = '/login';
              }
            }
          } else {
            // Ingen refresh token, logg ut
            if (typeof localStorage !== 'undefined') {
              localStorage.removeItem('auth_token');
            }
            // Omdiriger til innloggingssiden hvis vi er i en nettleser
            if (typeof window !== 'undefined') {
              window.location.href = '/login';
            }
          }
        }
        
        return Promise.reject(error);
      }
    );
  }

  /**
   * Hent en instans av PropertyService (Singleton-mønster)
   */
  public static getInstance(): PropertyService {
    if (!PropertyService.instance) {
      PropertyService.instance = new PropertyService();
    }
    return PropertyService.instance;
  }

  /**
   * Analyser en eiendom basert på adresse
   */
  async analyzeByAddress(address: string): Promise<PropertyAnalysisResult> {
    const cacheKey = `address_${address}`;
    const cachedResult = ApiCache.get<PropertyAnalysisResult>(cacheKey);
    
    if (cachedResult) {
      console.log('Returnerer cachet resultat for adresse:', address);
      return cachedResult;
    }
    
    try {
      const propertyData: PropertyData = {
        address,
        lot_size: 0,
        current_utilization: 0,
        building_height: 0,
        floor_area_ratio: 0,
      };
      
      const result = await this.analyzeProperty(propertyData);
      ApiCache.set(cacheKey, result);
      
      return result;
    } catch (error) {
      this.handleError(error as Error, 'analyzeByAddress');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Analyser en eiendom basert på opplastede filer
   */
  async analyzeFiles(files: File[]): Promise<PropertyAnalysisResult> {
    try {
      const formData = new FormData();
      files.forEach((file, index) => {
        formData.append(`file_${index}`, file);
      });
      
      const response = await this.apiClient.post<PropertyAnalysisResult>(
        '/property/analyze/files',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      return this.processResponse(response);
    } catch (error) {
      this.handleError(error as Error, 'analyzeFiles');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Hent detaljert informasjon om en eiendom
   */
  async getPropertyDetails(propertyId: string): Promise<any> {
    const cacheKey = `property_${propertyId}`;
    const cachedResult = ApiCache.get<any>(cacheKey);
    
    if (cachedResult) {
      return cachedResult;
    }
    
    try {
      const response = await this.apiClient.get<any>(`/property/${propertyId}`);
      const result = this.processResponse(response);
      
      ApiCache.set(cacheKey, result);
      return result;
    } catch (error) {
      this.handleError(error as Error, 'getPropertyDetails');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Generer dokumenter for en eiendom
   */
  async generateDocuments(propertyId: string, documentTypes: string[]): Promise<any> {
    try {
      const response = await this.apiClient.post<any>(`/property/${propertyId}/documents`, { document_types: documentTypes });
      return this.processResponse(response);
    } catch (error) {
      this.handleError(error as Error, 'generateDocuments');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Sjekk kommunale reguleringer for en eiendom
   */
  async checkMunicipalityRegulations(propertyId: string): Promise<RegulationRule[]> {
    const cacheKey = `regulations_${propertyId}`;
    const cachedResult = ApiCache.get<RegulationRule[]>(cacheKey);
    
    if (cachedResult) {
      return cachedResult;
    }
    
    try {
      const response = await this.apiClient.get<RegulationRule[]>(`/property/${propertyId}/regulations`);
      const result = this.processResponse(response);
      
      ApiCache.set(cacheKey, result);
      return result;
    } catch (error) {
      this.handleError(error as Error, 'checkMunicipalityRegulations');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Hent Enova-støtteinformasjon for en eiendom
   */
  async getEnovaSupport(propertyId: string): Promise<any> {
    const cacheKey = `enova_${propertyId}`;
    const cachedResult = ApiCache.get<any>(cacheKey);
    
    if (cachedResult) {
      return cachedResult;
    }
    
    try {
      const response = await this.apiClient.get<any>(`/property/${propertyId}/enova-support`);
      const result = this.processResponse(response);
      
      ApiCache.set(cacheKey, result);
      return result;
    } catch (error) {
      this.handleError(error as Error, 'getEnovaSupport');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Analyser 3D-modell for en eiendom
   */
  async analyze3DModel(propertyId: string): Promise<any> {
    const cacheKey = `3dmodel_${propertyId}`;
    const cachedResult = ApiCache.get<any>(cacheKey);
    
    if (cachedResult) {
      return cachedResult;
    }
    
    try {
      const response = await this.apiClient.get<any>(`/property/${propertyId}/3d-analysis`);
      const result = this.processResponse(response);
      
      ApiCache.set(cacheKey, result);
      return result;
    } catch (error) {
      this.handleError(error as Error, 'analyze3DModel');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Invalider cache for en eiendom eller all cache
   */
  invalidateCache(propertyId?: string): void {
    ApiCache.invalidate(propertyId);
  }

  /**
   * Analyser en eiendom basert på eiendomsdata
   */
  async analyzeProperty(propertyData: PropertyData): Promise<PropertyAnalysisResult> {
    try {
      const response = await this.apiClient.post<PropertyAnalysisResult>('/property/analyze', propertyData);
      return this.processResponse(response);
    } catch (error) {
      this.handleError(error as Error, 'analyzeProperty');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Hent reguleringsdata for en kommune
   */
  async getMunicipalityRegulations(municipalityId: string): Promise<RegulationRule[]> {
    const cacheKey = `municipality_regulations_${municipalityId}`;
    const cachedResult = ApiCache.get<RegulationRule[]>(cacheKey);
    
    if (cachedResult) {
      return cachedResult;
    }
    
    try {
      const response = await this.apiClient.get<RegulationRule[]>(`/property/municipality/${municipalityId}/regulations`);
      const result = this.processResponse(response);
      
      ApiCache.set(cacheKey, result);
      return result;
    } catch (error) {
      this.handleError(error as Error, 'getMunicipalityRegulations');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Hent kontaktinformasjon for en kommune
   */
  async getMunicipalityContacts(municipalityId: string): Promise<MunicipalityContact[]> {
    const cacheKey = `municipality_contacts_${municipalityId}`;
    const cachedResult = ApiCache.get<MunicipalityContact[]>(cacheKey);
    
    if (cachedResult) {
      return cachedResult;
    }
    
    try {
      const response = await this.apiClient.get<MunicipalityContact[]>(`/property/municipality/${municipalityId}/contacts`);
      const result = this.processResponse(response);
      
      ApiCache.set(cacheKey, result);
      return result;
    } catch (error) {
      this.handleError(error as Error, 'getMunicipalityContacts');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Hent liste over støttede kommuner
   */
  async getSupportedMunicipalities(): Promise<Municipality[]> {
    const cacheKey = 'supported_municipalities';
    const cachedResult = ApiCache.get<Municipality[]>(cacheKey);
    
    if (cachedResult) {
      return cachedResult;
    }
    
    try {
      const response = await this.apiClient.get<Municipality[]>('/property/supported-municipalities');
      const result = this.processResponse(response);
      
      ApiCache.set(cacheKey, result);
      return result;
    } catch (error) {
      this.handleError(error as Error, 'getSupportedMunicipalities');
      throw error; // For å tilfredsstille TypeScript
    }
  }

  /**
   * Feilhåndtering for API-kall
   */
  private handleError(error: Error | AxiosError, methodName: string): never {
    console.error(`PropertyService.${methodName} error:`, error);
    
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ApiErrorResponse>;
      
      if (axiosError.response) {
        // Serveren svarte med en feilkode
        const status = axiosError.response.status;
        const errorData = axiosError.response.data;
        
        if (status === 400) {
          throw new Error(`Ugyldig forespørsel: ${errorData.message || errorData.error || errorData.detail || 'Ukjent feil'}`);
        } else if (status === 401) {
          throw new Error('Du må være innlogget for å utføre denne operasjonen');
        } else if (status === 403) {
          throw new Error('Du har ikke tilgang til denne ressursen');
        } else if (status === 404) {
          throw new Error('Ressursen ble ikke funnet');
        } else if (status === 429) {
          throw new Error('For mange forespørsler. Prøv igjen senere.');
        } else if (status >= 500) {
          throw new Error('Serverfeil. Prøv igjen senere.');
        }
        
        // Generisk feilmelding
        throw new Error(errorData.message || errorData.error || errorData.detail || 'Det oppstod en feil');
      } else if (axiosError.request) {
        // Ingen respons mottatt
        throw new Error('Kunne ikke kontakte serveren. Sjekk nettverkstilkoblingen din.');
      }
    }
    
    // Generisk feil
    throw new Error(`En uventet feil oppstod: ${error.message}`);
  }

  /**
   * Behandle API-respons
   */
  private processResponse<T>(response: AxiosResponse<T>): T {
    return response.data;
  }
}

export default PropertyService;