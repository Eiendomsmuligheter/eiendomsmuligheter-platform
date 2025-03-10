/**
 * EdgeProcessing - Lokal databehandlingsmodul for Eiendomsmuligheter Platform
 * 
 * Denne modulen implementerer optimalisert databehandling på klientsiden (edge)
 * for å redusere belastning på backend, minimere nettverkstrafikk,
 * og maksimere responsiviteten i brukergrensesnittet.
 */

import { PropertyAnalysisResult } from './PropertyService';

// Verdier som brukes til caching
const CACHE_EXPIRY_TIME = 1000 * 60 * 30; // 30 minutter
const MAX_CACHE_ITEMS = 100;

// Typer
export interface CacheItem<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

export interface ProcessingOptions {
  cacheEnabled: boolean;
  prefetchEnabled: boolean;
  compressionEnabled: boolean;
  localComputationEnabled: boolean;
  cacheTTL?: number; // millisekunder
}

export interface CompressionOptions {
  level: 'low' | 'medium' | 'high';
  format: 'json' | 'binary' | 'protobuf';
}

export interface ProcessingResult<T> {
  data: T;
  source: 'cache' | 'network' | 'computed';
  processTime: number;
  cacheStatus?: 'hit' | 'miss' | 'expired';
}

// Hjelpefunksjoner
const deepClone = <T>(obj: T): T => {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return [...obj.map(item => deepClone(item))] as unknown as T;
  }

  return Object.keys(obj).reduce((result, key) => {
    const value = (obj as Record<string, any>)[key];
    result[key] = deepClone(value);
    return result;
  }, {} as Record<string, any>) as T;
};

/**
 * EdgeProcessing - Hovedklasse for optimalisert databehandling
 */
class EdgeProcessing {
  private static instance: EdgeProcessing;
  private cache: Map<string, CacheItem<any>> = new Map();
  private options: ProcessingOptions = {
    cacheEnabled: true,
    prefetchEnabled: true,
    compressionEnabled: true,
    localComputationEnabled: true,
    cacheTTL: CACHE_EXPIRY_TIME,
  };
  private compressionOptions: CompressionOptions = {
    level: 'medium',
    format: 'json',
  };
  private pendingRequests: Map<string, Promise<any>> = new Map();
  private isProxyEnabled: boolean = false;

  private constructor() {
    this.initializeCache();
    this.detectCapabilities();
  }

  /**
   * Hent singleton-instansen av EdgeProcessing
   */
  public static getInstance(): EdgeProcessing {
    if (!EdgeProcessing.instance) {
      EdgeProcessing.instance = new EdgeProcessing();
    }
    return EdgeProcessing.instance;
  }

  /**
   * Konfigurer EdgeProcessing med egendefinerte instillinger
   */
  public configure(options: Partial<ProcessingOptions>): void {
    this.options = { ...this.options, ...options };
    console.log('EdgeProcessing konfigurert:', this.options);
  }

  /**
   * Sett kompresjonsalternativer
   */
  public setCompressionOptions(options: Partial<CompressionOptions>): void {
    this.compressionOptions = { ...this.compressionOptions, ...options };
  }

  /**
   * Hent data med smart caching og bearbeiding
   */
  public async processData<T>(
    key: string,
    fetchFn: () => Promise<T>,
    computeFn?: (data: any) => T,
    options?: Partial<ProcessingOptions>
  ): Promise<ProcessingResult<T>> {
    const startTime = performance.now();
    const currentOptions = { ...this.options, ...options };

    // Sjekk cache
    if (currentOptions.cacheEnabled) {
      const cachedItem = this.getFromCache<T>(key);
      if (cachedItem) {
        return {
          data: deepClone(cachedItem.data),
          source: 'cache',
          processTime: performance.now() - startTime,
          cacheStatus: 'hit',
        };
      }
    }

    // Håndter pågående forespørsler (deduplicering)
    if (this.pendingRequests.has(key)) {
      const data = await this.pendingRequests.get(key);
      return {
        data,
        source: 'network',
        processTime: performance.now() - startTime,
        cacheStatus: 'miss',
      };
    }

    // Lokale beregninger hvis mulig og aktivert
    if (currentOptions.localComputationEnabled && computeFn) {
      try {
        const localInput = localStorage.getItem(`edge_data_${key}`);
        if (localInput) {
          const parsedInput = JSON.parse(localInput);
          const result = computeFn(parsedInput);
          
          // Lagre resultatet i cache
          if (currentOptions.cacheEnabled) {
            this.saveToCache(key, result, currentOptions.cacheTTL);
          }
          
          return {
            data: result,
            source: 'computed',
            processTime: performance.now() - startTime,
            cacheStatus: 'miss',
          };
        }
      } catch (error) {
        console.warn('Feil ved lokal beregning, fortsetter med nettverkshenting:', error);
      }
    }

    // Hent data via nettverket
    try {
      const fetchPromise = fetchFn();
      this.pendingRequests.set(key, fetchPromise);
      
      const data = await fetchPromise;
      
      // Lagre i cache
      if (currentOptions.cacheEnabled && data) {
        this.saveToCache(key, data, currentOptions.cacheTTL);
      }
      
      // Lagre rådataen for fremtidige lokale beregninger
      if (currentOptions.localComputationEnabled && data) {
        try {
          localStorage.setItem(`edge_data_${key}`, JSON.stringify(data));
        } catch (error) {
          console.warn('Kunne ikke lagre data i localStorage:', error);
        }
      }
      
      this.pendingRequests.delete(key);
      
      return {
        data,
        source: 'network',
        processTime: performance.now() - startTime,
        cacheStatus: 'miss',
      };
    } catch (error) {
      this.pendingRequests.delete(key);
      throw error;
    }
  }

  /**
   * Prefetch data som sannsynligvis vil bli brukt senere
   */
  public prefetchData<T>(
    key: string, 
    fetchFn: () => Promise<T>,
    priority: 'high' | 'medium' | 'low' = 'medium'
  ): void {
    if (!this.options.prefetchEnabled) return;
    
    // Bare prefetch hvis vi ikke allerede har dataen i cache
    if (this.options.cacheEnabled && this.getFromCache(key)) {
      return;
    }
    
    const delay = priority === 'high' ? 0 : priority === 'medium' ? 100 : 500;
    
    setTimeout(() => {
      // Bruk IdleCallback hvis tilgjengelig, ellers setTimeout
      const idleCallback = window.requestIdleCallback || ((cb) => setTimeout(cb, 1));
      
      idleCallback(() => {
        this.processData(key, fetchFn)
          .then(() => console.log(`Prefetch fullført for ${key}`))
          .catch(error => console.warn(`Prefetch feilet for ${key}:`, error));
      });
    }, delay);
  }

  /**
   * Komprimerer data for effektiv lagring og overføring
   */
  public compressData(data: any): Uint8Array | string {
    if (!this.options.compressionEnabled) {
      return JSON.stringify(data);
    }
    
    // I en faktisk implementasjon ville vi brukt ekte kompresjon
    // Her simulerer vi bare en enkel kompresjon
    const jsonString = JSON.stringify(data);
    
    switch (this.compressionOptions.format) {
      case 'binary':
        // Simulert binærkompresjon (faktisk implementasjon ville brukt pako, lz-string, etc.)
        return new TextEncoder().encode(jsonString);
      case 'protobuf':
        // Simulert protobuf (faktisk implementasjon ville brukt protobuf.js)
        return jsonString;
      case 'json':
      default:
        return jsonString;
    }
  }

  /**
   * Dekomprimerer data
   */
  public decompressData(data: Uint8Array | string): any {
    if (!this.options.compressionEnabled) {
      return JSON.parse(data as string);
    }
    
    // Konverter data tilbake
    let jsonString: string;
    
    if (data instanceof Uint8Array) {
      jsonString = new TextDecoder().decode(data);
    } else {
      jsonString = data;
    }
    
    return JSON.parse(jsonString);
  }

  /**
   * Beregn optimalisert visualiseringsdata (forenklet versjon) basert på eiendomsanalyse
   */
  public computeOptimizedVisualizationData(analysis: PropertyAnalysisResult): any {
    // Dette ville i praksis være mer avansert
    
    return {
      type: '3d_building',
      maxHeight: analysis.building_potential.max_height,
      maxArea: analysis.building_potential.max_buildable_area,
      floors: Math.ceil(analysis.building_potential.max_units / 2),
      position: {
        x: 0,
        y: 0,
        z: 0
      },
      dimensions: {
        width: Math.sqrt(analysis.building_potential.max_buildable_area / analysis.building_potential.max_units),
        height: analysis.building_potential.max_height,
        depth: Math.sqrt(analysis.building_potential.max_buildable_area / analysis.building_potential.max_units),
      },
      color: '#3388ff',
      opacity: 0.7
    };
  }

  /**
   * Tøm cachen
   */
  public clearCache(): void {
    this.cache.clear();
    console.log('Edge-cache tømt');
  }

  /**
   * Fjern utgått cache
   */
  public pruneCache(): void {
    const now = Date.now();
    let expiredCount = 0;
    
    for (const [key, item] of this.cache.entries()) {
      if (item.expiresAt < now) {
        this.cache.delete(key);
        expiredCount++;
      }
    }
    
    console.log(`Fjernet ${expiredCount} utgåtte cache-elementer`);
  }

  /**
   * Aktiver/deaktiver service worker proxy for bedre offline-støtte
   */
  public setProxyEnabled(enabled: boolean): void {
    this.isProxyEnabled = enabled;
    
    if (enabled && 'serviceWorker' in navigator) {
      // I praksis ville vi registrert en service worker her
      console.log('Service Worker proxy aktivert for offline-støtte');
    } else if (!enabled && 'serviceWorker' in navigator) {
      // Avregistrer eksisterende service worker
      console.log('Service Worker proxy deaktivert');
    }
  }

  // Private metoder
  private getFromCache<T>(key: string): CacheItem<T> | null {
    if (!this.options.cacheEnabled) return null;
    
    const item = this.cache.get(key);
    if (!item) return null;
    
    const now = Date.now();
    if (item.expiresAt < now) {
      this.cache.delete(key);
      return null;
    }
    
    return item;
  }

  private saveToCache<T>(key: string, data: T, ttl?: number): void {
    if (!this.options.cacheEnabled) return;
    
    const now = Date.now();
    const expiryTime = ttl || this.options.cacheTTL || CACHE_EXPIRY_TIME;
    
    const item: CacheItem<T> = {
      data,
      timestamp: now,
      expiresAt: now + expiryTime
    };
    
    // Hvis cachen er full, fjern det eldste elementet
    if (this.cache.size >= MAX_CACHE_ITEMS) {
      let oldestKey: string | null = null;
      let oldestTimestamp = Infinity;
      
      for (const [cacheKey, cacheItem] of this.cache.entries()) {
        if (cacheItem.timestamp < oldestTimestamp) {
          oldestTimestamp = cacheItem.timestamp;
          oldestKey = cacheKey;
        }
      }
      
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }
    
    this.cache.set(key, item);
  }

  private initializeCache(): void {
    // Prøv å gjenopprette cache fra sessionStorage
    try {
      const savedCache = sessionStorage.getItem('edge_processing_cache');
      if (savedCache) {
        const parsed = JSON.parse(savedCache);
        for (const [key, value] of Object.entries(parsed)) {
          this.cache.set(key, value as CacheItem<any>);
        }
        console.log(`Gjenopprettet ${this.cache.size} cache-elementer fra session`);
      }
    } catch (error) {
      console.warn('Kunne ikke gjenopprette cache:', error);
    }
    
    // Lagre cachen ved utlogging/lukking
    window.addEventListener('beforeunload', () => {
      try {
        // Konverter Map til objekt for lagring
        const cacheObj: Record<string, CacheItem<any>> = {};
        for (const [key, value] of this.cache.entries()) {
          cacheObj[key] = value;
        }
        sessionStorage.setItem('edge_processing_cache', JSON.stringify(cacheObj));
      } catch (error) {
        console.warn('Kunne ikke lagre cache:', error);
      }
    });
    
    // Sett opp periodisk pruning av utgått cache
    setInterval(() => this.pruneCache(), 60000); // Hvert minutt
  }

  private detectCapabilities(): void {
    // Oppdager nettverkshastighet og enhetens yteevne
    // Dette ville vært mer avansert i praksis
    
    const connectionType = (navigator as any).connection?.type;
    const deviceMemory = (navigator as any).deviceMemory;
    const isLowEndDevice = deviceMemory && deviceMemory < 4;
    
    if (connectionType === 'slow-2g' || connectionType === '2g' || isLowEndDevice) {
      // For trege tilkoblinger eller enheter med lav ytelse
      this.options.compressionEnabled = true;
      this.options.localComputationEnabled = false;
      this.compressionOptions.level = 'high';
      console.log('Optimaliserer for enhet med lav ytelse / treg tilkobling');
    }
    
    // Sjekk offline-tilstand
    if (!navigator.onLine) {
      console.log('Enheten er offline, aktiverer offline-modus');
      this.setProxyEnabled(true);
    }
    
    // Lytt på endringer i tilkobling
    window.addEventListener('online', () => {
      console.log('Enheten er nå online');
      this.setProxyEnabled(false);
    });
    
    window.addEventListener('offline', () => {
      console.log('Enheten er nå offline, aktiverer offline-modus');
      this.setProxyEnabled(true);
    });
  }
}

export default EdgeProcessing; 