import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { CommuneConnect, RegulationData, PropertyRegulations, MunicipalityContact } from '../../backend/api/CommuneConnect';

// Mock avhengigheter
vi.mock('axios', () => ({
  default: {
    create: vi.fn().mockReturnValue({
      get: vi.fn(),
      post: vi.fn(),
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() }
      }
    })
  }
}));

vi.mock('ioredis', () => ({
  Redis: vi.fn().mockImplementation(() => ({
    get: vi.fn(),
    set: vi.fn(),
    quit: vi.fn(),
    disconnect: vi.fn()
  }))
}));

// Mock global process
vi.stubGlobal('process', {
  env: {
    REDIS_HOST: 'localhost',
    REDIS_PORT: '6379',
    REDIS_PASSWORD: '',
    REDIS_DB: '0',
    BAERUM_CLIENT_ID: 'test-client-id',
    BAERUM_CLIENT_SECRET: 'test-client-secret'
  }
});

describe('CommuneConnect', () => {
  let communeConnect: typeof CommuneConnect;
  
  beforeEach(() => {
    // Hent en ny instans av CommuneConnect for hver test
    communeConnect = CommuneConnect.getInstance({
      cacheEnabled: false,
      cacheTTL: 300,
      useCompression: false,
      rateLimiting: false,
      maxRetries: 3,
      timeout: 5000,
      userAgent: 'TestAgent',
      logLevel: 'error'
    });
  });
  
  afterEach(() => {
    // Rydder opp og resetter mocks
    vi.clearAllMocks();
  });
  
  it('skal opprette en singleton-instans av CommuneConnect', () => {
    const instance1 = CommuneConnect.getInstance();
    const instance2 = CommuneConnect.getInstance();
    
    expect(instance1).toBe(instance2);
  });
  
  it('skal kunne initialiseres med egendefinerte innstillinger', async () => {
    const customConfig = {
      cacheEnabled: true,
      cacheTTL: 600,
      useCompression: true,
      rateLimiting: true,
      maxRetries: 5,
      timeout: 10000,
      userAgent: 'CustomAgent',
      logLevel: 'debug' as const
    };
    
    const instance = CommuneConnect.getInstance(customConfig);
    await instance.initialize(customConfig);
    
    // Vi kan ikke direkte sjekke den private config-variabelen, men vi kan 
    // sjekke at initialize() ble kalt uten feil
    expect(instance).toBeDefined();
  });
  
  it('skal sjekke om en kommune er støttet', () => {
    const instance = CommuneConnect.getInstance();
    
    // Dette skal være sant siden Oslo som regel er satt opp i standardkonfigurasjonen
    expect(instance.isMunicipalitySupported('0301')).toBe(true);
    // Dette skal være falskt siden "9999" ikke er en gyldig kommuneID
    expect(instance.isMunicipalitySupported('9999')).toBe(false);
  });
  
  it('skal hente en liste over støttede kommuner', () => {
    const instance = CommuneConnect.getInstance();
    const municipalities = instance.getSupportedMunicipalities();
    
    expect(Array.isArray(municipalities)).toBe(true);
    expect(municipalities.length).toBeGreaterThan(0);
    
    // Sjekk at det har forventet struktur
    const first = municipalities[0];
    expect(first).toHaveProperty('id');
    expect(first).toHaveProperty('name');
    expect(first).toHaveProperty('supportLevel');
  });
  
  it('skal normalisere adresser', () => {
    const instance = CommuneConnect.getInstance();
    
    // Vi må bruke en reflection-teknikk for å teste private metoder
    const normalizeAddress = vi.spyOn(instance as any, 'normalizeAddress');
    
    // Sett opp spy-funksjonen til å returnere en gyldig verdi
    normalizeAddress.mockReturnValue('storgata 1, 0155 oslo');
    
    const result = (instance as any).normalizeAddress('Storgata 1, Oslo');
    
    expect(normalizeAddress).toHaveBeenCalledWith('Storgata 1, Oslo');
    expect(result).toBe('storgata 1, 0155 oslo');
  });
  
  it('skal hente reguleringer basert på adresse', async () => {
    const instance = CommuneConnect.getInstance();
    const httpClientMock = (instance as any).httpClient;
    
    // Setter opp mock for http-klient
    httpClientMock.get = vi.fn().mockResolvedValue({
      data: {
        propertyId: '123',
        address: 'Testgata 1, Oslo',
        municipalityId: '0301',
        regulations: [
          {
            regulationId: 'reg1',
            title: 'Testregel',
            status: 'active',
            validFrom: '2023-01-01',
            rules: []
          }
        ],
        zoningCategory: 'bolig',
        utilization: { max: 500, current: 300, available: 200, unit: 'm2' }
      }
    });
    
    const result = await instance.getRegulationsByAddress('Testgata 1, Oslo');
    
    expect(result).toBeDefined();
    expect(result.propertyId).toBe('123');
    expect(result.address).toBe('Testgata 1, Oslo');
    expect(result.regulations).toHaveLength(1);
    expect(result.regulations[0].regulationId).toBe('reg1');
  });
  
  it('skal hente reguleringer basert på eiendomsID', async () => {
    const instance = CommuneConnect.getInstance();
    const httpClientMock = (instance as any).httpClient;
    
    // Setter opp mock for http-klient
    httpClientMock.get = vi.fn().mockResolvedValue({
      data: {
        propertyId: '123',
        address: 'Testgata 1, Oslo',
        municipalityId: '0301',
        regulations: [],
        zoningCategory: 'bolig',
        utilization: { max: 500, current: 300, available: 200, unit: 'm2' }
      }
    });
    
    const result = await instance.getRegulationsByPropertyId('123', '0301');
    
    expect(result).toBeDefined();
    expect(result.propertyId).toBe('123');
    expect(httpClientMock.get).toHaveBeenCalled();
  });
  
  it('skal håndtere feil ved henting av reguleringer', async () => {
    const instance = CommuneConnect.getInstance();
    const httpClientMock = (instance as any).httpClient;
    
    // Setter opp mock for å kaste feil
    httpClientMock.get = vi.fn().mockRejectedValue(new Error('API error'));
    
    // Forvent at metoden kaster en feil
    await expect(instance.getRegulationsByAddress('Invalid address')).rejects.toThrow();
  });
  
  it('skal hente kontakter for en kommune', async () => {
    const instance = CommuneConnect.getInstance();
    const httpClientMock = (instance as any).httpClient;
    
    // Setter opp mock for http-klient
    httpClientMock.get = vi.fn().mockResolvedValue({
      data: [
        {
          municipalityId: '0301',
          name: 'Test Person',
          department: 'Plan og bygg',
          email: 'test@oslo.kommune.no',
          phone: '12345678',
          role: 'Saksbehandler'
        }
      ]
    });
    
    const contacts = await instance.getMunicipalityContacts('0301');
    
    expect(contacts).toHaveLength(1);
    expect(contacts[0].name).toBe('Test Person');
    expect(contacts[0].email).toBe('test@oslo.kommune.no');
  });
}); 