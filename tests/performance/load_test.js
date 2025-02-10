import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Definer ytelsesmetrics
const errorRate = new Rate('errors');

// Definer ytelseskriterier
export const options = {
  stages: [
    { duration: '1m', target: 20 }, // Bygg opp til 20 brukere
    { duration: '3m', target: 20 }, // Hold 20 brukere i 3 minutter
    { duration: '1m', target: 0 },  // Bygg ned til 0 brukere
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500'], // 95% av forespørslene skal være under 500ms
    'http_req_failed': ['rate<0.01'],    // Feilrate under 1%
    'errors': ['rate<0.05'],             // Total feilrate under 5%
  },
};

// Simulert brukeroppførsel
export default function() {
  // Initialiser testsesjon
  const baseUrl = 'http://localhost:8000';
  let response;

  // 1. Bruker logger inn
  response = http.post(`${baseUrl}/api/auth/login`, {
    email: 'test@example.com',
    password: 'testpassword123'
  });

  check(response, {
    'innlogging vellykket': (r) => r.status === 200,
  }) || errorRate.add(1);

  const authToken = response.json('token');
  const params = {
    headers: {
      'Authorization': `Bearer ${authToken}`,
      'Content-Type': 'application/json',
    },
  };

  sleep(1);

  // 2. Hent eiendomsdata
  response = http.get(`${baseUrl}/api/property/info?address=Testgata%201`, params);
  check(response, {
    'eiendomsdata hentet': (r) => r.status === 200,
  }) || errorRate.add(1);

  sleep(1);

  // 3. Last opp bilde
  const imageData = open('test-data/test-property.jpg', 'b');
  response = http.post(`${baseUrl}/api/property/upload`, imageData, {
    ...params,
    headers: {
      ...params.headers,
      'Content-Type': 'multipart/form-data',
    },
  });
  
  check(response, {
    'bilde lastet opp': (r) => r.status === 200,
  }) || errorRate.add(1);

  sleep(1);

  // 4. Start analyse
  response = http.post(`${baseUrl}/api/property/analyze`, {
    propertyId: 'test123',
    options: {
      includeEnergy: true,
      include3D: true,
    }
  }, params);

  check(response, {
    'analyse startet': (r) => r.status === 200,
  }) || errorRate.add(1);

  sleep(3);

  // 5. Hent analyseresultater
  response = http.get(`${baseUrl}/api/property/analysis-results/test123`, params);
  check(response, {
    'analyseresultater hentet': (r) => r.status === 200,
  }) || errorRate.add(1);

  sleep(1);

  // 6. Last ned rapport
  response = http.get(`${baseUrl}/api/property/report/test123`, params);
  check(response, {
    'rapport lastet ned': (r) => r.status === 200,
  }) || errorRate.add(1);

  sleep(1);

  // 7. Hent 3D-modell
  response = http.get(`${baseUrl}/api/property/3d-model/test123`, params);
  check(response, {
    '3D-modell hentet': (r) => r.status === 200,
  }) || errorRate.add(1);

  // Legg inn pause mellom iterasjoner
  sleep(3);
}