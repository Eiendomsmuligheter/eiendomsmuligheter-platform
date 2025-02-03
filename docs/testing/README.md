# Test Dokumentasjon - Eiendomsmuligheter Platform

## Testarkitektur

### Teststrategi
- Enhetstest: 90% dekning
- Integrasjonstest: Alle API-endepunkter
- E2E-test: Kritiske brukerflyter
- Performance-test: API og frontend ytelse
- Sikkerhetstesting: OWASP Top 10

### Testmiljøer
1. Lokal utvikling
2. CI/CD pipeline
3. Staging
4. Produksjon

## Test Suiter

### 1. Frontend Testing

#### Unit Tests (Jest + React Testing Library)
```typescript
// PropertyAnalyzer.test.tsx
describe('PropertyAnalyzer Component', () => {
  test('renders upload area', () => {
    render(<PropertyAnalyzer />);
    expect(screen.getByText(/last opp bilder/i)).toBeInTheDocument();
  });

  test('handles file upload', async () => {
    render(<PropertyAnalyzer />);
    const file = new File(['dummy content'], 'example.png', {type: 'image/png'});
    const uploader = screen.getByLabelText(/filopplasting/i);
    
    await userEvent.upload(uploader, file);
    expect(screen.getByText(/analyzing/i)).toBeInTheDocument();
  });
});
```

#### Integration Tests
```typescript
// PropertyService.test.ts
describe('PropertyService', () => {
  test('fetches property data', async () => {
    const service = new PropertyService();
    const data = await service.getProperty('123');
    expect(data).toHaveProperty('address');
  });
});
```

#### E2E Tests (Cypress)
```typescript
// property-analysis.spec.ts
describe('Property Analysis Flow', () => {
  it('completes full analysis', () => {
    cy.visit('/');
    cy.login();
    cy.get('[data-test="new-analysis"]').click();
    cy.get('[data-test="address-input"]').type('Testveien 1');
    cy.get('[data-test="start-analysis"]').click();
    cy.get('[data-test="analysis-results"]', { timeout: 10000 }).should('be.visible');
  });
});
```

### 2. Backend Testing

#### Unit Tests (pytest)
```python
# test_property_analyzer.py
def test_analyze_property():
    analyzer = PropertyAnalyzer()
    result = analyzer.analyze("Test Address")
    assert result.get("status") == "success"
    assert "property_info" in result

@pytest.mark.asyncio
async def test_municipality_service():
    service = MunicipalityService()
    regulations = await service.get_regulations("3005")
    assert len(regulations) > 0
```

#### API Tests (pytest + httpx)
```python
# test_api.py
@pytest.mark.asyncio
async def test_property_endpoint():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/properties/analyze", json={
            "address": "Test Street 1"
        })
        assert response.status_code == 200
        assert "property_info" in response.json()
```

#### Database Tests
```python
# test_models.py
def test_property_model():
    property = Property(
        address="Test Address",
        municipality_code="3005"
    )
    db.session.add(property)
    db.session.commit()
    
    saved = Property.query.filter_by(address="Test Address").first()
    assert saved is not None
```

### 3. Performance Testing

#### Load Testing (k6)
```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 100,
  duration: '5m'
};

export default function() {
  let res = http.post('http://test-api/properties/analyze', {
    address: 'Test Address'
  });
  
  check(res, {
    'is status 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500
  });
  
  sleep(1);
}
```

#### Frontend Performance (Lighthouse)
```javascript
// lighthouse-config.js
module.exports = {
  extends: 'lighthouse:default',
  settings: {
    onlyCategories: ['performance', 'accessibility'],
    formFactor: 'desktop',
    throttling: {
      rttMs: 40,
      throughputKbps: 10240,
      cpuSlowdownMultiplier: 1
    }
  }
};
```

### 4. Sikkerhetstesting

#### OWASP ZAP Scanning
```yaml
# zap-config.yaml
scan:
  endpoints:
    - /api/properties/*
    - /api/auth/*
  rules:
    - xss
    - sql-injection
    - csrf
```

#### Dependency Scanning
```yaml
# dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/frontend"
    schedule:
      interval: "daily"
  
  - package-ecosystem: "pip"
    directory: "/backend"
    schedule:
      interval: "daily"
```

## Test Data

### Mock Data
```json
{
  "properties": [
    {
      "id": "test-1",
      "address": "Testveien 1",
      "municipality_code": "3005",
      "property_number": "1/1"
    }
  ],
  "regulations": [
    {
      "type": "zoning",
      "description": "Test regulation"
    }
  ]
}
```

### Fixtures
```python
# conftest.py
@pytest.fixture
def mock_property():
    return Property(
        id="test-1",
        address="Test Address",
        municipality_code="3005"
    )

@pytest.fixture
def mock_analysis():
    return Analysis(
        property_id="test-1",
        status="completed"
    )
```

## Kontinuerlig Testing

### GitHub Actions Workflow
```yaml
# test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Frontend Tests
        run: |
          cd frontend
          npm install
          npm test
          
      - name: Backend Tests
        run: |
          cd backend
          pip install -r requirements.txt
          pytest
          
      - name: E2E Tests
        run: |
          npm run cypress
          
      - name: Performance Tests
        run: |
          k6 run load-test.js
```

## Testrapportering

### Coverage Report
```bash
# Frontend coverage
jest --coverage

# Backend coverage
pytest --cov=src tests/
```

### Test Results
```json
{
  "summary": {
    "total": 500,
    "passed": 495,
    "failed": 5,
    "skipped": 0
  },
  "coverage": {
    "statements": 90.5,
    "branches": 85.3,
    "functions": 88.7,
    "lines": 90.1
  }
}
```

## Regressjonstesting

### Automatiske Regresjonstester
```python
# test_regression.py
def test_known_issues():
    """Test tidligere kjente feil"""
    property_id = "test-1"
    result = analyze_property(property_id)
    assert "error_xyz" not in result
```

## Performance Benchmarks

### API Response Times
- GET endpoints: < 100ms
- POST endpoints: < 500ms
- Analyse: < 2000ms

### Frontend Metrics
- First Contentful Paint: < 1.5s
- Time to Interactive: < 3.5s
- Total Blocking Time: < 300ms

## Testautomatisering

### Test Scheduling
```yaml
# schedule.yml
name: Scheduled Tests
on:
  schedule:
    - cron: '0 */6 * * *'  # Hver 6. time

jobs:
  automated-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run Test Suite
        run: |
          npm test
          pytest
```

## Feilsøking og Debugging

### Logging Setup
```python
# test_logging.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('test.log'),
        logging.StreamHandler()
    ]
)
```

### Debug Tools
- Chrome DevTools
- Vue.js DevTools
- Python debugger (pdb)
- Network analysis tools

## Testmiljø Setup

### Docker Compose
```yaml
# docker-compose.test.yml
version: '3.8'
services:
  test-db:
    image: postgres:13
    environment:
      POSTGRES_DB: test_db
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_pass
      
  test-redis:
    image: redis:6
    
  test-app:
    build: .
    command: pytest
    depends_on:
      - test-db
      - test-redis
```