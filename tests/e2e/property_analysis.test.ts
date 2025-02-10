import { test, expect } from '@playwright/test';

test.describe('Eiendomsanalyse E2E Tester', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000');
  });

  test('Fullstendig eiendomsanalyse flyt', async ({ page }) => {
    // 1. Innlogging
    await test.step('Bruker kan logge inn', async () => {
      await page.getByRole('button', { name: 'Logg inn' }).click();
      await page.fill('input[name="username"]', 'test@example.com');
      await page.fill('input[name="password"]', 'testpassword123');
      await page.getByRole('button', { name: 'Logg inn' }).click();
      await expect(page.getByText('Velkommen')).toBeVisible();
    });

    // 2. Last opp bilde
    await test.step('Bruker kan laste opp bilde', async () => {
      await page.setInputFiles('input[type="file"]', 'test-data/test-property.jpg');
      await expect(page.getByText('Bilde lastet opp')).toBeVisible();
    });

    // 3. Fyll inn adresse
    await test.step('Bruker kan søke etter adresse', async () => {
      await page.fill('input[name="address"]', 'Testgata 1, 0123 Oslo');
      await page.getByRole('button', { name: 'Søk' }).click();
      await expect(page.getByText('Adresse funnet')).toBeVisible();
    });

    // 4. Start analyse
    await test.step('Bruker kan starte analyse', async () => {
      await page.getByRole('button', { name: 'Start analyse' }).click();
      await expect(page.getByText('Analyserer eiendom')).toBeVisible();
    });

    // 5. Vis resultater
    await test.step('Bruker kan se analyseresultater', async () => {
      // Vent på at analysen skal fullføres
      await page.waitForSelector('.analysis-results', { timeout: 30000 });
      
      // Sjekk at alle hovedkomponenter er synlige
      await expect(page.getByText('Eiendomsinformasjon')).toBeVisible();
      await expect(page.getByText('Reguleringsdata')).toBeVisible();
      await expect(page.getByText('Utviklingspotensial')).toBeVisible();
      await expect(page.getByText('Energianalyse')).toBeVisible();
    });

    // 6. Last ned dokumenter
    await test.step('Bruker kan laste ned genererte dokumenter', async () => {
      await page.getByRole('button', { name: 'Last ned rapport' }).click();
      await expect(page.getByText('Rapport lastet ned')).toBeVisible();
    });

    // 7. Vis 3D-modell
    await test.step('Bruker kan vise 3D-modell', async () => {
      await page.getByRole('button', { name: 'Vis 3D' }).click();
      await expect(page.locator('#3d-viewer')).toBeVisible();
    });

    // 8. Sjekk betalingsflyt
    await test.step('Bruker kan kjøpe analyse', async () => {
      await page.getByRole('button', { name: 'Kjøp analyse' }).click();
      await expect(page.getByText('Velg abonnement')).toBeVisible();
      
      // Velg abonnement
      await page.getByText('Pro').click();
      
      // Fyll inn betalingsinformasjon
      await page.fill('input[name="cardNumber"]', '4242424242424242');
      await page.fill('input[name="cardExpiry"]', '12/25');
      await page.fill('input[name="cardCvc"]', '123');
      
      // Fullfør betaling
      await page.getByRole('button', { name: 'Betal' }).click();
      await expect(page.getByText('Betaling godkjent')).toBeVisible();
    });
  });

  test('Feilhåndtering', async ({ page }) => {
    // Test feilhåndtering ved ugyldig adresse
    await test.step('Viser feilmelding ved ugyldig adresse', async () => {
      await page.fill('input[name="address"]', 'Ugyldig adresse 123');
      await page.getByRole('button', { name: 'Søk' }).click();
      await expect(page.getByText('Kunne ikke finne adressen')).toBeVisible();
    });

    // Test feilhåndtering ved ugyldig bilde
    await test.step('Viser feilmelding ved ugyldig bilde', async () => {
      await page.setInputFiles('input[type="file"]', 'test-data/invalid.txt');
      await expect(page.getByText('Ugyldig filformat')).toBeVisible();
    });

    // Test feilhåndtering ved mislykket betaling
    await test.step('Viser feilmelding ved mislykket betaling', async () => {
      await page.getByRole('button', { name: 'Kjøp analyse' }).click();
      await page.getByText('Pro').click();
      
      // Bruk et ugyldig kortnummer
      await page.fill('input[name="cardNumber"]', '4242424242424241');
      await page.fill('input[name="cardExpiry"]', '12/25');
      await page.fill('input[name="cardCvc"]', '123');
      
      await page.getByRole('button', { name: 'Betal' }).click();
      await expect(page.getByText('Betalingen ble avvist')).toBeVisible();
    });
  });
});