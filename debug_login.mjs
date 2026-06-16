import { chromium } from 'playwright-core';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  page.on('console', msg => console.log('CONSOLE:', msg.type(), msg.text()));
  page.on('response', async response => {
    const url = response.url();
    const status = response.status();
    if (url.includes('/auth/login') || url.includes('/api/')) {
      const body = await response.text().catch(() => '');
      console.log('NETWORK:', status, url, body.slice(0, 200));
    }
  });

  await page.goto('http://localhost:3000/login');
  await page.fill('input#email', 'testorbit500@example.com');
  await page.fill('input#password', 'testpass123');
  await page.click('button[type="submit"]');
  await page.waitForTimeout(3000);

  console.log('Final URL:', page.url());
  await browser.close();
})();
