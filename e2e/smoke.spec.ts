import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('nn-tour-seen-v2', '1');
  });
});

test('homepage loads and tabs render', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /Neural networks/i })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Architecture' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Datasets' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Training' })).toBeVisible();
});

test('switch to datasets and load XOR', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Datasets' }).click();
  await page.getByRole('combobox').first().click();
  await page.getByRole('option', { name: /XOR/i }).click();
  await expect(page.getByRole('combobox').first()).toContainText(/XOR \(4 points\)/);
});
