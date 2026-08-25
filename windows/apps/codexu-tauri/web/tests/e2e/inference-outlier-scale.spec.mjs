import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { expect, test } from '@playwright/test';

test('keeps regular throughput groups readable when one group is a high outlier', async ({ page }) => {
  await page.addInitScript((data) => {
    window.__CODEXU_VISUAL_DATA__ = data;
  }, createInferenceFixture());
  await page.goto('/visual-test.html?surface=inference');

  const tab = page.getByRole('tab', { name: 'Inference performance', exact: true });
  await tab.click();
  await expect(tab).toHaveAttribute('aria-selected', 'true');

  const chart = page.locator('#dashboard-home-panel-inference .inference-scatter');
  await expect(chart).toBeVisible();
  await expect(chart).toHaveAttribute('data-inference-scale-mode', 'single-outlier');

  const spark = chart.locator('[data-inference-group-id="spark"] circle.inference-bubble');
  const sol = chart.locator('[data-inference-group-id="sol"] circle.inference-bubble');
  const luna = chart.locator('[data-inference-group-id="luna"] circle.inference-bubble');
  const [sparkX, sparkY, solY, lunaY] = await Promise.all([
    numericAttribute(spark, 'cx'),
    numericAttribute(spark, 'cy'),
    numericAttribute(sol, 'cy'),
    numericAttribute(luna, 'cy'),
  ]);

  expect(sparkY).toBeCloseTo(42, 0);
  expect(solY).toBeLessThan(140);
  expect(lunaY - solY).toBeGreaterThan(70);
  expect(sparkX).toBeCloseTo(88.4, 1);

  const regularGrid = chart.locator('[data-inference-horizontal-grid]');
  await expect(regularGrid).toHaveCount(5);
  expect(await numericAttribute(regularGrid.first(), 'y1')).toBeCloseTo(82, 0);
  expect(await numericAttribute(regularGrid.last(), 'y1')).toBeCloseTo(278, 0);

  const regularTicks = chart.locator('[data-inference-regular-tick]');
  await expect(regularTicks).toHaveCount(5);
  await expect(regularTicks).toHaveText(['70', '52', '35', '17', '0']);

  const durationTicks = chart.locator('[data-inference-x-tick]');
  await expect(durationTicks).toHaveCount(5);
  await expect(durationTicks).toHaveText(['0', '14s', '28s', '42s', '56s']);
  expect(await numericAttribute(durationTicks.first(), 'x')).toBeCloseTo(54, 0);
  expect(await numericAttribute(durationTicks.last(), 'x')).toBeCloseTo(696, 0);
});

test('shows the outlier value and axis break without overflow copy', async ({ page }, testInfo) => {
  await page.addInitScript((data) => {
    window.__CODEXU_VISUAL_DATA__ = data;
  }, createInferenceFixture());
  await page.goto('/visual-test.html?surface=inference');

  const tab = page.getByRole('tab', { name: 'Inference performance', exact: true });
  await tab.click();
  const panel = page.locator('#dashboard-home-panel-inference');
  const chart = panel.locator('.inference-scatter');

  await expect(chart.locator('[data-inference-outlier-tick]')).toHaveText('240');
  await expect(chart.locator('[data-inference-axis-break]')).toHaveCount(1);
  await expect(chart.getByText('5.3-codex-spark · Medium', { exact: true })).toBeVisible();
  await expect(panel).not.toContainText(/overflow|超额/i);

  const screenshotPath = path.join(testInfo.outputDir, 'evidence', 'inference-outlier-scale', 'focused.png');
  await mkdir(path.dirname(screenshotPath), { recursive: true });
  await panel.screenshot({ path: screenshotPath });
});

async function numericAttribute(locator, name) {
  const value = await locator.getAttribute(name);
  expect(value).not.toBeNull();
  return Number(value);
}

function createInferenceFixture() {
  const refreshedAt = Date.now();
  const groups = [
    inferenceGroup('spark', 'gpt-5.3-codex-spark', 'medium', 240, 3, 5, 3),
    inferenceGroup('sol', 'gpt-5.6-sol', 'xhigh', 60, 12, 40, 18),
    inferenceGroup('terra', 'gpt-5.6-terra', 'high', 45, 9, 24, 12),
    inferenceGroup('luna', 'gpt-5.6-luna', 'medium', 30, 6, 14, 9),
  ];
  const period = {
    period: 'today',
    coverage_day_count: 1,
    groups,
    total_call_count: groups.reduce((total, group) => total + group.call_count, 0),
  };

  return {
    dashboard: {
      refreshed_at: refreshedAt,
      codex: {
        scope: 'codex',
        status: 'local_only',
        quota_source_label: 'Codex local fixture',
        snapshot: {
          refreshed_at: refreshedAt,
          account: { type: 'codex-local', plan_type: null, email_present: false },
          limit_id: 'codex-local',
          limit_name: 'Codex local fixture',
          quota_read_succeeded: false,
          five_hour_quota: null,
          seven_day_quota: null,
          monthly_quota: null,
          local: {
            lifetime_tokens: 150,
            today_tokens: 150,
            seven_day_tokens: 150,
            thread_count: 1,
            last_updated_at: refreshedAt,
            daily_buckets: [],
            recent_threads: [],
            detailed_usage: null,
            usage_trend: null,
            inference_performance: {
              recording_started_at: refreshedAt,
              today: period,
              seven_days: { ...period, period: 'seven_days', coverage_day_count: 7 },
              twenty_eight_days: { ...period, period: 'twenty_eight_days', coverage_day_count: 28 },
            },
            project_board: null,
            tool_usages: [],
            skill_usages: [],
          },
          task_board: null,
          messages: [],
        },
      },
      leadership: {
        score: null,
        evidence_coverage: 0,
        active_day_count: 0,
        period: 'today',
        model_version: 'fixture',
        report: null,
      },
      messages: [],
    },
    settings: {
      config: {
        codex_root: '<fixture Codex data>',
        cache_dir: '<fixture codexU cache>',
        theme: 'light',
        refresh_interval_secs: 60,
        tray_density: 'classic',
        language: 'en',
        palette_id: 'codexu.default',
      },
      app_data_dir: '<fixture codexU app data>',
    },
    source: { mode: 'live-readonly', provider: 'codex-dashboard' },
  };
}

function inferenceGroup(id, model, effort, throughput, p50, p90, calls) {
  return {
    id,
    model,
    effort,
    call_count: calls,
    average_daily_call_count: calls,
    average_duration_seconds: p50,
    p50_duration_seconds: p50,
    p90_duration_seconds: p90,
    effective_output_tokens_per_second: throughput,
    output_tokens: Math.round(throughput * p50 * calls),
    reasoning_output_tokens: Math.round(throughput * p50 * calls * 0.4),
  };
}
