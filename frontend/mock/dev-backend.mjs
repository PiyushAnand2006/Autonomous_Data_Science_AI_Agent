/**
 * Dev-only mock AUDAS backend (no dependencies).
 * Lets you preview and test the full frontend flow — sample dataset, live
 * agent log stream, run summary, leaderboard, visualizations, report,
 * artifacts — without the real FastAPI backend.
 *
 *   node mock/dev-backend.mjs        # serves on http://localhost:8000
 *
 * It only ever serves fixture data; it trains nothing.
 */
import http from 'node:http';

const PORT = process.env.MOCK_PORT || 8000;

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,PUT,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

const RUN_ID = 'RUN_MOCK_20261118_0142';

/* ------------------------------ fixtures ------------------------------ */

const COLUMNS = ['customer_id', 'age', 'monthly_charges', 'tenure_months', 'contract_type', 'payment_method', 'total_services', 'is_senior', 'churn'];
const PREVIEW_ROWS = [
  ['C-10001', 44, 89.5, 12, 'Month-to-month', 'Credit card', 4, 0, 'Yes'],
  ['C-10002', 31, 52.25, 36, 'Two year', 'Bank transfer', 7, 0, 'No'],
  ['C-10003', 67, 104.9, 3, 'Month-to-month', 'Electronic check', 2, 1, 'Yes'],
  ['C-10004', 25, 39.75, 48, 'Two year', 'Credit card', 8, 0, 'No'],
  ['C-10005', 58, 78.1, 9, 'One year', 'Mailed check', 5, 1, 'No'],
];

const SETTINGS = {
  settings: {
    execution_mode: 'local',
    selected_provider: 'openrouter',
    selected_model: 'anthropic/claude-3.5-sonnet',
    autonomy: 'fully_autonomous',
    engine: 'pandas',
    random_seed: 42,
    api_keys: {},
    provider_models_cache: {},
  },
  default_models: {
    openrouter: [
      'google/gemini-2.0-flash-001',
      'meta-llama/llama-3.3-70b-instruct',
      'openai/gpt-4o-mini',
      'anthropic/claude-3.5-sonnet',
      'deepseek/deepseek-r1',
    ],
    nvidia: ['meta/llama-3.3-70b-instruct', 'deepseek-ai/deepseek-r1'],
    google: ['gemini-2.0-flash', 'gemini-1.5-pro'],
    openai: ['gpt-4o-mini', 'gpt-4o'],
    anthropic: ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022'],
    ollama: ['llama3.2', 'llama3.1', 'mistral'],
    custom: ['custom-model'],
  },
};

const MODEL_NAMES = ['LightGBM', 'XGBoost', 'RandomForest', 'ExtraTrees', 'GradientBoosting', 'CatBoost', 'LogisticRegression'];

function mkMetrics(quality) {
  // Deterministic pseudo-metrics descending by quality
  return {
    accuracy: +(0.72 + quality * 0.17).toFixed(4),
    f1: +(0.68 + quality * 0.19).toFixed(4),
    precision: +(0.70 + quality * 0.15).toFixed(4),
    recall: +(0.66 + quality * 0.21).toFixed(4),
    roc_auc: +(0.75 + quality * 0.18).toFixed(4),
  };
}

const ALL_CANDIDATES = MODEL_NAMES.map((name, i) => ({
  rank: i + 1,
  model: name,
  training_time: +(1.9 - i * 0.22 + (i % 2) * 0.07).toFixed(6),
  ...mkMetrics(1 - i / MODEL_NAMES.length),
}));

const TOP_MODELS = ALL_CANDIDATES.slice(0, 4).map((c, i) => ({
  rank: i + 1,
  model_name: c.model,
  metrics: { accuracy: c.accuracy, f1: c.f1, roc_auc: c.roc_auc },
  training_time: c.training_time,
  selection_reason:
    i === 0
      ? 'Champion: highest holdout F1 and ROC-AUC across all 7 candidates with stable cross-validation variance.'
      : `Rank ${i + 1}: strong holdout metrics with complementary error profile; kept for ensemble diversity.`,
  filename: `model_0${i + 1}_${c.model.toLowerCase()}.pkl`,
}));

const RESULT = {
  run_id: RUN_ID,
  best_model_name: 'LightGBM',
  best_metrics: { f1: ALL_CANDIDATES[0].f1, accuracy: ALL_CANDIDATES[0].accuracy },
  data_quality_score: 87,
  top_models: TOP_MODELS,
  all_candidates: ALL_CANDIDATES,
  executive_summary: `# Executive Summary — Customer Churn Study

**Run** \`${RUN_ID}\` · **Mode** Local (deterministic) · **Engine** Pandas

## Objective
Predict customer churn, discover key behavioral drivers, and export top performing ML models.

## Data Quality Assessment
The Data Quality agent scored the raw dataset **87/100**. Nine duplicate rows were removed and
two implicit-null columns were re-typed. The \`customer_id\` column was flagged as an identifier
and excluded from feature space.

## Key Behavioral Drivers
- **Tenure** — churn risk decays sharply after month 12; new cohorts churn at ~3.1x the rate.
- **Contract type** — month-to-month contracts dominate the churn class (61.8% of churners).
- **Monthly charges** — the top price quartile churns disproportionately when paired with low service counts.

## Model Leaderboard
| Model | F1 | Accuracy | ROC-AUC |
| --- | --- | --- | --- |
| LightGBM | ${ALL_CANDIDATES[0].f1} | ${ALL_CANDIDATES[0].accuracy} | ${ALL_CANDIDATES[0].roc_auc} |
| XGBoost | ${ALL_CANDIDATES[1].f1} | ${ALL_CANDIDATES[1].accuracy} | ${ALL_CANDIDATES[1].roc_auc} |
| RandomForest | ${ALL_CANDIDATES[2].f1} | ${ALL_CANDIDATES[2].accuracy} | ${ALL_CANDIDATES[2].roc_auc} |

## Strategic Recommendations
1. **Target the first-year window** — deploy a retention offer within the first 90 days of tenure.
2. **Contract conversion campaign** — incentivize month-to-month customers toward annual terms.
3. **Bundle monitoring** — customers with fewer than 3 services and high charges warrant proactive outreach.

### Reproducibility
The exported Jupyter notebook re-runs end-to-end with \`random_seed = 42\`; all scalers were
fitted strictly on the training partition (zero-leakage guarantee verified by the Leakage Guard).`,
};

function warmSvg(title, bars) {
  const max = Math.max(...bars.map((b) => b[1]));
  const width = 640;
  const height = 360;
  const barW = 54;
  const gap = (width - 80 - bars.length * barW) / (bars.length + 1);
  const colors = ['#d97706', '#c2410c', '#b45309', '#0e7a4f', '#9a3412', '#8a8f3c', '#b8863b'];
  const rects = bars
    .map(([label, v], i) => {
      const h = (v / max) * 220;
      const x = 55 + gap + i * (barW + gap);
      const y = 290 - h;
      return `<rect x="${x}" y="${y}" width="${barW}" height="${h}" rx="6" fill="${colors[i % colors.length]}" opacity="0.88"/>
<text x="${x + barW / 2}" y="${y - 8}" text-anchor="middle" font-size="13" fill="#57503f" font-family="monospace">${v}</text>
<text x="${x + barW / 2}" y="315" text-anchor="middle" font-size="11.5" fill="#8b8270" font-family="monospace">${label}</text>`;
    })
    .join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="${width}" height="${height}" fill="#fffdf7"/>
<text x="24" y="36" font-size="16" font-weight="600" fill="#231f16" font-family="Georgia,serif">${title}</text>
<line x1="40" y1="290" x2="${width - 24}" y2="290" stroke="#d2c9b2" stroke-width="1.5"/>
${rects}
</svg>`;
}

const VIZ = [
  { name: 'target_distribution.png', category: 'EDA', svg: warmSvg('Churn distribution', [['No', 430], ['Yes', 70]]) },
  { name: 'correlation_heatmap.png', category: 'EDA', svg: warmSvg('Feature correlation', [['tenure', 0.62], ['charges', 0.55], ['contract', 0.74], ['services', 0.41], ['age', 0.22], ['senior', 0.18]]) },
  { name: 'numeric_distributions.png', category: 'EDA', svg: warmSvg('Numeric feature skew', [['age', 0.3], ['charges', 1.2], ['tenure', 0.8], ['services', 0.5]]) },
  { name: 'confusion_matrix.png', category: 'DIAGNOSTICS', svg: warmSvg('Confusion matrix (holdout)', [['TP', 812], ['FP', 61], ['FN', 84], ['TN', 243]]) },
  { name: 'feature_importance.png', category: 'MODEL', svg: warmSvg('Feature importance (top 7)', [['tenure', 0.31], ['contract', 0.24], ['charges', 0.18], ['services', 0.11], ['payment', 0.08], ['age', 0.05], ['senior', 0.03]]) },
  { name: 'calibration_curve.png', category: 'DIAGNOSTICS', svg: warmSvg('Probability calibration', [['0.2', 0.19], ['0.4', 0.42], ['0.6', 0.57], ['0.8', 0.83], ['1.0', 0.97]]) },
];

const FOLDERS = {
  '01_Raw_Data': {
    description: 'Immutable copy of the raw uploaded dataset',
    files: [{ path: '01_Raw_Data/sample_churn.csv', name: 'sample_churn.csv', size_kb: 148.2 }],
  },
  '02_Cleaned_Data': {
    description: 'Sanitized, type-validated dataset',
    files: [{ path: '02_Cleaned_Data/cleaned.csv', name: 'cleaned.csv', size_kb: 146.9 }],
  },
  '03_Feature_Engineered_Data': {
    description: 'Interaction & temporal engineered features',
    files: [{ path: '03_Feature_Engineered_Data/features.csv', name: 'features.csv', size_kb: 212.4 }],
  },
  '04_ML_Ready_Data': {
    description: 'Imputed, scaled, encoded ML matrices',
    files: [
      { path: '04_ML_Ready_Data/X_train.csv', name: 'X_train.csv', size_kb: 189.1 },
      { path: '04_ML_Ready_Data/X_test.csv', name: 'X_test.csv', size_kb: 48.3 },
    ],
  },
  '05_Notebook': {
    description: 'Standalone validated Jupyter notebook',
    files: [{ path: '05_Notebook/audas_run.ipynb', name: 'audas_run.ipynb', size_kb: 96.7 }],
  },
  '06_Models': {
    description: 'Serialized top-model artifacts',
    files: TOP_MODELS.map((m) => ({
      path: `06_Models/${m.filename}`,
      name: m.filename,
      size_kb: +(2.4 + Math.random()).toFixed(1),
    })),
  },
  '07_Visualizations': {
    description: 'High-resolution EDA & diagnostic charts',
    files: VIZ.map((v) => ({ path: `07_Visualizations/${v.name}`, name: v.name, size_kb: 84 })),
  },
  '08_Reports': {
    description: 'Chief AI Scientist executive summaries',
    files: [
      { path: '08_Reports/executive_summary.md', name: 'executive_summary.md', size_kb: 6.1 },
      { path: '08_Reports/executive_summary.pdf', name: 'executive_summary.pdf', size_kb: 118.0 },
    ],
  },
  '09_Experiments': {
    description: 'Full candidate benchmark logs',
    files: [{ path: '09_Experiments/experiments.csv', name: 'experiments.csv', size_kb: 3.8 }],
  },
  '10_Metadata': {
    description: 'Complete agent execution state history',
    files: [{ path: '10_Metadata/run_state.json', name: 'run_state.json', size_kb: 41.2 }],
  },
};

const PROGRESS_LINES = [
  '[INIT] AUDAS master orchestrator online · run state sealed (immutable audit trail)',
  '[INIT] Engine: pandas · autonomy: fully_autonomous · seed: 42',
  '[PROFILER] Schema resolved: 500 rows × 9 columns · 6 numeric, 3 categorical',
  '[PROFILER] Cardinality check: customer_id flagged as identifier → excluded from feature space',
  '[AI PLANNER] Objective parsed: churn prediction + driver discovery · task type: binary classification',
  '[AI PLANNER] Strategy: stratified holdout 70/15/15 · primary metric F1',
  '[DATA QUALITY] Health score computed: 87/100',
  '[DATA QUALITY] Anomalies: 9 duplicate rows, 2 implicit-null columns · audit logged',
  '[EDA] Distribution & correlation plots queued for 07_Visualizations',
  '[CLEANER] Deduplicated 9 rows · re-typed total_services to integer · 491 rows retained',
  '[SPLITTER] Train 343 / val 74 / test 74 (stratified)',
  '[LEAKAGE GUARD] Pre-fit boundary audit passed · zero target leakage across partitions',
  '[AI FEATURE ENG] Synthesized 4 features: tenure_band, charge_per_service, is_month_to_month, services_bucket',
  '[PREPROCESS] One-hot encoding (3 cols) · median imputation · robust scaling fitted on train only',
  '[ML EXPERIMENT] Training candidate machine learning models — 7 algorithms queued',
  '[EVALUATION] Holdout metrics computed · residual diagnostics clean',
  '[REPLANNING] Metric threshold met on attempt 1 · converged',
  '[NOTEBOOK] Executable notebook generated and validated (reproducible, seed=42)',
  '[EXECUTIVE REPORT] Chief AI Scientist narrative synthesized · 8_Reports/executive_summary.md',
  '[COMPLETE] 10-folder project archive sealed · 4 models exported',
];

/* ------------------------------ server ------------------------------ */

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const path = url.pathname;
  const json = (obj, status = 200) => {
    res.writeHead(status, { 'Content-Type': 'application/json', ...CORS });
    res.end(JSON.stringify(obj));
  };

  if (req.method === 'OPTIONS') {
    res.writeHead(204, CORS);
    return res.end();
  }

  if (path === '/api/health') return json({ status: 'ok', mock: true });

  if (path === '/api/settings' && req.method === 'GET') return json(SETTINGS);
  if (path === '/api/settings' && req.method === 'PUT') return json({ saved: true });

  if (path === '/api/test-connection') return json({ success: true, message: 'Connection OK — mock provider responded in 42ms' });
  if (path === '/api/fetch-models') return json({ models: SETTINGS.default_models.openrouter });

  if (path === '/api/sample-data') {
    return json({ path: 'mock://sample_churn.csv', filename: 'sample_churn.csv', rows: 500, features: 9 });
  }

  if (path === '/api/preview') {
    return json({ columns: COLUMNS, rows: PREVIEW_ROWS.slice(0, Number(url.searchParams.get('rows') || 5)), total_rows: 500, total_columns: 9 });
  }

  if (path === '/api/pipeline/run') return json({ run_id: RUN_ID });

  if (path.startsWith('/api/pipeline/') && path.endsWith('/stream')) {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      ...CORS,
    });
    let i = 0;
    const timer = setInterval(() => {
      if (i < PROGRESS_LINES.length) {
        res.write(`data: ${JSON.stringify({ type: 'progress', message: PROGRESS_LINES[i++] })}\n\n`);
      } else {
        clearInterval(timer);
        res.write(`data: ${JSON.stringify({ type: 'complete', message: 'done' })}\n\n`);
        res.end();
      }
    }, 420);
    req.on('close', () => clearInterval(timer));
    return;
  }

  if (path.startsWith('/api/pipeline/') && path.endsWith('/result')) {
    return json({ status: 'completed', result: RESULT });
  }

  if (path.startsWith('/api/pipeline/') && path.endsWith('/files')) {
    return json({ folders: FOLDERS });
  }

  if (path.startsWith('/api/pipeline/') && path.endsWith('/visualizations')) {
    return json({ images: VIZ.map(({ name, category, svg }) => ({ name, category, url: `/mock-charts/${encodeURIComponent(name)}` })) });
  }

  if (path.startsWith('/mock-charts/')) {
    const name = decodeURIComponent(path.replace('/mock-charts/', ''));
    const found = VIZ.find((v) => v.name === name);
    if (found) {
      res.writeHead(200, { 'Content-Type': 'image/svg+xml', ...CORS });
      return res.end(found.svg);
    }
  }

  // Model / file / zip downloads: serve a tiny placeholder binary
  if (path.includes('/download') || path.endsWith('/zip')) {
    res.writeHead(200, {
      'Content-Type': 'application/octet-stream',
      'Content-Disposition': 'attachment; filename="mock_artifact.bin"',
      ...CORS,
    });
    return res.end('mock artifact');
  }

  json({ detail: `Mock backend has no handler for ${req.method} ${path}` }, 404);
});

server.listen(PORT, () => {
  console.log(`Mock AUDAS backend ready on http://localhost:${PORT}`);
});
