# AUDAS Pitch Demo Video Script

**Product:** Autonomous AI Data Scientist (AUDAS)
**Format:** 16:9 product demo, screen recording with voice-over
**Target duration:** 3:30–4:00
**Primary demo path:** React/Vite dashboard in **Local (Offline)** mode using the built-in `sample_churn.csv` dataset
**Core message:** Turn a dataset and a plain-English goal into a complete, reproducible machine-learning project—without stitching together a dozen tools.

## Production guardrails

- Use the built-in Customer Churn sample for a deterministic recording: 500 rows, 9 features.
- Run Local mode for the main take; it needs no API key and avoids network latency.
- Do not invent metrics. Capture the actual values shown in the run summary and substitute them for `[BEST_MODEL]`, `[PRIMARY_METRIC]`, `[DATA_HEALTH]`, and `[RUN_ID]`.
- If showing AI mode, show it as an optional configuration path and avoid recording API keys.
- Record the dashboard, not the settings drawer, unless demonstrating provider configuration with redacted credentials.

## Shot-by-shot script

| Time | Visual / action | Voice-over | On-screen text |
|---|---|---|---|
| 0:00–0:12 | Cold open: AUDAS dashboard, dark obsidian and purple glass UI. Slow zoom toward the title. | “Most data-science projects do not fail because the model is impossible. They fail because profiling, cleaning, experimentation, reporting, and handoff are scattered across disconnected tools. AUDAS brings that entire lifecycle into one autonomous workspace.” | **AUDAS — Autonomous AI Data Scientist** |
| 0:12–0:24 | Point to the five navigation surfaces: Launch Pipeline, Executive Report, Visualizations, Top Models, Artifacts. | “From one interface, you can launch a run, inspect the report, explore diagnostics, compare models, and download every generated artifact.” | **One workspace. Full ML lifecycle.** |
| 0:24–0:42 | In Data Ingestion, switch to **Use Sample Dataset** and select Customer Churn. Expand Dataset Preview. | “Start with your own CSV, Excel, or Parquet file—or use the built-in sample to get moving instantly. AUDAS uploads the file with progress feedback, previews the schema and rows, and preserves the original dataset as an auditable raw copy.” | **CSV · XLSX · Parquet · preview before you run** |
| 0:42–0:58 | Move to Task & Objective. Keep the prefilled churn objective; leave target blank, then briefly type `churn` or show auto-detection. | “Next, describe the outcome in natural language. The goal planner turns that sentence into a structured task strategy, while the target can be detected automatically or specified explicitly.” | **Natural-language objective → structured plan** |
| 0:58–1:12 | Show the Local/AI mode switch. Keep Local selected; optionally open settings briefly and show provider list without entering secrets. | “Choose Local Deterministic mode for an offline, reproducible run, or switch to AI-Powered mode for LLM-assisted planning, feature suggestions, preprocessing rationale, and richer report narrative. Providers are pluggable—from OpenRouter and OpenAI to Gemini, Anthropic, NVIDIA NIM, Ollama, and custom OpenAI-compatible endpoints.” | **Local for speed. AI when you want deeper reasoning.** |
| 1:12–1:18 | Click **Launch Autonomous Pipeline**. | “With the data and objective ready, launch the autonomous pipeline.” | **One click to start** |
| 1:18–1:58 | Keep the live pipeline panel on screen. Let the log stream show INIT, PROFILER, PLANNER, DATA QUALITY, EDA, CLEANER, SPLITTER, LEAKAGE GUARD, FEATURE ENG, PREPROCESS, ML EXPERIMENT, EVALUATION, NOTEBOOK, and REPORT. | “The supervisor orchestrator coordinates specialist agents end to end. AUDAS profiles columns, detects missingness and anomalies, plans the task, generates exploratory charts, cleans and deduplicates data, creates train/validation/test partitions, checks for leakage, engineers features, and fits an adaptive preprocessing pipeline.” | **Live agent trace · real-time progress · run ID** |
| 1:58–2:14 | Pause on ML experiment progress as candidate models increment. | “Then the experiment agent benchmarks multiple candidate algorithms, ranks them using task-appropriate metrics, and exports the strongest four models. Classification runs can surface accuracy, precision, recall, F1, and ROC-AUC; regression runs can surface RMSE, MAE, and R-squared.” | **Automated model benchmarking, not one-shot guessing** |
| 2:14–2:28 | Show completed Run Summary. Zoom to winning model, primary metric, candidate count, health score, and ZIP button. | “When the run completes, you get a concise decision surface: the winning model, the primary score, the number of exported candidates, the data-health score, and a one-click project archive.” | **Winner: [BEST_MODEL] · Score: [PRIMARY_METRIC] · Health: [DATA_HEALTH]/100** |
| 2:28–2:48 | Open **Top Models Leaderboard**. Scroll from top-four model cards to the complete benchmark table. Click one model download if useful. | “The leaderboard makes model selection explainable. Each card shows rank, metrics, training time, selection rationale, and a download action. The full benchmark table keeps every evaluated candidate visible instead of hiding the comparison.” | **Ranked candidates · metrics · selection rationale** |
| 2:48–3:03 | Open **Visualizations**. Click a category filter, open a correlation or confusion-matrix chart in the lightbox, and show the high-resolution PNG download. | “EDA and evaluation are visual by default: distributions, categorical frequencies, correlations, outliers, target relationships, confusion matrices, and regression diagnostics. Filter by category, open a chart full screen, and download the high-resolution image.” | **Explore evidence, not just scores** |
| 3:03–3:22 | Open **In-Depth Executive Report**. Scroll through objective, data profile, quality, model comparison, preprocessing, leakage, and recommendations. Show PDF, DOCX, and Markdown buttons. | “AUDAS turns the run into an executive narrative covering the objective, dataset profile, quality findings, model comparison, preprocessing, leakage controls, and recommendations. The same report is available as Markdown, PDF, and Word.” | **From experiment output to decision-ready briefing** |
| 3:22–3:45 | Open **Dataset Versions & Artifacts**. Expand folders 01, 05, 06, 07, 08, 09, and 10. End on the ZIP download. | “Finally, the artifact explorer exposes the complete lineage: immutable raw data, cleaned and feature-engineered data, ML-ready matrices, a generated and validated Jupyter notebook, serialized models, visualizations, reports, experiment logs, and run metadata—organized into ten predictable project folders.” | **Reproducible delivery, not a screenshot of a result** |
| 3:45–3:58 | Fast montage: dashboard → pipeline trace → leaderboard → report → ZIP. End on AUDAS logo/title. | “Whether you are a data scientist, analyst, or product team, AUDAS compresses the path from raw table to reproducible machine-learning project into one guided run.” | **Raw data in. Reproducible ML project out.** |
| 3:58–4:05 | End card with repository/demo URLs if desired. | “This is autonomous data science with a visible audit trail.” | **AUDAS — Discover. Build. Explain. Export.** |

## Feature coverage checklist

### Input and planning

- Upload or drag-and-drop CSV, XLSX, and Parquet datasets.
- Upload progress and backend-ready confirmation.
- Built-in Customer Churn sample dataset.
- Dataset preview with rows, columns, and values.
- Natural-language objective entry.
- Optional target-column override and automatic target inference.
- Automatic task planning and objective decomposition.

### Execution and intelligence

- Local deterministic execution with no API key.
- AI-powered execution with provider/model selection.
- OpenRouter, NVIDIA NIM, Google Gemini, OpenAI, Anthropic, Ollama, and custom OpenAI-compatible endpoints.
- Dynamic model fetching and API connection testing.
- Autonomy levels: fully autonomous, semi-autonomous, and manual approval.
- Configurable processing engine selection: Pandas, Polars, and DuckDB.
- Configurable random seed and storage location.

### Autonomous data-science lifecycle

- Dataset profiling: schema, types, missingness, cardinality, skew, and candidate targets.
- Data-quality scoring and anomaly audit.
- EDA chart generation.
- Cleaning: placeholder null handling, type fixes, deduplication, imputation, date parsing, and outlier treatment.
- Train/validation/test splitting and stratification where applicable.
- Leakage audit and proxy pruning.
- Feature engineering: interactions, mathematical transforms, differences, and temporal features.
- Adaptive preprocessing: imputation, scaling, one-hot encoding, and frequency encoding.
- Candidate model training, ranking, and top-model export.
- Holdout evaluation and diagnostic plots.
- Replanning decision record.
- Live Server-Sent Events progress stream with tagged phases and error details.

### Results and delivery

- Run ID and completion summary.
- Primary metric and best-model display.
- Top-four model cards plus full candidate benchmark table.
- Model downloads and serialized `.pkl` artifacts.
- High-resolution EDA and evaluation visualizations with filters, fullscreen lightbox, and PNG download.
- Executive report rendered in the web UI.
- Markdown, PDF, and DOCX report downloads.
- Generated, programmatically validated Jupyter notebook.
- Ten-folder artifact contract with raw, cleaned, engineered, ML-ready, notebook, models, visualizations, reports, experiments, and metadata.
- One-click ZIP export of the complete project.

## Optional 20-second architecture insert

Use this only if the audience is technical. Animate the flow:

`Dataset + objective → Orchestrator → Profiler → Planner → Quality + EDA → Cleaning → Split + Leakage Guard → Feature Engineering → Preprocessing → Model Experiments → Evaluation → Notebook + Report + ZIP`

Voice-over: “Under the interface is a supervisor–specialist architecture. Each phase records decisions, metrics, and artifacts, so the final answer is not only a prediction—it is a reproducible project with evidence.”

## Brag-ready generation prompt

Use the following as the input to the `brag` video-generation skill when it is available:

> Create a polished 16:9 startup-pitch product demo video for AUDAS, the Autonomous AI Data Scientist. Use a premium dark obsidian and radiant royal-purple visual language with restrained neon accents, crisp kinetic typography, subtle glassmorphism, smooth cursor-led screen transitions, and confident but factual narration. Follow `AADS_PITCH_DEMO_VIDEO_SCRIPT.md` shot by shot. Record the local React/Vite dashboard at `http://localhost:5173` backed by FastAPI at `http://localhost:8000`, using the built-in Customer Churn sample in Local mode. Capture the live multi-agent trace, completed run summary, model leaderboard, visualization lightbox, executive report downloads, and ten-folder artifact explorer. Use actual runtime values for the winning model, metric, health score, and run ID; never fabricate metrics. End with the message: “Raw data in. Reproducible ML project out.”

## Recording setup

```powershell
# Terminal 1 — backend
$env:PYTHONPATH="C:\Users\thaku\hackathon&projects\Agentic AI Data Scientist"
& .venv\Scripts\python.exe -m uvicorn aads.api.server:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd "C:\Users\thaku\hackathon&projects\Agentic AI Data Scientist\frontend"
npm run dev
```

Open `http://localhost:5173`, use Local mode, select the sample churn dataset, and run the pipeline before recording the result tabs.

## Accuracy notes for the presenter

- Say “leakage checks and boundary guards,” not “mathematically guaranteed zero leakage.”
- Say “candidate models” rather than promising a fixed number in every future configuration.
- Use captured runtime metrics instead of examples from the README.
- AI mode is optional and depends on a configured provider, valid credentials, and network access.
- The repository also includes a Streamlit dashboard and a CLI runner; mention these as alternate surfaces in a technical appendix rather than switching away from the main web demo.
