# 🧠 Autonomous AI Data Scientist (AADS)

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://autonomous-data-science-ai-agent.streamlit.app/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%20Core-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An Agentic AI Data Scientist that autonomously transforms raw tabular datasets and natural-language objectives into full-lifecycle, production-ready, reproducible Machine Learning projects.**

🚀 **Live Web Application**: [https://autonomous-data-science-ai-agent.streamlit.app/](https://autonomous-data-science-ai-agent.streamlit.app/)

</div>

---

## 📌 Table of Contents
- [✨ Key Features](#-key-features)
- [🤖 Agentic Multi-Agent Architecture](#-agentic-multi-agent-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [🎨 Modern UI & Experience](#-modern-ui--experience)
- [🚀 Live Demo & How to Use](#-live-demo--how-to-use)
- [📦 Standardized 10-Folder Artifact Contract](#-standardized-10-folder-artifact-contract)
- [⚙️ Local Installation & Quickstart](#️-local-installation--quickstart)
- [🛡️ Guardrails & Core Principles](#️-guardrails--core-principles)
- [📄 License](#-license)

---

## ✨ Key Features

- 🧠 **Autonomous Decision-Making**: Coordinates 14+ specialist agents to profile, clean, engineer features, benchmark ML models, and generate analytical executive reports without manual intervention.
- 🌐 **Multi-Provider LLM Intelligence**: Pluggable agent reasoning powered by **OpenRouter, Google Gemini, Anthropic Claude, OpenAI, Groq, or Ollama (Local)** with dynamic searchable model discovery.
- ⚡ **Dual Execution Engines (AI vs Local)**:
  - **✨ AI-Powered Agentic Mode**: Uses LLM cognition to reason over edge cases, formulate strategic analytical plans, and write custom diagnostic interpretations.
  - **💻 Local (Deterministic) Mode**: Fully offline, rule-based, deterministic pipeline executing instantly without external API keys.
- 🛡️ **Zero Data-Leakage Guarantee**: Enforces rigorous train/val/test boundary guards before scaling and encoding transformations.
- 🏆 **Comprehensive Model Leaderboard**: Evaluates 10+ candidate algorithms (RandomForest, ExtraTrees, GradientBoosting, XGBoost, CatBoost, LightGBM, Linear/Logistic models) with multi-metric ranking and download capabilities.
- 🔍 **Interactive Dataset Exploration**: Instant dataset previews with collapsible accordion viewers and authentic raw feature casing preservation.
- 📦 **1-Click Full Project Export**: Exports a self-contained `.ZIP` bundle containing cleaned CSVs, serialized model weights (`.pkl`), an interactive validated Jupyter Notebook (`.ipynb`), high-res EDA charts (`.png`), and executive markdown reports (`.md`).
- 📁 **High-Capacity Dataset Uploads**: Supports CSV, XLSX, and Parquet uploads with support for up to 500 MB datasets.

---

## 🤖 Agentic Multi-Agent Architecture

AADS is built on a **Supervisor–Specialist Agentic Pattern**, where a central Orchestrator manages specialized autonomous workers through every phase of the data science lifecycle:

```mermaid
flowchart TB
    %% Inputs
    subgraph Inputs ["📥 1. Input & Objective Ingestion"]
        direction TB
        RawData["📁 Raw Tabular Dataset<br/><i>(CSV, XLSX, Parquet up to 500MB)</i>"]
        UserGoal["🎯 Natural Language Objective<br/><i>(e.g., 'Predict churn & find drivers')</i>"]
    end

    %% Supervisor
    subgraph Supervisor ["🧠 Master Supervisor Layer"]
        direction TB
        Orchestrator["⚡ AADS Master Orchestrator<br/><i>(Lifecycle Coordination & Event Routing)</i>"]
        LLM["🤖 Multi-LLM Reasoning Engine<br/><i>(Claude 3.7 / GPT-4.5 / Gemini 2.0 / DeepSeek / Groq / Ollama)</i>"]
        RunState["🗄️ Immutable Run State & Audit Trail<br/><i>(Decisions, Lineage & Metrics)</i>"]
    end

    %% Phase 1: Understanding & Planning
    subgraph Discovery ["🔍 Phase 1: Profiling & Planning"]
        direction TB
        Profiler["📊 Profiler Agent<br/><i>Schema, Statistical Skew, Card</i>"]
        Planner["📋 Goal Planner Agent<br/><i>Hypotheses & Task Strategy Plan</i>"]
        DQ["🛡️ Data Quality Agent<br/><i>Health Scoring (0-100) & Anomalies</i>"]
        EDA["📈 EDA & Viz Agent<br/><i>Distributions, Outliers & Correlations</i>"]
    end

    %% Phase 2: Data Curation & Integrity
    subgraph Prep ["🧹 Phase 2: Data Curation & Integrity"]
        direction TB
        Cleaning["🧼 Data Cleaning Agent<br/><i>Deduplication & Sanitization</i>"]
        Split["✂️ Split Manager<br/><i>Train / Val / Test Partitioning</i>"]
        LeakGuard["🔒 Leakage Guard Agent<br/><i>Pre-fitting Boundary Audit</i>"]
    end

    %% Phase 3: Feature Engineering & Preprocessing
    subgraph MLPipeline ["⚙️ Phase 3: Feature Engineering & Preprocessing"]
        direction TB
        FE["🔧 Feature Engineering Agent<br/><i>Interactions, Math & Temporal</i>"]
        PrepAgent["🛠️ Adaptive Preprocessor Agent<br/><i>Encoding, Imputation & Scaling</i>"]
    end

    %% Phase 4: Modeling & Replanning
    subgraph Experimentation ["🏆 Phase 4: Autonomous ML Benchmarking"]
        direction TB
        MLExp["🤖 ML Experiment Agent<br/><i>10+ Models: RF, ExtraTrees, XGB, CatBoost, LGBM</i>"]
        Eval["📉 Evaluation & Diagnostics Agent<br/><i>Holdout Metrics, F1/RMSE & Residuals</i>"]
        Replan{"🔄 Replanning Loop<br/><i>Metric Threshold Met?</i>"}
    end

    %% Phase 5: Synthesis & Export
    subgraph Packaging ["📦 Phase 5: Synthesis & 10-Folder Delivery"]
        direction TB
        Notebook["📓 Notebook Generator<br/><i>Executable, Tested .ipynb</i>"]
        Reporter["📄 Chief AI Report Agent<br/><i>Executive Business Narrative .md</i>"]
        Artifacts["🗂️ 10-Folder Standardized Project ZIP<br/><i>Models (.pkl), Datasets, Visuals, Code</i>"]
    end

    %% Flow Connections
    Inputs --> Orchestrator
    Orchestrator <--> LLM
    Orchestrator <--> RunState

    Orchestrator --> Discovery
    Profiler --> Planner --> DQ --> EDA

    Discovery --> Prep
    Cleaning --> Split --> LeakGuard

    Prep --> MLPipeline
    FE --> PrepAgent

    MLPipeline --> Experimentation
    MLExp --> Eval --> Replan
    Replan -- "Iterate / Tune" --> MLExp
    Replan -- "Converged / Optimal" --> Packaging

    Notebook --> Artifacts
    Reporter --> Artifacts
```

### 👥 The Specialist Agents:
1. **Goal Planner Agent**: Deconstructs natural language user goals into structured hypotheses and analytical roadmaps.
2. **Profiler Agent**: Inspects dataset shapes, types, sparsity, skewness, and cardinality.
3. **Data Quality Agent**: Calculates holistic data health scores (0-100) and detects anomalies.
4. **EDA Agent**: Generates publication-grade distribution, correlation, and categorical visualization plots.
5. **Cleaning Agent**: Sanitizes null values, fixes types, and eliminates duplicates with transparent audit logs.
6. **Split Manager**: Partitions data into strictly separated Train, Validation, and Test holdouts.
7. **Leakage Guard**: Validates zero feature overlap or target leakage across partitions.
8. **Feature Engineering Agent**: Synthesizes interaction terms, polynomial features, and domain-specific indicators.
9. **Preprocessing Agent**: Applies adaptive one-hot / target encoding, imputation, and feature scaling.
10. **ML Experiment Agent**: Trains candidate algorithms and maintains comprehensive benchmark rankings.
11. **Evaluation Agent**: Computes holdout metrics (F1, ROC-AUC, RMSE, MAE, R²) and residual diagnostics.
12. **Replanning Agent**: Iteratively refines modeling strategies based on validation feedback.
13. **Notebook Generator Agent**: Auto-synthesizes a standalone, executable, reproducible `.ipynb` notebook.
14. **Chief AI Report Agent**: Synthesizes in-depth business narratives and strategic recommendations.

---

## 🛠️ Tech Stack

### 🧠 Agentic AI & LLM Orchestration
- **LangChain / LangChain Core**: Standardized prompt formatting, messaging, and multi-provider agent interfaces.
- **Multi-LLM Provider Engine**: Zero-dependency direct integration with **OpenRouter, Anthropic Claude 3.7 / 3.5, OpenAI GPT-4.5 / GPT-4o, Google Gemini 2.0, DeepSeek, Groq, and Ollama**.
- **Pydantic & Pydantic-Settings v2**: Robust type validation, state schemas, and environment configuration.

### 📊 Machine Learning & Data Engines
- **Scikit-Learn**: Core ML algorithms, pipelines, cross-validation, and metrics evaluation.
- **XGBoost, CatBoost, LightGBM**: State-of-the-art gradient boosted decision trees.
- **Pandas, Polars, PyArrow, OpenPyXL**: High-performance multi-engine tabular processing.
- **DuckDB**: Fast in-process analytical SQL queries.

### 🎨 Frontend UI & Visualization
- **React 18 + Vite**: High-performance single-page app with floating navbar, real-time pipeline visualizer, and artifact manager.
- **Deep Black & Radiant Royal Purple Design System**: Modern dark-mode aesthetic with ambient purple glows, frosted double-bezel cards, and responsive micro-interactions.
- **FastAPI (Async Backend)**: High-speed ASGI REST API and Server-Sent Events (SSE) log streaming.
- **Streamlit**: Classic interactive web dashboard.
- **Plotly & Matplotlib**: Interactive and high-res static diagnostic visualizations.

---

## 🎨 Modern UI & Experience

AADS offers a **React 18 + Vite** web interface powered by a **FastAPI backend**:
- **Radiant Purple Aesthetic**: Built with deep obsidian surfaces (`#05020c`), radiant purple lighting, and frosted glass components.
- **Real-Time Pipeline Stepper**: Live tracking of all 8 pipeline phases with elapsed timers, progress percentages, and streaming audit logs.
- **Frontier Model Combobox**: Search, type, or pick top frontier models with instant keyboard navigation.
- **Dedicated Artifact Explorer**: In-browser tabs for inspecting EDA visuals, model leaderboards, data quality scores, and full project downloads.

---

## 🚀 Live Demo & How to Use

Try the live web application here:
🔗 **[https://autonomous-data-science-ai-agent.streamlit.app/](https://autonomous-data-science-ai-agent.streamlit.app/)**

### 3-Step Workflow:
1. **Upload Dataset / Pick Sample**: Upload your CSV, Excel, or Parquet dataset (or choose the built-in Customer Churn dataset).
2. **Define Goal**: Enter your goal in natural English (e.g. *"Predict customer churn, discover key behavioral drivers, and export top models"*).
3. **Execute & Download**: Run the autonomous pipeline and download the full 10-folder project archive with all trained models, charts, and notebooks.

---

## 📦 Standardized 10-Folder Artifact Contract

Every run creates a standardized, reproducible project directory:

```
Generated_Project_<RUN_ID>/
├── 01_Raw_Data/                 # Immutable copy of raw uploaded data
├── 02_Cleaned_Data/             # Sanitized and type-validated dataset
├── 03_Feature_Engineered_Data/  # Domain interactions and engineered features
├── 04_ML_Ready_Data/            # Imputed, scaled, and encoded ML matrices
├── 05_Notebook/                 # Standalone, validated Jupyter Notebook (.ipynb)
├── 06_Models/                   # Serialized top model artifacts (.pkl) & metadata
├── 07_Visualizations/           # High-resolution EDA and diagnostic charts
├── 08_Reports/                  # Chief AI Scientist executive summaries (.md / .json)
├── 09_Experiments/              # Full candidate experiment benchmark logs (.csv)
└── 10_Metadata/                 # Complete agent execution state history (.json)
```

---

## ⚙️ Local Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ and npm (for React UI)

### 1. Clone & Setup Python Environment
```bash
git clone https://github.com/PiyushAnand2006/Autonomous_Data_Science_AI_Agent.git
cd Autonomous_Data_Science_AI_Agent

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Option A: Run Full-Stack React + FastAPI App (Recommended)
```bash
# Terminal 1 — Start the FastAPI Backend (Port 8000)
python -m uvicorn aads.api.server:app --port 8000

# Terminal 2 — Start the Vite React Frontend (Port 5173)
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

### 3. Option B: Run Streamlit App
```bash
streamlit run aads/app/streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🛡️ Guardrails & Core Principles

- **Data Immutability**: The original raw dataset is treated as read-only and preserved untouched in `01_Raw_Data/`.
- **Strict Leakage Prevention**: Transformers and scalers are fitted strictly on training partitions to ensure zero data leakage.
- **Auditability**: Every decision, imputation rationale, and feature selection is recorded in `10_Metadata/run_state.json`.

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
