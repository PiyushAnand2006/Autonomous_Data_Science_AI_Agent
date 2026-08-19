# Autonomous AI Data Scientist (AADS)

> A reproducible autonomous AI Data Scientist that transforms a raw dataset and a natural-language objective into a complete, reusable, validated data-science project.

## Overview

AADS accepts a tabular dataset (CSV, Excel, Parquet) along with a natural-language objective and autonomously performs:

- **Profiling** — schema, distributions, missing values, memory estimation
- **EDA** — context-aware analysis with saved visualizations
- **Cleaning** — justified, logged transformations with approval modes
- **Leakage checks** — dedicated guards against train/test contamination
- **Feature engineering** — candidate generation, evaluation, acceptance/rejection
- **Preprocessing** — encoding, scaling, pipeline fitting (train-only)
- **ML experimentation** — baseline → experiment → evaluate → replan loop
- **Artifact export** — notebook, models, pipelines, reports, datasets, logs

## Quickstart

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Clone reference repositories
.\aads\scripts\clone_references.ps1

# 4. Run tests
python -m pytest aads/tests/ -v
```

## Project Structure

```
aads/
├── core/           # Config, state, schemas, logging, exceptions
├── agents/         # Specialist agents (orchestrator, profiler, etc.)
├── tools/          # Deterministic Python tools (loaders, viz, ML)
├── prompts/        # LLM prompt templates
├── templates/      # Notebook/report templates
├── storage/        # Generated runs and cache
├── app/            # UI (Streamlit) and API (FastAPI) layer
├── scripts/        # Utility scripts
└── tests/          # Unit, integration, and regression tests
```

## Design Principles

1. **Original data is immutable** — raw uploads are never modified
2. **Reproducibility first** — every decision logged, every run has a unique ID
3. **Agent actions are tool-backed** — LLM plans, Python tools execute
4. **No data leakage** — train/test boundaries enforced before fitting
5. **Dynamic workflows** — plans adapt to dataset characteristics
6. **Modular architecture** — agents are independently testable

## Architecture

The system uses a **supervisor + specialist agent** pattern orchestrated via LangGraph:

```
Orchestrator → Profiler → Planner → Data Quality → EDA
    → Cleaning → Split Manager → Leakage Guard
    → Feature Engineering → Preprocessing
    → ML Experiment → Evaluation → Replanning (loop)
    → Notebook Generator → Report Generator → Artifact Manager
```

## Autonomy Modes

| Mode | Behavior |
|------|----------|
| **Fully Autonomous** | All decisions automated within safety limits |
| **Semi-Autonomous** | Routine actions auto, high-impact decisions need approval |
| **Manual Approval** | Major decisions require explicit user approval |

## Contributing

See `MASTER_PLAN .md` for the complete design specification and phase-by-phase implementation plan.

## License

MIT
