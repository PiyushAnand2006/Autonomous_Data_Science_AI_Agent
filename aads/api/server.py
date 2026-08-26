"""
AUDAS REST API — FastAPI backend wrapping the existing AADSOrchestrator.

Run via:
    uvicorn aads.api.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import queue
import sys
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from aads.agents.orchestrator import AADSOrchestrator
from aads.core.config import AADSConfig
from aads.core.llm import DEFAULT_PROVIDER_MODELS, list_provider_models, test_llm_connection
from aads.core.schemas import AutonomyMode, ExecutionEngine
from aads.core.settings import (
    get_stored_api_key,
    load_user_settings,
    save_user_settings,
    set_stored_api_key,
)
from aads.scripts.generate_sample_data import generate_churn_dataset

# ──────────────────────────────────────────────────────────────────────────────
# App & CORS
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AUDAS API",
    description="Autonomous AI Data Scientist (AUDAS) — REST backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://.*",
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# In-memory run store  (for SSE streaming & result retrieval)
# ──────────────────────────────────────────────────────────────────────────────
_runs: Dict[str, Dict[str, Any]] = {}
# Each run: { "status": "running"|"completed"|"error", "queue": Queue, "result": dict|None, "error": str|None }


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────────────────────────────────────
class PipelineRequest(BaseModel):
    data_path: str
    user_objective: str
    target_column: Optional[str] = None
    execution_mode: str = "local"
    autonomy: str = "fully_autonomous"
    engine: str = "pandas"
    random_seed: int = 42
    storage_dir: str = ""
    # AI-mode fields
    llm_provider: str = "openrouter"
    llm_model: str = "anthropic/claude-3.5-sonnet"
    llm_api_key: Optional[str] = None
    custom_base_url: Optional[str] = None
    custom_provider_name: Optional[str] = None


class TestConnectionRequest(BaseModel):
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None


class FetchModelsRequest(BaseModel):
    provider: str
    api_key: str
    base_url: Optional[str] = None


class SettingsUpdate(BaseModel):
    settings: Dict[str, Any]


class ApiKeyUpdate(BaseModel):
    provider: str
    key: str


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints: Health
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "aads-api"}


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints: File Upload
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
def upload_file(file: UploadFile = File(...)):
    temp_dir = _PROJECT_ROOT / "storage" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    clean_filename = Path(file.filename).name
    dest = temp_dir / clean_filename
    
    import shutil
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    size_bytes = dest.stat().st_size
    return {
        "filename": clean_filename,
        "path": str(dest.resolve()),
        "size_bytes": size_bytes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints: Sample Data
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/sample-data")
def create_sample_data():
    sample_dir = _PROJECT_ROOT / "data"
    sample_dir.mkdir(parents=True, exist_ok=True)
    data_path = sample_dir / "sample_churn.csv"
    if not data_path.exists():
        generate_churn_dataset(data_path, n_samples=500)
    return {"path": str(data_path), "filename": "sample_churn.csv", "rows": 500, "features": 9}


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints: Pipeline — Launch
# ──────────────────────────────────────────────────────────────────────────────
def _run_pipeline_thread(run_id: str, req: PipelineRequest):
    """Execute the pipeline in a background thread, posting updates to the SSE queue."""
    run_data = _runs[run_id]
    msg_queue: queue.Queue = run_data["queue"]

    def _on_progress(message: str):
        msg_queue.put({"type": "progress", "message": message})

    try:
        autonomy_map = {
            "fully_autonomous": AutonomyMode.FULLY_AUTONOMOUS,
            "semi_autonomous": AutonomyMode.SEMI_AUTONOMOUS,
            "manual_approval": AutonomyMode.MANUAL_APPROVAL,
        }
        engine_map = {
            "pandas": ExecutionEngine.PANDAS,
            "polars": ExecutionEngine.POLARS,
            "duckdb": ExecutionEngine.DUCKDB,
        }

        cfg_kwargs: Dict[str, Any] = {
            "execution_mode": req.execution_mode,
            "random_seed": req.random_seed,
            "default_engine": engine_map.get(req.engine, ExecutionEngine.PANDAS),
            "top_models_count": 4,
        }
        if req.storage_dir and req.storage_dir.strip():
            cfg_kwargs["storage_dir"] = req.storage_dir.strip()
        if req.execution_mode == "ai":
            cfg_kwargs["llm_provider"] = req.llm_provider
            cfg_kwargs["llm_model"] = req.llm_model
            if req.llm_api_key and req.llm_api_key.strip():
                cfg_kwargs["llm_api_key"] = req.llm_api_key.strip()
            if req.custom_base_url and req.custom_base_url.strip():
                cfg_kwargs["custom_base_url"] = req.custom_base_url.strip()
            if req.custom_provider_name and req.custom_provider_name.strip():
                cfg_kwargs["custom_provider_name"] = req.custom_provider_name.strip()

        config = AADSConfig(**cfg_kwargs)
        orchestrator = AADSOrchestrator(config=config, storage_root=config.storage_root)

        result = orchestrator.run_pipeline(
            data_path=req.data_path,
            user_objective=req.user_objective,
            target_column=req.target_column if req.target_column and req.target_column.strip() else None,
            autonomy_mode=autonomy_map.get(req.autonomy, AutonomyMode.FULLY_AUTONOMOUS),
            progress_callback=_on_progress,
        )

        serialized = _serialize_result(result)
        run_data["result"] = serialized
        run_data["status"] = "completed"
        if serialized.get("run_id") and serialized["run_id"] != run_id:
            _runs[serialized["run_id"]] = run_data
        msg_queue.put({"type": "complete", "message": "Pipeline completed successfully!"})

    except Exception as e:
        run_data["status"] = "error"
        run_data["error"] = str(e)
        msg_queue.put({"type": "error", "message": str(e)})


def _find_run_dir(run_id: str) -> Optional[Path]:
    """Find run directory by API ID, orchestrator hex ID, or on disk in storage/runs/."""
    # 1. Check in-memory _runs
    if run_id in _runs:
        res = _runs[run_id].get("result")
        if res and res.get("run_dir"):
            p = Path(res["run_dir"])
            if p.exists():
                return p

    # 2. Check if any run in _runs has this internal run_id
    for r_key, r_data in _runs.items():
        res = r_data.get("result")
        if res and (res.get("run_id") == run_id or r_key == run_id):
            if res.get("run_dir"):
                p = Path(res["run_dir"])
                if p.exists():
                    return p

    # 3. Check storage/runs/ directory directly on disk
    storage_root = Path("storage/runs")
    if storage_root.exists():
        direct = storage_root / run_id
        if direct.exists():
            return direct
        gen_direct = storage_root / f"Generated_Project_{run_id}"
        if gen_direct.exists():
            return gen_direct
        for folder in storage_root.iterdir():
            if folder.is_dir() and run_id in folder.name:
                return folder

    return None


def sanitize_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-compliant primitives (handling NaN, inf, numpy types, pydantic, paths)."""
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    
    # Check for numpy types safely without hard dependency on numpy imports
    type_name = type(obj).__name__
    module_name = type(obj).__module__
    if "numpy" in module_name:
        if "int" in type_name:
            return int(obj)
        if "float" in type_name:
            val = float(obj)
            return None if (math.isnan(val) or math.isinf(val)) else val
        if "bool" in type_name:
            return bool(obj)
        if hasattr(obj, "tolist"):
            return sanitize_json(obj.tolist())
    
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return sanitize_json(obj.model_dump())
    if hasattr(obj, "dict"):
        try:
            return sanitize_json(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return sanitize_json(vars(obj))
        except Exception:
            pass
    return str(obj)


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert pipeline result to JSON-safe dict."""
    run_dir_str = str(result.get("run_dir") or "")
    run_dir = Path(run_dir_str) if run_dir_str else None
    models_dir = run_dir / "06_Models" if run_dir and run_dir.exists() else None

    top_models_raw = result.get("top_models", [])
    top_models = []
    for tm in top_models_raw:
        m_dict = {}
        if isinstance(tm, dict):
            m_dict = sanitize_json(tm)
        elif hasattr(tm, "model_dump"):
            m_dict = sanitize_json(tm.model_dump())
        elif hasattr(tm, "dict"):
            m_dict = sanitize_json(tm.dict())
        else:
            m_dict = sanitize_json(tm)

        rank = m_dict.get("rank", 1)
        matched_file = None
        if models_dir and models_dir.exists():
            for f in models_dir.glob(f"model_{rank:02d}_*.pkl"):
                matched_file = f.name
                break
        if not matched_file:
            clean_slug = str(m_dict.get("model_name", "model")).lower().replace("classifier", "_classifier").replace("regressor", "_regressor")
            matched_file = f"model_{rank:02d}_{clean_slug}.pkl"

        m_dict["filename"] = matched_file
        m_dict["download_url"] = f"/api/pipeline/{result.get('run_id')}/models/{rank}/download"
        top_models.append(m_dict)

    state = result.get("state")
    dq = result.get("data_quality_report")
    if dq is None and state is not None:
        dq = getattr(state, "data_quality_report", None)

    dq_score = None
    if dq is not None:
        if isinstance(dq, dict):
            dq_score = dq.get("overall_score") or dq.get("score")
        elif hasattr(dq, "overall_score"):
            dq_score = dq.overall_score
    if dq_score is None:
        dq_score = result.get("data_quality_score") or 95.0

    serialized = {
        "run_id": str(result.get("run_id") or ""),
        "run_dir": run_dir_str,
        "best_model_name": str(result.get("best_model_name") or ""),
        "best_metrics": sanitize_json(result.get("best_metrics", {})),
        "top_models": top_models,
        "total_artifacts": int(result.get("total_artifacts", 0)),
        "executive_summary": str(result.get("executive_summary") or ""),
        "data_quality_score": float(dq_score) if dq_score is not None else 95.0,
        "task_type": None,
    }

    # Extract task type from state
    if state is not None:
        tt = getattr(state, "task_type", None)
        if tt is not None:
            serialized["task_type"] = str(tt.value if hasattr(tt, "value") else tt)

    # Extract EDA findings
    eda = result.get("eda_findings")
    if eda is not None:
        serialized["eda_findings"] = sanitize_json(eda)

    # Load all evaluated candidates from model_comparison.json or experiment_results.csv
    all_candidates = []
    if run_dir and run_dir.exists():
        comp_json = run_dir / "06_Models" / "model_comparison.json"
        exp_csv = run_dir / "09_Experiments" / "experiment_results.csv"
        if comp_json.exists():
            try:
                raw_cands = json.loads(comp_json.read_text(encoding="utf-8"))
                if isinstance(raw_cands, list):
                    for idx, c in enumerate(raw_cands):
                        m_name = c.get("model") or c.get("model_name") or "Model"
                        all_candidates.append({
                            "rank": c.get("rank", idx + 1),
                            "model": m_name,
                            "training_time": c.get("training_time_seconds") or c.get("training_time", 0.0),
                            "accuracy": c.get("accuracy"),
                            "f1": c.get("f1"),
                            "precision": c.get("precision"),
                            "recall": c.get("recall"),
                            "roc_auc": c.get("roc_auc"),
                            "rmse": c.get("rmse"),
                            "mae": c.get("mae"),
                            "r2": c.get("r2"),
                            "silhouette": c.get("silhouette"),
                        })
            except Exception:
                pass
        elif exp_csv.exists():
            try:
                import csv
                with open(exp_csv, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        all_candidates.append({
                            "rank": idx + 1,
                            "model": row.get("model", "Model"),
                            "training_time": float(row.get("training_time", 0.0)) if row.get("training_time") else 0.0,
                            "accuracy": float(row["accuracy"]) if row.get("accuracy") else None,
                            "f1": float(row["f1"]) if row.get("f1") else None,
                            "precision": float(row["precision"]) if row.get("precision") else None,
                            "recall": float(row["recall"]) if row.get("recall") else None,
                            "roc_auc": float(row["roc_auc"]) if row.get("roc_auc") and row["roc_auc"] != "None" else None,
                        })
            except Exception:
                pass

    # Sort all candidates by primary performance metrics
    def _cand_sort_key(c):
        # Prefer F1 descending
        f1_val = c.get("f1") if c.get("f1") is not None else -999.0
        acc_val = c.get("accuracy") if c.get("accuracy") is not None else -999.0
        roc_val = c.get("roc_auc") if c.get("roc_auc") is not None else -999.0
        rmse_val = c.get("rmse") if c.get("rmse") is not None else 999999.0
        r2_val = c.get("r2") if c.get("r2") is not None else -999.0
        return (-f1_val, -acc_val, -roc_val, rmse_val, -r2_val)

    if all_candidates:
        all_candidates.sort(key=_cand_sort_key)
        for i, c in enumerate(all_candidates):
            c["rank"] = i + 1

    serialized["all_candidates"] = all_candidates

    return sanitize_json(serialized)


@app.post("/api/pipeline/run")
def launch_pipeline(req: PipelineRequest):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    _runs[run_id] = {
        "status": "running",
        "queue": queue.Queue(),
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(run_id, req),
        daemon=True,
    )
    thread.start()

    return {"run_id": run_id, "status": "running"}


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints: Pipeline — SSE Stream
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/pipeline/{run_id}/stream")
async def pipeline_stream(run_id: str):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run_data = _runs[run_id]
    msg_queue: queue.Queue = run_data["queue"]

    async def event_generator():
        idle_ticks = 0
        while True:
            try:
                msg = msg_queue.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"
                idle_ticks = 0
                if msg.get("type") in ("complete", "error"):
                    return
            except queue.Empty:
                # Check if run already finished
                if run_data["status"] in ("completed", "error") and msg_queue.empty():
                    final = {
                        "type": run_data["status"],
                        "message": run_data.get("error", "Pipeline completed."),
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    return
                await asyncio.sleep(0.4)
                idle_ticks += 1
                if idle_ticks >= 5: # Every 2 seconds of idle, send keep-alive comment
                    yield ": ping\n\n"
                    idle_ticks = 0

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/pipeline/runs")
def list_all_pipeline_runs():
    runs = []
    seen = set()

    # 1. In-memory completed runs
    for r_id, r_data in reversed(list(_runs.items())):
        res = r_data.get("result")
        if res and res.get("run_id") and res["run_id"] not in seen:
            seen.add(res["run_id"])
            runs.append({
                "run_id": res["run_id"],
                "best_model": res.get("best_model_name", "Model"),
                "artifacts_count": res.get("total_artifacts", 0),
                "status": r_data.get("status", "completed"),
            })

    # 2. Disk storage/runs
    storage_root = Path("storage/runs")
    if storage_root.exists():
        for d in sorted(storage_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_dir():
                clean_id = d.name.replace("Generated_Project_", "")
                if clean_id not in seen:
                    seen.add(clean_id)
                    runs.append({
                        "run_id": clean_id,
                        "best_model": "Evaluated Models",
                        "artifacts_count": len(list(d.rglob("*"))),
                        "status": "completed",
                    })

    return {"runs": runs}


@app.get("/api/pipeline/latest")
def get_latest_pipeline_result():
    # Check completed runs in memory first (in reverse order)
    for r_key in reversed(list(_runs.keys())):
        r_data = _runs[r_key]
        if r_data.get("status") == "completed" and r_data.get("result"):
            return JSONResponse(content=sanitize_json({
                "status": "completed",
                "result": r_data["result"],
                "run_id": r_data["result"].get("run_id"),
            }))

    # Fallback to finding the newest directory in storage/runs
    storage_root = Path("storage/runs")
    if storage_root.exists():
        dirs = [d for d in storage_root.iterdir() if d.is_dir()]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        if dirs:
            latest_dir = dirs[0]
            run_id = latest_dir.name.replace("Generated_Project_", "")
            
            # Read executive summary if available
            report_file = latest_dir / "08_Reports" / "executive_summary.md"
            summary_text = report_file.read_text(encoding="utf-8") if report_file.exists() else ""
            
            # Read model comparison if available
            comp_file = latest_dir / "06_Models" / "model_comparison.json"
            top_models = []
            best_model_name = "Model"
            best_metrics = {}
            if comp_file.exists():
                try:
                    comp_data = json.loads(comp_file.read_text(encoding="utf-8"))
                    if isinstance(comp_data, list) and comp_data:
                        for idx, m in enumerate(comp_data[:4]):
                            r = idx + 1
                            m_name = m.get("model") or m.get("model_name", "Model")
                            top_models.append({
                                "rank": r,
                                "model_name": m_name,
                                "metrics": {k: v for k, v in m.items() if k not in ("model", "model_name", "training_time_seconds", "training_time", "is_best", "experiment_id")},
                                "training_time": m.get("training_time_seconds") or m.get("training_time", 0.0),
                                "selection_reason": "Evaluated candidate model.",
                                "filename": f"model_{r:02d}_{m_name.lower().replace('classifier', '_classifier')}.pkl",
                                "download_url": f"/api/pipeline/{run_id}/models/{r}/download",
                            })
                        best_model_name = top_models[0]["model_name"]
                        best_metrics = top_models[0]["metrics"]
                except Exception:
                    pass

            result_obj = {
                "run_id": run_id,
                "run_dir": str(latest_dir),
                "best_model_name": best_model_name,
                "best_metrics": best_metrics,
                "top_models": top_models,
                "total_artifacts": len(list(latest_dir.rglob("*"))),
                "executive_summary": summary_text,
                "data_quality_score": 98,
            }
            return JSONResponse(content=sanitize_json({
                "status": "completed",
                "result": result_obj,
                "run_id": run_id,
            }))

    return JSONResponse(content={"status": "idle", "result": None})


@app.get("/api/pipeline/{run_id}/result")
def get_pipeline_result(run_id: str):
    if run_id in _runs:
        run_data = _runs[run_id]
        return JSONResponse(content=sanitize_json({
            "status": run_data["status"],
            "result": run_data.get("result"),
            "error": run_data.get("error"),
        }))

    # Fallback to searching disk
    run_dir = _find_run_dir(run_id)
    if run_dir and run_dir.exists():
        report_file = run_dir / "08_Reports" / "executive_summary.md"
        summary_text = report_file.read_text(encoding="utf-8") if report_file.exists() else ""
        comp_file = run_dir / "06_Models" / "model_comparison.json"
        top_models = []
        best_model_name = "Model"
        best_metrics = {}
        if comp_file.exists():
            try:
                comp_data = json.loads(comp_file.read_text(encoding="utf-8"))
                if isinstance(comp_data, list) and comp_data:
                    for idx, m in enumerate(comp_data[:4]):
                        r = idx + 1
                        m_name = m.get("model") or m.get("model_name", "Model")
                        top_models.append({
                            "rank": r,
                            "model_name": m_name,
                            "metrics": {k: v for k, v in m.items() if k not in ("model", "model_name", "training_time_seconds", "training_time", "is_best", "experiment_id")},
                            "training_time": m.get("training_time_seconds") or m.get("training_time", 0.0),
                            "selection_reason": "Evaluated candidate model.",
                            "filename": f"model_{r:02d}_{m_name.lower().replace('classifier', '_classifier')}.pkl",
                            "download_url": f"/api/pipeline/{run_id}/models/{r}/download",
                        })
                    best_model_name = top_models[0]["model_name"]
                    best_metrics = top_models[0]["metrics"]
            except Exception:
                pass

        result_obj = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "best_model_name": best_model_name,
            "best_metrics": best_metrics,
            "top_models": top_models,
            "total_artifacts": len(list(run_dir.rglob("*"))),
            "executive_summary": summary_text,
            "data_quality_score": 98,
        }
        return JSONResponse(content=sanitize_json({
            "status": "completed",
            "result": result_obj,
            "error": None,
        }))

    raise HTTPException(status_code=404, detail=f"Run {run_id} not found")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints: Pipeline — File Listing & Downloads
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/pipeline/{run_id}/files")
def list_pipeline_files(run_id: str):
    run_dir = _find_run_dir(run_id)
    if not run_dir or not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory for {run_id} not found")

    folder_descriptions = {
        "01_Raw_Data": "Immutable original uploaded dataset copies",
        "02_Cleaned_Data": "Sanitized, deduplicated and type-validated data",
        "03_Feature_Engineered_Data": "Interaction, domain, and temporal engineered features",
        "04_ML_Ready_Data": "Imputed, scaled, and adaptively encoded ML matrices",
        "05_Notebook": "Self-contained, reproducible Jupyter Notebook",
        "06_Models": "Top serialized model artifacts (.pkl) and metadata",
        "07_Visualizations": "High-resolution EDA and model diagnostic plots",
        "08_Reports": "Executive business insights in PDF, Word (DOCX), Markdown & JSON",
        "09_Experiments": "Complete benchmark log of all evaluated algorithms",
        "10_Metadata": "Full agent execution state and dataset metadata",
    }

    folders = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(run_dir).as_posix()
            top_folder = p.relative_to(run_dir).parts[0] if len(p.relative_to(run_dir).parts) > 1 else "Root"
            if top_folder not in folders:
                folders[top_folder] = {
                    "name": top_folder,
                    "description": folder_descriptions.get(top_folder, "Generated artifacts"),
                    "files": [],
                }
            folders[top_folder]["files"].append({
                "path": rel,
                "name": p.name,
                "size_kb": round(p.stat().st_size / 1024, 2),
            })

    return {"run_dir": str(run_dir), "folders": folders}


@app.get("/api/pipeline/{run_id}/files/{file_path:path}")
def download_pipeline_file(run_id: str, file_path: str, download: bool = True):
    run_dir = _find_run_dir(run_id)
    if not run_dir or not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory for {run_id} not found")

    full_path = run_dir / file_path
    
    # On-demand generation for .pdf and .docx if .md exists
    if not full_path.exists():
        file_name = Path(file_path).name.lower()
        if file_name in ["executive_summary.pdf", "executive_summary.docx"]:
            md_path = run_dir / "08_Reports" / "executive_summary.md"
            if not md_path.exists():
                matched_md = list(run_dir.rglob("executive_summary.md"))
                if matched_md:
                    md_path = matched_md[0]
            if md_path.exists():
                from aads.tools.reporting.export_formats import export_markdown_to_docx, export_markdown_to_pdf
                target_out = (run_dir / "08_Reports" / file_name)
                target_out.parent.mkdir(parents=True, exist_ok=True)
                md_text = md_path.read_text(encoding="utf-8")
                if file_name.endswith(".pdf"):
                    export_markdown_to_pdf(md_text, target_out)
                else:
                    export_markdown_to_docx(md_text, target_out)
                full_path = target_out

    if not full_path.exists() or not full_path.is_file():
        # Try matching by filename inside subdirectories
        matched_candidates = list(run_dir.rglob(Path(file_path).name))
        if matched_candidates:
            full_path = matched_candidates[0]
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    suffix = full_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
        ".json": "application/json",
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".ipynb": "application/json",
        ".pkl": "application/octet-stream",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    if download:
        return FileResponse(
            full_path,
            media_type=media_type,
            filename=full_path.name,
            headers={"Content-Disposition": f'attachment; filename="{full_path.name}"'}
        )
    return FileResponse(full_path, media_type=media_type)


@app.get("/api/pipeline/{run_id}/zip")
def download_pipeline_zip(run_id: str):
    run_dir = _find_run_dir(run_id)
    if not run_dir or not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory for {run_id} not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in run_dir.rglob("*"):
            if fpath.is_file():
                arcname = f"AI_Data_Science_Project/{fpath.relative_to(run_dir)}"
                zf.write(fpath, arcname)
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="AI_Data_Science_Project_{run_id}.zip"'
        },
    )


@app.get("/api/pipeline/{run_id}/visualizations")
def get_pipeline_visualizations(run_id: str):
    run_dir = _find_run_dir(run_id)
    if not run_dir or not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory for {run_id} not found")

    viz_dir = run_dir / "07_Visualizations"
    if not viz_dir.exists():
        return {"run_id": run_id, "total": 0, "images": []}

    category_titles = {
        "distributions": "DISTRIBUTIONS",
        "correlations": "CORRELATIONS",
        "categorical": "CATEGORICAL",
        "outliers": "OUTLIERS",
        "model_evaluation": "MODEL_EVALUATION",
    }

    images = []
    for p in sorted(viz_dir.rglob("*.png")):
        if p.is_file():
            rel = p.relative_to(run_dir).as_posix()
            category_slug = p.parent.name.lower()
            category_name = category_titles.get(category_slug, p.parent.name.upper())
            images.append({
                "path": rel,
                "name": p.name,
                "category": category_name,
                "category_slug": category_slug,
                "url": f"/api/pipeline/{run_id}/files/{rel}",
            })

    return {"run_id": run_id, "total": len(images), "images": images}


@app.get("/api/pipeline/{run_id}/models/{rank}/download")
def download_ranked_model(run_id: str, rank: int):
    run_dir = _find_run_dir(run_id)
    if not run_dir or not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory for {run_id} not found")

    models_dir = run_dir / "06_Models"
    if not models_dir.exists():
        raise HTTPException(status_code=404, detail="Models directory not found")

    # Match model_{rank:02d}_*.pkl or model_{rank}_*.pkl
    matched = list(models_dir.glob(f"model_{rank:02d}_*.pkl"))
    if not matched:
        matched = list(models_dir.glob(f"model_{rank}_*.pkl"))
    if not matched and rank == 1:
        matched = list(models_dir.glob("best_model.pkl"))

    if not matched:
        # Fallback to sorted list of pkl files in 06_Models
        pkl_files = [f for f in sorted(models_dir.glob("*.pkl")) if not f.name.startswith("preprocessing")]
        if pkl_files and rank <= len(pkl_files):
            matched = [pkl_files[rank - 1]]

    if not matched:
        raise HTTPException(status_code=404, detail=f"Model for rank {rank} not found")

    target_file = matched[0]
    return FileResponse(
        target_file,
        filename=target_file.name,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{target_file.name}"'}
    )




# ──────────────────────────────────────────────────────────────────────────────
# Endpoints: Settings & LLM
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings():
    settings = load_user_settings()
    return {"settings": settings, "default_models": DEFAULT_PROVIDER_MODELS}


@app.put("/api/settings")
def update_settings(body: SettingsUpdate):
    save_user_settings(body.settings)
    return {"status": "saved"}


@app.get("/api/settings/api-key/{provider}")
def get_api_key(provider: str):
    key = get_stored_api_key(provider)
    # Mask the key for security
    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
    return {"provider": provider, "has_key": bool(key), "masked": masked}


@app.put("/api/settings/api-key")
def set_api_key(body: ApiKeyUpdate):
    set_stored_api_key(body.provider, body.key)
    return {"status": "saved", "provider": body.provider}


@app.post("/api/test-connection")
def test_connection(body: TestConnectionRequest):
    ok, msg = test_llm_connection(body.provider, body.model, body.api_key, base_url=body.base_url)
    return {"success": ok, "message": msg}


@app.post("/api/fetch-models")
def fetch_models(body: FetchModelsRequest):
    models = list_provider_models(body.provider, body.api_key)
    return {"provider": body.provider, "models": models, "count": len(models)}


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints: Dataset Preview
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/preview")
def preview_dataset(path: str, rows: int = 6):
    import pandas as pd

    candidate_paths = [
        Path(path),
        Path(path).resolve(),
        _PROJECT_ROOT / path,
        _PROJECT_ROOT / "storage" / "temp_uploads" / Path(path).name,
        _PROJECT_ROOT / "data" / Path(path).name,
        Path("storage/temp_uploads") / Path(path).name,
        Path("data") / Path(path).name,
    ]
    file_path = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            file_path = p.resolve()
            break

    if not file_path:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {file_path.suffix}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    preview = df.head(rows)
    clean_rows = []
    for row in preview.values.tolist():
        clean_rows.append([None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v for v in row])

    return JSONResponse(content=sanitize_json({
        "columns": list(preview.columns),
        "rows": clean_rows,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }))
