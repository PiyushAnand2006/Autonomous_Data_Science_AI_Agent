"""
Autonomous AI Data Scientist (AADS) — Interactive Streamlit Dashboard.

Launch via:
    streamlit run aads/app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from aads.agents.orchestrator import AADSOrchestrator
from aads.core.config import AADSConfig
from aads.core.llm import DEFAULT_PROVIDER_MODELS, list_provider_models, test_llm_connection
from aads.core.schemas import AutonomyMode, ExecutionEngine
from aads.scripts.generate_sample_data import generate_churn_dataset

st.set_page_config(
    page_title="AADS — Autonomous AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .model-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .model-badge {
        background-color: #2563eb;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-pass {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


from aads.core.settings import (
    load_user_settings,
    save_user_settings,
    get_stored_api_key,
    set_stored_api_key,
)


def init_session_state():
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "provider_models" not in st.session_state:
        st.session_state.provider_models = DEFAULT_PROVIDER_MODELS["openrouter"]


init_session_state()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONFIGURATION (WITH PERSISTENT MEMORY)
# ──────────────────────────────────────────────────────────────────────────────
user_settings = load_user_settings()
st.sidebar.title("⚙️ AADS Configuration")

# 1. Execution Mode: Local vs AI
saved_mode = user_settings.get("execution_mode", "ai")
mode_options = ["💻 Local Machine Mode (Offline)", "✨ AI-Powered Mode (LLM Assisted)"]
mode_default_idx = 1 if saved_mode == "ai" else 0

execution_mode_ui = st.sidebar.radio(
    "Execution Mode",
    options=mode_options,
    index=mode_default_idx,
    help="Local mode runs offline with deterministic algorithms. AI mode uses OpenRouter or other LLMs for rich insights.",
)
is_ai_mode = "AI-Powered" in execution_mode_ui
execution_mode_val = "ai" if is_ai_mode else "local"

if saved_mode != execution_mode_val:
    save_user_settings({"execution_mode": execution_mode_val})

# 2. AI Provider Settings (if AI mode enabled)
selected_provider = "openrouter"
api_key_input = None
selected_model = "anthropic/claude-3.5-sonnet"

if is_ai_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Provider Settings")

    provider_options = ["openrouter", "google", "openai", "anthropic", "groq", "ollama"]
    saved_provider = user_settings.get("selected_provider", "openrouter")
    provider_idx = provider_options.index(saved_provider) if saved_provider in provider_options else 0

    selected_provider = st.sidebar.selectbox(
        "LLM Provider",
        options=provider_options,
        index=provider_idx,
    )

    if selected_provider != saved_provider:
        save_user_settings({"selected_provider": selected_provider})

    # Stored API key for this provider
    saved_key = get_stored_api_key(selected_provider)
    api_key_input = st.sidebar.text_input(
        f"{selected_provider.upper()} API Key",
        value=saved_key,
        type="password",
        help="Saved automatically to persistent memory (.aads_user_settings.json).",
    )

    # Auto-save API key if modified
    if api_key_input != saved_key:
        set_stored_api_key(selected_provider, api_key_input)

    # Stored models list for this provider
    stored_models_map = user_settings.get("provider_models_cache", {})
    if not isinstance(stored_models_map, dict):
        stored_models_map = {}
    cached_models = stored_models_map.get(selected_provider, DEFAULT_PROVIDER_MODELS.get(selected_provider, ["default"]))

    if "provider_models" not in st.session_state or st.session_state.get("_active_provider") != selected_provider:
        st.session_state.provider_models = cached_models
        st.session_state._active_provider = selected_provider

    # Dynamic model fetch & connection test buttons
    btn_col1, btn_col2 = st.sidebar.columns(2)
    with btn_col1:
        if st.button("🔄 Fetch Models", use_container_width=True):
            with st.status("Fetching models..."):
                fetched = list_provider_models(selected_provider, api_key_input)
                st.session_state.provider_models = fetched
                stored_models_map[selected_provider] = fetched
                save_user_settings({"provider_models_cache": stored_models_map})
                st.success(f"Fetched {len(fetched)} models!")

    with btn_col2:
        if st.button("⚡ Test API", use_container_width=True):
            with st.status("Testing connection..."):
                # Use current saved model or first available
                test_model = user_settings.get(f"selected_model_{selected_provider}") or (st.session_state.provider_models[0] if st.session_state.provider_models else "default")
                ok, msg = test_llm_connection(selected_provider, test_model, api_key_input)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    # Model selector with persistence
    available_models = list(st.session_state.provider_models)
    saved_model_for_provider = user_settings.get(f"selected_model_{selected_provider}", "")
    if saved_model_for_provider and saved_model_for_provider not in available_models:
        available_models.insert(0, saved_model_for_provider)

    model_idx = available_models.index(saved_model_for_provider) if saved_model_for_provider in available_models else 0
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=model_idx if model_idx < len(available_models) else 0,
    )
    if selected_model != saved_model_for_provider:
        save_user_settings({
            f"selected_model_{selected_provider}": selected_model,
            "selected_model": selected_model,
        })


st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Engine & Runtime")

autonomy_choice = st.sidebar.selectbox(
    "Autonomy Level",
    options=["Fully Autonomous", "Semi-Autonomous", "Manual"],
    index=0,
)
autonomy_mode = (
    AutonomyMode.FULLY_AUTONOMOUS
    if autonomy_choice == "Fully Autonomous"
    else AutonomyMode.SEMI_AUTONOMOUS
    if autonomy_choice == "Semi-Autonomous"
    else AutonomyMode.MANUAL_APPROVAL
)

engine_choice = st.sidebar.selectbox(
    "Processing Engine",
    options=["Pandas (Default)", "Polars", "DuckDB"],
    index=0,
)
selected_engine = (
    ExecutionEngine.PANDAS
    if "Pandas" in engine_choice
    else ExecutionEngine.POLARS
    if "Polars" in engine_choice
    else ExecutionEngine.DUCKDB
)

random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42)

# Storage location empty by default
storage_dir = st.sidebar.text_input(
    "Storage Location (Leave empty for default)",
    value="",
    help="Leave empty by default to prevent repository clutter. Will default to storage/runs.",
)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Autonomous AI Data Scientist (AADS)</div>', unsafe_allow_html=True)
mode_badge = "✨ AI-POWERED" if is_ai_mode else "💻 LOCAL MACHINE"
st.markdown(f'<div class="sub-header">Transform raw datasets and natural-language goals into complete, multi-model reproducible projects. &nbsp; <b>[{mode_badge}]</b></div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🚀 Launch Pipeline",
    "📊 In-Depth Executive Report",
    "📈 Visualizations",
    "🏆 Top Models Leaderboard",
    "📦 Dataset Versions & Artifacts",
])

# ---------------------------------------------------------------------------
# TAB 1: Launch Pipeline
# ---------------------------------------------------------------------------
with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Data Source")
        data_source_mode = st.radio("Select Dataset Mode", ["Upload My Dataset", "Use Sample Dataset"], horizontal=True)

        data_path = None
        if data_source_mode == "Upload My Dataset":
            uploaded_file = st.file_uploader("Upload CSV, Excel, or Parquet file", type=["csv", "xlsx", "parquet"])
            if uploaded_file:
                temp_dir = Path("storage/temp_uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)
                data_path = temp_dir / uploaded_file.name
                with open(data_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: `{uploaded_file.name}`")
        else:
            sample_type = st.selectbox("Sample Dataset", ["Customer Churn (Classification)"])
            sample_dir = Path("data")
            sample_dir.mkdir(parents=True, exist_ok=True)
            data_path = sample_dir / "sample_churn.csv"
            if not data_path.exists():
                generate_churn_dataset(data_path, n_samples=500)
            st.info("Loaded `sample_churn.csv` (500 rows, 9 features)")

        # Preview dataset
        if data_path and data_path.exists():
            try:
                preview_df = pd.read_csv(data_path) if data_path.suffix == ".csv" else pd.read_parquet(data_path)
                with st.expander("🔍 Dataset Preview", expanded=False):
                    st.dataframe(preview_df.head(6), use_container_width=True)
            except Exception:
                pass

    with col2:
        st.subheader("2. Objective & Target")
        user_objective = st.text_area(
            "Natural Language Goal",
            value="Predict customer churn, discover key behavioral drivers, and export top performing ML models.",
            height=100,
        )
        target_col = st.text_input("Target Column (Leave blank for auto-detection)", value="churn")

        run_btn = st.button("🚀 Run Autonomous Pipeline", type="primary", use_container_width=True)

    if run_btn:
        if not data_path or not data_path.exists():
            st.error("Please provide or upload a valid dataset.")
        else:
            with st.status(f"🚀 Executing Autonomous Pipeline in {mode_badge} mode...", expanded=True) as status_box:
                cfg_kwargs = {
                    "execution_mode": execution_mode_val,
                    "random_seed": random_seed,
                    "default_engine": selected_engine,
                    "top_models_count": 4,
                }
                if storage_dir and storage_dir.strip():
                    cfg_kwargs["storage_dir"] = storage_dir.strip()
                if is_ai_mode:
                    cfg_kwargs["llm_provider"] = selected_provider
                    cfg_kwargs["llm_model"] = selected_model
                    if api_key_input and api_key_input.strip():
                        cfg_kwargs["llm_api_key"] = api_key_input.strip()

                config = AADSConfig(**cfg_kwargs)
                orchestrator = AADSOrchestrator(config=config, storage_root=config.storage_root)

                def _on_phase_update(message: str) -> None:
                    st.write(message)

                try:
                    result = orchestrator.run_pipeline(
                        data_path=data_path,
                        user_objective=user_objective,
                        target_column=target_col if target_col.strip() else None,
                        autonomy_mode=autonomy_mode,
                        progress_callback=_on_phase_update,
                    )
                    st.session_state.latest_result = result
                    status_box.update(label="🎉 Full Autonomous Pipeline Completed Successfully!", state="complete", expanded=False)
                    st.success("🎉 Full Autonomous Pipeline Completed Successfully!")
                except Exception as e:
                    status_box.update(label=f"❌ Pipeline Execution Error", state="error", expanded=True)
                    st.error(f"Pipeline error: {e}")

    # Display results summary if available
    res = st.session_state.latest_result
    if res:
        st.markdown("---")
        st.subheader(f"Run Summary: `{res['run_id']}`")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Rank 1 Model", res["best_model_name"])
        with m_col2:
            primary_val = list(res["best_metrics"].values())[0] if res["best_metrics"] else "N/A"
            primary_key = list(res["best_metrics"].keys())[0].upper() if res["best_metrics"] else "SCORE"
            st.metric(f"Best {primary_key}", primary_val)
        with m_col3:
            st.metric("Top Models Exported", f"{len(res.get('top_models', [1]))} models")
        with m_col4:
            st.metric("Data Health Score", f"{res['data_quality_report'].overall_score}/100")

        # Top-level Full Project ZIP download
        run_dir = Path(res["run_dir"])
        if run_dir.exists():
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for fpath in run_dir.rglob("*"):
                    if fpath.is_file():
                        arcname = f"AI_Data_Science_Project/{fpath.relative_to(run_dir)}"
                        zf.write(fpath, arcname)
            zip_buffer.seek(0)

            st.download_button(
                "📦 Download Entire Project as ZIP (All Datasets, Notebook, Models & Reports)",
                data=zip_buffer,
                file_name=f"AI_Data_Science_Project_{res['run_id']}.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )


# ---------------------------------------------------------------------------
# TAB 2: Executive Report
# ---------------------------------------------------------------------------
with tabs[1]:
    res = st.session_state.latest_result
    if res:
        st.markdown(res["executive_summary"])
    else:
        st.info("No run executed yet. Launch a pipeline from the first tab to view the executive report.")

# ---------------------------------------------------------------------------
# TAB 3: Visualizations
# ---------------------------------------------------------------------------
with tabs[2]:
    res = st.session_state.latest_result
    if res:
        run_dir = Path(res["run_dir"])
        viz_dir = run_dir / "07_Visualizations"
        if viz_dir.exists():
            img_files = list(viz_dir.rglob("*.png"))
            if img_files:
                st.subheader(f"Generated Visualizations ({len(img_files)} charts)")
                cols = st.columns(2)
                for idx, img_p in enumerate(img_files):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        category = img_p.parent.name
                        st.image(str(img_p), caption=f"[{category.upper()}] {img_p.name}", use_container_width=True)
            else:
                st.warning("No charts found in visualizations folder.")
        else:
            st.warning("Visualization directory not found.")
    else:
        st.info("Run a pipeline to view generated charts and exploratory plots.")

# ---------------------------------------------------------------------------
# TAB 4: Model Leaderboard & Top Models
# ---------------------------------------------------------------------------
with tabs[3]:
    res = st.session_state.latest_result
    if res:
        run_dir = Path(res["run_dir"])
        models_dir = run_dir / "06_Models"

        # 1. Top Exported Models Cards
        top_models = res.get("top_models", [])
        if top_models:
            st.subheader(f"🏆 Top {len(top_models)} Selected Models (Exported to 06_Models/)")
            for tm in top_models:
                rank = tm.get("rank", 1)
                m_name = tm.get("model_name", "Model")
                metrics = tm.get("metrics", {})
                t_time = tm.get("training_time", 0.0)
                reason = tm.get("selection_reason", "")

                slug = m_name.lower().replace("classifier", "_classifier").replace("regressor", "_regressor")
                m_file = models_dir / f"model_{rank:02d}_{slug}.pkl"

                with st.container():
                    st.markdown(f"""
                    <div class="model-card">
                        <span class="model-badge">RANK {rank}</span> &nbsp; <b>{m_name}</b> &nbsp; <i>(Training: {t_time:.3f}s)</i><br>
                        <p style="margin-top:6px; color:#334155;"><b>Reason:</b> {reason}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    m_cols = st.columns(len(metrics) + 1 if metrics else 2)
                    for idx, (mk, mv) in enumerate(metrics.items()):
                        with m_cols[idx]:
                            st.metric(mk.upper(), mv)

                    with m_cols[-1]:
                        if m_file.exists():
                            with open(m_file, "rb") as mf:
                                st.download_button(
                                    f"📥 Download ({m_file.name})",
                                    data=mf,
                                    file_name=m_file.name,
                                    mime="application/octet-stream",
                                    key=f"dl_model_{rank}",
                                )
                    st.markdown("---")

        # 2. Full Experiment Benchmark Table — all evaluated models ranked
        st.subheader("📋 Complete Benchmark of All Evaluated Candidates")

        # Prefer model_comparison.json (richer: includes training times)
        comp_json = models_dir / "model_comparison.json"
        exp_csv = run_dir / "09_Experiments" / "experiment_results.csv"

        bench_df = None
        if comp_json.exists():
            try:
                bench_df = pd.read_json(comp_json)
            except Exception:
                pass

        if bench_df is None and exp_csv.exists():
            try:
                bench_df = pd.read_csv(exp_csv)
            except Exception:
                pass

        if bench_df is not None and not bench_df.empty:
            # Determine primary sort metric from task type
            task_type_val = res.get("state")
            is_regression = False
            is_clustering = False
            if task_type_val is not None:
                tt = getattr(task_type_val, "task_type", None)
                if tt is not None:
                    is_regression = tt.value == "regression" if hasattr(tt, "value") else str(tt) == "regression"
                    is_clustering = tt.value == "clustering" if hasattr(tt, "value") else str(tt) == "clustering"

            if is_regression:
                sort_col = "rmse" if "rmse" in bench_df.columns else None
                sort_ascending = True
            elif is_clustering:
                sort_col = "silhouette" if "silhouette" in bench_df.columns else None
                sort_ascending = False
            else:
                sort_col = "f1" if "f1" in bench_df.columns else ("accuracy" if "accuracy" in bench_df.columns else None)
                sort_ascending = False

            if sort_col and sort_col in bench_df.columns:
                bench_df = bench_df.sort_values(sort_col, ascending=sort_ascending).reset_index(drop=True)

            # Add rank column
            bench_df.insert(0, "Rank", range(1, len(bench_df) + 1))

            # Rename 'model' column to 'Model' for readability
            if "model" in bench_df.columns:
                bench_df = bench_df.rename(columns={"model": "Model"})

            # Rename training time column
            if "training_time_seconds" in bench_df.columns:
                bench_df = bench_df.rename(columns={"training_time_seconds": "Training Time (s)"})

            # Drop internal columns not useful for display
            for drop_col in ["experiment_id", "is_best"]:
                if drop_col in bench_df.columns:
                    bench_df = bench_df.drop(columns=[drop_col])

            # Round numeric columns for clean display
            numeric_cols = bench_df.select_dtypes(include=["float64", "float32"]).columns
            bench_df[numeric_cols] = bench_df[numeric_cols].round(4)

            # Highlight the best model row (Rank 1)
            def _highlight_best(row):
                if row["Rank"] == 1:
                    return ["background-color: #dcfce7; font-weight: bold"] * len(row)
                return [""] * len(row)

            styled_df = bench_df.style.apply(_highlight_best, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.caption(f"Showing all {len(bench_df)} evaluated model candidates, ranked by {'RMSE (lower is better)' if is_regression else 'Silhouette (higher is better)' if is_clustering else 'F1 Score (higher is better)'}. The best model is highlighted in green.")
        else:
            st.info("No benchmark data found for this run.")

    else:
        st.info("Run a pipeline to view the model leaderboard and top selected models.")

# ---------------------------------------------------------------------------
# TAB 5: Dataset Versions & Artifact Explorer
# ---------------------------------------------------------------------------
with tabs[4]:
    res = st.session_state.latest_result
    if res:
        run_dir = Path(res["run_dir"])

        # ── 1. Hero Full Project ZIP Download ─────────────────────────────────
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); padding: 24px; border-radius: 12px; color: white; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 6px;">📦 Complete AI Data Science Project Package</div>
            <div style="font-size: 0.95rem; color: #dbeafe; line-height: 1.5; margin-bottom: 4px;">
                Download the complete self-contained project archive containing all <b>10 standardized folders</b>:
                raw & cleaned CSV datasets, ML matrices, 4 top serialized models (.pkl), interactive Jupyter notebook (.ipynb), high-res visualizations (.png), executive summary (.md), and full benchmark logs.
            </div>
            <div style="font-size: 0.85rem; color: #93c5fd;">Project Identifier: <code>{res['run_id']}</code></div>
        </div>
        """, unsafe_allow_html=True)

        zip_buf_tab5 = io.BytesIO()
        with zipfile.ZipFile(zip_buf_tab5, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in run_dir.rglob("*"):
                if fpath.is_file():
                    arcname = f"AI_Data_Science_Project/{fpath.relative_to(run_dir)}"
                    zf.write(fpath, arcname)
        zip_buf_tab5.seek(0)
        zip_size_mb = round(len(zip_buf_tab5.getvalue()) / (1024 * 1024), 2)

        st.download_button(
            f"⬇️ Download Entire Project Archive (.ZIP) — [{zip_size_mb} MB]",
            data=zip_buf_tab5,
            file_name=f"AI_Data_Science_Project_{res['run_id']}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
            key="dl_zip_hero",
        )

        st.markdown("---")

        # ── 2. Folder-by-Folder File Explorer & Direct Downloaders ────────────
        st.subheader("📂 Browse & Download Individual Artifacts by Folder")

        # Group all files by top-level folder
        folder_groups: dict[str, list[Path]] = {}
        for p in sorted(list(run_dir.rglob("*"))):
            if p.is_file():
                rel = p.relative_to(run_dir)
                top_folder = rel.parts[0] if len(rel.parts) > 1 else "Root"
                folder_groups.setdefault(top_folder, []).append(p)

        folder_descriptions = {
            "01_Raw_Data": "Immutable original uploaded dataset copies",
            "02_Cleaned_Data": "Sanitized, deduplicated and type-validated data",
            "03_Feature_Engineered_Data": "Interaction, domain, and temporal engineered features",
            "04_ML_Ready_Data": "Imputed, scaled, and adaptively encoded ML matrices",
            "05_Notebook": "Self-contained, reproducible Jupyter Notebook and validation payload",
            "06_Models": "Top 4 serialized model artifacts (.pkl), pipeline, and metadata",
            "07_Visualizations": "High-resolution EDA and model diagnostic plots",
            "08_Reports": "Executive business insights markdown and JSON summaries",
            "09_Experiments": "Complete benchmark log of all evaluated algorithms",
            "10_Metadata": "Full agent execution state and dataset metadata",
        }

        for folder_name in sorted(folder_groups.keys()):
            files_in_folder = folder_groups[folder_name]
            desc = folder_descriptions.get(folder_name, "Generated artifacts")
            
            with st.expander(f"📁 **{folder_name}/** ({len(files_in_folder)} file{'s' if len(files_in_folder) != 1 else ''}) — *{desc}*", expanded=True):
                for fpath in files_in_folder:
                    rel_p = fpath.relative_to(run_dir)
                    f_size_kb = round(fpath.stat().st_size / 1024, 2)
                    fc1, fc2 = st.columns([3, 1])
                    with fc1:
                        st.markdown(f"📄 ` {str(rel_p)} ` &nbsp; <span style='color:#64748b; font-size:0.85rem;'>({f_size_kb} KB)</span>", unsafe_allow_html=True)
                    with fc2:
                        with open(fpath, "rb") as cur_f:
                            st.download_button(
                                f"📥 Download {fpath.name}",
                                data=cur_f,
                                file_name=fpath.name,
                                key=f"dl_file_{str(rel_p).replace(os.sep, '_')}",
                                use_container_width=True,
                            )

    else:
        st.info("Run a pipeline to explore generated files and download model/notebook artifacts.")

