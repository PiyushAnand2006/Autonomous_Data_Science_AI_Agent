"""
Autonomous AI Data Scientist (AADS) — Interactive Streamlit Dashboard.

Launch via:
    streamlit run aads/app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from aads.agents.orchestrator import AADSOrchestrator
from aads.core.config import AADSConfig
from aads.core.schemas import AutonomyMode, ExecutionEngine
from aads.scripts.generate_sample_data import generate_churn_dataset

st.set_page_config(
    page_title="AADS — Autonomous AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None


init_session_state()

st.sidebar.title("🧠 AADS Settings")
st.sidebar.markdown("Configure autonomous run behavior:")

autonomy_choice = st.sidebar.selectbox(
    "Autonomy Mode",
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
    "Preferred Engine",
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
storage_dir = st.sidebar.text_input("Storage Base Path", value="storage/runs")

st.markdown('<div class="main-header">Autonomous AI Data Scientist (AADS)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Transform raw datasets and natural-language goals into complete, reproducible data science projects.</div>', unsafe_allow_html=True)

tabs = st.tabs(["🚀 Launch Pipeline", "📊 Executive Report", "📈 Visualizations", "🏆 Model Leaderboard", "📦 Artifact Explorer"])

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
                with st.expander("Dataset Preview", expanded=False):
                    st.dataframe(preview_df.head(6), use_container_width=True)
            except Exception:
                pass

    with col2:
        st.subheader("2. Objective & Target")
        user_objective = st.text_area(
            "Natural Language Goal",
            value="Predict customer churn, discover key behavioral drivers, and train a high-performing baseline.",
            height=100,
        )
        target_col = st.text_input("Target Column (Leave blank for auto-detection)", value="churn")

        run_btn = st.button("🚀 Run Autonomous Pipeline", type="primary", use_container_width=True)

    if run_btn:
        if not data_path or not data_path.exists():
            st.error("Please provide or upload a valid dataset.")
        else:
            with st.spinner("Autonomous AI Data Scientist is executing the 10-phase pipeline..."):
                config = AADSConfig(
                    storage_root=Path(storage_dir),
                    random_seed=random_seed,
                    default_engine=selected_engine,
                )
                orchestrator = AADSOrchestrator(config=config, storage_root=storage_dir)

                try:
                    result = orchestrator.run_pipeline(
                        data_path=data_path,
                        user_objective=user_objective,
                        target_column=target_col if target_col.strip() else None,
                        autonomy_mode=autonomy_mode,
                    )
                    st.session_state.latest_result = result
                    st.success("🎉 Full Autonomous Pipeline Completed Successfully!")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    # Display results summary if available
    res = st.session_state.latest_result
    if res:
        st.markdown("---")
        st.subheader(f"Run Results: `{res['run_id']}`")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Winning Model", res["best_model_name"])
        with m_col2:
            primary_val = list(res["best_metrics"].values())[0] if res["best_metrics"] else "N/A"
            primary_key = list(res["best_metrics"].keys())[0].upper() if res["best_metrics"] else "SCORE"
            st.metric(f"Best {primary_key}", primary_val)
        with m_col3:
            st.metric("Artifacts Created", f"{res['total_artifacts']} files")
        with m_col4:
            st.metric("Data Health", f"{res['data_quality_report'].overall_score}/100")

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
                # Grid of images
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
# TAB 4: Model Leaderboard
# ---------------------------------------------------------------------------
with tabs[3]:
    res = st.session_state.latest_result
    if res:
        run_dir = Path(res["run_dir"])
        exp_csv = run_dir / "09_Experiments" / "experiment_results.csv"
        if exp_csv.exists():
            exp_df = pd.read_csv(exp_csv)
            st.subheader("Candidate Model Benchmark Leaderboard")
            st.dataframe(exp_df, use_container_width=True)
        else:
            st.info("Experiment results table not found.")
    else:
        st.info("Run a pipeline to view the candidate model leaderboard.")

# ---------------------------------------------------------------------------
# TAB 5: Artifact Explorer & Downloads
# ---------------------------------------------------------------------------
with tabs[4]:
    res = st.session_state.latest_result
    if res:
        run_dir = Path(res["run_dir"])
        st.subheader("Generated Project Folder Structure (§10 Contract)")
        st.code(f"Output Directory: {run_dir.resolve()}", language="bash")

        # Download Buttons for Key Files
        nb_path = run_dir / "05_Notebook" / "pipeline_notebook.ipynb"
        model_path = run_dir / "06_Models" / "best_model.pkl"
        summary_path = run_dir / "08_Reports" / "executive_summary.md"

        d_col1, d_col2, d_col3 = st.columns(3)
        if nb_path.exists():
            with d_col1:
                with open(nb_path, "rb") as f:
                    st.download_button(
                        "📥 Download Jupyter Notebook (.ipynb)",
                        data=f,
                        file_name="pipeline_notebook.ipynb",
                        mime="application/x-ipynb+json",
                        use_container_width=True,
                    )
        if model_path.exists():
            with d_col2:
                with open(model_path, "rb") as f:
                    st.download_button(
                        "📥 Download Best Model (.pkl)",
                        data=f,
                        file_name="best_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
        if summary_path.exists():
            with d_col3:
                with open(summary_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        "📥 Download Executive Report (.md)",
                        data=f.read(),
                        file_name="executive_summary.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

        # File tree display
        st.markdown("### Files Created in Run Directory")
        all_files = sorted(list(run_dir.rglob("*")))
        tree_rows = []
        for p in all_files:
            if p.is_file():
                rel_path = p.relative_to(run_dir)
                size_kb = round(p.stat().st_size / 1024, 2)
                tree_rows.append({"Path": str(rel_path), "Size (KB)": size_kb})
        st.dataframe(pd.DataFrame(tree_rows), use_container_width=True)
    else:
        st.info("Run a pipeline to explore generated files and download model/notebook artifacts.")
