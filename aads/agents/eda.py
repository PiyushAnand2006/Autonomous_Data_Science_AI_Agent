"""
AADS Exploratory Data Analysis (EDA) Agent — conducts automated statistical exploration,
generates visualizations, and records narrative data findings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord, EDAFindings
from aads.core.state import RunState
from aads.tools.visualization.charts import (
    plot_categorical,
    plot_correlations,
    plot_distributions,
    plot_outliers,
    plot_target_relationships,
)

logger = get_logger(__name__)


class EDAAgent:
    """Agent that performs automated EDA, renders visualizations, and extracts statistical findings."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager

    def run(
        self,
        df: pd.DataFrame,
        state: RunState,
    ) -> EDAFindings:
        """Execute exploratory analysis, generate charts, and record findings.

        Args:
            df: DataFrame to analyze.
            state: The current RunState.

        Returns:
            EDAFindings object containing insights and chart references.
        """
        logger.info("eda_agent_start", run_id=state.run_id, target=state.target_column)

        # 1. Classify columns
        num_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        cat_cols_all = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]

        # 2. Rank & cap to top-K most important features to reduce RAM
        MAX_NUM_CHARTS = 8
        MAX_CAT_CHARTS = 6

        # Rank numeric columns by absolute correlation with target (most relevant first)
        if state.target_column and state.target_column in num_cols_all and len(num_cols_all) > 1:
            try:
                corr_with_target = (
                    df[num_cols_all]
                    .corr()[state.target_column]
                    .drop(state.target_column, errors="ignore")
                    .abs()
                    .sort_values(ascending=False)
                )
                num_cols = list(corr_with_target.index[:MAX_NUM_CHARTS])
                # Ensure the target itself is always included for distribution plots
                if state.target_column not in num_cols:
                    num_cols = [state.target_column] + num_cols[:MAX_NUM_CHARTS - 1]
            except Exception:
                num_cols = num_cols_all[:MAX_NUM_CHARTS]
        else:
            num_cols = num_cols_all[:MAX_NUM_CHARTS]

        # Rank categorical columns by variance (nunique) — skip near-unique ID-like cols
        total_rows = len(df)
        cat_cols_filtered = [
            c for c in cat_cols_all
            if c in df.columns and (total_rows == 0 or df[c].nunique() / total_rows <= 0.4)
        ]
        cat_cols = cat_cols_filtered[:MAX_CAT_CHARTS]

        logger.info(
            "eda_column_selection",
            num_total=len(num_cols_all), num_selected=len(num_cols),
            cat_total=len(cat_cols_all), cat_selected=len(cat_cols),
        )

        generated_charts: list[str] = []

        # 3. Render and register charts if artifact manager is available
        if self.artifact_manager:
            try:
                viz_dir = self.artifact_manager.get_path("visualizations")

                # Distributions
                dist_files = plot_distributions(df, num_cols, viz_dir / "distributions")
                for f in dist_files:
                    rec = self.artifact_manager.register_artifact(
                        artifact_type=ArtifactType.VISUALIZATION,
                        path=f,
                        description=f"Distribution plot: {Path(f).name}",
                    )
                    generated_charts.append(rec.path)

                # Categorical frequencies
                cat_files = plot_categorical(df, cat_cols, viz_dir / "categorical")
                for f in cat_files:
                    rec = self.artifact_manager.register_artifact(
                        artifact_type=ArtifactType.VISUALIZATION,
                        path=f,
                        description=f"Categorical bar chart: {Path(f).name}",
                    )
                    generated_charts.append(rec.path)

                # Correlations heatmap
                corr_files = plot_correlations(df, num_cols, viz_dir / "correlations")
                for f in corr_files:
                    rec = self.artifact_manager.register_artifact(
                        artifact_type=ArtifactType.VISUALIZATION,
                        path=f,
                        description="Pearson correlation matrix heatmap",
                    )
                    generated_charts.append(rec.path)

                # Outliers boxplots
                box_files = plot_outliers(df, num_cols, viz_dir / "outliers")
                for f in box_files:
                    rec = self.artifact_manager.register_artifact(
                        artifact_type=ArtifactType.VISUALIZATION,
                        path=f,
                        description=f"Outlier box plot: {Path(f).name}",
                    )
                    generated_charts.append(rec.path)

                # Target relationships
                if state.target_column and state.target_column in df.columns:
                    feat_candidates = [c for c in df.columns if c != state.target_column]
                    target_files = plot_target_relationships(df, state.target_column, feat_candidates, viz_dir / "correlations")
                    for f in target_files:
                        rec = self.artifact_manager.register_artifact(
                            artifact_type=ArtifactType.VISUALIZATION,
                            path=f,
                            description=f"Target relationship plot: {Path(f).name}",
                        )
                        generated_charts.append(rec.path)

            except Exception as e:
                logger.warning("eda_chart_generation_failed", error=str(e))

        # 4. Formulate statistical findings (use ALL columns for insights, not just plotted ones)
        univariate_insights: list[str] = []
        for col in num_cols_all:
            non_null = df[col].dropna()
            if len(non_null) > 5 and non_null.std() > 0:
                skew = float(non_null.skew())
                if abs(skew) > 1.5:
                    univariate_insights.append(f"Column '{col}' is heavily skewed (skewness: {skew:.2f}). Consider log or power transformation.")

        correlation_insights: list[str] = []
        if len(num_cols_all) >= 2:
            corr_mat = df[num_cols_all].corr().abs().fillna(0.0)
            corr_values = corr_mat.to_numpy(copy=True)
            np.fill_diagonal(corr_values, 0.0)
            high_corr_pairs = []
            for i in range(len(num_cols_all)):
                for j in range(i + 1, len(num_cols_all)):
                    c_val = float(corr_values[i, j])
                    if c_val >= 0.75:
                        high_corr_pairs.append((num_cols_all[i], num_cols_all[j], c_val))

            for col1, col2, val in high_corr_pairs[:5]:
                correlation_insights.append(f"Strong collinearity between '{col1}' and '{col2}' (r = {val:.2f}).")

        bivariate_insights: list[str] = []
        if state.target_column and state.target_column in num_cols_all:
            target_corrs = df[num_cols_all].corr()[state.target_column].drop(state.target_column).abs().sort_values(ascending=False)
            top_target_feats = target_corrs.head(3)
            for f_name, c_val in top_target_feats.items():
                if not np.isnan(c_val):
                    bivariate_insights.append(f"Feature '{f_name}' exhibits strong linear correlation with target '{state.target_column}' (r = {c_val:.2f}).")

        summary = (
            f"EDA completed on {len(df)} rows and {len(df.columns)} columns "
            f"({len(num_cols_all)} numeric, {len(cat_cols_all)} categorical). "
            f"Plotted top {len(num_cols)} numeric and {len(cat_cols)} categorical features. "
            f"Generated {len(generated_charts)} visualization artifacts."
        )

        findings = EDAFindings(
            summary=summary,
            target_column=state.target_column,
            univariate_insights=univariate_insights,
            bivariate_insights=bivariate_insights,
            correlation_insights=correlation_insights,
            generated_visualizations=generated_charts,
        )

        state.mark_phase_complete("eda")

        # Save artifact to metadata
        if self.artifact_manager:
            try:
                meta_dir = self.artifact_manager.get_path("metadata")
                findings_path = meta_dir / "eda_findings.json"
                findings_path.write_text(findings.model_dump_json(indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=findings_path,
                    description="EDA statistical findings and insight registry",
                )
            except Exception as e:
                logger.warning("eda_findings_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="eda",
                action="exploratory_analysis",
                reason=f"Generated {len(generated_charts)} visual artifacts and identified key distribution/correlation drivers.",
                approval_mode=state.autonomy_mode,
                details={
                    "numeric_count": len(num_cols),
                    "categorical_count": len(cat_cols),
                    "charts_generated": len(generated_charts),
                },
            )
        )

        logger.info("eda_agent_completed", charts_count=len(generated_charts))
        return findings
