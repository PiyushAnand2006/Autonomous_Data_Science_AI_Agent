"""
AADS Report Generator Agent — produces the executive summary and project documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.llm import get_llm
from aads.core.schemas import ArtifactType, DecisionRecord
from aads.core.state import RunState
from aads.tools.reporting.report_builder import (
    build_executive_summary_md,
    build_project_readme_md,
)

logger = get_logger(__name__)


class ReportGeneratorAgent:
    """Agent that compiles all findings, charts, top models, and AI insights into executive reports."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
        llm: Any = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager
        self.llm = llm

    def _generate_ai_narrative(
        self,
        state: RunState,
        best_model_name: str,
        best_metrics: dict[str, float],
    ) -> Optional[str]:
        """Optionally generate deep AI domain narrative if in AI Mode."""
        if getattr(self.config, "execution_mode", "local") != "ai":
            return None

        try:
            llm_instance = self.llm
            if llm_instance is None:
                llm_instance = get_llm(self.config)

            prompt = (
                f"You are an expert Chief AI Scientist. Write a 2-3 paragraph executive analytical summary "
                f"for a data science project.\n"
                f"User Objective: {state.user_objective}\n"
                f"Task Type: {state.task_type.value if state.task_type else 'N/A'}\n"
                f"Target Column: {state.target_column}\n"
                f"Dataset: {state.dataset_meta.n_rows if state.dataset_meta else 'N/A'} rows, "
                f"{state.dataset_meta.n_cols if state.dataset_meta else 'N/A'} cols\n"
                f"Top Model: {best_model_name}\n"
                f"Metrics: {best_metrics}\n\n"
                f"Highlight key data insights, business implications of model performance, "
                f"and strategic recommendations for operationalizing this model."
            )

            try:
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=prompt)]
            except ImportError:
                messages = [{"role": "user", "content": prompt}]

            response = llm_instance.invoke(messages)
            content = getattr(response, "content", "") or str(response)
            return content.strip() if content else None
        except Exception as e:
            logger.debug("ai_narrative_generation_skipped", error=str(e))
            return None

    def run(
        self,
        state: RunState,
        best_model_name: str,
        best_metrics: dict[str, float],
        eval_report: Optional[dict[str, Any]] = None,
        top_models: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Generate executive summary and project README.

        Args:
            state: The current RunState.
            best_model_name: Name of winning model.
            best_metrics: Holdout evaluation metrics.
            eval_report: Optional evaluation diagnostics.
            top_models: Optional list of top ranked models.

        Returns:
            Executive summary Markdown content.
        """
        logger.info("report_generator_agent_start", run_id=state.run_id, model=best_model_name)

        ai_narrative = self._generate_ai_narrative(state, best_model_name, best_metrics)

        summary_md = build_executive_summary_md(
            state=state,
            best_model_name=best_model_name,
            best_metrics=best_metrics,
            eval_report=eval_report,
            top_models=top_models,
            ai_narrative=ai_narrative,
        )
        readme_md = build_project_readme_md(state, best_model_name, best_metrics)

        state.mark_phase_complete("report_generation")

        if self.artifact_manager:
            try:
                rep_dir = self.artifact_manager.get_path("reports")
                summary_path = rep_dir / "executive_summary.md"
                summary_path.write_text(summary_md, encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.REPORT,
                    path=summary_path,
                    description="Executive summary and comprehensive project conclusions",
                )

                # Write README at run root
                if self.artifact_manager.current_run_dir:
                    readme_path = self.artifact_manager.current_run_dir / "README.md"
                    readme_path.write_text(readme_md, encoding="utf-8")

            except Exception as e:
                logger.warning("report_save_failed", error=str(e))

        state.add_decision(
            DecisionRecord(
                agent="report_generator",
                action="generate_reports",
                reason=f"Compiled executive summary and project documentation for {best_model_name}.",
                approval_mode=state.autonomy_mode,
            )
        )

        logger.info("report_generator_agent_completed")
        return summary_md
