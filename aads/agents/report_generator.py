"""
AADS Report Generator Agent — produces the executive summary and project documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord
from aads.core.state import RunState
from aads.tools.reporting.report_builder import (
    build_executive_summary_md,
    build_project_readme_md,
)

logger = get_logger(__name__)


class ReportGeneratorAgent:
    """Agent that compiles all findings, charts, and metrics into executive reports."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager

    def run(
        self,
        state: RunState,
        best_model_name: str,
        best_metrics: dict[str, float],
        eval_report: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate executive summary and project README.

        Args:
            state: The current RunState.
            best_model_name: Name of winning model.
            best_metrics: Holdout evaluation metrics.
            eval_report: Optional evaluation diagnostics.

        Returns:
            Executive summary Markdown content.
        """
        logger.info("report_generator_agent_start", run_id=state.run_id, model=best_model_name)

        summary_md = build_executive_summary_md(state, best_model_name, best_metrics, eval_report)
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
                    description="Executive summary and project conclusions",
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
