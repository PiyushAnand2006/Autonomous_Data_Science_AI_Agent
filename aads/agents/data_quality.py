"""
AADS Data Quality Agent — audits raw and transformed datasets,
identifies anomalies, and records comprehensive health metrics.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DataQualityReport, DecisionRecord
from aads.core.state import RunState
from aads.tools.quality.checker import audit_data_quality

logger = get_logger(__name__)


class DataQualityAgent:
    """Agent that performs data quality audits and surfaces anomalies and hygiene issues."""

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
    ) -> DataQualityReport:
        """Execute data quality checks and persist report artifact.

        Args:
            df: DataFrame to audit.
            state: The current RunState.

        Returns:
            DataQualityReport object.
        """
        logger.info("data_quality_agent_start", run_id=state.run_id, target=state.target_column)

        report = audit_data_quality(df, target_column=state.target_column)

        state.mark_phase_complete("data_quality")

        # Save artifact to metadata folder
        if self.artifact_manager:
            try:
                meta_dir = self.artifact_manager.get_path("metadata")
                report_path = meta_dir / "data_quality_report.json"
                report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=report_path,
                    description=f"Data quality report (Health Score: {report.overall_score}/100, Issues: {len(report.issues)})",
                )
            except Exception as e:
                logger.warning("data_quality_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="data_quality",
                action="audit_dataset",
                reason=f"Data health score: {report.overall_score}/100 with {len(report.issues)} identified issue(s).",
                approval_mode=state.autonomy_mode,
                details={
                    "overall_score": report.overall_score,
                    "issue_count": len(report.issues),
                    "has_critical": report.has_critical_issues,
                    "constant_columns": report.constant_columns,
                },
            )
        )

        logger.info(
            "data_quality_agent_completed",
            score=report.overall_score,
            issues=len(report.issues),
            has_critical=report.has_critical_issues,
        )

        return report
