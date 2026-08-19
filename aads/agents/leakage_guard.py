"""
AADS Leakage Guard Agent — enforces strict boundaries between features and targets,
and prevents train/test contamination.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.exceptions import LeakageError
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord
from aads.core.state import RunState
from aads.tools.processing.leakage import audit_leakage

logger = get_logger(__name__)


class LeakageGuard:
    """Agent that guards against data leakage, proxy target leakage, and test contamination."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        state: RunState,
        raise_on_critical: bool = False,
    ) -> dict[str, Any]:
        """Perform leakage checks across training and test splits.

        Args:
            X_train: Training features.
            X_test: Testing features.
            y_train: Training target.
            y_test: Testing target.
            state: The current RunState.
            raise_on_critical: If True, raises LeakageError upon critical findings.

        Returns:
            Audit result dictionary.

        Raises:
            LeakageError: When critical leakage is detected and raise_on_critical is True.
        """
        target = state.target_column or "target"
        logger.info("leakage_guard_start", run_id=state.run_id, target=target)

        report = audit_leakage(X_train, X_test, y_train, y_test, target_name=target)

        state.mark_phase_complete("leakage_check")

        # Save metadata report if artifact manager present
        if self.artifact_manager:
            try:
                meta_dir = self.artifact_manager.get_path("metadata")
                rep_path = meta_dir / "leakage_audit.json"
                rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=rep_path,
                    description=f"Leakage audit (Has Leakage: {report['has_leakage']}, Critical: {len(report['critical_issues'])})",
                )
            except Exception as e:
                logger.warning("leakage_artifact_save_failed", error=str(e))

        # Log decision
        action_summary = "Blocked due to leakage" if report["has_leakage"] else "Passed leakage guard"
        state.add_decision(
            DecisionRecord(
                agent="leakage_guard",
                action="leakage_check",
                reason=(
                    f"{action_summary}. Critical issues: {len(report['critical_issues'])}, "
                    f"Warnings: {len(report['warnings'])}."
                ),
                approval_mode=state.autonomy_mode,
                approved=not report["has_leakage"],
                details=report,
            )
        )

        if report["has_leakage"] and raise_on_critical:
            error_msg = "; ".join(report["critical_issues"])
            logger.error("leakage_guard_critical_failure", errors=report["critical_issues"])
            raise LeakageError(f"Data leakage detected: {error_msg}")

        logger.info("leakage_guard_completed", has_leakage=report["has_leakage"])
        return report
