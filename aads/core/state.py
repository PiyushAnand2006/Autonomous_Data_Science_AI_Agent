"""
AADS Run State — the single source of truth for a running workflow.

RunState tracks everything about a single execution: identity, objective,
current phase, artifacts, decisions, and experiment history. It is serializable
to JSON so it can be persisted and resumed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from aads.core.schemas import (
    ArtifactRecord,
    AutonomyMode,
    DatasetMeta,
    DecisionRecord,
    ExperimentRecord,
    TaskType,
)


class RunState(BaseModel):
    """Immutable-style state object for a single AADS run.

    Create with ``RunState.create(...)`` and update via ``state.update(...)``
    which returns a *new* RunState (functional style), or mutate in place
    for simpler orchestration code.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    run_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Unique run identifier",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the run started",
    )

    # ── User inputs ───────────────────────────────────────────────────────
    user_objective: str = Field(
        default="",
        description="Natural-language objective provided by the user",
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Explicitly specified target column, if any",
    )
    autonomy_mode: AutonomyMode = Field(
        default=AutonomyMode.SEMI_AUTONOMOUS,
    )
    user_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional constraints (metric, model family, time budget, etc.)",
    )

    # ── Task understanding ────────────────────────────────────────────────
    task_type: Optional[TaskType] = Field(
        default=None,
        description="Detected or confirmed task type",
    )
    task_plan: list[str] = Field(
        default_factory=list,
        description="Ordered list of planned workflow steps",
    )
    columns_to_drop: list[str] = Field(
        default_factory=list,
        description="Columns to drop due to user instructions, IDs, or autonomous triage",
    )
    column_triage_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="Explicit justifications for each dropped column",
    )
    user_guidelines: list[str] = Field(
        default_factory=list,
        description="User-specified constraints and domain directives extracted from objective",
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_meta: Optional[DatasetMeta] = Field(
        default=None,
        description="Metadata captured from the uploaded dataset",
    )

    # ── Progress ──────────────────────────────────────────────────────────
    current_phase: str = Field(
        default="initialized",
        description="Name of the current workflow phase",
    )
    completed_phases: list[str] = Field(
        default_factory=list,
        description="Phases that have been completed",
    )

    # ── Registries ────────────────────────────────────────────────────────
    artifacts: list[ArtifactRecord] = Field(
        default_factory=list,
        description="All generated artifacts",
    )
    decisions: list[DecisionRecord] = Field(
        default_factory=list,
        description="All autonomous or approved decisions",
    )
    experiments: list[ExperimentRecord] = Field(
        default_factory=list,
        description="ML experiment history",
    )

    # ── Artifact root ─────────────────────────────────────────────────────
    run_dir: Optional[str] = Field(
        default=None,
        description="Absolute path to the run output directory",
    )

    # ── Metadata ──────────────────────────────────────────────────────────
    software_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Key library versions captured at run start",
    )
    random_seed: int = Field(default=42)

    # ──────────────────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        user_objective: str,
        autonomy_mode: AutonomyMode = AutonomyMode.SEMI_AUTONOMOUS,
        target_column: Optional[str] = None,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> RunState:
        """Create a new RunState with a fresh run_id and timestamp."""
        return cls(
            user_objective=user_objective,
            autonomy_mode=autonomy_mode,
            target_column=target_column,
            random_seed=random_seed,
            **kwargs,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def add_artifact(self, record: ArtifactRecord) -> None:
        """Register a new artifact."""
        self.artifacts.append(record)

    def add_decision(self, record: DecisionRecord) -> None:
        """Register a new decision."""
        self.decisions.append(record)

    def add_experiment(self, record: ExperimentRecord) -> None:
        """Register a new experiment result."""
        self.experiments.append(record)

    def mark_phase_complete(self, phase: str) -> None:
        """Mark a phase as completed and update current_phase."""
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)

    def get_best_experiment(self) -> Optional[ExperimentRecord]:
        """Return the experiment marked as best, if any."""
        for exp in self.experiments:
            if exp.is_best:
                return exp
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Serialize state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunState:
        """Deserialize state from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
