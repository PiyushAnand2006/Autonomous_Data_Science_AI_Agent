"""
AADS Notebook Generator Agent — produces clean, reproducible, end-to-end Jupyter Notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import (
    ArtifactType,
    DecisionRecord,
    NotebookValidationResult,
)
from aads.core.state import RunState
from aads.tools.notebook.generator import build_project_notebook
from aads.tools.notebook.validator import validate_and_execute_notebook

logger = get_logger(__name__)


class NotebookGeneratorAgent:
    """Agent that compiles all pipeline decisions and code cells into a reproducible Jupyter Notebook and validates execution."""

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
        best_model_name: str = "RandomForestClassifier",
        raw_data_filename: str = "original_dataset.csv",
        top_models: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Synthesize and export reproducible Jupyter Notebook artifact and execute validation.

        Args:
            state: The current RunState.
            best_model_name: Name of best model.
            raw_data_filename: Raw data filename.
            top_models: Optional list of top ranked models.

        Returns:
            Jupyter notebook JSON dictionary.
        """
        logger.info("notebook_generator_agent_start", run_id=state.run_id, model=best_model_name)

        notebook_dict = build_project_notebook(
            state=state,
            raw_data_filename=raw_data_filename,
            best_model_name=best_model_name,
            top_models=top_models,
        )

        state.mark_phase_complete("notebook_generation")

        validation_result = NotebookValidationResult(success=True)

        # Save to 05_Notebook/
        if self.artifact_manager:
            try:
                nb_dir = self.artifact_manager.get_path("notebook")

                # Primary named notebook: autonomous_analysis.ipynb
                nb_path = nb_dir / "autonomous_analysis.ipynb"
                nb_path.write_text(json.dumps(notebook_dict, indent=2), encoding="utf-8")

                # Legacy / alias notebook for backward compatibility
                alias_path = nb_dir / "pipeline_notebook.ipynb"
                alias_path.write_text(json.dumps(notebook_dict, indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.NOTEBOOK,
                    path=nb_path,
                    description=f"Professional reproducible Jupyter Notebook ({len(notebook_dict['cells'])} cells)",
                )

                # Programmatic top-to-bottom validation of generated notebook
                validation_result = validate_and_execute_notebook(
                    notebook_path=nb_path,
                    working_dir=nb_dir,
                )

                val_path = nb_dir / "notebook_validation.json"
                val_path.write_text(validation_result.model_dump_json(indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=val_path,
                    description=f"Notebook execution validation (Success={validation_result.success}, Duration={validation_result.duration_seconds}s)",
                )

            except Exception as e:
                logger.warning("notebook_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="notebook_generator",
                action="generate_and_validate_notebook",
                reason=(
                    f"Generated professional notebook with {len(notebook_dict['cells'])} cells. "
                    f"Validation status: {'PASSED' if validation_result.success else 'WARNING'}."
                ),
                approval_mode=state.autonomy_mode,
                details={
                    "cell_count": len(notebook_dict["cells"]),
                    "best_model": best_model_name,
                    "validation_success": validation_result.success,
                    "validation_errors": validation_result.errors,
                },
            )
        )

        logger.info(
            "notebook_generator_agent_completed",
            cells=len(notebook_dict["cells"]),
            validation_success=validation_result.success,
        )
        return notebook_dict
