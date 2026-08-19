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
from aads.core.schemas import ArtifactType, DecisionRecord
from aads.core.state import RunState
from aads.tools.notebook.generator import build_project_notebook

logger = get_logger(__name__)


class NotebookGeneratorAgent:
    """Agent that compiles all pipeline decisions and code cells into a reproducible Jupyter Notebook."""

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
        raw_data_filename: str = "dataset.csv",
    ) -> dict[str, Any]:
        """Synthesize and export reproducible Jupyter Notebook artifact.

        Args:
            state: The current RunState.
            best_model_name: Name of best model.
            raw_data_filename: Raw data filename.

        Returns:
            Jupyter notebook JSON dictionary.
        """
        logger.info("notebook_generator_agent_start", run_id=state.run_id, model=best_model_name)

        notebook_dict = build_project_notebook(
            state=state,
            raw_data_filename=raw_data_filename,
            best_model_name=best_model_name,
        )

        state.mark_phase_complete("notebook_generation")

        # Save to 05_Notebooks/
        if self.artifact_manager:
            try:
                nb_dir = self.artifact_manager.get_path("notebooks")
                nb_path = nb_dir / "pipeline_notebook.ipynb"
                nb_path.write_text(json.dumps(notebook_dict, indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.NOTEBOOK,
                    path=nb_path,
                    description=f"Reproducible end-to-end Jupyter Notebook ({len(notebook_dict['cells'])} cells)",
                )
            except Exception as e:
                logger.warning("notebook_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="notebook_generator",
                action="generate_notebook",
                reason=f"Synthesized reproducible pipeline notebook with {len(notebook_dict['cells'])} cells.",
                approval_mode=state.autonomy_mode,
                details={"cell_count": len(notebook_dict["cells"]), "best_model": best_model_name},
            )
        )

        logger.info("notebook_generator_agent_completed", cells=len(notebook_dict["cells"]))
        return notebook_dict
