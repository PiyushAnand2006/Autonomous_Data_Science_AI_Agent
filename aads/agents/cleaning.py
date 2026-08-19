"""
AADS Cleaning Agent — performs safe, logged data cleaning and exports cleaned artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord
from aads.core.state import RunState
from aads.tools.processing.cleaner import clean_dataset

logger = get_logger(__name__)


class CleaningAgent:
    """Agent responsible for cleaning raw data, recording decisions, and persisting cleaned artifacts."""

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
    ) -> tuple[pd.DataFrame, dict]:
        """Clean dataset and export artifacts into 02_Cleaned_Data/.

        Args:
            df: Input raw DataFrame.
            state: The current RunState.

        Returns:
            Tuple of (cleaned_dataframe, cleaning_log).
        """
        logger.info("cleaning_agent_start", run_id=state.run_id, target=state.target_column)

        cleaned_df, cleaning_log = clean_dataset(
            df,
            target_column=state.target_column,
            missing_drop_threshold=0.80,
            winsorize_outliers=True,
        )

        state.mark_phase_complete("cleaning")

        # Save artifacts to 02_Cleaned_Data/
        if self.artifact_manager:
            try:
                clean_dir = self.artifact_manager.get_path("cleaned_data")

                # Save parquet / csv
                clean_data_path = clean_dir / "cleaned_dataset.parquet"
                cleaned_df.to_parquet(clean_data_path, index=False)

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.CLEANED_DATA,
                    path=clean_data_path,
                    description=f"Cleaned dataset ({len(cleaned_df)} rows × {len(cleaned_df.columns)} cols)",
                )

                # Save log
                log_path = clean_dir / "cleaning_log.json"
                log_path.write_text(json.dumps(cleaning_log, indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=log_path,
                    description="Detailed log of all data cleaning transformations",
                )
            except Exception as e:
                logger.warning("cleaning_artifact_save_failed", error=str(e))

        # Record decision
        state.add_decision(
            DecisionRecord(
                agent="cleaning",
                action="clean_dataset",
                reason=(
                    f"Cleaned data: dropped {len(cleaning_log['dropped_columns'])} column(s), "
                    f"deduplicated {cleaning_log['deduplicated_rows']} row(s), "
                    f"imputed {len(cleaning_log['imputations'])} feature(s)."
                ),
                approval_mode=state.autonomy_mode,
                details=cleaning_log,
            )
        )

        logger.info("cleaning_agent_completed", final_shape=cleaning_log["final_shape"])
        return cleaned_df, cleaning_log
