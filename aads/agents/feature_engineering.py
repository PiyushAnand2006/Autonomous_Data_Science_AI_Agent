"""
AADS Feature Engineering Agent — discovers, constructs, and selects candidate features,
exporting transformed datasets to 03_Feature_Engineered_Data/.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord, TaskType
from aads.core.state import RunState
from aads.tools.processing.feature_engineer import generate_candidate_features

logger = get_logger(__name__)


class FeatureEngineeringAgent:
    """Agent that creates interaction, transformation, and domain features."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
        llm: Any = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager
        self.llm = llm

    def _consult_ai_feature_engineering(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        state: RunState,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Consult LLM to identify noisy/non-predictive columns to drop and domain interactions to create."""
        if getattr(self.config, "execution_mode", "local") != "ai":
            return [], []

        llm_client = self.llm
        if llm_client is None:
            try:
                from aads.core.llm import get_llm
                llm_client = get_llm(self.config)
            except Exception:
                return [], []

        try:
            col_info = []
            for col in list(X_train.columns)[:35]:
                nunique = X_train[col].nunique(dropna=False)
                dtype = str(X_train[col].dtype)
                col_info.append(f"- {col} (dtype={dtype}, distinct={nunique})")

            prompt = (
                f"You are a World-Class Kaggle Grandmaster and Principal Data Scientist.\n"
                f"Objective: {state.user_objective}\n"
                f"Task Type: {state.task_type.value if state.task_type else 'classification'}\n"
                f"Target Column: {state.target_column}\n\n"
                f"Feature Columns:\n" + "\n".join(col_info) + "\n\n"
                f"Your goal is to MAXIMIZE predictive accuracy and eliminate noise.\n"
                f"Return valid JSON ONLY with this exact structure:\n"
                f"{{\n"
                f'  "drop_features": ["list", "of", "useless", "or", "non_predictive_id_columns"],\n'
                f'  "domain_interactions": [\n'
                f'    {{"name": "interaction_col_name", "col1": "existing_col", "op": "/", "col2": "existing_col"}}\n'
                f"  ]\n"
                f"}}\n"
                f"Note: For 'op', supported operators are '/', '*', '+', '-'. Only select numerical columns for domain_interactions."
            )

            response = llm_client.invoke(prompt)
            raw = str(getattr(response, "content", "") or response).strip()
            
            # Extract JSON from potential markdown fence
            if "```" in raw:
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)

            parsed = json.loads(raw)
            drops = [c for c in parsed.get("drop_features", []) if c in X_train.columns]
            interactions = parsed.get("domain_interactions", [])
            return drops, interactions
        except Exception as e:
            logger.debug("ai_feature_engineering_consultation_skipped", error=str(e))
            return [], []

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        state: RunState,
        X_val: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict[str, Any]]:
        """Run feature engineering on training features and project to test/val sets.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            state: The current RunState.
            X_val: Optional validation features.

        Returns:
            Tuple of (X_train_fe, X_test_fe, X_val_fe, feature_log).
        """
        logger.info("feature_engineering_agent_start", run_id=state.run_id)
        task_type = state.task_type or TaskType.CLASSIFICATION

        X_train_fe, X_test_fe, log = generate_candidate_features(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            task_type=task_type,
            max_new_features=10,
        )

        X_val_fe = None
        if X_val is not None and len(X_val) > 0:
            _, X_val_fe, _ = generate_candidate_features(
                X_train=X_train,
                X_test=X_val,
                y_train=y_train,
                task_type=task_type,
                max_new_features=10,
            )

        # AI Cognitive Feature Selection & Domain Synthesis
        ai_drops, ai_interactions = self._consult_ai_feature_engineering(X_train, y_train, state)

        # 1. Apply AI Domain Interactions
        for inter in ai_interactions:
            try:
                name = inter.get("name")
                c1 = inter.get("col1")
                c2 = inter.get("col2")
                op = inter.get("op", "/")
                if c1 in X_train_fe.columns and c2 in X_train_fe.columns and name:
                    if op == "/":
                        X_train_fe[name] = X_train_fe[c1] / (X_train_fe[c2].abs() + 1e-6)
                        X_test_fe[name] = X_test_fe[c1] / (X_test_fe[c2].abs() + 1e-6)
                        if X_val_fe is not None:
                            X_val_fe[name] = X_val_fe[c1] / (X_val_fe[c2].abs() + 1e-6)
                    elif op == "*":
                        X_train_fe[name] = X_train_fe[c1] * X_train_fe[c2]
                        X_test_fe[name] = X_test_fe[c1] * X_test_fe[c2]
                        if X_val_fe is not None:
                            X_val_fe[name] = X_val_fe[c1] * X_val_fe[c2]
                    elif op == "+":
                        X_train_fe[name] = X_train_fe[c1] + X_train_fe[c2]
                        X_test_fe[name] = X_test_fe[c1] + X_test_fe[c2]
                        if X_val_fe is not None:
                            X_val_fe[name] = X_val_fe[c1] + X_val_fe[c2]
                    elif op == "-":
                        X_train_fe[name] = X_train_fe[c1] - X_train_fe[c2]
                        X_test_fe[name] = X_test_fe[c1] - X_test_fe[c2]
                        if X_val_fe is not None:
                            X_val_fe[name] = X_val_fe[c1] - X_val_fe[c2]
                    log["created_features"].append({"name": name, "type": f"ai_domain_{op}", "source": [c1, c2]})
            except Exception as e:
                logger.debug("ai_interaction_application_failed", error=str(e))

        # 2. Apply AI Noise Pruning (Drop non-predictive features)
        if ai_drops:
            valid_drops = [c for c in ai_drops if c in X_train_fe.columns and len(X_train_fe.columns) > len(ai_drops) + 1]
            if valid_drops:
                X_train_fe.drop(columns=valid_drops, inplace=True, errors="ignore")
                X_test_fe.drop(columns=valid_drops, inplace=True, errors="ignore")
                if X_val_fe is not None:
                    X_val_fe.drop(columns=valid_drops, inplace=True, errors="ignore")
                log["dropped_features"].extend(valid_drops)

        state.mark_phase_complete("feature_engineering")

        # Save artifacts to 03_Feature_Engineered_Data/
        if self.artifact_manager:
            try:
                fe_dir = self.artifact_manager.get_path("feature_engineered_data")

                # Export full feature engineered dataset as CSV and Parquet
                full_fe_df = pd.concat([X_train_fe, X_test_fe], axis=0).reset_index(drop=True)
                fe_csv_path = fe_dir / "feature_engineered_dataset.csv"
                full_fe_df.to_csv(fe_csv_path, index=False)

                fe_parquet_path = fe_dir / "feature_engineered_dataset.parquet"
                full_fe_df.to_parquet(fe_parquet_path, index=False)

                X_train_fe.to_parquet(fe_dir / "X_train_fe.parquet", index=False)

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.FEATURE_ENGINEERED_DATA,
                    path=fe_csv_path,
                    description=f"Feature-engineered dataset CSV ({len(full_fe_df)} rows × {len(full_fe_df.columns)} cols)",
                )
                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.FEATURE_ENGINEERED_DATA,
                    path=fe_dir / "X_train_fe.parquet",
                    description=f"Feature-engineered training data ({len(X_train_fe.columns)} features)",
                )

                log_path = fe_dir / "feature_engineering_log.json"
                log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=log_path,
                    description="Detailed log of generated feature candidates",
                )
            except Exception as e:
                logger.warning("feature_engineering_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="feature_engineering",
                action="generate_features",
                reason=f"Engineered {len(log['created_features'])} candidate features (retained {len(log['retained_features'])}).",
                approval_mode=state.autonomy_mode,
                details=log,
            )
        )

        logger.info(
            "feature_engineering_agent_completed",
            new_features=len(log["created_features"]),
            total_features=len(X_train_fe.columns),
        )
        return X_train_fe, X_test_fe, X_val_fe, log
