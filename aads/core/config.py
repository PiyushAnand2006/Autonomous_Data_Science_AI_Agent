"""
AADS Configuration — application-level settings loaded from environment,
.env files, and defaults.

Usage:
    from aads.core.config import AADSConfig
    cfg = AADSConfig()                # loads from env / .env
    cfg = AADSConfig(random_seed=42)  # explicit override
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from aads.core.schemas import AutonomyMode, ExecutionEngine


class AADSConfig(BaseSettings):
    """Central configuration for an AADS session.

    Values are resolved in order: explicit kwargs → environment variables →
    .env file → defaults defined here.
    """

    # ── Project paths ──────────────────────────────────────────────────────
    project_root: Path = Field(
        default=Path("."),
        description="Root directory of the AADS project",
    )
    storage_dir: str = Field(
        default="aads/storage/runs",
        description="Directory where generated run folders are created",
    )

    # ── LLM provider ──────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="google",
        description="LLM provider key (google, openai, anthropic, ollama)",
    )
    llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Model identifier to use with the provider",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM provider (prefer env var)",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM inference",
    )

    # ── Autonomy ──────────────────────────────────────────────────────────
    default_autonomy_mode: AutonomyMode = Field(
        default=AutonomyMode.SEMI_AUTONOMOUS,
        description="Default human-oversight level for new runs",
    )

    # ── Execution ─────────────────────────────────────────────────────────
    default_engine: ExecutionEngine = Field(
        default=ExecutionEngine.PANDAS,
        description="Default data-processing engine",
    )
    pandas_row_limit: int = Field(
        default=500_000,
        description="Row threshold above which Polars/DuckDB is preferred",
    )

    # ── Reproducibility ───────────────────────────────────────────────────
    random_seed: int = Field(
        default=42,
        description="Global random seed for reproducibility",
    )

    # ── Experiment budget ─────────────────────────────────────────────────
    max_experiment_iterations: int = Field(
        default=10,
        description="Maximum replanning iterations before stopping",
    )
    test_size: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Default test split ratio",
    )
    validation_size: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Default validation split ratio (0 = no separate validation set)",
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_json: bool = Field(
        default=False,
        description="If True, emit JSON-formatted log lines",
    )

    model_config = {
        "env_prefix": "AADS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
