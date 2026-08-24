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


_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AADSConfig(BaseSettings):
    """Central configuration for an AADS session.

    Values are resolved in order: explicit kwargs → environment variables →
    .env file → defaults defined here.
    """

    # ── Project paths ──────────────────────────────────────────────────────
    project_root: Path = Field(
        default_factory=lambda: _DEFAULT_PROJECT_ROOT,
        description="Root directory of the AADS project",
    )
    storage_dir: str = Field(
        default="",
        description="Directory where generated run folders are created",
    )

    @property
    def storage_root(self) -> Path:
        """Return safe resolved absolute Path for storage directory anchored to project_root."""
        if self.storage_dir and str(self.storage_dir).strip():
            p = Path(self.storage_dir)
            if p.is_absolute():
                return p.resolve()
            return (self.project_root / p).resolve()
        return (self.project_root / "storage" / "runs").resolve()

    # ── Execution Mode ────────────────────────────────────────────────────
    execution_mode: str = Field(
        default="local",
        description="Execution mode: 'local' (deterministic offline) or 'ai' (LLM-assisted)",
    )

    # ── LLM provider ──────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider key (openrouter, nvidia, google, openai, anthropic, ollama, custom)",
    )
    llm_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="Model identifier to use with the provider",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM provider (prefer env var)",
    )
    custom_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for custom OpenAI-compatible endpoint (e.g. https://api.together.xyz/v1, http://localhost:1234/v1)",
    )
    custom_provider_name: Optional[str] = Field(
        default="Custom Provider",
        description="Display name for custom provider",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM inference",
    )

    # ── Model Export ──────────────────────────────────────────────────────
    top_models_count: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of top-performing ML models to export (default 4)",
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
