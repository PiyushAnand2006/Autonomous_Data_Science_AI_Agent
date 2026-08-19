"""Tests for aads.core.config — AADSConfig."""

import os

import pytest

from aads.core.config import AADSConfig
from aads.core.schemas import AutonomyMode, ExecutionEngine


class TestAADSConfigDefaults:
    """Verify default configuration values."""

    def test_default_random_seed(self):
        cfg = AADSConfig()
        assert cfg.random_seed == 42

    def test_default_autonomy_mode(self):
        cfg = AADSConfig()
        assert cfg.default_autonomy_mode == AutonomyMode.SEMI_AUTONOMOUS

    def test_default_engine(self):
        cfg = AADSConfig()
        assert cfg.default_engine == ExecutionEngine.PANDAS

    def test_default_llm_provider(self):
        cfg = AADSConfig()
        assert cfg.llm_provider == "google"

    def test_default_llm_temperature(self):
        cfg = AADSConfig()
        assert cfg.llm_temperature == 0.1

    def test_default_test_size(self):
        cfg = AADSConfig()
        assert 0.05 <= cfg.test_size <= 0.5

    def test_default_log_level(self):
        cfg = AADSConfig()
        assert cfg.log_level == "INFO"


class TestAADSConfigOverrides:
    """Verify that explicit overrides work."""

    def test_override_random_seed(self):
        cfg = AADSConfig(random_seed=123)
        assert cfg.random_seed == 123

    def test_override_autonomy_mode(self):
        cfg = AADSConfig(default_autonomy_mode=AutonomyMode.FULLY_AUTONOMOUS)
        assert cfg.default_autonomy_mode == AutonomyMode.FULLY_AUTONOMOUS

    def test_override_engine(self):
        cfg = AADSConfig(default_engine=ExecutionEngine.POLARS)
        assert cfg.default_engine == ExecutionEngine.POLARS

    def test_override_llm_model(self):
        cfg = AADSConfig(llm_model="gpt-4o")
        assert cfg.llm_model == "gpt-4o"

    def test_override_max_experiment_iterations(self):
        cfg = AADSConfig(max_experiment_iterations=5)
        assert cfg.max_experiment_iterations == 5


class TestAADSConfigEnvVars:
    """Verify environment variable loading with AADS_ prefix."""

    def test_env_override_random_seed(self, monkeypatch):
        monkeypatch.setenv("AADS_RANDOM_SEED", "99")
        cfg = AADSConfig()
        assert cfg.random_seed == 99

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("AADS_LOG_LEVEL", "DEBUG")
        cfg = AADSConfig()
        assert cfg.log_level == "DEBUG"
