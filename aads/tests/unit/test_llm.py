"""
Tests for AADS LLM abstraction factory.
"""

import pytest

from aads.core.config import AADSConfig
from aads.core.exceptions import ConfigError
from aads.core.llm import get_llm


class TestLLMFactory:
    """Verify LLM factory behavior and error handling."""

    def test_unsupported_provider_raises_config_error(self):
        cfg = AADSConfig(llm_provider="unsupported_vendor_xyz")
        with pytest.raises(ConfigError, match="Unsupported LLM provider"):
            get_llm(cfg)

    def test_factory_case_insensitive(self):
        cfg = AADSConfig(llm_provider="GOOGLE")
        try:
            llm = get_llm(cfg)
            assert llm is not None
        except ConfigError as e:
            assert "requires" in str(e)

    def test_openrouter_factory(self):
        cfg = AADSConfig(llm_provider="openrouter", llm_api_key="sk-test-fake-key", llm_model="openai/gpt-4o")
        try:
            llm = get_llm(cfg)
            assert llm is not None
        except ConfigError as e:
            assert "requires" in str(e)

    def test_list_provider_models_fallback(self):
        from aads.core.llm import list_provider_models
        models = list_provider_models("openrouter")
        assert isinstance(models, list)
        assert len(models) >= 3
        assert any("claude" in m or "gpt" in m or "llama" in m for m in models)

        groq_models = list_provider_models("groq")
        assert len(groq_models) >= 2
