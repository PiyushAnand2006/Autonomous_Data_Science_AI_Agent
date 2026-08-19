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
        # If langchain_google_genai is installed, it initializes; otherwise raises ConfigError with helpful message
        try:
            llm = get_llm(cfg)
            assert llm is not None
        except ConfigError as e:
            assert "requires" in str(e)
