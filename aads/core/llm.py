"""
AADS LLM Abstraction — provider-agnostic LLM factory.

Supports Google Gemini, OpenAI, Anthropic, and Ollama behind a clean
interface. The provider is selected via ``AADSConfig.llm_provider``.

Usage:
    from aads.core.llm import get_llm
    from aads.core.config import AADSConfig

    cfg = AADSConfig()
    llm = get_llm(cfg)  # returns a LangChain BaseChatModel
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aads.core.exceptions import ConfigError
from aads.core.logging import get_logger

if TYPE_CHECKING:
    from aads.core.config import AADSConfig
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)

# Registry of supported providers and their factory functions
_PROVIDER_FACTORIES: dict[str, str] = {
    "google": "_create_google_llm",
    "openai": "_create_openai_llm",
    "anthropic": "_create_anthropic_llm",
    "ollama": "_create_ollama_llm",
}


def get_llm(config: "AADSConfig") -> "BaseChatModel":
    """Create a LangChain chat model from AADS configuration.

    Args:
        config: AADS configuration with LLM provider settings.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ConfigError: If the provider is unsupported or dependencies are missing.
    """
    provider = config.llm_provider.lower().strip()

    if provider not in _PROVIDER_FACTORIES:
        raise ConfigError(
            f"Unsupported LLM provider '{provider}'. "
            f"Supported: {list(_PROVIDER_FACTORIES.keys())}"
        )

    factory_name = _PROVIDER_FACTORIES[provider]
    factory_fn = globals()[factory_name]
    llm = factory_fn(config)

    logger.info(
        "llm_initialized",
        provider=provider,
        model=config.llm_model,
    )
    return llm


def _create_google_llm(config: "AADSConfig") -> "BaseChatModel":
    """Create a Google Gemini chat model."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ConfigError(
            "Google Gemini requires 'langchain-google-genai'. "
            "Install with: pip install langchain-google-genai"
        )

    kwargs: dict = {
        "model": config.llm_model,
        "temperature": config.llm_temperature,
    }
    if config.llm_api_key:
        kwargs["google_api_key"] = config.llm_api_key

    return ChatGoogleGenerativeAI(**kwargs)


def _create_openai_llm(config: "AADSConfig") -> "BaseChatModel":
    """Create an OpenAI chat model."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ConfigError(
            "OpenAI requires 'langchain-openai'. "
            "Install with: pip install langchain-openai"
        )

    kwargs: dict = {
        "model": config.llm_model,
        "temperature": config.llm_temperature,
    }
    if config.llm_api_key:
        kwargs["openai_api_key"] = config.llm_api_key

    return ChatOpenAI(**kwargs)


def _create_anthropic_llm(config: "AADSConfig") -> "BaseChatModel":
    """Create an Anthropic Claude chat model."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ConfigError(
            "Anthropic requires 'langchain-anthropic'. "
            "Install with: pip install langchain-anthropic"
        )

    kwargs: dict = {
        "model": config.llm_model,
        "temperature": config.llm_temperature,
    }
    if config.llm_api_key:
        kwargs["anthropic_api_key"] = config.llm_api_key

    return ChatAnthropic(**kwargs)


def _create_ollama_llm(config: "AADSConfig") -> "BaseChatModel":
    """Create an Ollama local chat model."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ConfigError(
            "Ollama requires 'langchain-ollama'. "
            "Install with: pip install langchain-ollama"
        )

    return ChatOllama(
        model=config.llm_model,
        temperature=config.llm_temperature,
    )
