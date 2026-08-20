"""
AADS LLM Abstraction — provider-agnostic, zero-dependency LLM engine.

Supports OpenRouter, Google Gemini, OpenAI, Anthropic, Groq, and Ollama behind a clean
interface with automatic fallback between native SDKs, OpenAI-compatible endpoints,
and standard HTTP clients.

Usage:
    from aads.core.llm import get_llm
    from aads.core.config import AADSConfig

    cfg = AADSConfig(execution_mode="ai", llm_provider="openrouter", llm_model="anthropic/claude-3.5-sonnet")
    llm = get_llm(cfg)
    response = llm.invoke("Summarize dataset findings.")
    print(response.content)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Optional

from aads.core.exceptions import ConfigError
from aads.core.logging import get_logger

if TYPE_CHECKING:
    from aads.core.config import AADSConfig

logger = get_logger(__name__)

# Registry of supported providers
_SUPPORTED_PROVIDERS = {"openrouter", "google", "openai", "anthropic", "groq", "ollama"}

# Curated models per provider
DEFAULT_PROVIDER_MODELS: dict[str, list[str]] = {
    "openrouter": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "deepseek/deepseek-r1",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "openai/gpt-4o-mini",
        "mistralai/mistral-large-2411",
    ],
    "google": [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-pro-exp-02-05",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
        "o1",
        "gpt-4-turbo",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768",
    ],
    "ollama": [
        "llama3.2",
        "llama3.1",
        "deepseek-r1",
        "mistral",
        "qwen2.5",
        "phi3",
    ],
}


class AADSLLMResponse:
    """Standardized response container compatible with LangChain BaseMessage."""

    def __init__(self, content: str, raw_response: Optional[dict[str, Any]] = None) -> None:
        self.content = content
        self.raw_response = raw_response or {}

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"AADSLLMResponse(content={self.content[:60]!r}...)"


class BaseAADSLLM:
    """Base interface for all AADS LLM providers."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        base_url: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.base_url = base_url
        self.extra_headers = extra_headers or {}

    def _normalize_messages(self, input_data: Any) -> list[dict[str, str]]:
        """Normalize string, LangChain messages, or dict list into chat format."""
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]

        if isinstance(input_data, list):
            normalized = []
            for item in input_data:
                if isinstance(item, dict):
                    normalized.append(item)
                elif hasattr(item, "content") and hasattr(item, "type"):
                    role = "system" if getattr(item, "type") == "system" else "user" if getattr(item, "type") == "human" else "assistant"
                    normalized.append({"role": role, "content": str(item.content)})
                elif hasattr(item, "content"):
                    normalized.append({"role": "user", "content": str(item.content)})
                else:
                    normalized.append({"role": "user", "content": str(item)})
            return normalized

        return [{"role": "user", "content": str(input_data)}]

    def invoke(self, input_data: Any) -> AADSLLMResponse:
        """Synchronously execute chat completion."""
        raise NotImplementedError


class OpenAILikeLLM(BaseAADSLLM):
    """Universal OpenAI-compatible LLM client (works with OpenRouter, Groq, OpenAI, Ollama)."""

    def invoke(self, input_data: Any) -> AADSLLMResponse:
        messages = self._normalize_messages(input_data)
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("AADS_LLM_API_KEY") or "dummy-key"
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions" if self.base_url else "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "AADS-Autonomous-AI-Data-Scientist/1.0",
            **self.extra_headers,
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        logger.info("sending_api_request_to_provider", endpoint=endpoint, model=self.model)

        req = urllib.request.Request(
            endpoint,
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choice = resp_data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                logger.info("api_response_received_successfully", model=self.model, length=len(content))
                return AADSLLMResponse(content=content, raw_response=resp_data)
        except urllib.error.HTTPError as http_err:
            error_body = ""
            try:
                error_body = http_err.read().decode("utf-8")
            except Exception:
                pass
            err_msg = f"HTTP {http_err.code} from {endpoint}: {http_err.reason}. {error_body}"
            logger.error("api_request_http_error", error=err_msg, model=self.model)
            raise RuntimeError(err_msg) from http_err
        except Exception as e:
            logger.error("api_request_connection_error", error=str(e), model=self.model)
            raise RuntimeError(f"Failed to communicate with provider at {endpoint}: {e}") from e


class GoogleGeminiLLM(BaseAADSLLM):
    """Direct Google Gemini REST API client."""

    def invoke(self, input_data: Any) -> AADSLLMResponse:
        messages = self._normalize_messages(input_data)
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("AADS_LLM_API_KEY")
        if not api_key:
            raise ConfigError("Google Gemini requires GEMINI_API_KEY or GOOGLE_API_KEY")

        # Format messages for Gemini API
        contents = []
        for m in messages:
            role = "user" if m.get("role") in ["user", "system"] else "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        model_name = self.model if self.model.startswith("gemini") else f"gemini-2.0-flash"
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": self.temperature},
        }

        req = urllib.request.Request(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                candidates = resp_data.get("candidates", [{}])
                parts = candidates[0].get("content", {}).get("parts", [{}])
                content = parts[0].get("text", "")
                return AADSLLMResponse(content=content, raw_response=resp_data)
        except Exception as e:
            logger.error("gemini_api_error", error=str(e))
            raise RuntimeError(f"Google Gemini API error: {e}") from e


class AnthropicLLM(BaseAADSLLM):
    """Direct Anthropic Claude REST API client."""

    def invoke(self, input_data: Any) -> AADSLLMResponse:
        messages = self._normalize_messages(input_data)
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("AADS_LLM_API_KEY")
        if not api_key:
            raise ConfigError("Anthropic Claude requires ANTHROPIC_API_KEY")

        system_prompt = ""
        chat_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_prompt = m.get("content", "")
            else:
                chat_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": 4096,
            "temperature": self.temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        req = urllib.request.Request(
            endpoint,
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content_blocks = resp_data.get("content", [{}])
                text = "".join([b.get("text", "") for b in content_blocks if b.get("type") == "text"])
                return AADSLLMResponse(content=text, raw_response=resp_data)
        except Exception as e:
            logger.error("anthropic_api_error", error=str(e))
            raise RuntimeError(f"Anthropic API error: {e}") from e


def list_provider_models(provider: str, api_key: Optional[str] = None) -> list[str]:
    """Dynamically fetch available models for a given provider or return curated fallbacks."""
    prov = provider.lower().strip()
    fallback = DEFAULT_PROVIDER_MODELS.get(prov, ["default-model"])

    # Live query for OpenRouter
    if prov == "openrouter":
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={
                    "User-Agent": "AADS-Agent/1.0",
                    **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    model_ids = [m["id"] for m in data.get("data", []) if "id" in m]
                    if model_ids:
                        top_hints = ["anthropic/claude-3.5", "openai/gpt-4o", "deepseek/deepseek-r1", "google/gemini-2.0", "meta-llama/llama-3.3"]
                        sorted_models = sorted(
                            model_ids,
                            key=lambda m: (not any(h in m for h in top_hints), m),
                        )
                        return sorted_models
        except Exception as e:
            logger.debug("openrouter_model_fetch_failed", error=str(e))
            return fallback

    # Live query for Ollama
    if prov == "ollama":
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m["name"] for m in data.get("models", []) if "name" in m]
                    if models:
                        return models
        except Exception:
            return fallback

    return fallback


def get_llm(config: "AADSConfig") -> BaseAADSLLM:
    """Create a high-performing, provider-agnostic chat model instance.

    Args:
        config: AADS configuration with LLM provider settings.

    Returns:
        A BaseAADSLLM instance ready for `.invoke(messages)`.

    Raises:
        ConfigError: If provider is unsupported.
    """
    provider = (config.llm_provider or "openrouter").lower().strip()

    if provider not in _SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"Unsupported LLM provider '{provider}'. "
            f"Supported: {list(_SUPPORTED_PROVIDERS)}"
        )

    # 1. OpenRouter
    if provider == "openrouter":
        return OpenAILikeLLM(
            model=config.llm_model or "anthropic/claude-3.5-sonnet",
            api_key=config.llm_api_key,
            temperature=config.llm_temperature,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "https://github.com/PiyushAnand2006/Autonomous_Data_Science_AI_Agent",
                "X-Title": "AADS Autonomous AI Data Scientist",
            },
        )

    # 2. Groq
    elif provider == "groq":
        return OpenAILikeLLM(
            model=config.llm_model or "llama-3.3-70b-versatile",
            api_key=config.llm_api_key or os.getenv("GROQ_API_KEY"),
            temperature=config.llm_temperature,
            base_url="https://api.groq.com/openai/v1",
        )

    # 3. OpenAI
    elif provider == "openai":
        return OpenAILikeLLM(
            model=config.llm_model or "gpt-4o",
            api_key=config.llm_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=config.llm_temperature,
            base_url="https://api.openai.com/v1",
        )

    # 4. Ollama (local)
    elif provider == "ollama":
        return OpenAILikeLLM(
            model=config.llm_model or "llama3.2",
            api_key="ollama",
            temperature=config.llm_temperature,
            base_url="http://localhost:11434/v1",
        )

    # 5. Google Gemini
    elif provider == "google":
        return GoogleGeminiLLM(
            model=config.llm_model or "gemini-2.0-flash",
            api_key=config.llm_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            temperature=config.llm_temperature,
        )

    # 6. Anthropic
    elif provider == "anthropic":
        return AnthropicLLM(
            model=config.llm_model or "claude-3-5-sonnet-20241022",
            api_key=config.llm_api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=config.llm_temperature,
        )

    raise ConfigError(f"Unsupported provider: {provider}")


def test_llm_connection(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> tuple[bool, str]:
    """Test connection to an LLM provider and verify the API key and model availability.

    Args:
        provider: Provider name (e.g. 'openrouter', 'google', 'openai', 'groq').
        model: Target model identifier.
        api_key: Optional API key.

    Returns:
        Tuple of (is_successful: bool, message: str).
    """
    from aads.core.config import AADSConfig

    cfg = AADSConfig(
        execution_mode="ai",
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key if api_key and api_key.strip() else None,
    )
    try:
        llm = get_llm(cfg)
        resp = llm.invoke("Respond with only the word: OK")
        content = getattr(resp, "content", "") or str(resp)
        return True, f"Successfully connected to {provider.upper()} ({model})! Model response: {content.strip()[:60]}"
    except Exception as e:
        return False, f"Connection failed: {e}"
