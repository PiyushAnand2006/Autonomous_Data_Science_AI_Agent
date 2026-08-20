"""
Persistent settings manager for AADS.
Stores and retrieves user configuration, API keys, selected models, and preferences
in a local JSON file so settings persist across browser refreshes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / ".aads_user_settings.json"


def get_settings_file_path() -> Path:
    """Return the path to the persistent settings JSON file."""
    return _SETTINGS_PATH


def load_user_settings() -> Dict[str, Any]:
    """Load user settings from disk.

    Returns:
        Dictionary of stored settings, or default structure if not found.
    """
    if not _SETTINGS_PATH.is_file():
        return {}
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user_settings(settings: Dict[str, Any]) -> None:
    """Save or update user settings on disk.

    Args:
        settings: Key-value pairs to merge into settings.
    """
    try:
        current = load_user_settings()
        current.update(settings)
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def get_stored_api_key(provider: str) -> str:
    """Get the stored API key for a specific provider.

    Checks:
    1. Stored user settings (.aads_user_settings.json)
    2. Environment variables (e.g. OPENROUTER_API_KEY, AADS_LLM_API_KEY)

    Args:
        provider: Provider name (e.g. 'openrouter', 'google', 'openai')

    Returns:
        API key string or empty string.
    """
    settings = load_user_settings()
    api_keys = settings.get("api_keys", {})
    if isinstance(api_keys, dict) and provider in api_keys and api_keys[provider]:
        return str(api_keys[provider]).strip()

    # Fallback to environment variables
    env_keys = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    env_var = env_keys.get(provider.lower())
    if env_var and os.getenv(env_var):
        return os.getenv(env_var, "").strip()

    return os.getenv("AADS_LLM_API_KEY", "").strip()


def set_stored_api_key(provider: str, key: str) -> None:
    """Store an API key for a provider.

    Args:
        provider: Provider name
        key: API key string
    """
    settings = load_user_settings()
    api_keys = settings.get("api_keys", {})
    if not isinstance(api_keys, dict):
        api_keys = {}
    api_keys[provider.lower()] = key.strip()
    settings["api_keys"] = api_keys
    save_user_settings(settings)
