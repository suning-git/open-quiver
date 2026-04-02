"""Shared provider registry and factory helpers."""

import os

from .llm_provider import OpenAICompatProvider

PROVIDERS = {
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "gpt-5.4-mini": {
        "model": "gpt-5.4-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-5.4": {
        "model": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-5.4-nano": {
        "model": "gpt-5.4-nano",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def list_provider_names() -> list[str]:
    """Return provider names in display order."""
    return list(PROVIDERS.keys())


def is_known_provider(name: str) -> bool:
    """Check whether a provider name exists in the registry."""
    return name in PROVIDERS


def get_provider_config(name: str) -> dict:
    """Return config for a provider name.

    Raises:
        ValueError: If provider is unknown.
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")
    return PROVIDERS[name]


def create_provider(name: str) -> OpenAICompatProvider:
    """Build an OpenAI-compatible provider from environment config.

    Raises:
        ValueError: If provider is unknown.
        RuntimeError: If API key is missing.
    """
    cfg = get_provider_config(name)
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"Set {cfg['api_key_env']} in .env first.")

    return OpenAICompatProvider(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=api_key,
    )

