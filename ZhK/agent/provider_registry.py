"""Shared provider registry and factory helpers."""

import os

from .llm_provider import OpenAICompatProvider

PROVIDERS = {
    "deepseek-chat": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "deepseek-reasoner": {
        "model": "deepseek-reasoner",
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
    "qwen3-max": {
        "model": "qwen3-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "qwen-flash": {
        "model": "qwen-flash",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "qwen-plus-latest": {
        "model": "qwen-plus-latest",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "qwen3-0.6b": {
        "model": "qwen3-0.6b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "qwen3-1.7b": {
        "model": "qwen3-1.7b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "qwen3-4b": {
        "model": "qwen3-4b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "qwen3-8b": {
        "model": "qwen3-8b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "qwen3-14b": {
        "model": "qwen3-14b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "qwen3-32b": {
        "model": "qwen3-32b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
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

    extra_kwargs = {}
    if "extra_body" in cfg:
        extra_kwargs["extra_body"] = cfg["extra_body"]

    return OpenAICompatProvider(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=api_key,
        **extra_kwargs,
    )
