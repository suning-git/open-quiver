"""Integration tests that hit real LLM APIs.

Skipped by default. Run explicitly with:
    pytest -m integration ning/agent/tests/integration/

Each test loads .env for API keys and is skipped if the key is missing.
"""

import os

import pytest
from dotenv import load_dotenv

from ning.agent import catalog
from ning.agent.game_session_runner import run_game_from_matrix
from ning.agent.provider_registry import create_provider, get_provider_config

load_dotenv()


def _has_key(provider_name: str) -> bool:
    cfg = get_provider_config(provider_name)
    return bool(os.getenv(cfg["api_key_env"], ""))


def _run(provider_name: str, graph_name: str, max_steps: int = 20):
    """Helper: load graph, create provider, run game."""
    g = catalog.get_graph(graph_name)
    provider = create_provider(provider_name)
    return run_game_from_matrix(g["B_A"], provider, max_steps=max_steps)


# ── Smoke tests: simplest game on each provider ────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _has_key("gpt-5.4-mini"), reason="OPENAI_API_KEY not set")
def test_gpt_5_4_mini_solves_linear_2():
    result = _run("gpt-5.4-mini", "linear_2")
    assert result.won, f"Failed: {result.reason}, moves={result.move_history}"
