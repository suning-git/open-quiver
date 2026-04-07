"""Single-turn primitive: one LLM-driven mutation step.

Used by both the session runner (game_session_runner.py) and the
Streamlit frontend so that step semantics live in exactly one place.
"""

from dataclasses import dataclass

from .engine import QuiverEngine
from .harness import (
    build_user_message,
    format_error,
    parse_action,
    render_diff,
)
from .initial_prompts import get_system_prompt
from .llm_provider import LLMProvider


@dataclass
class TurnResult:
    """Result of executing exactly one LLM-driven mutation turn."""

    game_over: bool
    reason: str = ""  # "won", "parse_failure", ""
    diff_text: str = ""


def initialize_messages(engine: QuiverEngine) -> list[dict]:
    """Build initial conversation messages from current engine state."""
    state = engine.get_state()
    return [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": build_user_message(state)},
    ]


def run_turn(
    engine: QuiverEngine,
    messages: list[dict],
    provider: LLMProvider,
    max_retries: int = 3,
) -> TurnResult:
    """Execute one turn: LLM response -> parse -> mutate -> feedback.

    Mutates `messages` in place by appending assistant/user messages.
    """
    # Guard: do not ask the model for another move after game is already won.
    if engine.is_won():
        return TurnResult(game_over=True, reason="won")

    n = engine.n

    response = provider.chat(messages)
    messages.append({"role": "assistant", "content": response})

    k = parse_action(response, n)
    retries = 0
    while k is None:
        if retries >= max_retries:
            return TurnResult(game_over=True, reason="parse_failure")
        error_msg = format_error(response, n)
        messages.append({"role": "user", "content": error_msg})
        response = provider.chat(messages)
        messages.append({"role": "assistant", "content": response})
        k = parse_action(response, n)
        retries += 1

    state = engine.mutate(k)
    diff_text = render_diff(state)

    if engine.is_won():
        messages.append({"role": "user", "content": diff_text})
        return TurnResult(game_over=True, reason="won", diff_text=diff_text)

    user_msg = build_user_message(state, diff_text=diff_text)
    messages.append({"role": "user", "content": user_msg})
    return TurnResult(game_over=False, reason="", diff_text=diff_text)
