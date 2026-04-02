"""Shared game-loop helpers used by different frontends."""

from dataclasses import dataclass

from .engine import QuiverEngine
from .harness import (
    SYSTEM_PROMPT,
    build_user_message,
    format_error,
    parse_action,
    render_diff,
)
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
        {"role": "system", "content": SYSTEM_PROMPT},
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
    n = engine.n

    response = provider.chat(messages)
    messages.append({"role": "assistant", "content": response})

    k = parse_action(response, n)
    retries = 0
    while k is None and retries < max_retries:
        error_msg = format_error(response, n)
        messages.append({"role": "user", "content": error_msg})
        response = provider.chat(messages)
        messages.append({"role": "assistant", "content": response})
        k = parse_action(response, n)
        retries += 1

    if k is None:
        return TurnResult(game_over=True, reason="parse_failure")

    state = engine.mutate(k)
    diff_text = render_diff(state)

    if engine.is_won():
        messages.append({"role": "user", "content": diff_text})
        return TurnResult(game_over=True, reason="won", diff_text=diff_text)

    user_msg = build_user_message(state, diff_text=diff_text)
    messages.append({"role": "user", "content": user_msg})
    return TurnResult(game_over=False, reason="", diff_text=diff_text)

