"""Shared game-loop helpers used by different frontends."""

from dataclasses import dataclass

from .engine import QuiverEngine
from .harness import (
    build_user_message,
    format_error,
    parse_action,
    parse_action_or_undo,
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


# Keep undo implementation available, but disable it for now.
ALLOW_UNDO = False


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

    command = parse_action_or_undo(response, n) if ALLOW_UNDO else parse_action(response, n)
    retries = 0
    state = None

    while state is None:
        if isinstance(command, int):
            state = engine.mutate(command)
            break

        if ALLOW_UNDO and command == "undo":
            try:
                state = engine.undo()
                break
            except ValueError:
                error_msg = (
                    "Cannot undo: no previous move exists. "
                    f"Reply with an integer between 1 and {n}, or 'undo'."
                )
        else:
            error_msg = format_error(response, n, allow_undo=ALLOW_UNDO)

        if retries >= max_retries:
            return TurnResult(game_over=True, reason="parse_failure")

        messages.append({"role": "user", "content": error_msg})
        response = provider.chat(messages)
        messages.append({"role": "assistant", "content": response})
        command = parse_action_or_undo(response, n) if ALLOW_UNDO else parse_action(response, n)
        retries += 1

    diff_text = render_diff(state)

    if engine.is_won():
        messages.append({"role": "user", "content": diff_text})
        return TurnResult(game_over=True, reason="won", diff_text=diff_text)

    user_msg = build_user_message(state, diff_text=diff_text)
    messages.append({"role": "user", "content": user_msg})
    return TurnResult(game_over=False, reason="", diff_text=diff_text)
