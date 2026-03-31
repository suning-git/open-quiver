"""Main game loop: connects engine, harness, and LLM provider."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

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
class GameResult:
    """Result of a completed game run."""

    won: bool
    steps: int
    move_history: list[int]
    messages: list[dict] = field(repr=False)
    reason: str = ""  # "won", "max_steps", "parse_failure"


def run_game(
    n: int,
    edges: list[tuple[int, int]],
    provider: LLMProvider,
    max_steps: int = 50,
    max_retries: int = 3,
) -> GameResult:
    """Run one complete game from an edge list.

    Args:
        n: Number of mutable vertices.
        edges: Edge list for graph A.
        provider: LLM provider to use.
        max_steps: Maximum mutation steps before giving up.
        max_retries: Max consecutive parse failures per turn.

    Returns:
        GameResult with outcome, history, and full message log.
    """
    engine = QuiverEngine()
    engine.reset(n, edges)
    return _run_game_loop(engine, provider, max_steps, max_retries)


def run_game_from_matrix(
    B_A: NDArray[np.int64],
    provider: LLMProvider,
    max_steps: int = 50,
    max_retries: int = 3,
) -> GameResult:
    """Run one complete game from an exchange matrix.

    Args:
        B_A: n×n antisymmetric exchange matrix.
        provider: LLM provider to use.
        max_steps: Maximum mutation steps before giving up.
        max_retries: Max consecutive parse failures per turn.

    Returns:
        GameResult with outcome, history, and full message log.
    """
    engine = QuiverEngine()
    engine.reset_from_matrix(B_A)
    return _run_game_loop(engine, provider, max_steps, max_retries)


def _run_game_loop(
    engine: QuiverEngine,
    provider: LLMProvider,
    max_steps: int,
    max_retries: int,
) -> GameResult:
    """Internal: run the LLM game loop on an already-initialized engine."""
    state = engine.get_state()
    n = engine.n

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_msg = build_user_message(state)
    messages.append({"role": "user", "content": user_msg})

    while not engine.is_won() and state["step"] < max_steps:
        # Get LLM response
        response = provider.chat(messages)
        messages.append({"role": "assistant", "content": response})

        # Parse action with retries
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
            s = engine.get_state()
            return GameResult(
                won=False,
                steps=s["step"],
                move_history=s["move_history"],
                messages=messages,
                reason="parse_failure",
            )

        # Execute mutation
        state = engine.mutate(k)
        diff_text = render_diff(state)

        if engine.is_won():
            # Record final feedback for completeness
            messages.append({"role": "user", "content": diff_text})
            break

        # Build next turn message
        user_msg = build_user_message(state, diff_text=diff_text)
        messages.append({"role": "user", "content": user_msg})

    state = engine.get_state()

    if engine.is_won():
        reason = "won"
    else:
        reason = "max_steps"

    return GameResult(
        won=engine.is_won(),
        steps=state["step"],
        move_history=state["move_history"],
        messages=messages,
        reason=reason,
    )
