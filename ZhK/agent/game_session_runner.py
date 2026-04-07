"""Session-level game loop: drives run_turn until the game ends."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .engine import QuiverEngine
from .game_turn_runner import initialize_messages, run_turn
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
    """Run one complete game from an edge list."""
    engine = QuiverEngine()
    engine.reset(n, edges)
    return _run_game_loop(engine, provider, max_steps, max_retries)


def run_game_from_matrix(
    B_A: NDArray[np.int64],
    provider: LLMProvider,
    max_steps: int = 50,
    max_retries: int = 3,
) -> GameResult:
    """Run one complete game from an n×n exchange matrix."""
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
    messages = initialize_messages(engine)

    while not engine.is_won() and engine.total_steps < max_steps:
        turn = run_turn(engine, messages, provider, max_retries=max_retries)

        if turn.reason == "parse_failure":
            s = engine.get_state()
            return GameResult(
                won=False,
                steps=s["step"],
                move_history=s["move_history"],
                messages=messages,
                reason="parse_failure",
            )
        if turn.reason == "won":
            break

    state = engine.get_state()
    reason = "won" if engine.is_won() else "max_steps"

    return GameResult(
        won=engine.is_won(),
        steps=state["step"],
        move_history=state["move_history"],
        messages=messages,
        reason=reason,
    )
