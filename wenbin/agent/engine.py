"""Stateful game engine for the green-red mutation game.

Wraps the pure functions in mutation.py with state management,
history tracking, and cycle detection.
"""

import numpy as np

from .mutation import (
    Matrix,
    make_exchange_matrix,
    make_framed,
    mutate,
    get_colors,
    is_all_red,
    matrix_to_edges,
)


class QuiverEngine:
    """Manages the state of a green-red mutation game."""

    def __init__(self):
        self._B: Matrix | None = None
        self._B_A: Matrix | None = None
        self._n: int = 0
        self._history: list[Matrix] = []
        self._visited: set[bytes] = set()
        self._move_history: list[int] = []

    @property
    def n(self) -> int:
        """Number of mutable vertices."""
        return self._n

    @property
    def total_steps(self) -> int:
        """Number of mutations performed so far."""
        return len(self._move_history)

    def reset(self, n: int, edges: list[tuple[int, int]]) -> dict:
        """Initialize a new game from an edge list.

        Args:
            n: Number of mutable vertices.
            edges: Edge list for A, as (src, dst) pairs, 1-indexed.

        Returns:
            Initial state dict.
        """
        return self.reset_from_matrix(make_exchange_matrix(n, edges))

    def reset_from_matrix(self, B_A: Matrix) -> dict:
        """Initialize a new game from an exchange matrix B_A.

        Args:
            B_A: n×n antisymmetric exchange matrix.

        Returns:
            Initial state dict.
        """
        self._n = B_A.shape[0]
        self._B_A = B_A.copy()
        self._B = make_framed(self._B_A)
        self._history = [self._B.copy()]
        self._visited = {self._B.tobytes()}
        self._move_history = []
        return self.get_state()

    def mutate(self, k: int) -> dict:
        """Execute mutation μ_k and return the result with diff info.

        Args:
            k: Mutable vertex to mutate (1-indexed, must be in 1..n).

        Returns:
            Dict with new state and diff information.
        """
        if self._B is None:
            raise RuntimeError("Engine not initialized. Call reset() first.")
        if not (1 <= k <= self._n):
            raise ValueError(f"Vertex {k} is not a mutable vertex (must be 1..{self._n}).")

        old_B = self._B
        old_colors = get_colors(old_B)

        self._B = mutate(old_B, k)
        self._move_history.append(k)

        # Cycle detection
        key = self._B.tobytes()
        cycle_step = None
        if key in self._visited:
            for i, hist_B in enumerate(self._history):
                if hist_B.tobytes() == key:
                    cycle_step = i
                    break

        self._visited.add(key)
        self._history.append(self._B.copy())

        new_colors = get_colors(self._B)

        # Compute diff
        color_changes = {}
        for v in range(1, self._n + 1):
            if old_colors[v] != new_colors[v]:
                color_changes[v] = (old_colors[v], new_colors[v])

        old_red = sum(1 for c in old_colors.values() if c == "red")
        new_red = sum(1 for c in new_colors.values() if c == "red")

        state = self.get_state()
        state["diff"] = {
            "mutated_vertex": k,
            "color_changes": color_changes,
            "red_count_before": old_red,
            "red_count_after": new_red,
        }
        if cycle_step is not None:
            state["diff"]["cycle_warning"] = cycle_step

        return state

    def get_state(self) -> dict:
        """Return the current game state."""
        if self._B is None:
            raise RuntimeError("Engine not initialized. Call reset() first.")

        colors = get_colors(self._B)
        red_count = sum(1 for c in colors.values() if c == "red")

        return {
            "matrix": self._B.copy(),
            "edges": matrix_to_edges(self._B),
            "colors": colors,
            "red_count": red_count,
            "total_mutable": self._n,
            "step": len(self._move_history),
            "move_history": list(self._move_history),
        }

    def get_state_at(self, step: int) -> dict:
        """Return the game state at a given step in history.

        Args:
            step: Step index (0 = initial state, len(move_history) = current).

        Returns:
            State dict with matrix, edges, colors, and the move that led here.
        """
        if self._B is None:
            raise RuntimeError("Engine not initialized. Call reset() first.")
        if not (0 <= step <= len(self._move_history)):
            raise ValueError(
                f"Step {step} out of range (0..{len(self._move_history)})."
            )

        B = self._history[step]
        colors = get_colors(B)
        red_count = sum(1 for c in colors.values() if c == "red")

        state = {
            "matrix": B.copy(),
            "edges": matrix_to_edges(B),
            "colors": colors,
            "red_count": red_count,
            "total_mutable": self._n,
            "step": step,
            "move_history": list(self._move_history[:step]),
        }
        if step > 0:
            state["last_move"] = self._move_history[step - 1]
        return state

    def is_won(self) -> bool:
        """Check if the game is won (all vertices red)."""
        if self._B is None:
            return False
        return is_all_red(self._B)

    def get_history(self) -> list[Matrix]:
        """Return list of all visited matrices (copies)."""
        return [B.copy() for B in self._history]
