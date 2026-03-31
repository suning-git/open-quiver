"""Unit tests for mutation.py and engine.py.

Test cases are drawn from ning/graph_rule.md and ning/graph_matrix_rule.md.
"""

import numpy as np
import pytest

from .mutation import (
    make_exchange_matrix,
    make_framed,
    make_coframed,
    mutate,
    get_colors,
    is_all_red,
    matrix_to_edges,
)
from .engine import QuiverEngine


# ── mutation.py tests ──────────────────────────────────────────────


class TestMakeExchangeMatrix:
    def test_linear_3(self):
        """Graph: 1→2→3"""
        B = make_exchange_matrix(3, [(1, 2), (2, 3)])
        expected = np.array([
            [0,  1,  0],
            [-1, 0,  1],
            [0, -1,  0],
        ])
        np.testing.assert_array_equal(B, expected)

    def test_antisymmetric(self):
        B = make_exchange_matrix(3, [(1, 2), (2, 3)])
        np.testing.assert_array_equal(B, -B.T)

    def test_multiple_edges(self):
        """Two edges 1→2"""
        B = make_exchange_matrix(2, [(1, 2), (1, 2)])
        assert B[0, 1] == 2
        assert B[1, 0] == -2


class TestFraming:
    def test_framed(self):
        B_A = make_exchange_matrix(3, [(1, 2), (2, 3)])
        B = make_framed(B_A)
        assert B.shape == (3, 6)
        np.testing.assert_array_equal(B[:, 3:], np.eye(3))

    def test_coframed(self):
        B_A = make_exchange_matrix(3, [(1, 2), (2, 3)])
        B = make_coframed(B_A)
        assert B.shape == (3, 6)
        np.testing.assert_array_equal(B[:, 3:], -np.eye(3))


class TestMutate:
    def test_doc_example(self):
        """Verify μ_2 on 1→2→3 matches graph_matrix_rule.md Section 7."""
        B_A = make_exchange_matrix(3, [(1, 2), (2, 3)])
        B = make_framed(B_A)
        B_prime = mutate(B, 2)

        expected = np.array([
            [0, -1,  1,  1,  1,  0],
            [1,  0, -1,  0, -1,  0],
            [-1, 1,  0,  0,  0,  1],
        ])
        np.testing.assert_array_equal(B_prime, expected)

    def test_involution(self):
        """μ_k ∘ μ_k = id"""
        B_A = make_exchange_matrix(3, [(1, 2), (2, 3)])
        B = make_framed(B_A)
        for k in [1, 2, 3]:
            B_double = mutate(mutate(B, k), k)
            np.testing.assert_array_equal(B_double, B, err_msg=f"Involution failed for k={k}")

    def test_involution_with_multiple_edges(self):
        """Involution holds for multigraphs too."""
        B_A = make_exchange_matrix(3, [(1, 2), (1, 2), (2, 3)])
        B = make_framed(B_A)
        for k in [1, 2, 3]:
            B_double = mutate(mutate(B, k), k)
            np.testing.assert_array_equal(B_double, B, err_msg=f"Involution failed for k={k}")

    def test_long_sequence_n10(self):
        """Regression: 13-step mutation on a 10-vertex quiver (ZhK/TestSamples/Test1_02)."""
        B = np.array([
            [0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, -1, -1, 1, 0, 0, 0, 0, 0],
            [-1, 1, 0, 0, -1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, -1, 0, -1, 1, 0, 0],
            [0, -1, 1, 1, 0, -1, 0, -1, 1, 0],
            [0, 0, -1, 0, 1, 0, 0, 0, -1, 1],
            [0, 0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, -1, 1, 0, 1, 0, -1, 0],
            [0, 0, 0, 0, -1, 1, 0, 1, 0, -1],
            [0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
        ], dtype=np.int64)
        for k in [1, 5, 7, 10, 9, 6, 8, 4, 3, 2, 9, 8, 4]:
            B = mutate(B, k)
        expected = np.array([
            [0, 0, 1, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, -1, 1, -1],
            [-1, -1, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
            [0, 0, 0, -1, 0, 0, 0, 0, 1, 0],
            [1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, -1, 1, 0, 0, 0, -1, 0],
            [0, -1, 0, 1, -1, -1, 0, 1, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int64)
        np.testing.assert_array_equal(B, expected)


class TestColors:
    def test_framed_all_green(self):
        B_A = make_exchange_matrix(3, [(1, 2), (2, 3)])
        B = make_framed(B_A)
        colors = get_colors(B)
        assert all(c == "green" for c in colors.values())

    def test_coframed_all_red(self):
        B_A = make_exchange_matrix(3, [(1, 2), (2, 3)])
        B = make_coframed(B_A)
        colors = get_colors(B)
        assert all(c == "red" for c in colors.values())

    def test_after_mutation(self):
        """After μ_2 on 1→2→3: vertex 2 is red, 1 and 3 are green."""
        B_A = make_exchange_matrix(3, [(1, 2), (2, 3)])
        B = make_framed(B_A)
        B_prime = mutate(B, 2)
        colors = get_colors(B_prime)
        assert colors == {1: "green", 2: "red", 3: "green"}


class TestIsAllRed:
    def test_framed_not_red(self):
        B = make_framed(make_exchange_matrix(3, [(1, 2), (2, 3)]))
        assert not is_all_red(B)

    def test_coframed_is_red(self):
        B = make_coframed(make_exchange_matrix(3, [(1, 2), (2, 3)]))
        assert is_all_red(B)


class TestEdgeExtraction:
    def test_framed_edges(self):
        B = make_framed(make_exchange_matrix(3, [(1, 2), (2, 3)]))
        edges = matrix_to_edges(B)
        # Edges: 1→2, 2→3, 1→1', 2→2', 3→3' (1-indexed, frozen are 4,5,6)
        assert (1, 2, 1) in edges
        assert (2, 3, 1) in edges
        assert (1, 4, 1) in edges
        assert (2, 5, 1) in edges
        assert (3, 6, 1) in edges
        assert len(edges) == 5


# ── engine.py tests ────────────────────────────────────────────────


class TestEngine:
    def test_reset(self):
        engine = QuiverEngine()
        state = engine.reset(3, [(1, 2), (2, 3)])
        assert state["total_mutable"] == 3
        assert state["red_count"] == 0
        assert state["step"] == 0
        assert all(c == "green" for c in state["colors"].values())

    def test_mutate_returns_diff(self):
        engine = QuiverEngine()
        engine.reset(3, [(1, 2), (2, 3)])
        result = engine.mutate(2)
        assert result["diff"]["mutated_vertex"] == 2
        assert result["diff"]["red_count_before"] == 0
        assert result["diff"]["red_count_after"] == 1
        assert 2 in result["diff"]["color_changes"]
        assert result["diff"]["color_changes"][2] == ("green", "red")

    def test_invalid_vertex(self):
        engine = QuiverEngine()
        engine.reset(3, [(1, 2), (2, 3)])
        with pytest.raises(ValueError):
            engine.mutate(0)
        with pytest.raises(ValueError):
            engine.mutate(4)

    def test_uninitialized(self):
        engine = QuiverEngine()
        with pytest.raises(RuntimeError):
            engine.mutate(1)
        with pytest.raises(RuntimeError):
            engine.get_state()

    def test_cycle_detection(self):
        """μ_k ∘ μ_k returns to start — should trigger cycle warning."""
        engine = QuiverEngine()
        engine.reset(3, [(1, 2), (2, 3)])
        engine.mutate(2)
        result = engine.mutate(2)  # back to initial state
        assert "cycle_warning" in result["diff"]
        assert result["diff"]["cycle_warning"] == 0  # returned to step 0

    def test_full_game_n2(self):
        """Complete game on the simplest graph: 1→2 (n=2).

        framed: 1→2, 1→1', 2→2'
        μ_1: reverse 1's edges → 2→1, 1'→1, 2→1'(new from path 2→1→1'), 2→2'
             vertex 1: 1'→1 exists → red; vertex 2: 2→2' exists → green
        μ_2: reverse 2's edges → 1→2, 1'→1, 1'→2(new from path 1'→1→2... wait
             let's just run it and check.
        """
        engine = QuiverEngine()
        engine.reset(2, [(1, 2)])
        assert not engine.is_won()

        engine.mutate(1)
        engine.mutate(2)
        assert engine.is_won()

    def test_move_history(self):
        engine = QuiverEngine()
        engine.reset(3, [(1, 2), (2, 3)])
        engine.mutate(2)
        engine.mutate(1)
        state = engine.get_state()
        assert state["move_history"] == [2, 1]
        assert state["step"] == 2
