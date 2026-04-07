"""Unit tests for engine.py and catalog.py.

Pure mutation tests live in common/quiver/tests/test_mutation.py.
"""

import numpy as np
import pytest

from common.quiver.mutation import make_exchange_matrix
from ning.agent.engine import QuiverEngine
from ning.agent import catalog


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

    def test_reset_from_matrix(self):
        """reset_from_matrix produces same result as reset with edges."""
        engine1 = QuiverEngine()
        engine1.reset(3, [(1, 2), (2, 3)])

        engine2 = QuiverEngine()
        B_A = make_exchange_matrix(3, [(1, 2), (2, 3)])
        engine2.reset_from_matrix(B_A)

        np.testing.assert_array_equal(
            engine1.get_state()["matrix"],
            engine2.get_state()["matrix"],
        )

    def test_get_state_at(self):
        """Browse history: get_state_at returns correct state for each step."""
        engine = QuiverEngine()
        engine.reset(3, [(1, 2), (2, 3)])

        # Capture states as we go
        state0 = engine.get_state()
        engine.mutate(2)
        state1 = engine.get_state()
        engine.mutate(1)
        state2 = engine.get_state()

        # Browse back
        np.testing.assert_array_equal(engine.get_state_at(0)["matrix"], state0["matrix"])
        np.testing.assert_array_equal(engine.get_state_at(1)["matrix"], state1["matrix"])
        np.testing.assert_array_equal(engine.get_state_at(2)["matrix"], state2["matrix"])

        # Step metadata
        assert engine.get_state_at(0)["step"] == 0
        assert engine.get_state_at(0)["move_history"] == []
        assert engine.get_state_at(1)["step"] == 1
        assert engine.get_state_at(1)["move_history"] == [2]
        assert engine.get_state_at(1)["last_move"] == 2
        assert engine.get_state_at(2)["last_move"] == 1

        # Out of range
        with pytest.raises(ValueError):
            engine.get_state_at(3)
        with pytest.raises(ValueError):
            engine.get_state_at(-1)


# ── catalog.py tests ──────────────────────────────────────────────


class TestCatalog:
    def test_list_graphs(self):
        graphs = catalog.list_graphs()
        assert len(graphs) >= 11  # 4 presets + 7 test1
        names = [g["name"] for g in graphs]
        assert "linear_1t2" in names
        assert "triangle_inside_triangle_equiv_to_D6 (test1_03)" in names

    def test_list_sorted_by_n(self):
        graphs = catalog.list_graphs()
        ns = [g["n"] for g in graphs]
        assert ns == sorted(ns)

    def test_get_graph(self):
        g = catalog.get_graph("linear_1t2t3")
        assert g["n"] == 3
        assert g["B_A"].shape == (3, 3)
        # Antisymmetric
        np.testing.assert_array_equal(g["B_A"], -g["B_A"].T)

    def test_get_graph_test1(self):
        g = catalog.get_graph("4_loop (test1_07)")
        assert g["n"] == 4
        assert g["B_A"].shape == (4, 4)

    def test_get_solution(self):
        sol = catalog.get_solution("triangle_inside_triangle_equiv_to_D6 (test1_03)")
        assert sol is not None
        assert isinstance(sol, list)
        assert len(sol) == 10

    def test_no_solution_for_preset(self):
        sol = catalog.get_solution("linear_1t2")
        assert sol is None

    def test_get_graph_not_found(self):
        with pytest.raises(FileNotFoundError):
            catalog.get_graph("nonexistent")

    def test_solutions_reach_all_red(self):
        """Verify all games with solutions actually reach all-red."""
        for info in catalog.list_graphs():
            sol = catalog.get_solution(info["name"])
            if sol is None:
                continue
            g = catalog.get_graph(info["name"])
            engine = QuiverEngine()
            engine.reset_from_matrix(g["B_A"])
            for k in sol:
                engine.mutate(k)
            assert engine.is_won(), f"{info['name']} solution does not reach all-red"
