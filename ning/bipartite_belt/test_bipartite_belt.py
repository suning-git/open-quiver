"""Tests for bipartite belt: square product generation and solution verification.

Verifies that the bipartite belt mutation sequence is a maximal green sequence
for square products of Dynkin diagrams, using QuiverEngine.
"""

import numpy as np
import pytest

from ning.agent.engine import QuiverEngine
from ning.bipartite_belt.square_product import (
    DynkinGraph,
    SquareProduct,
    dynkin_graph,
    square_product,
)
from ning.bipartite_belt.solver import bipartite_belt_solution


# ── Dynkin graph tests ──────────────────────────────────────────────


class TestDynkinGraph:
    def test_A1(self):
        g = dynkin_graph("A", 1)
        assert g.num_vertices == 1
        assert g.edges == []
        assert g.coxeter == 2

    def test_A3(self):
        g = dynkin_graph("A", 3)
        assert g.num_vertices == 3
        assert len(g.edges) == 2
        assert g.coxeter == 4
        # Adjacent vertices must have different colors
        for u, v in g.edges:
            assert g.color[u] != g.color[v]

    def test_D4(self):
        g = dynkin_graph("D", 4)
        assert g.num_vertices == 4
        assert len(g.edges) == 3  # chain 0-1-2 + branch 1-3
        assert g.coxeter == 6
        for u, v in g.edges:
            assert g.color[u] != g.color[v]

    def test_D5(self):
        g = dynkin_graph("D", 5)
        assert g.num_vertices == 5
        assert len(g.edges) == 4
        assert g.coxeter == 8

    def test_E6(self):
        g = dynkin_graph("E", 6)
        assert g.num_vertices == 6
        assert len(g.edges) == 5
        assert g.coxeter == 12
        for u, v in g.edges:
            assert g.color[u] != g.color[v]

    def test_E7(self):
        g = dynkin_graph("E", 7)
        assert g.num_vertices == 7
        assert len(g.edges) == 6
        assert g.coxeter == 18

    def test_E8(self):
        g = dynkin_graph("E", 8)
        assert g.num_vertices == 8
        assert len(g.edges) == 7
        assert g.coxeter == 30

    def test_directed_edges_source_to_sink(self):
        """Every directed edge goes from a white (source) to a black (sink)."""
        for typ, rank in [("A", 5), ("D", 4), ("D", 6), ("E", 6), ("E", 7), ("E", 8)]:
            g = dynkin_graph(typ, rank)
            for src, dst in g.directed_edges:
                assert g.color[src] == 0, f"{typ}{rank}: src {src} not white"
                assert g.color[dst] == 1, f"{typ}{rank}: dst {dst} not black"

    def test_invalid_types(self):
        with pytest.raises(ValueError):
            dynkin_graph("A", 0)
        with pytest.raises(ValueError):
            dynkin_graph("D", 3)
        with pytest.raises(ValueError):
            dynkin_graph("E", 5)
        with pytest.raises(ValueError):
            dynkin_graph("X", 3)


# ── Square product tests ────────────────────────────────────────────


class TestSquareProduct:
    def test_A2_square_A3_matrix(self):
        """Cross-validate against common/games/A2_square_A3 (test1_05).json."""
        G = dynkin_graph("A", 2)
        Gp = dynkin_graph("A", 3)
        sp = square_product(G, Gp)

        assert sp.n == 6
        assert sp.B_A.shape == (6, 6)

        # Expected matrix from A2_square_A3 (test1_05).json (row-major numbering)
        expected = np.array([
            [ 0,  1,  0, -1,  0,  0],
            [-1,  0, -1,  0,  1,  0],
            [ 0,  1,  0,  0,  0, -1],
            [ 1,  0,  0,  0, -1,  0],
            [ 0, -1,  0,  1,  0,  1],
            [ 0,  0,  1,  0, -1,  0],
        ], dtype=np.int64)
        np.testing.assert_array_equal(sp.B_A, expected)

    def test_bipartite_groups_no_internal_edges(self):
        """Within each bipartite group, all pairwise entries in B_A are 0."""
        G = dynkin_graph("A", 3)
        Gp = dynkin_graph("A", 4)
        sp = square_product(G, Gp)

        for u in sp.black_group:
            for v in sp.black_group:
                assert sp.B_A[u - 1, v - 1] == 0
        for u in sp.white_group:
            for v in sp.white_group:
                assert sp.B_A[u - 1, v - 1] == 0

    def test_antisymmetric(self):
        """B_A is antisymmetric for various products."""
        for (t1, r1), (t2, r2) in [
            (("A", 2), ("D", 4)),
            (("A", 3), ("E", 6)),
            (("D", 4), ("D", 4)),
        ]:
            G = dynkin_graph(t1, r1)
            Gp = dynkin_graph(t2, r2)
            sp = square_product(G, Gp)
            np.testing.assert_array_equal(sp.B_A, -sp.B_A.T)

    def test_vertex_count(self):
        G = dynkin_graph("A", 2)
        Gp = dynkin_graph("E", 7)
        sp = square_product(G, Gp)
        assert sp.n == 14
        assert len(sp.black_group) + len(sp.white_group) == 14


# ── Bipartite belt solution verification ─────────────────────────────


def _verify_solution(B_A: np.ndarray, solution: list[int]):
    """Run a mutation sequence on QuiverEngine and assert it wins."""
    engine = QuiverEngine()
    engine.reset_from_matrix(B_A)
    for k in solution:
        engine.mutate(k)
    assert engine.is_won(), (
        f"Solution did not reach all-red after {len(solution)} steps. "
        f"Move history: {solution}"
    )


# Pairs to test: (type1, rank1, type2, rank2)
_TEST_PAIRS = [
    ("A", 1, "A", 1),
    ("A", 1, "A", 2),
    ("A", 2, "A", 2),
    ("A", 2, "A", 3),
    ("A", 2, "A", 4),
    ("A", 3, "A", 3),
    ("A", 3, "A", 4),
    ("A", 2, "D", 4),
    ("A", 2, "E", 6),
    ("A", 2, "E", 7),
    ("A", 2, "E", 8),
    ("D", 4, "D", 4),
    ("A", 3, "E", 6),
]


class TestBipartiteBeltSolution:
    @pytest.mark.parametrize("t1,r1,t2,r2", _TEST_PAIRS,
                             ids=[f"{t1}{r1}_x_{t2}{r2}" for t1, r1, t2, r2 in _TEST_PAIRS])
    def test_start_white(self, t1, r1, t2, r2):
        """Bipartite belt starting with white group (h_{G'} rounds) wins."""
        G = dynkin_graph(t1, r1)
        Gp = dynkin_graph(t2, r2)
        sp = square_product(G, Gp)
        sol = bipartite_belt_solution(sp, start_white=True)
        _verify_solution(sp.B_A, sol)

    @pytest.mark.parametrize("t1,r1,t2,r2", _TEST_PAIRS,
                             ids=[f"{t1}{r1}_x_{t2}{r2}" for t1, r1, t2, r2 in _TEST_PAIRS])
    def test_start_black(self, t1, r1, t2, r2):
        """Bipartite belt starting with black group (h_G rounds) wins."""
        G = dynkin_graph(t1, r1)
        Gp = dynkin_graph(t2, r2)
        sp = square_product(G, Gp)
        sol = bipartite_belt_solution(sp, start_white=False)
        _verify_solution(sp.B_A, sol)

    def test_A2_A3_solution_lengths(self):
        """A2□A3: start_white → h_A3=4 rounds, start_black → h_A2=3 rounds."""
        G = dynkin_graph("A", 2)
        Gp = dynkin_graph("A", 3)
        sp = square_product(G, Gp)
        sol_w = bipartite_belt_solution(sp, start_white=True)
        sol_b = bipartite_belt_solution(sp, start_white=False)
        assert len(sol_w) == 4 * 3  # h_A3=4 rounds × 3 vertices/round = 12
        assert len(sol_b) == 3 * 3  # h_A2=3 rounds × 3 vertices/round = 9

    def test_A2_E7_solution_lengths(self):
        """A2□E7: start_white → h_E7=18 rounds, start_black → h_A2=3 rounds."""
        G = dynkin_graph("A", 2)
        Gp = dynkin_graph("E", 7)
        sp = square_product(G, Gp)
        sol_w = bipartite_belt_solution(sp, start_white=True)
        sol_b = bipartite_belt_solution(sp, start_white=False)
        assert len(sol_w) == 18 * 7  # h_E7=18 rounds × 7 vertices/round = 126
        assert len(sol_b) == 3 * 7   # h_A2=3 rounds × 7 vertices/round = 21
