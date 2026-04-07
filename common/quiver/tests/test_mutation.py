"""Unit tests for common.quiver.mutation.

Test cases drawn from common/quiver/MATH.md.
"""

import numpy as np

from common.quiver.mutation import (
    make_exchange_matrix,
    make_framed,
    make_coframed,
    mutate,
    get_colors,
    is_all_red,
    matrix_to_edges,
)


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
        """Verify μ_2 on 1→2→3 matches MATH.md Section 7."""
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

    def test_coframed_edges(self):
        """Coframed has frozen→mutable edges (negative B_f entries)."""
        B = make_coframed(make_exchange_matrix(3, [(1, 2), (2, 3)]))
        edges = matrix_to_edges(B)
        # Frozen→mutable: 4→1, 5→2, 6→3
        assert (4, 1, 1) in edges
        assert (5, 2, 1) in edges
        assert (6, 3, 1) in edges
        # Mutable: 1→2, 2→3
        assert (1, 2, 1) in edges
        assert (2, 3, 1) in edges
        assert len(edges) == 5
