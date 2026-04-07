"""Pure functions for quiver mutation on exchange matrices.

All functions are stateless: matrix in, matrix out.
See wenbin/graph_matrix_rule.md for the math.
"""

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.int64]


def make_exchange_matrix(n: int, edges: list[tuple[int, int]]) -> Matrix:
    """Build the n×n antisymmetric exchange matrix B_A from an edge list.

    Args:
        n: Number of mutable vertices (labeled 1..n).
        edges: List of (src, dst) pairs, 1-indexed. Duplicate entries
               represent multiple arrows in the same direction.

    Returns:
        n×n matrix B_A where b[i][j] = #(i→j) - #(j→i).
    """
    B = np.zeros((n, n), dtype=np.int64)
    for src, dst in edges:
        B[src - 1, dst - 1] += 1
        B[dst - 1, src - 1] -= 1
    return B


def make_framed(B_A: Matrix) -> Matrix:
    """Construct framed(A): [B_A | I_n]."""
    n = B_A.shape[0]
    return np.hstack([B_A, np.eye(n, dtype=np.int64)])


def make_coframed(B_A: Matrix) -> Matrix:
    """Construct coframed(A): [B_A | -I_n]."""
    n = B_A.shape[0]
    return np.hstack([B_A, -np.eye(n, dtype=np.int64)])


def mutate(B: Matrix, k: int) -> Matrix:
    """Apply mutation μ_k to exchange matrix B.

    Args:
        B: n×2n exchange matrix.
        k: Mutable vertex to mutate (1-indexed).

    Returns:
        New matrix B' after mutation.
    """
    B_new = B.copy()
    ki = k - 1  # 0-indexed row
    n = B.shape[0]

    # Step 1: path completion — update non-k entries
    for i in range(n):
        if i == ki:
            continue
        for j in range(B.shape[1]):
            if j == ki:
                continue
            b_ik = B[i, ki]
            b_kj = B[ki, j]
            B_new[i, j] = B[i, j] + max(b_ik, 0) * max(b_kj, 0) - max(-b_ik, 0) * max(-b_kj, 0)

    # Step 2: reverse edges at k — negate row k and column k
    B_new[ki, :] = -B[ki, :]
    B_new[:, ki] = -B[:, ki]

    return B_new


def get_colors(B: Matrix) -> dict[int, str]:
    """Determine green/red status of each mutable vertex.

    Looks at B_f (the frozen block, columns n..2n).
    - Green: row i of B_f is all >= 0
    - Red:   row i of B_f is all <= 0

    Args:
        B: n×2n exchange matrix.

    Returns:
        Dict mapping vertex (1-indexed) to "green" or "red".
    """
    n = B.shape[0]
    B_f = B[:, n:]
    colors = {}
    for i in range(n):
        row = B_f[i]
        if np.all(row >= 0):
            colors[i + 1] = "green"
        elif np.all(row <= 0):
            colors[i + 1] = "red"
        else:
            raise ValueError(
                f"Vertex {i + 1} is neither green nor red: B_f row = {row}. "
                "This violates the sign-coherence theorem."
            )
    return colors


def is_all_red(B: Matrix) -> bool:
    """Check if all mutable vertices are red (game won)."""
    n = B.shape[0]
    B_f = B[:, n:]
    return bool(np.all(B_f <= 0))


def matrix_to_edges(B: Matrix) -> list[tuple[int, int, int]]:
    """Extract edge list from exchange matrix.

    For the mutable block (columns 0..n-1), positive B[i,j] means i→j.
    For the frozen block (columns n..2n-1), negative B[i,j] means j→i
    (frozen vertex pointing to mutable vertex), which has no other row
    to record it.

    Returns:
        List of (src, dst, count) triples (1-indexed).
    """
    n = B.shape[0]
    edges = []
    for i in range(n):
        for j in range(B.shape[1]):
            if B[i, j] > 0:
                edges.append((i + 1, j + 1, int(B[i, j])))
            elif B[i, j] < 0 and j >= n:
                # Frozen→mutable edge: column j has no row, so extract from negative entry
                edges.append((j + 1, i + 1, int(-B[i, j])))
    return edges
