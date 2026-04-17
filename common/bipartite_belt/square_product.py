"""Square product of Dynkin diagrams — game generation.

Constructs the exchange matrix B_A for the square product G □ G' of two
Dynkin diagrams, along with the bipartite group structure.

The output B_A can be fed to QuiverEngine (or any other consumer) to play
the green-red mutation game.  This module knows nothing about solvers.

References:
- [1] Keller, "The periodicity conjecture for pairs of Dynkin diagrams",
      Annals of Math 177(1), 2013. arXiv:1001.1531 — §8: square product.
- [2] Keller, "Quiver mutation and combinatorial DT-invariants", 2017.
      arXiv:1709.03143 — §5.2: square product definition.
"""

import collections
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.int64]


def _coxeter_number(dynkin_type: str, rank: int) -> int:
    """Coxeter number h for a Dynkin diagram.

    Ref: Humphreys, "Reflection Groups and Coxeter Groups", §3.18.
    """
    if dynkin_type == "A":
        return rank + 1
    if dynkin_type == "D":
        return 2 * rank - 2
    if dynkin_type == "E":
        return {6: 12, 7: 18, 8: 30}[rank]
    raise ValueError(f"Unknown Dynkin type: {dynkin_type}")


@dataclass
class DynkinGraph:
    """A Dynkin diagram with alternating orientation (bipartite 2-coloring).

    Attributes:
        dynkin_type: "A", "D", or "E".
        rank: n for A_n / D_n, or 6/7/8 for E.
        num_vertices: Same as rank.
        color: color[i] ∈ {0, 1}. 0 = white (source), 1 = black (sink).
        edges: Undirected edges as (u, v) pairs, 0-indexed.
        directed_edges: Directed edges (source → sink), 0-indexed.
        coxeter: Coxeter number h.
    """
    dynkin_type: str
    rank: int
    num_vertices: int
    color: list[int]
    edges: list[tuple[int, int]]
    directed_edges: list[tuple[int, int]] = field(default_factory=list)
    coxeter: int = 0


def dynkin_graph(dynkin_type: str, rank: int) -> DynkinGraph:
    """Generate a Dynkin diagram with alternating orientation.

    Args:
        dynkin_type: "A", "D", or "E".
        rank: n for A_n (n >= 1), n for D_n (n >= 4), or 6/7/8 for E.

    Returns:
        DynkinGraph with bipartite coloring and directed edges.
    """
    if dynkin_type == "A":
        if rank < 1:
            raise ValueError(f"A_n requires n >= 1, got {rank}")
        edges = [(i, i + 1) for i in range(rank - 1)]
    elif dynkin_type == "D":
        if rank < 4:
            raise ValueError(f"D_n requires n >= 4, got {rank}")
        # Chain: 0-1-2-..-(n-2), branch: (n-3)-(n-1)
        edges = [(i, i + 1) for i in range(rank - 2)]
        edges.append((rank - 3, rank - 1))
    elif dynkin_type == "E":
        if rank not in (6, 7, 8):
            raise ValueError(f"E_n requires n in {{6,7,8}}, got {rank}")
        # Main chain: 0-1-2-3-4-..-(rank-2), branch: 2-(rank-1)
        edges = [(i, i + 1) for i in range(rank - 2)]
        edges.append((2, rank - 1))
    else:
        raise ValueError(f"Unknown Dynkin type: {dynkin_type}")

    # BFS 2-coloring from vertex 0
    color = [-1] * rank
    color[0] = 0  # white
    queue = collections.deque([0])
    while queue:
        u = queue.popleft()
        for a, b in edges:
            v = b if a == u else (a if b == u else None)
            if v is not None and color[v] == -1:
                color[v] = 1 - color[u]
                queue.append(v)

    assert all(c >= 0 for c in color), "BFS coloring incomplete"
    for u, v in edges:
        assert color[u] != color[v], f"Bad coloring: edge ({u},{v})"

    # Directed edges: source (white, 0) → sink (black, 1)
    directed_edges = []
    for u, v in edges:
        if color[u] == 0:
            directed_edges.append((u, v))
        else:
            directed_edges.append((v, u))

    return DynkinGraph(
        dynkin_type=dynkin_type,
        rank=rank,
        num_vertices=rank,
        color=color,
        edges=edges,
        directed_edges=directed_edges,
        coxeter=_coxeter_number(dynkin_type, rank),
    )


@dataclass
class SquareProduct:
    """Square product G □ G' with bipartite structure.

    Attributes:
        B_A: n×n antisymmetric exchange matrix (the game).
        n: Total number of vertices = |G| × |G'|.
        black_group: Black group vertex indices (1-indexed).
                     Same-color pairs: (W,W) ∪ (B,B).
        white_group: White group vertex indices (1-indexed).
                     Cross-color pairs: (W,B) ∪ (B,W).
        vertex_map: vertex_map[k] = (i, j), 0-indexed grid coordinates.
        G: First Dynkin graph.
        G_prime: Second Dynkin graph.
    """
    B_A: Matrix
    n: int
    black_group: list[int]
    white_group: list[int]
    vertex_map: list[tuple[int, int]]
    G: DynkinGraph
    G_prime: DynkinGraph


def square_product(G: DynkinGraph, G_prime: DynkinGraph) -> SquareProduct:
    """Construct the square product G □ G'.

    Vertices are numbered in row-major order:
        (i, j) → i * |G'| + j  (0-indexed internally, 1-indexed externally).

    Arrow directions follow the cyclic rule (verified against A2□A3 data):
        (W,W) →[horizontal] (W,B) →[vertical] (B,B) →[horizontal] (B,W) →[vertical] (W,W)

    Concretely:
    - Horizontal edges (from G' edge j1-j2, at each i ∈ G):
        i is source (W) → keep G' direction
        i is sink   (B) → reverse G' direction
    - Vertical edges (from G edge i1-i2, at each j ∈ G'):
        j is source (W) → reverse G direction
        j is sink   (B) → keep G direction

    Ref: [1, §8], [2, §5.2].
    """
    m1 = G.num_vertices
    m2 = G_prime.num_vertices
    n = m1 * m2

    def idx(i: int, j: int) -> int:
        return i * m2 + j

    B_A = np.zeros((n, n), dtype=np.int64)

    # Horizontal edges: from G' edges, replicated at each row i of G
    for j1, j2 in G_prime.directed_edges:
        for i in range(m1):
            src = idx(i, j1)
            dst = idx(i, j2)
            if G.color[i] == 0:  # i is W: keep G' direction
                B_A[src, dst] += 1
                B_A[dst, src] -= 1
            else:  # i is B: reverse G' direction
                B_A[dst, src] += 1
                B_A[src, dst] -= 1

    # Vertical edges: from G edges, replicated at each column j of G'
    for i1, i2 in G.directed_edges:
        for j in range(m2):
            src = idx(i1, j)
            dst = idx(i2, j)
            if G_prime.color[j] == 0:  # j is W: reverse G direction
                B_A[dst, src] += 1
                B_A[src, dst] -= 1
            else:  # j is B: keep G direction
                B_A[src, dst] += 1
                B_A[dst, src] -= 1

    # Bipartite groups
    black_group = []  # (W,W) ∪ (B,B): same color
    white_group = []  # (W,B) ∪ (B,W): different color
    vertex_map = []
    for i in range(m1):
        for j in range(m2):
            vertex_map.append((i, j))
            v = idx(i, j) + 1  # 1-indexed
            if G.color[i] == G_prime.color[j]:
                black_group.append(v)
            else:
                white_group.append(v)

    assert B_A.shape == (n, n)
    assert np.array_equal(B_A, -B_A.T), "B_A must be antisymmetric"

    return SquareProduct(
        B_A=B_A,
        n=n,
        black_group=black_group,
        white_group=white_group,
        vertex_map=vertex_map,
        G=G,
        G_prime=G_prime,
    )
