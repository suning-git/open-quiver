"""Bipartite belt solver for square product quivers.

Given a square product G □ G', generates the bipartite belt mutation
sequence — a known maximal green sequence (analytical solution).

References:
- [1] Keller, "The periodicity conjecture for pairs of Dynkin diagrams",
      Annals of Math 177(1), 2013. arXiv:1001.1531
- [2] Casbi-Hosaka-Ikeda, "Half-Periodicity of Zamolodchikov Periodic
      Cluster Algebras", 2026. arXiv:2602.15140 — Proposition 3.1
"""

from common.bipartite_belt.square_product import SquareProduct


def bipartite_belt_solution(
    sp: SquareProduct,
    start_white: bool = True,
) -> list[int]:
    """Generate the bipartite belt mutation sequence.

    By Proposition 3.1 of [2], there are two maximal green sequences:
    - Starting with white group (μ○), alternating for h_{G'} rounds.
    - Starting with black group (μ●), alternating for h_G rounds.

    Within each round, the order of mutations does not matter because
    all vertices in the same group have b_{uv} = 0 (mutations commute).

    Args:
        sp: SquareProduct instance.
        start_white: If True, start with white group (h_{G'} rounds).
                     If False, start with black group (h_G rounds).

    Returns:
        List of vertex indices (1-indexed) forming a maximal green sequence.
    """
    if start_white:
        groups = [sp.white_group, sp.black_group]
        num_rounds = sp.G_prime.coxeter
    else:
        groups = [sp.black_group, sp.white_group]
        num_rounds = sp.G.coxeter

    sequence = []
    for r in range(num_rounds):
        group = groups[r % 2]
        sequence.extend(group)

    return sequence
