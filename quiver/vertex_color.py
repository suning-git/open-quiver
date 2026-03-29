"""
Green / red criteria for mutable vertices in a framed ice quiver (Docs/MutationRules.md).

Convention: |Q_0| = 2n, mutable vertices 1..n, frozen (frame) vertices n+1..2n,
with vertex i' labeled n+i.
"""

from __future__ import annotations

from .core import IceQuiver


# def framed_layout_n(q: IceQuiver) -> int | None:
#     """
#     If q is a framed layout (vertices 1..2n, frozen exactly {n+1,..,2n}), return n;
#     otherwise None.
#     """
#     m = len(q.vertices)
#     if m == 0 or m % 2 != 0:
#         return None
#     n = m // 2
#     if q.vertices != list(range(1, m + 1)):
#         return None
#     if q.frozen != set(range(n + 1, m + 1)):
#         return None
#     return n


def is_green_framed(q: IceQuiver, k: int) -> bool:
    """Mutable i is green if there is no arrow j' -> i for any j in [n]."""
    if k in q.frozen:
        return False
    for j in q.frozen:
        if q.arrow_counts.get((j, k), 0) > 0:
            return False
    return True


def is_red_framed(q: IceQuiver, k: int) -> bool:
    """Mutable i is red if there is no arrow i -> j' for any j in [n]."""
    if k in q.frozen:
        return False
    for j in q.frozen:
        if q.arrow_counts.get((k, j), 0) > 0:
            return False
    return True
