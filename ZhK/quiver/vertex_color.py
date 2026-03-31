"""
Green / red criteria for mutable vertices in a framed ice quiver (Docs/MutationRules.md).

Convention: |Q_0| = 2n, mutable vertices 1..n, frozen (frame) vertices n+1..2n,
with vertex i' labeled n+i.
"""

from __future__ import annotations

from enum import Enum, auto
from .core import IceQuiver

class VColor(Enum):
    DEFAULT = auto()
    YELLOW = auto()
    GREEN = auto()
    RED = auto()
    FROZEN = auto()


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

def vertex_color(quiver: IceQuiver, vid: int) -> VColor:
    if vid in quiver.frozen:
        return VColor.FROZEN

    if is_green_framed(quiver, vid):
        if is_red_framed(quiver, vid):
            return VColor.YELLOW
        else:
            return VColor.GREEN
    elif is_red_framed(quiver, vid):
        return VColor.RED

    return VColor.DEFAULT