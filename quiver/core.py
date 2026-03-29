import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple, Union


Arrow = Tuple[int, int]
ArrowCounts = Dict[Arrow, int]


@dataclass(frozen=True)
class IceQuiver:
    vertices: List[int]
    frozen: Set[int]
    arrow_counts: ArrowCounts

    def to_dict(self) -> Dict:
        arrows = [
            {"from": a, "to": b, "label": c}
            for (a, b), c in sorted(self.arrow_counts.items())
            if c > 0
        ]
        return {
            "vertices": self.vertices,
            "mutable_vertices": [v for v in self.vertices if v not in self.frozen],
            "frozen_vertices": sorted(self.frozen),
            "arrows": arrows,
        }
    
    def to_exmat(self) -> np.ndarray:
        """
        Exchange matrix B with shape (n, m): B_ij = #(i→j) − #(j→i)
        for mutable rows i ∈ [n] and columns j ∈ [m] (1-based vertex labels).
        """
        m = len(self.vertices)
        if self.vertices != list(range(1, m + 1)):
            raise ValueError("exmat requires vertices to be exactly [1..m]")

        n = len([v for v in self.vertices if v not in self.frozen])
        expected_frozen = set(range(n + 1, m + 1))
        if set(self.frozen) != expected_frozen:
            raise ValueError("exmat requires frozen vertices to be exactly [n+1..m]")

        mat = np.zeros((n, m), dtype=int)
        for i0 in range(n):
            i = i0 + 1
            for j0 in range(m):
                j = j0 + 1
                mat[i0, j0] = self.arrow_counts.get((i, j), 0) - self.arrow_counts.get((j, i), 0)
        return mat


def ice_quiver_from_exmat(mat: Union[np.ndarray, Sequence[Sequence[int]]]) -> IceQuiver:
    """
    Build an ice quiver from an exchange matrix B of shape (n, m), with the
    same convention as `.exmat` / `IceQuiver.to_exmat()`:

    - Q_0 = [m], frozen vertices F = [n+1, m], mutable [n].
    - For i ∈ [n], j ∈ [m], B_ij = #(i→j) − #(j→i).

    Does not validate the input matrix. Only upper trapezoid part is used.
    """
    arr = np.asarray(mat, dtype=int)
    if arr.ndim != 2:
        raise ValueError("exchange matrix must be 2-dimensional")
    n_mutable, m_total = int(arr.shape[0]), int(arr.shape[1])
    if n_mutable <= 0:
        raise ValueError("exchange matrix must have at least one row (n >= 1)")
    if m_total < n_mutable:
        raise ValueError(f"m must be >= n (got m={m_total}, n={n_mutable})")

    counts: ArrowCounts = {}
    for i0 in range(n_mutable):
        i = i0 + 1
        for j0 in range(m_total):
            j = j0 + 1
            bij = int(arr[i0, j0])
            if i >= j or bij == 0:
                continue
            if bij > 0:
                counts[(i, j)] = counts.get((i, j), 0) + bij
            else:
                counts[(j, i)] = counts.get((i, j), 0) - bij

    vertices = list(range(1, m_total + 1))
    frozen = set(range(n_mutable + 1, m_total + 1))
    return IceQuiver(vertices=vertices, frozen=frozen, arrow_counts=counts)