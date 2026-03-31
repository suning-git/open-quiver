from __future__ import annotations

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

    def m_total(self) -> int:
        return len(self.vertices)
    
    def n_mutable(self) -> int:
        return len(self.vertices) - len(self.frozen)

    @staticmethod
    def empty() -> IceQuiver:
        """
        Create an empty IceQuiver with no vertices or arrows.
        """
        return IceQuiver(vertices=[], frozen=set(), arrow_counts={})
    
    def add_vertex(self, frozen: bool = False) -> tuple[IceQuiver, int]:
        """
        Return a new IceQuiver with one additional vertex.

        Vertices are labeled as 1..m. The new vertex gets label m+1.

        Returns:
            (new_quiver, new_vertex_id)
        """
        new_id = self.m_total() + 1

        new_vertices = self.vertices + [new_id]
        new_frozen = set(self.frozen)

        if frozen:
            new_frozen.add(new_id)

        return (
            IceQuiver(
                vertices=new_vertices,
                frozen=new_frozen,
                arrow_counts=dict(self.arrow_counts),
            ),
            new_id,
        )
    
    def add_arrow(self, a: int, b: int, count: int = 1) -> IceQuiver:
        """
        Return a new IceQuiver with an arrow a -> b.
        If the arrow already exists, increase its multiplicity.
        """
        if a not in self.vertices or b not in self.vertices:
            raise ValueError("Vertices must exist")
        
        if a == b:
            return self

        new_counts = self.arrow_counts

        if (b, a) in new_counts:
            cancel = min(count, new_counts[(b, a)])

            new_counts[(b, a)] -= cancel
            if new_counts[(b, a)] <= 0:
                del new_counts[(b, a)]

            count -= cancel

        if count > 0:
            new_counts[(a, b)] = new_counts.get((a, b), 0) + count

        return IceQuiver(
            vertices=list(self.vertices),
            frozen=set(self.frozen),
            arrow_counts=new_counts,
        )

    def remove_arrow(self, a: int, b: int, count: int = 1) -> IceQuiver:
        """
        Decrease multiplicity of arrow a -> b.
        Remove entry if count becomes zero or negative.
        """
        new_counts = self.arrow_counts

        if (a, b) in new_counts:
            new_counts[(a, b)] -= count
            if new_counts[(a, b)] <= 0:
                new_counts.pop((a, b), 0)

        return IceQuiver(
            vertices=list(self.vertices),
            frozen=set(self.frozen),
            arrow_counts=new_counts,
        )
    
    def remove_vertex(self, v: int) -> IceQuiver:
        if v not in self.vertices:
            return self

        m = len(self.vertices)

        # map vertices to new vertex number
        def remap(x: int) -> int:
            if x < v:
                return x
            elif x > v:
                return x - 1
            else:
                raise ValueError("should not map removed vertex")

        new_vertices = list(range(1, m))  # [m-1]
        new_frozen = {remap(x) for x in self.frozen if x != v}
        new_counts: ArrowCounts = {}

        for (a, b), c in self.arrow_counts.items():
            if a == v or b == v:
                continue

            a2 = remap(a)
            b2 = remap(b)
            new_counts[(a2, b2)] = new_counts.get((a2, b2), 0) + c

        return IceQuiver(
            vertices=new_vertices,
            frozen=new_frozen,
            arrow_counts=new_counts,
        )

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
    
    def freeze_vertex(self, v: int) -> IceQuiver:
        if v not in self.vertices:
            return self

        new_frozen = set(self.frozen)
        new_frozen.add(v)

        new_counts: ArrowCounts = {}

        for (a, b), c in self.arrow_counts.items():
            if a in new_frozen and b in new_frozen:
                continue
            new_counts[(a, b)] = c

        return IceQuiver(
            vertices=self.vertices,
            frozen=new_frozen,
            arrow_counts=new_counts,
        )
    
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