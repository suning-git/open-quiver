import random
from typing import List, Set, Tuple
from .core import Arrow, ArrowCounts, IceQuiver


def _validate_2_acyclic_counts(counts: ArrowCounts) -> None:
    seen: Set[Arrow] = set()
    for (a, b), c in counts.items():
        if c < 0:
            raise ValueError(f"Negative multiplicity: {a} -> {b} [{c}]")
        if c == 0:
            counts.pop((a, b), None)
            continue
        if a == b:
            raise ValueError(f"Loop detected: {a} -> {b}")
        if (b, a) in seen:
            raise ValueError(f"2-cycle detected: {b} -> {a} and {a} -> {b}")
        seen.add((a, b))

# def _cleanup_counts(counts: ArrowCounts) -> ArrowCounts:
#     return {(a, b): c for (a, b), c in counts.items() if c != 0}

def random_quiver(
        n: int,
        *,
        rng: random.Random,
        edge_probability: float | None,
        edges: int | None,
        allow_multi_arrows: bool,
        max_parallel: int,
    ) -> ArrowCounts:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if (edge_probability is None) == (edges is None):
        raise ValueError("Specify exactly one of edge_probability or edges")
    if max_parallel <= 0:
        raise ValueError("max_parallel must be >= 1")

    pairs: List[Tuple[int, int]] = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]

    if edge_probability is not None:
        if not (0.0 <= edge_probability <= 1.0):
            raise ValueError("edge_probability must be in [0, 1]")
        chosen = [pair for pair in pairs if rng.random() < edge_probability]
    else:
        max_edges = len(pairs)
        if edges is None or edges < 0:
            raise ValueError("edges must be >= 0")
        if edges > max_edges:
            raise ValueError(f"edges too large: max is {max_edges} for n={n}")
        chosen = rng.sample(pairs, k=edges)

    counts: ArrowCounts = {}
    for i, j in chosen:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        mult = rng.randint(1, max_parallel) if allow_multi_arrows else 1
        counts[(a, b)] = counts.get((a, b), 0) + mult

    return counts

def frame_quiver_old(n: int, base_counts: ArrowCounts) -> IceQuiver:
    vertices = list(range(1, 2*n + 1))
    frozen = set(range(n + 1, 2*n + 1))

    counts = dict(base_counts)
    for i in range(1, n + 1):
        counts[(i, n + i)] = counts.get((i, n + i), 0) + 1

    return IceQuiver(vertices=vertices, frozen=frozen, arrow_counts=counts)

def frame_quiver(q: IceQuiver) -> IceQuiver:
    m = q.m_total()
    new_vertices = list(range(1, 2*m + 1))
    new_frozen = set(range(m + 1, 2*m + 1))
    new_counts = dict(q.arrow_counts)

    for i in q.vertices:
        new_counts[(i, m + i)] = new_counts.get((i, m + i), 0) + 1
    
    return IceQuiver(
            vertices=new_vertices,
            frozen=new_frozen,
            arrow_counts=new_counts,
        )

def coframe_quiver(n: int, base_counts: ArrowCounts) -> IceQuiver:
    vertices = list(range(1, 2*n + 1))
    frozen = set(range(n + 1, 2*n + 1))

    counts = dict(base_counts)
    for i in range(1, n + 1):
        counts[(n + i, i)] = counts.get((n + i, i), 0) + 1

    return IceQuiver(vertices=vertices, frozen=frozen, arrow_counts=counts)

def mutate_ice_quiver(q: IceQuiver, k: int) -> IceQuiver:
    if k in q.frozen:
        raise ValueError(f"Cannot mutate at frozen vertex {k}")
    if k not in q.vertices:
        raise ValueError(f"Vertex {k} is not in the quiver")

    counts = dict(q.arrow_counts)

    incoming: List[Tuple[int, int]] = []
    outgoing: List[Tuple[int, int]] = []
    for (a, b), c in counts.items():
        if c <= 0:
            continue
        if b == k and a != k:
            incoming.append((a, c))
        if a == k and b != k:
            outgoing.append((b, c))

    # Step 1: add arrow i->j for any i->k->j.
    # Notice that the number of new arrows i->j is multiplicity(i->k) * multiplicity(k->j).
    for i, c1 in incoming:
        for j, c2 in outgoing:
            if i == j:
                continue
            add = c1 * c2
            counts[(i, j)] = counts.get((i, j), 0) + add

    # Step 2: reverse all arrows incident to k.
    flipped = dict(counts)
    for i, c in incoming:
        prev = flipped.pop((i, k), 0)
        if prev != c:
            raise ValueError("Internal error: unexpected multiplicity mismatch for incoming edge.")
        flipped[(k, i)] = flipped.get((k, i), 0) + c
    for j, c in outgoing:
        prev = flipped.pop((k, j), 0)
        if prev != c:
            raise ValueError("Internal error: unexpected multiplicity mismatch for outgoing edge.")
        flipped[(j, k)] = flipped.get((j, k), 0) + c
    counts = flipped

    # Step 3: remove newly generated loops and 2-cycles, and arrows between frozen vertices.
    for (a, b) in list(counts.keys()):
        if a == b:
            counts.pop((a, b), None)
            continue
        if (b, a) not in counts:
            continue
        x = counts.get((a, b), 0)
        y = counts.get((b, a), 0)
        m = min(x, y)
        x -= m
        y -= m
        if x > 0:
            counts[(a, b)] = x
        else:
            counts.pop((a, b), None)
        if y > 0:
            counts[(b, a)] = y
        else:
            counts.pop((b, a), None)

    for (a, b) in list(counts.keys()):
        if a in q.frozen and b in q.frozen:
            counts.pop((a, b), None)

    _validate_2_acyclic_counts(counts)
    return IceQuiver(vertices=q.vertices, frozen=set(q.frozen), arrow_counts=counts)