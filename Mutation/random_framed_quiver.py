import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


Arrow = Tuple[int, int]
ArrowCounts = Dict[Arrow, int]
EDGELIST_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)(?:\s*\[\s*(\d+)\s*\])?\s*$")


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


def _validate_2_acyclic_counts(counts: ArrowCounts) -> None:
    seen: Set[Arrow] = set()
    for (a, b), c in counts.items():
        if c < 0:
            raise ValueError(f"Negative multiplicity: {a} -> {b} [{c}]")
        if c == 0:
            continue
        if a == b:
            raise ValueError(f"Loop detected: {a} -> {b}")
        if (b, a) in seen:
            raise ValueError(f"2-cycle detected: {b} -> {a} and {a} -> {b}")
        seen.add((a, b))


def _cleanup_counts(counts: ArrowCounts) -> ArrowCounts:
    return {(a, b): c for (a, b), c in counts.items() if c != 0}


# def _normalize_counts(counts: ArrowCounts, *, allow_multi_arrows: bool) -> ArrowCounts:
#     out: ArrowCounts = {}
#     for (a, b), c in counts.items():
#         if c <= 0:
#             continue
#         if a == b:
#             raise ValueError(f"Loop detected: {a} -> {b}")
#         out[(a, b)] = c if allow_multi_arrows else 1
#     _validate_2_acyclic_counts(out)
#     return out


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


def frame_quiver(n: int, base_counts: ArrowCounts, *, allow_multi_arrows: bool) -> IceQuiver:
    vertices = list(range(1, 2 * n + 1))
    frozen = set(range(n + 1, 2 * n + 1))

    counts = dict(base_counts)
    for i in range(1, n + 1):
        counts[(i, n + i)] = counts.get((i, n + i), 0) + 1

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

    for i, c1 in incoming:
        for j, c2 in outgoing:
            if i == j:
                continue
            # In ice-quiver mutation, the number of new arrows i->j
            # is multiplicity(i->k) * multiplicity(k->j).
            add = c1 * c2
            counts[(i, j)] = counts.get((i, j), 0) + add

    # Step 2: reverse all arrows incident to k.
    # Reuse the already-computed incident lists (incoming/outgoing).
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


def format_edgelist(q: IceQuiver) -> str:
    lines: List[str] = []
    n_mut = len([v for v in q.vertices if v not in q.frozen])
    lines.append(f"vertices: {len(q.vertices)} (mutable={n_mut}, frozen={len(q.frozen)})")
    lines.append(f"frozen: {', '.join(map(str, sorted(q.frozen)))}")
    lines.append("arrows:")
    for (a, b), c in sorted(q.arrow_counts.items()):
        if c > 0:
            lines.append(f"  {a} -> {b} [{c}]")
    return "\n".join(lines) + "\n"


def format_exchange_matrix_exmat(q: IceQuiver) -> str:
    """
    Output the exchange matrix in the same header+body style as `.exmat`.

    Requires the convention:
      - vertices == [1..m]
      - frozen == {n+1, ..., m} for some n
    The matrix is n x m with b_ij = #(i->j) - #(j->i).
    """
    m = len(q.vertices)
    if q.vertices != list(range(1, m + 1)):
        raise ValueError("exmat output requires vertices to be exactly [1..m]")

    n = len([v for v in q.vertices if v not in q.frozen])
    expected_frozen = set(range(n + 1, m + 1))
    if set(q.frozen) != expected_frozen:
        raise ValueError("exmat output requires frozen vertices to be exactly [n+1..m]")

    counts = _cleanup_counts(q.arrow_counts)
    _validate_2_acyclic_counts(counts)

    rows: List[str] = []
    for i in range(1, n + 1):
        vals: List[int] = []
        for j in range(1, m + 1):
            bij = counts.get((i, j), 0) - counts.get((j, i), 0)
            vals.append(bij)
        rows.append(" ".join(str(x) for x in vals))

    header = [f"vertices: {m}", f"mutable: {n}", "matrix:"]
    return "\n".join(header + rows) + "\n"


def parse_quiver_json(text: str) -> IceQuiver:
    data = json.loads(text)
    vertices = [int(v) for v in data["vertices"]]
    if vertices != list(range(1, len(vertices) + 1)):
        raise ValueError("JSON input requires vertices to be exactly [1,2,...,n]")
    frozen = {int(v) for v in data.get("frozen_vertices", [])}
    n_total = len(vertices)
    for v in frozen:
        if not (1 <= v <= n_total):
            raise ValueError(f"Frozen vertex out of range 1..{n_total}: {v}")
    counts: ArrowCounts = {}
    for e in data.get("arrows", []):
        a = int(e["from"])
        b = int(e["to"])
        c = int(e.get("label", e.get("count", 1)))
        if c < 0:
            raise ValueError(f"Negative label in JSON input: {a} -> {b} [{c}]")
        if c == 0:
            continue  # explicitly drop 0-labeled arrows
        if not (1 <= a <= n_total) or not (1 <= b <= n_total):
            raise ValueError(f"Arrow endpoint out of range 1..{n_total}: {a} -> {b}")
        counts[(a, b)] = counts.get((a, b), 0) + c
    _validate_2_acyclic_counts(counts)
    return IceQuiver(vertices=vertices, frozen=frozen, arrow_counts=counts)


def parse_quiver_edgelist(text: str) -> IceQuiver:
    r"""
    Parse an ice-quiver edgelist file.

    ## Expected structure (flexible)

    Headers (order doesn't matter; all optional):
    - `vertices: <N>` declares the vertex set to be \( \{1,\dots,N\} \).
      Extra text is allowed after the number.
    - `frozen: v1, v2, ...` declares frozen vertices as a comma-separated list.
    - `arrows:` starts the arrow section.

    Arrow lines (after `arrows:`):
    - `a -> b` (label defaults to 1)
    - `a -> b [c]` where `c` is a positive integer multiplicity label

    Blank lines are ignored. Other non-matching lines inside the arrow section
    raise an error.
    """
    vertices: List[int] = []
    frozen: Set[int] = set()
    counts: ArrowCounts = {}
    in_arrows = False
    declared_n_total: int | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("vertices:"):
            num_match = re.search(r"vertices:\s*(\d+)", line)
            if num_match:
                n_total = int(num_match.group(1))
                declared_n_total = n_total
                vertices = list(range(1, n_total + 1))
            continue
        if line.startswith("frozen:"):
            rest = line.split(":", 1)[1].strip()
            if rest:
                frozen = {int(x.strip()) for x in rest.split(",") if x.strip()}
            continue
        if line.startswith("arrows:"):
            in_arrows = True
            continue
        if not in_arrows:
            continue
        m = EDGELIST_RE.match(line)
        if not m:
            raise ValueError(f"Invalid edge line: {line}")
        a = int(m.group(1))
        b = int(m.group(2))
        c = int(m.group(3) or 1)
        if declared_n_total is not None:
            if not (1 <= a <= declared_n_total) or not (1 <= b <= declared_n_total):
                raise ValueError(f"Edge endpoint out of range 1..{declared_n_total}: {a} -> {b}")
        counts[(a, b)] = counts.get((a, b), 0) + c

    if not vertices:
        all_vertices = set()
        for (a, b) in counts.keys():
            all_vertices.add(a)
            all_vertices.add(b)
        all_vertices |= frozen
        vertices = sorted(all_vertices)
    else:
        n_total = len(vertices)
        for v in frozen:
            if not (1 <= v <= n_total):
                raise ValueError(f"Frozen vertex out of range 1..{n_total}: {v}")

    _validate_2_acyclic_counts(counts)
    return IceQuiver(vertices=vertices, frozen=frozen, arrow_counts=counts)


def _parse_int_matrix(text: str) -> List[List[int]]:
    rows: List[List[int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.strip("[]()")
        line = line.replace(",", " ")
        parts = [p for p in line.split() if p]
        if not parts:
            continue
        rows.append([int(x) for x in parts])
    if not rows:
        raise ValueError("Empty matrix input")
    m = len(rows[0])
    if m == 0:
        raise ValueError("Matrix must have at least one column")
    for r in rows:
        if len(r) != m:
            raise ValueError("Matrix rows must all have the same length")
    return rows


def parse_exchange_matrix_exmat(text: str) -> IceQuiver:
    """
    Parse an exchange matrix file with a header similar to `.edgelist`.

    ## Expected structure (flexible)

    - `vertices: <m>` (or `m: <m>`)
    - `mutable: <n>` (or `n: <n>`)
    - Optional marker line: `matrix:` (or `exchange_matrix:`)
    - Then the matrix body: **n rows**, each with **m integers**

    Whitespace-separated integers are accepted. Commas and simple brackets
    are tolerated (e.g. `[0, 1, -1]`).
    """
    n_mutable: int | None = None
    m_total: int | None = None
    matrix_lines: List[str] = []
    in_matrix = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        low = line.lower()
        if low.startswith("vertices:") or low.startswith("m:"):
            m = re.search(r"(\d+)", line)
            if not m:
                raise ValueError(f"Could not parse m from line: {line}")
            m_total = int(m.group(1))
            continue

        if low.startswith("mutable:") or low.startswith("n:"):
            n = re.search(r"(\d+)", line)
            if not n:
                raise ValueError(f"Could not parse n from line: {line}")
            n_mutable = int(n.group(1))
            continue

        if low.startswith("matrix:") or low.startswith("exchange_matrix:"):
            in_matrix = True
            continue

        if in_matrix:
            matrix_lines.append(line)

    # If no explicit matrix marker, treat any non-header lines as matrix lines.
    if not matrix_lines and (n_mutable is not None or m_total is not None):
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if (
                low.startswith("vertices:")
                or low.startswith("m:")
                or low.startswith("mutable:")
                or low.startswith("n:")
                or low.startswith("matrix:")
                or low.startswith("exchange_matrix:")
            ):
                continue
            matrix_lines.append(line)

    if n_mutable is None or m_total is None:
        raise ValueError("exmat input requires both 'mutable: n' and 'vertices: m' headers")
    if n_mutable <= 0:
        raise ValueError("mutable: n must be >= 1")
    if m_total < n_mutable:
        raise ValueError(f"vertices: m must satisfy m >= n (got m={m_total}, n={n_mutable})")

    mat = _parse_int_matrix("\n".join(matrix_lines))
    if len(mat) != n_mutable:
        raise ValueError(f"Exchange matrix must have exactly n={n_mutable} rows, got {len(mat)}")
    if len(mat[0]) != m_total:
        raise ValueError(f"Exchange matrix must have exactly m={m_total} columns, got {len(mat[0])}")

    counts: ArrowCounts = {}
    for i0 in range(n_mutable):
        i = i0 + 1
        row = mat[i0]
        for j0, bij in enumerate(row):
            j = j0 + 1
            if i == j or bij == 0:
                continue
            if j <= n_mutable and i >= j:
                continue
            if bij > 0:
                counts[(i, j)] = counts.get((i, j), 0) + bij
            else:
                counts[(j, i)] = counts.get((j, i), 0) + (-bij)

    vertices = list(range(1, m_total + 1))
    frozen = set(range(n_mutable + 1, m_total + 1))
    _validate_2_acyclic_counts(counts)
    return IceQuiver(vertices=vertices, frozen=frozen, arrow_counts=counts)


def load_quiver(path: str, fmt: str) -> IceQuiver:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if fmt == "json":
        return parse_quiver_json(text)
    if fmt == "exmat":
        return parse_exchange_matrix_exmat(text)
    return parse_quiver_edgelist(text)


def _parse_mutation_sequence(s: str) -> List[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate framed quivers and mutate ice quivers per Rules.md."
    )
    p.add_argument("--n", type=int, help="Number of mutable vertices (for generation mode).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--p", type=float, help="Edge probability among mutable pairs (generation mode).")
    g.add_argument("--edges", type=int, help="Number of base edges among mutable vertices.")

    p.add_argument("--input-file", type=str, default="", help="Path to an existing ice quiver.")
    p.add_argument(
        "--input-format",
        choices=["json", "edgelist", "exmat"],
        default="json",
        help="Format of --input-file.",
    )
    p.add_argument(
        "--allow-multi-arrows",
        action="store_true",
        help="Generation mode only: allow multiple arrows with the same source and target.",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Generation mode: max multiplicity per chosen base edge direction.",
    )
    p.add_argument(
        "--mutate",
        type=str,
        default="",
        help="Comma-separated sequence of mutable vertices to mutate at, e.g. '2,1,3'.",
    )
    p.add_argument("--seed", type=int, default=None, help="PRNG seed for generation mode.")
    p.add_argument("--format", choices=["json", "edgelist", "exmat"], default="json", help="Output format.")
    p.add_argument("--output-file", type=str, default="", help="If set, write output to this file.")
    args = p.parse_args()

    if args.input_file:
        # --allow-multi-arrows only affects random generation, not mutation on input quivers.
        q = load_quiver(args.input_file, args.input_format)
    else:
        if args.n is None:
            raise ValueError("Generation mode requires --n")
        if (args.p is None) == (args.edges is None):
            raise ValueError("Generation mode requires exactly one of --p or --edges")
        rng = random.Random(args.seed)
        base = random_quiver(
            args.n,
            rng=rng,
            edge_probability=args.p,
            edges=args.edges,
            allow_multi_arrows=args.allow_multi_arrows,
            max_parallel=args.max_parallel,
        )
        q = frame_quiver(args.n, base, allow_multi_arrows=args.allow_multi_arrows)

    for k in _parse_mutation_sequence(args.mutate):
        q = mutate_ice_quiver(q, k)

    if args.format == "json":
        out = json.dumps(q.to_dict(), indent=2, sort_keys=True) + "\n"
    elif args.format == "edgelist":
        out = format_edgelist(q)
    else:
        out = format_exchange_matrix_exmat(q)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

