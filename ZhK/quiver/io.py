import json
import re
from typing import List, Optional, Set
from .core import ArrowCounts, IceQuiver, ice_quiver_from_exmat
from .operations import _validate_2_acyclic_counts


EDGELIST_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)(?:\s*\[\s*(\d+)\s*\])?\s*$")


def _reject_arrows_between_frozen(frozen: Set[int], counts: ArrowCounts) -> None:
    """Ice quivers forbid arrows with both endpoints frozen (MutationRules / quiver def)."""
    for (a, b), c in counts.items():
        if c > 0 and a in frozen and b in frozen:
            raise ValueError(f"Ice quiver forbids arrows between frozen vertices: {a} -> {b}")


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


def format_exchange_matrix(q: IceQuiver) -> str:
    """
    Output the exchange matrix in the same header+body style as `.exmat`.

    Uses `IceQuiver.to_exmat()` for entries. Requires the same convention as
    `to_exmat`: vertices [1..m], frozen [n+1..m], and B is n×m with
    B_ij = #(i→j) − #(j→i).
    """
    mat = q.to_exmat()
    n, m = mat.shape
    rows = [" ".join(str(int(x)) for x in mat[i]) for i in range(n)]
    header = [f"vertices: {m}", f"mutable: {n}", "matrix:"]
    return "\n".join(header + rows) + "\n"


def parse_quiver_json(text: str) -> IceQuiver:
    """
    Requires vertices [1..n], valid labels; rejects arrows between two frozen
    vertices (ice quiver convention).
    """
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
    _reject_arrows_between_frozen(frozen, counts)
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

    No arrow may have both endpoints frozen (ice quiver convention).

    Blank lines are ignored. Other non-matching lines inside the arrow section
    raise an error.
    """
    vertices: List[int] = []
    frozen: Set[int] = set()
    counts: ArrowCounts = {}
    in_arrows = False
    declared_n_total: Optional[int] = None

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

    _reject_arrows_between_frozen(frozen, counts)
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


def parse_exchange_matrix(text: str) -> IceQuiver:
    """
    Parse an exchange matrix file with a header `.exmat`.

    ## Expected structure (flexible)

    - `vertices: <m>` (or `m: <m>`)
    - `mutable: <n>` (or `n: <n>`)
    - Optional marker line: `matrix:` (or `exchange_matrix:`)
    - Then the matrix body: **n rows**, each with **m integers**

    Whitespace-separated integers are accepted. Commas and simple brackets
    are tolerated (e.g. `[0, 1, -1]`).
    """
    n_mutable: Optional[int] = None
    m_total: Optional[int] = None
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

    q = ice_quiver_from_exmat(mat)
    _validate_2_acyclic_counts(q.arrow_counts)
    return q


def load_quiver(path: str, fmt: str) -> IceQuiver:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if fmt == "json":
        return parse_quiver_json(text)
    if fmt == "exmat":
        return parse_exchange_matrix(text)
    return parse_quiver_edgelist(text)