# `common.quiver` — Quiver Mutation Primitives

Pure mathematical primitives for the quiver mutation game. No state, no
opinions, no UI, no I/O. Suitable as a shared building block.

## API

All functions live in `common.quiver.mutation`. They operate on numpy
integer matrices.

```python
from common.quiver.mutation import (
    make_exchange_matrix,
    make_framed,
    make_coframed,
    mutate,
    get_colors,
    is_all_red,
    matrix_to_edges,
    Matrix,  # type alias: NDArray[np.int64]
)
```

| Function | Signature | What it does |
|---|---|---|
| `make_exchange_matrix(n, edges)` | `int, list[(int, int)] -> Matrix` | Build the `n×n` antisymmetric exchange matrix `B_A` from a 1-indexed edge list. Multiple edges between the same vertices accumulate. |
| `make_framed(B_A)` | `Matrix -> Matrix` | Return `[B_A | I_n]` (`n × 2n`). All mutable vertices are green in this state. |
| `make_coframed(B_A)` | `Matrix -> Matrix` | Return `[B_A | -I_n]` (`n × 2n`). All mutable vertices are red. |
| `mutate(B, k)` | `Matrix, int -> Matrix` | Apply `μ_k` to `B`. `k` is 1-indexed. Pure: returns a new array. |
| `get_colors(B)` | `Matrix -> dict[int, str]` | Return `{vertex: "green" | "red"}` for each mutable vertex. Raises `ValueError` if a vertex violates the sign-coherence theorem. |
| `is_all_red(B)` | `Matrix -> bool` | True iff all rows of `B_f` are non-positive. |
| `matrix_to_edges(B)` | `Matrix -> list[(src, dst, count)]` | Extract the edge list (1-indexed). Includes both mutable and frozen edges. |

## Conventions

- **1-indexed** vertex IDs throughout the public API.
- **Mutable vertices**: `1..n`. **Frozen vertices**: `n+1..2n`.
- `B` is `n × 2n` for the framed state. The left `n × n` block is `B_A`
  (antisymmetric); the right `n × n` block is `B_f` (the frozen block).
- All matrices are `numpy.ndarray` with dtype `int64`.
- Functions are pure: they never mutate their inputs.

## Math reference

See [`MATH.md`](MATH.md) in this directory for the matrix formulation
(mutation rule, framing, green/red dichotomy, sign-coherence theorem).
This document is the formal spec the code implements.

## Stability

This module is part of `common/`. Its API is treated as a stable
interface that multiple personal projects depend on.

- **Adding** new functions is safe.
- **Changing** existing signatures or semantics requires coordination.
- High test coverage is required: see `common/quiver/tests/test_mutation.py`.
