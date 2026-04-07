# Games — Shared Game Catalog

Each `.json` file in this directory defines one **green-red mutation game**: a graph
together with (optionally) a known winning mutation sequence.

This is **shared data**: any modification to existing files must be coordinated
across collaborators. Adding new files is generally safe.

## File format

```json
{
  "n": 6,
  "B_A": [
    [ 0, -1,  1,  0,  0,  0],
    [ 1,  0, -1, -1,  1,  0],
    [-1,  1,  0,  0, -1,  1],
    [ 0,  1,  0,  0, -1,  0],
    [ 0, -1,  1,  1,  0, -1],
    [ 0,  0, -1,  0,  1,  0]
  ],
  "solution": [1, 2, 3, 4, 5, 6, 3, 2, 1, 3]
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `n` | int | yes | Number of mutable vertices. Frozen vertex count is also `n` (always). |
| `B_A` | int[n][n] | yes | The base exchange matrix. **Must be antisymmetric** (`B_A[i][j] == -B_A[j][i]`). Diagonal must be zero. |
| `solution` | int[] | optional | A mutation sequence (1-indexed vertex numbers, in `1..n`) such that applying `μ_k1 ∘ μ_k2 ∘ ...` to `framed(B_A)` reaches the all-red state. |

### Conventions

- Filenames are graph names. Use lowercase with underscores. Suffix `_n<k>` (e.g. `test1_07_n4.json`) is recommended when several graphs share a base name.
- The `solution` is **not necessarily optimal** — it just needs to be a valid winning sequence. Multiple valid solutions may exist; storing any one of them is enough.
- The framing convention is: full state is `[B_A | I_n]`; the goal state is `[? | -P]` for some permutation matrix `P`.
- See `ning/graph_rule.md` (graph-theoretic rules) and [`common/quiver/MATH.md`](../quiver/MATH.md) (matrix formulation) for the math.

## Adding a new game

1. Create a new JSON file matching the format above.
2. Verify it: load `B_A`, build `framed(B_A)`, apply `solution`, check that the result is all-red.
3. Ning's `ning/agent/tests/test_engine.py::TestCatalog::test_solutions_reach_all_red` does this automatically for every game with a `solution`.
