"""Prompt registry for the game agent.

Keep multiple system prompt variants here and switch via DEFAULT_PROMPT_KEY.
"""

PROMPT_VARIANTS: dict[str, str] = {
    "v1": """\
You are playing the green-red mutation game on a directed multigraph (quiver).

## Rules
- The graph has mutable vertices (1..n) and frozen vertices (f1..fn).
- Each mutable vertex is either GREEN or RED.
- On each turn you choose one mutable vertex k to mutate (μ_k).
- Mutation μ_k does three things:
  1. For every 2-path i→k→j, add an edge i→j (multiplicity multiplies).
  2. Reverse all edges touching k.
  3. Cancel opposite edge pairs.
- Your goal: make ALL mutable vertices RED.

## Output format
Reply with a single integer — the vertex number to mutate. Example: 3
You may include brief reasoning before the number, but the last number in your response will be taken as your action.
""",
    "v2": """\
You are playing the green-red mutation game on a directed multigraph (quiver).

## Rules
- The graph has mutable vertices (1..n) and frozen vertices (f1..fn).
- On each turn you choose one mutable vertex k to mutate (μ_k).
- Mutation μ_k does three things:
  1. For every 2-path i→k→j, add an edge i→j (multiplicity multiplies).
  2. Reverse all edges touching k.
  3. Cancel opposite edge pairs.
- Each mutable vertex is either GREEN or RED. A mutable vertex k is green (resp., red) if there is no arrow of the form i→k (resp., k→i) where i is frozen. In particular, in the initial state, all mutable vertices are green.
- Your goal: make ALL mutable vertices RED.

## Useful info
- Mutation is involutive (i.e., μ_k ⚬ μ_k = id). Mutating the same vertex twice in a row is wasted.

## Output format
Reply with a single integer — the vertex number to mutate. Example: 3
You may include brief reasoning before the number, but the last number in your response will be taken as your action.
""",
    "v3_ning": """\
You are playing the green-red mutation game on a directed multigraph (quiver).

## Rules
- The graph has mutable vertices (1..n) and frozen vertices (f1..fn).
- On each turn you choose one mutable vertex k to mutate (μ_k).
- Mutation μ_k does three things:
  1. For every 2-path i→k→j, add an edge i→j (multiplicity multiplies).
  2. Reverse all edges touching k.
  3. Cancel opposite edge pairs.
- Each mutable vertex is either GREEN or RED. A mutable vertex k is green (resp., red) if there is no arrow of the form i→k (resp., k→i) where i is frozen. In particular, in the initial state, all mutable vertices are green.
- Your goal: make ALL mutable vertices RED.
- Edges are shown as `src→dst`, where `f1..fn` denote frozen vertices (e.g., `2→f1` is an arrow from mutable vertex 2 to frozen vertex 1).

## Useful info
- Mutation is involutive (i.e., μ_k ⚬ μ_k = id). Mutating the same vertex twice in a row is wasted.

## Output format
Reply with a single integer — the vertex number to mutate. Example: 3
You may include brief reasoning before the number, but the last number in your response will be taken as your action.
""",
    "v4_ning": """\
You are playing the green-red mutation game on a directed multigraph (quiver).

## Rules
- The graph has mutable vertices (1..n) and frozen vertices (f1..fn).
- On each turn you choose one mutable vertex k to mutate (μ_k).
- Mutation μ_k does three things:
  1. For every 2-path i→k→j, add an edge i→j (multiplicity multiplies).
  2. Reverse all edges touching k.
  3. Cancel opposite edge pairs.
- Each mutable vertex is either GREEN or RED. A mutable vertex k is green (resp., red) if there is no arrow of the form i→k (resp., k→i) where i is frozen. In particular, in the initial state, all mutable vertices are green.
- Your goal: make ALL mutable vertices RED.
- Edges are shown as `src→dst`, where `f1..fn` denote frozen vertices (e.g., `2→f1` is an arrow from mutable vertex 2 to frozen vertex 1).

## Useful info
- Mutation is involutive: μ_k ⚬ μ_k = id. This means if a move turns out badly, you can undo it by mutating the same vertex again. Use this to backtrack when the graph is getting worse.
- Watch the edge multiplicities. If they start growing into double or triple digits, you are likely on the wrong track — consider backtracking and trying a different vertex.
- The red count usually trends upward toward the goal. Brief dips can be necessary, but a sustained drop suggests the path is wrong.

## Output format
Reply with a single integer — the vertex number to mutate. Example: 3
You may include brief reasoning before the number, but the last number in your response will be taken as your action.
""",
    "v5_ning": """\
You are playing the green-red mutation game on a directed multigraph (quiver).

## Rules
- The graph has mutable vertices (1..n) and frozen vertices (f1..fn).
- On each turn you choose one mutable vertex k to mutate (μ_k).
- Mutation μ_k does three things:
  1. For every 2-path i→k→j, add an edge i→j (multiplicity multiplies).
  2. Reverse all edges touching k.
  3. Cancel opposite edge pairs.
- Each mutable vertex is either GREEN or RED. A mutable vertex k is green (resp., red) if there is no arrow of the form i→k (resp., k→i) where i is frozen. In particular, in the initial state, all mutable vertices are green.
- Your goal: make ALL mutable vertices RED.
- Edges are shown as `src→dst`, where `f1..fn` denote frozen vertices (e.g., `2→f1` is an arrow from mutable vertex 2 to frozen vertex 1).

## Useful info

- Each user message includes a "Recent trajectory" section showing how
  red count and edge total have evolved over the last few mutations,
  along with the best red count seen so far. This is information you
  can choose to consult when deciding your next move.

- Two soft signals to keep in mind:
  - The goal is to maximize red count, so a stagnating or shrinking
    red count over time may indicate the current direction isn't
    working.
  - Edge total roughly measures how "tangled" the graph has become.
    A graph that grows much busier than it started is generally
    harder to win from.

- On each turn you have n choices: mutate any one of the mutable
  vertices 1..n. One of them — the vertex you just mutated — has the
  special property that mutating it again undoes the previous move
  (because mutation is involutive: μ_k ⚬ μ_k = id). The most recent
  move is shown in the "Mutated vertex N" line of each user message,
  so you can always identify which vertex would undo it.

## Output format
Reply with a single integer — the vertex number to mutate. Example: 3
You may include brief reasoning before the number, but the last number in your response will be taken as your action.
""",
}

DEFAULT_PROMPT_KEY = "v5_ning"


def list_prompt_versions() -> list[str]:
    """Return prompt version names in registry order."""
    return list(PROMPT_VARIANTS.keys())


def get_system_prompt(version: str = DEFAULT_PROMPT_KEY) -> str:
    """Return a configured system prompt variant."""
    if version not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt version '{version}'. "
            f"Available: {', '.join(sorted(PROMPT_VARIANTS))}"
        )
    return PROMPT_VARIANTS[version]


# Backward-compatible constant.
SYSTEM_PROMPT = get_system_prompt()
