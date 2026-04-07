"""Prompt registry for the game agent.

Keep multiple system prompt variants here and switch via DEFAULT_PROMPT_KEY.
"""

PROMPT_VARIANTS: dict[str, str] = {
    "v1": """\
You are playing the green-red mutation game on a directed multigraph (quiver).

## Rules
- The graph has mutable vertices (1..n) and frozen vertices (1'..n').
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
- The graph has mutable vertices (1..n) and frozen vertices (1'..n').
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
}

DEFAULT_PROMPT_KEY = "v2"


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
