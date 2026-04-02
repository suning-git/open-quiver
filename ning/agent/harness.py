"""Harness layer: translates between engine state and LLM text.

Pure functions — no LLM calls, no state.
"""

import re

SYSTEM_PROMPT = """\
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

## Output format
Reply with a single integer — the vertex number to mutate. Example: 3
You may include brief reasoning before the number, but the last number in your response will be taken as your action.
"""


def render_state(state: dict) -> str:
    """Render engine state as text for the LLM.

    Args:
        state: Dict from engine.get_state() or engine.mutate().
    """
    n = state["total_mutable"]
    colors = state["colors"]
    edges = state["edges"]
    red_count = state["red_count"]
    step = state["step"]

    lines = []
    lines.append(f"Step {step} | Red: {red_count}/{n}")

    # Vertex colors
    color_parts = []
    for v in range(1, n + 1):
        c = colors[v]
        tag = "R" if c == "red" else "G"
        color_parts.append(f"{v}({tag})")
    lines.append("Vertices: " + " ".join(color_parts))

    if edges:
        edge_strs = []
        for s, d, c in edges:
            if c == 1:
                edge_strs.append(f"{s}→{d}")
            else:
                edge_strs.append(f"{s}→{d} (×{c})")
        lines.append("Edges: " + ", ".join(edge_strs))
    else:
        lines.append("Edges: (none)")

    return "\n".join(lines)


def render_diff(state: dict) -> str:
    """Render the diff portion of a mutate() result as text.

    Args:
        state: Dict from engine.mutate() (must contain "diff" key).
    """
    diff = state["diff"]
    k = diff["mutated_vertex"]
    before = diff["red_count_before"]
    after = diff["red_count_after"]
    n = state["total_mutable"]

    lines = []
    lines.append(f"Mutated vertex {k}.")

    # Color changes
    for v, (old, new) in sorted(diff["color_changes"].items()):
        lines.append(f"  Vertex {v}: {old} → {new}")

    lines.append(f"Red: {before}/{n} → {after}/{n}")

    if "cycle_warning" in diff:
        step = diff["cycle_warning"]
        lines.append(f"⚠ You returned to the state from step {step}.")

    return "\n".join(lines)


def parse_action(text: str, n: int) -> int | None:
    """Extract a mutable vertex number from LLM output.

    Takes the last integer found in the text. Returns None if no valid
    vertex number is found.

    Args:
        text: Raw LLM output.
        n: Number of mutable vertices (valid range is 1..n).
    """
    numbers = re.findall(r"\d+", text)
    for num_str in reversed(numbers):
        k = int(num_str)
        if 1 <= k <= n:
            return k
    return None


def format_error(text: str, n: int) -> str:
    """Generate an error message when parse_action fails.

    Args:
        text: The raw LLM output that failed to parse.
        n: Number of mutable vertices.
    """
    return (
        f"Could not parse a valid vertex from your response. "
        f"Please reply with a single integer between 1 and {n}."
    )


def build_user_message(state: dict, diff_text: str | None = None) -> str:
    """Assemble the user message for one turn.

    Args:
        state: Current engine state dict.
        diff_text: Output of render_diff() from the previous move, or None for the first turn.
    """
    parts = []
    if diff_text is not None:
        parts.append(diff_text)
        parts.append("")
    parts.append(render_state(state))
    parts.append("")
    parts.append("Your move?")
    return "\n".join(parts)
