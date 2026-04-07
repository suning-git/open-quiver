"""Harness layer: translates between engine state and LLM text.

Pure functions — no LLM calls, no state.
"""

import re

# Re-exported for backward compatibility. Source of truth: initial_prompts.py.
from .initial_prompts import SYSTEM_PROMPT  # noqa: F401


def _format_vertex(v: int, n: int) -> str:
    """Format vertex ID: mutable as int, frozen as f1..fn."""
    return str(v) if v <= n else f"f{v - n}"


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

    # Vertex colors (mutable only)
    color_parts = []
    for v in range(1, n + 1):
        c = colors[v]
        tag = "R" if c == "red" else "G"
        color_parts.append(f"{v}({tag})")
    lines.append("Vertices: " + " ".join(color_parts))

    # All edges (mutable-mutable and frozen edges in either direction)
    if edges:
        edge_strs = []
        for s, d, c in edges:
            sf = _format_vertex(s, n)
            df = _format_vertex(d, n)
            if c == 1:
                edge_strs.append(f"{sf}→{df}")
            else:
                edge_strs.append(f"{sf}→{df} (×{c})")
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
    # Exclude digits that are part of frozen-vertex labels like "f1", "f2".
    numbers = re.findall(r"(?<![a-zA-Z])\d+", text)
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
