"""Harness layer: translates between engine state and LLM text.

Pure functions — no LLM calls, no state.
"""

import re


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

    # # Edges among mutable vertices only
    # mutable_edges = [(s, d, c) for s, d, c in edges if s <= n and d <= n]
    # if mutable_edges:
    #     edge_strs = []
    #     for s, d, c in mutable_edges:
    #         if c == 1:
    #             edge_strs.append(f"{s}→{d}")
    #         else:
    #             edge_strs.append(f"{s}→{d} (×{c})")
    #     lines.append("Edges: " + ", ".join(edge_strs))
    # else:
    #     lines.append("Edges: (none)")

    # All edges
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
    action_type = diff.get("action_type", "mutate")
    before = diff["red_count_before"]
    after = diff["red_count_after"]
    n = state["total_mutable"]

    lines = []
    if action_type == "undo":
        k = diff["undone_vertex"]
        lines.append(f"Undid last move on vertex {k}.")
    else:
        k = diff["mutated_vertex"]
        lines.append(f"Mutated vertex {k}.")

    # Color changes
    for v, (old, new) in sorted(diff["color_changes"].items()):
        lines.append(f"  Vertex {v}: {old} → {new}")

    lines.append(f"Red: {before}/{n} → {after}/{n}")

    if "cycle_warning" in diff:
        step = diff["cycle_warning"]
        lines.append(f"⚠ You returned to the state from step {step}.")

    return "\n".join(lines)


def parse_action_or_undo(text: str, n: int) -> int | str | None:
    """Extract either a mutable vertex number or an undo command.

    Returns:
        int: Vertex ID in 1..n.
        "undo": If text requests undo.
        None: If no valid command found.
    """
    candidates: list[tuple[int, int | str]] = []

    # Collect undo commands.
    for m in re.finditer(r"\bundo\b|\bu\b", text.strip().lower()):
        candidates.append((m.end(), "undo"))

    # Collect valid vertex numbers.
    for m in re.finditer(r"\d+", text):
        k = int(m.group())
        if 1 <= k <= n:
            candidates.append((m.end(), k))

    if not candidates:
        return None

    # Use the last explicit command in the response.
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def parse_action(text: str, n: int) -> int | None:
    """Extract a mutable vertex number from LLM output.

    Takes the last integer found in the text. Returns None if no valid
    vertex number is found.

    Args:
        text: Raw LLM output.
        n: Number of mutable vertices (valid range is 1..n).
    """
    command = parse_action_or_undo(text, n)
    if isinstance(command, int):
        return command
    return None


def format_error(text: str, n: int, allow_undo: bool = False) -> str:
    """Generate an error message when parse_action fails.

    Args:
        text: The raw LLM output that failed to parse.
        n: Number of mutable vertices.
    """
    return (
        f"Could not parse a valid vertex from your response. "
        + (
            f"Please reply with a single integer between 1 and {n}, or 'undo'."
            if allow_undo
            else f"Please reply with a single integer between 1 and {n}."
        )
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
