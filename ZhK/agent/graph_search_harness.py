"""UI/format helpers for graph-search mode."""

from __future__ import annotations

from typing import Any

from common.quiver.mutation import matrix_to_edges

from .dual_agent_runner import IterationResult, SearchResult
from .state_graph_store import StateGraphStore


def _format_action_path(actions: list[int]) -> str:
    if not actions:
        return "[]"
    return "[" + " ".join(str(k) for k in actions) + "]"


def _state_ref(store: StateGraphStore, state_id: str) -> str:
    node = store.get_state(state_id)
    return f"{node.created_iter} {_format_action_path(node.shortest_path_actions)}"


def node_to_render_state(node: Any) -> dict:
    """Convert a StateGraphStore node to graph_viz-compatible state dict."""
    return {
        "matrix": node.matrix.copy(),
        "edges": matrix_to_edges(node.matrix),
        "colors": dict(node.colors),
        "red_count": node.red_count,
        "total_mutable": node.n,
        "step": node.created_iter,
        "move_history": [],
    }


def summarize_snapshot(snapshot: SearchResult, store: StateGraphStore) -> dict:
    """Build compact metric summary for the web sidebar."""
    best = store.get_state(snapshot.best_state_id)
    return {
        "won": snapshot.won,
        "reason": snapshot.reason,
        "iterations": snapshot.iterations,
        "parse_failures": snapshot.parse_failures,
        "expansions": snapshot.expansions,
        "state_count": store.state_count,
        "edge_count": store.edge_count,
        "best_red_count": best.red_count,
        "best_total_mutable": best.n,
        "best_state_created_iter": best.created_iter,
        "best_state_created_path": _format_action_path(best.shortest_path_actions),
    }


def format_iteration_log(it: IterationResult, store: StateGraphStore) -> str:
    """Render a short, human-readable iteration log line."""
    if it.parse_failed:
        selected = (
            _state_ref(store, it.selected_state_id)
            if it.selected_state_id is not None
            else "(none)"
        )
        return (
            f"Iter {it.iteration}: selected {selected}, "
            "parse failure (no valid action)."
        )
    if it.selected_state_id is None:
        return f"Iter {it.iteration}: stopped ({it.reason})."
    if it.to_state_id is None or it.action is None:
        return (
            f"Iter {it.iteration}: selected {_state_ref(store, it.selected_state_id)}, "
            "no transition."
        )

    src = store.get_state(it.selected_state_id)
    dst = store.get_state(it.to_state_id)
    delta_red = dst.red_count - src.red_count
    return (
        f"Iter {it.iteration}: "
        f"{src.created_iter} {_format_action_path(src.shortest_path_actions)} "
        f"--u{it.action}--> "
        f"{dst.created_iter} {_format_action_path(dst.shortest_path_actions)} "
        f"red {src.red_count}->{dst.red_count} "
        f"(delta {delta_red:+d}), reward={it.reward:.3f}."
    )
