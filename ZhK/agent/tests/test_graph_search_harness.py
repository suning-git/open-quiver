from common.quiver.mutation import make_exchange_matrix, make_framed, mutate
from ZhK.agent.dual_agent_runner import IterationResult, SearchResult
from ZhK.agent.graph_search_harness import (
    format_iteration_log,
    node_to_render_state,
    summarize_snapshot,
)
from ZhK.agent.state_graph_store import StateGraphStore


def test_node_to_render_state_shape_and_keys():
    store = StateGraphStore()
    root = make_framed(make_exchange_matrix(2, [(1, 2)]))
    state_id, _ = store.add_state(root)
    node = store.get_state(state_id)

    rendered = node_to_render_state(node)
    assert rendered["total_mutable"] == 2
    assert rendered["red_count"] == 0
    assert "edges" in rendered
    assert "colors" in rendered


def test_summarize_snapshot_extracts_best_metrics():
    store = StateGraphStore()
    root = make_framed(make_exchange_matrix(2, [(1, 2)]))
    root_id, _ = store.add_state(root)

    snapshot = SearchResult(
        won=False,
        reason="in_progress",
        iterations=3,
        root_state_id=root_id,
        best_state_id=root_id,
        best_path=[root_id],
        parse_failures=1,
        expansions=2,
    )
    summary = summarize_snapshot(snapshot, store)
    assert summary["iterations"] == 3
    assert summary["state_count"] == 1
    assert summary["best_state_created_iter"] == 0
    assert summary["best_state_created_path"] == "[]"


def test_format_iteration_log_parse_failure():
    store = StateGraphStore()
    root = make_framed(make_exchange_matrix(2, [(1, 2)]))
    state_id, _ = store.add_state(root)

    msg = format_iteration_log(
        IterationResult(
            iteration=1,
            selected_state_id=state_id,
            parse_failed=True,
        ),
        store,
    )
    assert "parse failure" in msg.lower()


def test_format_iteration_log_transition_contains_delta():
    store = StateGraphStore()
    root = make_framed(make_exchange_matrix(2, [(1, 2)]))
    src_id, _ = store.add_state(root)
    dst = mutate(root, 1)
    dst_id, _, _, _ = store.add_mutation_transition(src_id, 1, dst, created_iter=1)

    msg = format_iteration_log(
        IterationResult(
            iteration=1,
            selected_state_id=src_id,
            action=1,
            to_state_id=dst_id,
            reward=0.1,
            state_created=True,
        ),
        store,
    )
    assert "--u1-->" in msg
    assert "delta" in msg
    assert "[1]" in msg
