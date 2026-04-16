import pytest

from common.quiver.mutation import (
    make_coframed,
    make_exchange_matrix,
    make_framed,
    mutate,
)
from ZhK.agent.state_graph_store import StateGraphStore


def _matrix_n2():
    return make_framed(make_exchange_matrix(2, [(1, 2)]))


def _matrix_n3():
    return make_framed(make_exchange_matrix(3, [(1, 2), (2, 3)]))


def test_add_state_deduplicates_by_matrix_hash():
    store = StateGraphStore()
    matrix = _matrix_n2()

    state_id_1, created_1 = store.add_state(matrix, created_iter=0)
    state_id_2, created_2 = store.add_state(matrix, created_iter=99)

    assert created_1
    assert not created_2
    assert state_id_1 == state_id_2
    assert store.state_count == 1


def test_reverse_transition_reuses_same_undirected_edge():
    store = StateGraphStore()
    root = _matrix_n2()
    root_id, _ = store.add_state(root)

    next_matrix = mutate(root, 1)
    next_id, edge_id_1, state_created_1, edge_created_1 = store.add_mutation_transition(
        root_id,
        1,
        next_matrix,
    )

    back_id, edge_id_2, state_created_2, edge_created_2 = store.add_mutation_transition(
        next_id,
        1,
        root,
    )

    assert state_created_1
    assert edge_created_1
    assert back_id == root_id
    assert edge_id_2 == edge_id_1
    assert not state_created_2
    assert not edge_created_2
    assert store.edge_count == 1


def test_same_endpoints_with_different_actions_create_distinct_edges():
    store = StateGraphStore()
    root = _matrix_n2()
    root_id, _ = store.add_state(root)
    next_id, _, _, _ = store.add_mutation_transition(root_id, 1, mutate(root, 1))

    edge_id_1, created_1 = store.add_undirected_edge(root_id, next_id, 1)
    edge_id_2, created_2 = store.add_undirected_edge(root_id, next_id, 2)

    assert created_1 is False  # already created via add_mutation_transition
    assert created_2 is True
    assert edge_id_1 != edge_id_2
    assert store.edge_count == 2


def test_update_stats_updates_mean_values():
    store = StateGraphStore()
    root = _matrix_n2()
    root_id, _ = store.add_state(root)
    _, edge_id, _, _ = store.add_mutation_transition(root_id, 1, mutate(root, 1))

    store.update_state_stats(root_id, 1.0)
    store.update_state_stats(root_id, -0.5)
    node = store.get_state(root_id)
    assert node.visit_count == 2
    assert node.value_sum == pytest.approx(0.5)
    assert node.value_mean == pytest.approx(0.25)

    store.update_edge_stats(edge_id, 0.5)
    store.update_edge_stats(edge_id, 1.5)
    edge = store.get_edge(edge_id)
    assert edge.visit_count == 2
    assert edge.value_sum == pytest.approx(2.0)
    assert edge.value_mean == pytest.approx(1.0)


def test_extract_best_path_prefers_unvisited_neighbor():
    store = StateGraphStore()
    a = _matrix_n3()
    a_id, _ = store.add_state(a)

    b = mutate(a, 1)
    b_id, edge_ab, _, _ = store.add_mutation_transition(a_id, 1, b)

    c = mutate(b, 2)
    c_id, edge_bc, _, _ = store.add_mutation_transition(b_id, 2, c)

    # Make AB appear better than BC; extraction should still avoid revisiting A at B.
    store.update_edge_stats(edge_ab, 1.0)
    store.update_edge_stats(edge_bc, 0.1)

    path = store.extract_best_path(a_id, max_hops=4)
    assert path == [a_id, b_id, c_id]


def test_won_state_is_marked_solved_on_insert():
    store = StateGraphStore()
    B_A = make_exchange_matrix(2, [(1, 2)])
    coframed = make_coframed(B_A)
    state_id, created = store.add_state(coframed)

    assert created
    node = store.get_state(state_id)
    assert node.is_won
    assert node.status == "solved"


def test_invalid_action_vertex_raises():
    store = StateGraphStore()
    matrix = _matrix_n2()
    state_id, _ = store.add_state(matrix)
    next_id, _, _, _ = store.add_mutation_transition(state_id, 1, mutate(matrix, 1))

    with pytest.raises(ValueError):
        store.add_undirected_edge(state_id, next_id, 0)
    with pytest.raises(ValueError):
        store.add_undirected_edge(state_id, next_id, 3)


def test_add_mutation_transition_sets_created_path_actions():
    store = StateGraphStore()
    root = _matrix_n2()
    root_id, _ = store.add_state(root, created_iter=0)

    next_id, _, _, _ = store.add_mutation_transition(
        root_id,
        1,
        mutate(root, 1),
        created_iter=1,
    )
    node = store.get_state(next_id)
    assert node.created_iter == 1
    assert node.shortest_path_actions == [1]
