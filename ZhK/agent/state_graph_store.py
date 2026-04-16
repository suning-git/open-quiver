"""State graph store for mutation search.

This module stores quiver states as graph nodes and mutation relations as
undirected edges. The undirected choice is valid because mutation is
involutive: applying the same mutation vertex twice returns to the previous
state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np

from common.quiver.mutation import Matrix, get_colors, is_all_red


@dataclass
class _StateNode:
    """One unique exchange-matrix state."""

    state_id: str
    matrix_hash: str
    matrix: Matrix
    n: int
    red_count: int
    is_won: bool
    colors: dict[int, str]
    created_iter: int
    visit_count: int = 0
    value_sum: float = 0.0
    value_mean: float = 0.0
    status: str = "unexpanded"
    shortest_path_actions: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _MutationEdge:
    """Undirected relation between two states induced by a mutation vertex."""

    edge_id: str
    state_a_id: str
    state_b_id: str
    action_vertex: int
    created_iter: int
    visit_count: int = 0
    value_sum: float = 0.0
    value_mean: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def other(self, state_id: str) -> str:
        """Return the opposite endpoint."""
        if state_id == self.state_a_id:
            return self.state_b_id
        if state_id == self.state_b_id:
            return self.state_a_id
        raise ValueError(f"State {state_id} is not incident to edge {self.edge_id}.")


class StateGraphStore:
    """Graph storage for mutation search.

    - States are deduplicated by matrix hash.
    - Mutation edges are stored as undirected relations with a canonical key:
      (min_state_id, max_state_id, action_vertex)
    """

    def __init__(self) -> None:
        self._states_by_id: dict[str, _StateNode] = {}
        self._state_id_by_hash: dict[str, str] = {}
        self._edges_by_id: dict[str, _MutationEdge] = {}
        self._edge_id_by_key: dict[tuple[str, str, int], str] = {}
        self._incident_edge_ids: dict[str, set[str]] = {}

    @property
    def state_count(self) -> int:
        return len(self._states_by_id)

    @property
    def edge_count(self) -> int:
        return len(self._edges_by_id)

    def list_state_ids(self) -> list[str]:
        """Return all state ids in insertion order."""
        return list(self._states_by_id.keys())

    def list_edge_ids(self) -> list[str]:
        """Return all edge ids in insertion order."""
        return list(self._edges_by_id.keys())

    def list_states(self) -> list[_StateNode]:
        """Return all state objects in insertion order."""
        return list(self._states_by_id.values())

    def list_states_by_created_iter(self) -> list[_StateNode]:
        """Return all state objects sorted by (created_iter, state_id)."""
        return sorted(
            self._states_by_id.values(),
            key=lambda s: (s.created_iter, s.state_id),
        )

    def list_edges(self) -> list[_MutationEdge]:
        """Return all edge objects in insertion order."""
        return list(self._edges_by_id.values())

    def add_state(
        self,
        matrix: Matrix,
        *,
        created_iter: int = 0,
        shortest_path_actions: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Insert a state if absent.

        Returns:
            (state_id, created)
        """
        mat = self._normalize_matrix(matrix)
        n = mat.shape[0]
        if mat.shape[1] != 2 * n:
            raise ValueError(
                f"State matrix must be n x 2n, got shape={mat.shape}."
            )

        matrix_hash = self._matrix_hash(mat)
        existing = self._state_id_by_hash.get(matrix_hash)
        if existing is not None:
            if shortest_path_actions is not None:
                node = self._states_by_id[existing]
                if (
                    not node.shortest_path_actions
                    or len(shortest_path_actions) < len(node.shortest_path_actions)
                ):
                    node.shortest_path_actions = list(shortest_path_actions)
            return existing, False

        state_id = f"s_{matrix_hash[:16]}"
        colors = get_colors(mat)
        red_count = sum(1 for c in colors.values() if c == "red")
        won = bool(is_all_red(mat))
        status = "solved" if won else "unexpanded"

        node = _StateNode(
            state_id=state_id,
            matrix_hash=matrix_hash,
            matrix=mat,
            n=n,
            red_count=red_count,
            is_won=won,
            colors=colors,
            created_iter=created_iter,
            status=status,
            shortest_path_actions=list(shortest_path_actions or []),
            metadata=dict(metadata or {}),
        )
        self._states_by_id[state_id] = node
        self._state_id_by_hash[matrix_hash] = state_id
        self._incident_edge_ids[state_id] = set()
        return state_id, True

    def add_undirected_edge(
        self,
        state_a_id: str,
        state_b_id: str,
        action_vertex: int,
        *,
        created_iter: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Insert an undirected mutation edge if absent.

        Returns:
            (edge_id, created)
        """
        node_a = self.get_state(state_a_id)
        node_b = self.get_state(state_b_id)
        if node_a.n != node_b.n:
            raise ValueError(
                f"Cannot connect states with different n: {node_a.n} vs {node_b.n}."
            )
        if not (1 <= action_vertex <= node_a.n):
            raise ValueError(
                f"Vertex {action_vertex} is not valid for n={node_a.n}."
            )

        key = self._canonical_edge_key(state_a_id, state_b_id, action_vertex)
        existing = self._edge_id_by_key.get(key)
        if existing is not None:
            return existing, False

        edge_hash = hashlib.sha1(
            f"{key[0]}|{key[1]}|{action_vertex}".encode("utf-8")
        ).hexdigest()
        edge_id = f"e_{edge_hash[:16]}"
        edge = _MutationEdge(
            edge_id=edge_id,
            state_a_id=key[0],
            state_b_id=key[1],
            action_vertex=action_vertex,
            created_iter=created_iter,
            metadata=dict(metadata or {}),
        )
        self._edges_by_id[edge_id] = edge
        self._edge_id_by_key[key] = edge_id
        self._incident_edge_ids[key[0]].add(edge_id)
        self._incident_edge_ids[key[1]].add(edge_id)
        return edge_id, True

    def add_mutation_transition(
        self,
        from_state_id: str,
        action_vertex: int,
        to_matrix: Matrix,
        *,
        created_iter: int = 0,
        to_state_metadata: dict[str, Any] | None = None,
        edge_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str, bool, bool]:
        """Convenience API: add destination state + undirected edge.

        Returns:
            (to_state_id, edge_id, state_created, edge_created)
        """
        if from_state_id not in self._states_by_id:
            raise KeyError(f"Unknown source state: {from_state_id}")
        from_node = self._states_by_id[from_state_id]
        candidate_path = list(from_node.shortest_path_actions) + [action_vertex]

        to_state_id, state_created = self.add_state(
            to_matrix,
            created_iter=created_iter,
            shortest_path_actions=candidate_path,
            metadata=to_state_metadata,
        )
        edge_id, edge_created = self.add_undirected_edge(
            from_state_id,
            to_state_id,
            action_vertex,
            created_iter=created_iter,
            metadata=edge_metadata,
        )

        if from_node.status == "unexpanded":
            from_node.status = "expanded"
        
        if self._is_fully_expanded(from_state_id):
            from_node.status = "fully_expanded"

        return to_state_id, edge_id, state_created, edge_created

    def update_state_stats(self, state_id: str, reward: float) -> None:
        node = self.get_state(state_id)
        node.visit_count += 1
        node.value_sum += float(reward)
        node.value_mean = node.value_sum / node.visit_count

    def update_edge_stats(self, edge_id: str, reward: float) -> None:
        edge = self.get_edge(edge_id)
        edge.visit_count += 1
        edge.value_sum += float(reward)
        edge.value_mean = edge.value_sum / edge.visit_count

    def mark_state_solved(self, state_id: str) -> None:
        node = self.get_state(state_id)
        node.status = "solved"

    def mark_state_dead_end(self, state_id: str) -> None:
        node = self.get_state(state_id)
        if node.status != "solved":
            node.status = "dead_end"

    def get_state(self, state_id: str) -> _StateNode:
        try:
            return self._states_by_id[state_id]
        except KeyError as e:
            raise KeyError(f"Unknown state_id: {state_id}") from e

    def get_edge(self, edge_id: str) -> _MutationEdge:
        try:
            return self._edges_by_id[edge_id]
        except KeyError as e:
            raise KeyError(f"Unknown edge_id: {edge_id}") from e

    def neighbors(self, state_id: str) -> list[tuple[str, str]]:
        """Return [(neighbor_state_id, edge_id), ...] sorted by edge id."""
        _ = self.get_state(state_id)
        result: list[tuple[str, str]] = []
        for edge_id in sorted(self._incident_edge_ids[state_id]):
            edge = self._edges_by_id[edge_id]
            result.append((edge.other(state_id), edge_id))
        return result

    def extract_best_path(
        self,
        root_state_id: str,
        *,
        max_hops: int = 64,
        stop_on_solved: bool = True,
    ) -> list[str]:
        """Greedy path extraction by edge stats with no revisits."""
        current = self.get_state(root_state_id).state_id
        path = [current]
        visited = {current}

        for _ in range(max_hops):
            current_node = self._states_by_id[current]
            if stop_on_solved and current_node.is_won:
                break

            candidates: list[tuple[bool, float, int, int, str]] = []
            for edge_id in sorted(self._incident_edge_ids[current]):
                edge = self._edges_by_id[edge_id]
                nxt = edge.other(current)
                nxt_node = self._states_by_id[nxt]
                candidates.append(
                    (
                        nxt not in visited,
                        edge.value_mean,
                        edge.visit_count,
                        nxt_node.red_count,
                        nxt,
                    )
                )

            if not candidates:
                break

            candidates.sort(reverse=True)
            chosen = candidates[0][4]
            if chosen in visited:
                break

            path.append(chosen)
            visited.add(chosen)
            current = chosen

        return path
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph store into JSON-compatible dict."""
        states: list[dict[str, Any]] = []
        for node in self.list_states():
            states.append(
                {
                    "state_id": node.state_id,
                    "matrix_hash": node.matrix_hash,
                    "matrix": np.asarray(node.matrix, dtype=np.int64).tolist(),
                    "n": node.n,
                    "red_count": node.red_count,
                    "is_won": node.is_won,
                    "colors": {str(k): v for k, v in node.colors.items()},
                    "created_iter": node.created_iter,
                    "visit_count": node.visit_count,
                    "value_sum": node.value_sum,
                    "value_mean": node.value_mean,
                    "status": node.status,
                    "shortest_path_actions": list(node.shortest_path_actions),
                    "metadata": dict(node.metadata),
                }
            )

        edges: list[dict[str, Any]] = []
        for edge in self.list_edges():
            edges.append(
                {
                    "edge_id": edge.edge_id,
                    "state_a_id": edge.state_a_id,
                    "state_b_id": edge.state_b_id,
                    "action_vertex": edge.action_vertex,
                    "created_iter": edge.created_iter,
                    "visit_count": edge.visit_count,
                    "value_sum": edge.value_sum,
                    "value_mean": edge.value_mean,
                    "metadata": dict(edge.metadata),
                }
            )

        return {
            "states": states,
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateGraphStore":
        """Restore a graph store from to_dict() payload."""
        store = cls()

        for item in payload.get("states", []):
            matrix = np.asarray(item["matrix"], dtype=np.int64)
            node = _StateNode(
                state_id=item["state_id"],
                matrix_hash=item["matrix_hash"],
                matrix=matrix,
                n=int(item["n"]),
                red_count=int(item["red_count"]),
                is_won=bool(item["is_won"]),
                colors={int(k): str(v) for k, v in item.get("colors", {}).items()},
                created_iter=int(item["created_iter"]),
                visit_count=int(item.get("visit_count", 0)),
                value_sum=float(item.get("value_sum", 0.0)),
                value_mean=float(item.get("value_mean", 0.0)),
                status=str(item.get("status", "unexpanded")),
                shortest_path_actions=[int(k) for k in item.get("shortest_path_actions", [])],
                metadata=dict(item.get("metadata", {})),
            )
            store._states_by_id[node.state_id] = node
            store._state_id_by_hash[node.matrix_hash] = node.state_id
            store._incident_edge_ids[node.state_id] = set()

        for item in payload.get("edges", []):
            edge = _MutationEdge(
                edge_id=item["edge_id"],
                state_a_id=item["state_a_id"],
                state_b_id=item["state_b_id"],
                action_vertex=int(item["action_vertex"]),
                created_iter=int(item["created_iter"]),
                visit_count=int(item.get("visit_count", 0)),
                value_sum=float(item.get("value_sum", 0.0)),
                value_mean=float(item.get("value_mean", 0.0)),
                metadata=dict(item.get("metadata", {})),
            )
            store._edges_by_id[edge.edge_id] = edge
            key = store._canonical_edge_key(edge.state_a_id, edge.state_b_id, edge.action_vertex)
            store._edge_id_by_key[key] = edge.edge_id
            if edge.state_a_id in store._incident_edge_ids:
                store._incident_edge_ids[edge.state_a_id].add(edge.edge_id)
            if edge.state_b_id in store._incident_edge_ids:
                store._incident_edge_ids[edge.state_b_id].add(edge.edge_id)

        return store
        
    def _normalize_matrix(self, matrix: Matrix) -> Matrix:
        arr = np.asarray(matrix, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError(f"State matrix must be 2-D, got ndim={arr.ndim}.")
        # Keep C-contiguous for stable hashing and predictable serialization later.
        return np.array(arr, dtype=np.int64, copy=True, order="C")

    def _matrix_hash(self, matrix: Matrix) -> str:
        payload = (
            f"{matrix.dtype}|{matrix.shape[0]}|{matrix.shape[1]}|".encode("utf-8")
            + matrix.tobytes()
        )
        return hashlib.sha256(payload).hexdigest()

    def _canonical_edge_key(
        self,
        state_a_id: str,
        state_b_id: str,
        action_vertex: int,
    ) -> tuple[str, str, int]:
        if state_a_id <= state_b_id:
            return (state_a_id, state_b_id, action_vertex)
        return (state_b_id, state_a_id, action_vertex)
    
    def _is_fully_expanded(self, state_id: str) -> bool:
        node = self.get_state(state_id)
        expanded_actions = {
            self.get_edge(edge_id).action_vertex
            for _, edge_id in self.neighbors(state_id)
        }
        return len(expanded_actions) >= node.n

