from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from common.quiver.mutation import make_framed, matrix_to_edges, mutate
from .graph_search_initial_prompts import (
    get_graph_search_mutator_prompt,
    get_graph_search_state_selector_prompt,
)
from .harness import build_user_message, format_error, parse_action, render_state
from .llm_provider import LLMProvider
from .state_graph_store import StateGraphStore


@dataclass
class SearchConfig:
    """Config for phase-1 search."""

    max_iterations: int = 50
    max_parse_retries: int = 2
    exploration_c: float = 1.4
    solved_reward: float = 1.0
    progress_reward_weight: float = 0.2
    stagnation_penalty: float = -0.05
    parse_failure_penalty: float = -0.2
    revisit_penalty: float = -0.05


@dataclass
class SearchResult:
    """Summary of one search run."""

    won: bool
    reason: str
    iterations: int
    root_state_id: str
    best_state_id: str
    best_path: list[str]
    parse_failures: int
    expansions: int


@dataclass
class IterationResult:
    """Result of one incremental iteration."""

    iteration: int
    selected_state_id: str | None = None
    action: int | None = None
    to_state_id: str | None = None
    edge_id: str | None = None
    reward: float | None = None
    state_created: bool | None = None
    parse_failed: bool = False
    done: bool = False
    reason: str = ""
    llm_messages: list[dict] = field(default_factory=list)


class DualAgentSearchRunner:
    """Search loop with LLM state selection + LLM mutation selection."""

    def __init__(self, provider: LLMProvider, config: SearchConfig | None = None):
        self._provider = provider
        self._cfg = config or SearchConfig()
        self._mutator_system_prompt = get_graph_search_mutator_prompt()
        self._selector_system_prompt = get_graph_search_state_selector_prompt()
        self.store = StateGraphStore()
        self._root_state_id: str | None = None
        self._iterations = 0
        self._parse_failures = 0
        self._expansions = 0
        self._done = False
        self._reason = ""
        self._selector_messages: list[dict] = []

    def initialize_from_exchange_matrix(self, B_A: NDArray[np.int64]) -> str:
        self.store = StateGraphStore()
        self._iterations = 0
        self._parse_failures = 0
        self._expansions = 0
        self._done = False
        self._reason = ""
        self._selector_messages = [{"role": "system", "content": self._selector_system_prompt}]

        root_matrix = make_framed(np.asarray(B_A, dtype=np.int64))
        root_state_id, _ = self.store.add_state(root_matrix, created_iter=0)
        self._root_state_id = root_state_id

        if self._has_solved_state():
            self._done = True
            self._reason = "won"
        return root_state_id

    def is_done(self) -> bool:
        return self._done

    def export_search(self) -> dict[str, Any]:
        self._ensure_initialized()
        return {
            "version": 1,
            "config": {
                "max_iterations": self._cfg.max_iterations,
                "max_parse_retries": self._cfg.max_parse_retries,
                "exploration_c": self._cfg.exploration_c,
                "solved_reward": self._cfg.solved_reward,
                "progress_reward_weight": self._cfg.progress_reward_weight,
                "stagnation_penalty": self._cfg.stagnation_penalty,
                "parse_failure_penalty": self._cfg.parse_failure_penalty,
                "revisit_penalty": self._cfg.revisit_penalty,
            },
            "store": self.store.to_dict(),
            "root_state_id": self._root_state_id,
            "iterations": self._iterations,
            "parse_failures": self._parse_failures,
            "expansions": self._expansions,
            "done": self._done,
            "reason": self._reason,
            "selector_messages": [dict(m) for m in self._selector_messages],
            "prompts": {
                "mutator_system_prompt": self._mutator_system_prompt,
                "selector_system_prompt": self._selector_system_prompt,
            },
        }

    def import_search(self, payload: dict[str, Any]) -> None:
        self.store = StateGraphStore.from_dict(dict(payload["store"]))
        self._root_state_id = payload.get("root_state_id")
        self._iterations = int(payload.get("iterations", 0))
        self._parse_failures = int(payload.get("parse_failures", 0))
        self._expansions = int(payload.get("expansions", 0))
        self._done = bool(payload.get("done", False))
        self._reason = str(payload.get("reason", ""))
        self._selector_messages = [dict(m) for m in payload.get("selector_messages", [])]

        prompts = payload.get("prompts", {})
        self._mutator_system_prompt = str(
            prompts.get("mutator_system_prompt", get_graph_search_mutator_prompt())
        )
        self._selector_system_prompt = str(
            prompts.get("selector_system_prompt", get_graph_search_state_selector_prompt())
        )
        if not self._selector_messages:
            self._selector_messages = [{"role": "system", "content": self._selector_system_prompt}]

        cfg_payload = payload.get("config", {})
        self._cfg = SearchConfig(
            max_iterations=int(cfg_payload.get("max_iterations", self._cfg.max_iterations)),
            max_parse_retries=int(cfg_payload.get("max_parse_retries", self._cfg.max_parse_retries)),
            exploration_c=float(cfg_payload.get("exploration_c", self._cfg.exploration_c)),
            solved_reward=float(cfg_payload.get("solved_reward", self._cfg.solved_reward)),
            progress_reward_weight=float(
                cfg_payload.get("progress_reward_weight", self._cfg.progress_reward_weight)
            ),
            stagnation_penalty=float(cfg_payload.get("stagnation_penalty", self._cfg.stagnation_penalty)),
            parse_failure_penalty=float(
                cfg_payload.get("parse_failure_penalty", self._cfg.parse_failure_penalty)
            ),
            revisit_penalty=float(cfg_payload.get("revisit_penalty", self._cfg.revisit_penalty)),
        )

    def get_result_snapshot(self) -> SearchResult:
        self._ensure_initialized()
        root_state_id = self._root_state_id
        assert root_state_id is not None
        best_state_id = self._best_state_id()
        best_path = self.store.extract_best_path(root_state_id)
        won = self._has_solved_state()
        reason = self._reason if self._done else "in_progress"
        return SearchResult(
            won=won,
            reason=reason,
            iterations=self._iterations,
            root_state_id=root_state_id,
            best_state_id=best_state_id,
            best_path=best_path,
            parse_failures=self._parse_failures,
            expansions=self._expansions,
        )

    def run_from_exchange_matrix(self, B_A: NDArray[np.int64]) -> SearchResult:
        self.initialize_from_exchange_matrix(B_A)
        while not self.is_done():
            self.run_one_iteration()
        return self.get_result_snapshot()

    def run_one_iteration(self) -> IterationResult:
        self._ensure_initialized()

        if self._done:
            return IterationResult(iteration=self._iterations, done=True, reason=self._reason)

        if self._iterations >= self._cfg.max_iterations:
            self._done = True
            self._reason = "max_iterations"
            return IterationResult(iteration=self._iterations, done=True, reason=self._reason)

        self._iterations += 1
        it = self._iterations

        selected, selector_messages = self._select_state_id()
        if selected is None:
            self._done = True
            self._reason = "no_selectable_state"
            return IterationResult(iteration=it, done=True, reason=self._reason, llm_messages=selector_messages)

        action, mutator_messages = self._ask_action_for_state(selected)
        llm_messages = selector_messages + mutator_messages
        if action is None:
            self._parse_failures += 1
            self.store.update_state_stats(selected, self._cfg.parse_failure_penalty)
            if self._iterations >= self._cfg.max_iterations:
                self._done = True
                self._reason = "max_iterations"
            return IterationResult(
                iteration=it,
                selected_state_id=selected,
                parse_failed=True,
                done=self._done,
                reason=self._reason,
                llm_messages=llm_messages,
            )

        src = self.store.get_state(selected)
        to_matrix = mutate(src.matrix, action)
        to_state_id, edge_id, state_created, _ = self.store.add_mutation_transition(
            selected,
            action,
            to_matrix,
            created_iter=it,
        )
        dst = self.store.get_state(to_state_id)

        reward = self._compute_reward(src.red_count, dst.red_count, dst.is_won, state_created)
        self.store.update_edge_stats(edge_id, reward)
        self.store.update_state_stats(selected, reward)
        self.store.update_state_stats(to_state_id, reward)
        self._expansions += 1

        if dst.is_won:
            self._done = True
            self._reason = "won"
        elif self._iterations >= self._cfg.max_iterations:
            self._done = True
            self._reason = "max_iterations"

        return IterationResult(
            iteration=it,
            selected_state_id=selected,
            action=action,
            to_state_id=to_state_id,
            edge_id=edge_id,
            reward=reward,
            state_created=state_created,
            parse_failed=False,
            done=self._done,
            reason=self._reason,
            llm_messages=llm_messages,
        )

    def _select_state_id(self) -> tuple[str | None, list[dict]]:
        candidates = [
            s for s in self.store.list_states() if s.status != "solved" and s.status != "dead_end" and s.status != "fully_expanded"
        ]
        if not candidates:
            return None, []

        user_msg = self._build_selector_user_message(candidates)
        self._selector_messages.append({"role": "user", "content": user_msg})
        round_messages: list[dict] = [{"role": "user", "content": user_msg}]

        candidate_id_map = {s.created_iter: s.state_id for s in candidates}

        for _ in range(self._cfg.max_parse_retries + 1):
            response = self._provider.chat(self._selector_messages)
            self._selector_messages.append({"role": "assistant", "content": response})
            round_messages.append({"role": "assistant", "content": response})
            created_iter = self._parse_selected_created_iter(response, candidate_id_map)
            if created_iter is not None:
                return candidate_id_map[created_iter], round_messages

            hint = ", ".join(str(i) for i in sorted(candidate_id_map))
            error_msg = (
                f"Please select one candidate id from: {hint}. "
                "You may explain briefly, and end with `Selected ID: <id>`."
            )
            self._selector_messages.append({"role": "user", "content": error_msg})
            round_messages.append({"role": "user", "content": error_msg})

        return self._select_state_id_heuristic(candidates), round_messages

    def _select_state_id_heuristic(self, candidates: list) -> str:
        if not candidates:
            raise ValueError("No candidates available for heuristic selection.")

        total_visits = sum(s.visit_count for s in candidates)
        log_term = math.log(total_visits + 2.0)

        best_score = float("-inf")
        best: list[tuple[int, int, str]] = []
        for node in candidates:
            ucb = self._cfg.exploration_c * math.sqrt(log_term / (node.visit_count + 1.0))
            score = node.value_mean + ucb
            if score > best_score:
                best_score = score
                best = [(node.red_count, -node.created_iter, node.state_id)]
            elif score == best_score:
                best.append((node.red_count, -node.created_iter, node.state_id))

        best.sort(reverse=True)
        return best[0][2]

    def _build_selector_user_message(self, candidates: list) -> str:
        all_states = self.store.list_states_by_created_iter()
        candidate_ids = sorted(s.created_iter for s in candidates)

        lines: list[str] = []
        lines.append("State graph snapshot (all known states):")
        lines.append("")

        for node in all_states:
        # for node in candidates:
            payload = {
                "total_mutable": node.n,
                "colors": node.colors,
                "edges": matrix_to_edges(node.matrix),
                "red_count": node.red_count,
                "step": node.created_iter,
                "move_history": list(node.shortest_path_actions),
            }
            path_str = self._format_action_path(node.shortest_path_actions)
            lines.append(f"State ID: {node.created_iter}")
            lines.append(f"Created shortest path: {path_str}")
            lines.append(render_state(payload))
            lines.append("")

        lines.append("Candidate IDs to expand now: " + ", ".join(str(i) for i in candidate_ids))
        lines.append("You may include brief reasoning, but end with a line like: Selected ID: <candidate_id>.")
        return "\n".join(lines)

    def _parse_selected_created_iter(self, text: str, candidate_id_map: dict[int, str]) -> int | None:
        numbers = re.findall(r"\d+", text)
        for num_str in reversed(numbers):
            cid = int(num_str)
            if cid in candidate_id_map:
                return cid
        return None

    def _ask_action_for_state(self, state_id: str) -> tuple[int | None, list[dict]]:
        node = self.store.get_state(state_id)
        payload = {
            "total_mutable": node.n,
            "colors": node.colors,
            "edges": matrix_to_edges(node.matrix),
            "red_count": node.red_count,
            "step": node.created_iter,
            "move_history": [],
        }
        expanded_actions = sorted(
            {self.store.get_edge(edge_id).action_vertex for _, edge_id in self.store.neighbors(state_id)}
        )
        if expanded_actions:
            action_line = "Already-expanded actions from this state: " + ", ".join(
                f"u{k}" for k in expanded_actions
            )
        else:
            action_line = "Already-expanded actions from this state: (none)"

        user_msg = (
            build_user_message(payload)
            + "\n\n"
            + action_line
            + "\nYou must choose an unexplored action if available."
        )
        messages = [
            {"role": "system", "content": self._mutator_system_prompt},
            {"role": "user", "content": user_msg},
        ]

        for _ in range(self._cfg.max_parse_retries + 1):
            response = self._provider.chat(messages)
            messages.append({"role": "assistant", "content": response})
            action = parse_action(response, node.n)
            if action is not None:
                return action, messages
            messages.append({"role": "user", "content": format_error(response, node.n)})

        return None, messages

    def _compute_reward(self, red_before: int, red_after: int, is_won: bool, state_created: bool) -> float:
        if is_won:
            return self._cfg.solved_reward
        delta_red = red_after - red_before
        reward = self._cfg.progress_reward_weight * float(delta_red)
        if delta_red <= 0:
            reward += self._cfg.stagnation_penalty
        if not state_created:
            reward += self._cfg.revisit_penalty
        return reward

    def _has_solved_state(self) -> bool:
        return any(s.is_won for s in self.store.list_states())

    def _best_state_id(self) -> str:
        ranked = sorted(
            self.store.list_states(),
            key=lambda s: (s.is_won, s.red_count, s.value_mean, s.visit_count, s.state_id),
            reverse=True,
        )
        return ranked[0].state_id

    def _ensure_initialized(self) -> None:
        if self._root_state_id is None:
            raise RuntimeError("Runner is not initialized. Call initialize_from_exchange_matrix() first.")

    def _format_action_path(self, actions: list[int]) -> str:
        if not actions:
            return "[]"
        return "[" + " ".join(str(k) for k in actions) + "]"
