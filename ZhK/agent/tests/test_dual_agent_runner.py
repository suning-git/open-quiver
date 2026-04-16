import pytest
import re

from common.quiver.mutation import make_exchange_matrix, mutate
from ZhK.agent.dual_agent_runner import DualAgentSearchRunner, SearchConfig
from ZhK.agent.llm_provider import MockProvider


class _RecorderProvider:
    def __init__(self, replies: list[str] | None = None):
        self.replies = replies or ["1"]
        self.calls: list[list[dict]] = []
        self._call_count = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        idx = min(self._call_count, len(self.replies) - 1)
        self._call_count += 1
        return self.replies[idx]


class _SelectorLatestThenMutatorScriptProvider:
    """Selector picks latest candidate id; mutator follows a scripted list."""

    def __init__(self, mutator_replies: list[str]):
        self._mutator_replies = list(mutator_replies)
        self._mutator_idx = 0

    def chat(self, messages: list[dict]) -> str:
        system = messages[0]["content"].lower()
        if "state selector" in system:
            user_text = messages[-1]["content"]
            line = next(
                (ln for ln in user_text.splitlines() if ln.startswith("Candidate IDs to expand now:")),
                "",
            )
            ids = [int(x) for x in re.findall(r"\d+", line)]
            return str(max(ids)) if ids else "0"

        idx = min(self._mutator_idx, len(self._mutator_replies) - 1)
        self._mutator_idx += 1
        return self._mutator_replies[idx]


def test_runner_can_solve_linear_1_to_2():
    B_A = make_exchange_matrix(2, [(1, 2)])
    provider = MockProvider(["0", "1", "1", "2"])
    runner = DualAgentSearchRunner(provider, SearchConfig(max_iterations=10))

    result = runner.run_from_exchange_matrix(B_A)

    assert result.won
    assert result.reason == "won"
    assert result.expansions >= 2
    assert len(result.best_path) >= 2


def test_runner_continues_after_parse_failure():
    B_A = make_exchange_matrix(2, [(1, 2)])
    # Iter 1: parse failure after retries. Iter 2+: recover and solve.
    provider = _SelectorLatestThenMutatorScriptProvider(["bad", "still bad", "1", "2"])
    runner = DualAgentSearchRunner(
        provider,
        SearchConfig(max_iterations=6, max_parse_retries=1),
    )

    result = runner.run_from_exchange_matrix(B_A)

    assert result.won
    assert result.parse_failures >= 1
    assert result.expansions >= 2


def test_runner_respects_undirected_edge_storage_on_reverse_move():
    B_A = make_exchange_matrix(2, [(1, 2)])
    provider = MockProvider(["0", "1", "1", "1"])
    runner = DualAgentSearchRunner(provider, SearchConfig(max_iterations=2))

    result = runner.run_from_exchange_matrix(B_A)

    assert not result.won
    assert result.reason == "max_iterations"
    # root --(1)-- state1, then reverse via same action should reuse the same edge
    assert runner.store.state_count == 2
    assert runner.store.edge_count == 1


def test_incremental_api_runs_until_done():
    B_A = make_exchange_matrix(2, [(1, 2)])
    provider = MockProvider(["0", "1", "1", "2"])
    runner = DualAgentSearchRunner(provider, SearchConfig(max_iterations=10))

    root_id = runner.initialize_from_exchange_matrix(B_A)
    assert root_id.startswith("s_")
    assert not runner.is_done()

    # Step-by-step execution
    while not runner.is_done():
        step = runner.run_one_iteration()
        assert step.iteration >= 1

    snap = runner.get_result_snapshot()
    assert snap.won
    assert snap.reason == "won"
    assert snap.root_state_id == root_id


def test_run_one_iteration_requires_initialize():
    provider = MockProvider(["0", "1"])
    runner = DualAgentSearchRunner(provider, SearchConfig(max_iterations=1))

    with pytest.raises(RuntimeError):
        runner.run_one_iteration()


def test_snapshot_is_in_progress_before_done():
    B_A = make_exchange_matrix(2, [(1, 2)])
    provider = MockProvider(["0", "1", "1", "2"])
    runner = DualAgentSearchRunner(provider, SearchConfig(max_iterations=10))
    runner.initialize_from_exchange_matrix(B_A)

    snap0 = runner.get_result_snapshot()
    assert snap0.reason == "in_progress"
    assert not snap0.won

    runner.run_one_iteration()
    snap1 = runner.get_result_snapshot()
    assert snap1.reason in {"in_progress", "won"}


def test_mutation_prompt_uses_linear_style_user_message():
    B_A = make_exchange_matrix(2, [(1, 2)])
    provider = _RecorderProvider(replies=["0", "1"])
    runner = DualAgentSearchRunner(provider, SearchConfig(max_iterations=1))
    runner.initialize_from_exchange_matrix(B_A)
    runner.run_one_iteration()

    assert provider.calls, "Expected at least one LLM call."
    mutation_call = next(
        c
        for c in provider.calls
        if c and c[0]["role"] == "system" and "mutation proposer" in c[0]["content"].lower()
    )
    assert mutation_call[1]["role"] == "user"
    user_text = mutation_call[1]["content"]
    assert "Step " in user_text
    assert "Vertices:" in user_text
    assert "Edges:" in user_text
    assert "Your move?" in user_text


def test_mutation_prompt_contains_already_expanded_actions():
    B_A = make_exchange_matrix(2, [(1, 2)])
    provider = _RecorderProvider(replies=["2"])
    runner = DualAgentSearchRunner(provider, SearchConfig(max_iterations=1))
    root_id = runner.initialize_from_exchange_matrix(B_A)

    root = runner.store.get_state(root_id)
    runner.store.add_mutation_transition(root_id, 1, mutate(root.matrix, 1), created_iter=1)

    action, _ = runner._ask_action_for_state(root_id)  # noqa: SLF001
    assert action == 2

    first_call = provider.calls[0]
    user_text = first_call[1]["content"]
    assert "Already-expanded actions from this state:" in user_text
    assert "u1" in user_text
