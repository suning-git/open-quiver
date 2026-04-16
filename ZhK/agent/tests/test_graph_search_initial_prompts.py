from ZhK.agent.dual_agent_runner import DualAgentSearchRunner
from ZhK.agent.graph_search_initial_prompts import (
    get_graph_search_mutator_prompt,
    get_graph_search_state_selector_prompt,
)
from ZhK.agent.llm_provider import MockProvider


def test_graph_search_mutator_prompt_contains_mode_overview():
    prompt = get_graph_search_mutator_prompt()
    assert "Mode Overview" in prompt
    assert "state graph" in prompt.lower()
    assert "single integer" in prompt.lower()


def test_graph_search_state_selector_prompt_contains_selector_role():
    prompt = get_graph_search_state_selector_prompt()
    assert "state selector" in prompt.lower()
    assert "candidate ids" in prompt.lower()
    assert "mutation" in prompt.lower()


def test_runner_uses_graph_search_mutator_prompt():
    runner = DualAgentSearchRunner(MockProvider.from_actions([1]))
    assert "graph-search mode" in runner._mutator_system_prompt.lower()  # noqa: SLF001
    assert "state selector" in runner._selector_system_prompt.lower()  # noqa: SLF001
