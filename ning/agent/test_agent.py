"""Tests for the agent game loop using MockProvider."""

from .agent import run_game
from .llm_provider import MockProvider


class TestRunGameWin:
    def test_n2_known_sequence(self):
        """n=2, graph 1→2. Known solution: μ_1, μ_2."""
        provider = MockProvider.from_actions([1, 2])
        result = run_game(2, [(1, 2)], provider)
        assert result.won
        assert result.reason == "won"
        assert result.move_history == [1, 2]
        assert result.steps == 2

    def test_n3_known_sequence(self):
        """n=3, graph 1→2→3. Known solution: μ_1, μ_2, μ_3."""
        provider = MockProvider.from_actions([1, 2, 3])
        result = run_game(3, [(1, 2), (2, 3)], provider)
        assert result.won
        assert result.reason == "won"
        assert result.move_history == [1, 2, 3]


class TestRunGameMaxSteps:
    def test_exceeds_max_steps(self):
        """Repeating μ_1 on 1→2→3 never wins; should stop at max_steps."""
        provider = MockProvider.from_actions([1])  # cycles: always pick 1
        result = run_game(3, [(1, 2), (2, 3)], provider, max_steps=10)
        assert not result.won
        assert result.reason == "max_steps"
        assert result.steps <= 10


class TestRunGameParseRetry:
    def test_retry_then_succeed(self):
        """First response is garbage, second is valid."""
        provider = MockProvider(["I don't know", "1", "2"])
        result = run_game(2, [(1, 2)], provider)
        assert result.won
        assert result.move_history == [1, 2]
        # Should have error message in conversation
        assert any("Could not parse" in m["content"] for m in result.messages if m["role"] == "user")

    def test_all_retries_fail(self):
        """All responses are garbage; should abort."""
        provider = MockProvider(["nope", "still nope", "???", "!!!", "no"])
        result = run_game(2, [(1, 2)], provider, max_retries=2)
        assert not result.won
        assert result.reason == "parse_failure"


class TestRunGameMessages:
    def test_message_structure(self):
        """Verify message roles follow valid patterns."""
        provider = MockProvider.from_actions([1, 2])
        result = run_game(2, [(1, 2)], provider)

        assert result.messages[0]["role"] == "system"
        assert result.messages[1]["role"] == "user"
        # After system+user, roles must alternate assistant/user
        for i in range(2, len(result.messages)):
            prev = result.messages[i - 1]["role"]
            curr = result.messages[i]["role"]
            if prev == "user":
                assert curr == "assistant", f"Expected assistant after user at index {i}"
            else:
                assert curr == "user", f"Expected user after assistant at index {i}"

    def test_message_structure_with_retry(self):
        """Role alternation holds even when retries insert extra exchanges."""
        provider = MockProvider(["garbage", "1", "2"])
        result = run_game(2, [(1, 2)], provider)
        for i in range(2, len(result.messages)):
            prev = result.messages[i - 1]["role"]
            curr = result.messages[i]["role"]
            if prev == "user":
                assert curr == "assistant"
            else:
                assert curr == "user"

    def test_provider_receives_full_history(self):
        """MockProvider call count should match number of LLM calls."""
        provider = MockProvider.from_actions([1, 2])
        run_game(2, [(1, 2)], provider)
        assert provider.call_count == 2
