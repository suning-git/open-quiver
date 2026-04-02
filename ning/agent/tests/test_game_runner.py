"""Unit tests for one-turn game runner helpers."""

from ning.agent.engine import QuiverEngine
from ning.agent.game_turn_runner import initialize_messages, run_turn
from ning.agent.llm_provider import MockProvider


def _init_messages(engine: QuiverEngine) -> list[dict]:
    return initialize_messages(engine)


class TestRunTurn:
    def test_parse_failure_ends_turn(self):
        engine = QuiverEngine()
        engine.reset(2, [(1, 2)])
        messages = _init_messages(engine)

        provider = MockProvider(["nope", "still nope", "??", "none"])
        result = run_turn(engine, messages, provider, max_retries=2)

        assert result.game_over
        assert result.reason == "parse_failure"
        assert engine.total_steps == 0

    def test_success_appends_next_user_prompt(self):
        engine = QuiverEngine()
        engine.reset(2, [(1, 2)])
        messages = _init_messages(engine)

        provider = MockProvider.from_actions([1])
        result = run_turn(engine, messages, provider, max_retries=3)

        assert not result.game_over
        assert result.reason == ""
        assert engine.total_steps == 1
        assert messages[-1]["role"] == "user"
        assert "Your move?" in messages[-1]["content"]

    def test_winning_turn_records_final_diff(self):
        engine = QuiverEngine()
        engine.reset(2, [(1, 2)])
        messages = _init_messages(engine)

        provider = MockProvider.from_actions([1, 2])
        first = run_turn(engine, messages, provider, max_retries=3)
        assert not first.game_over

        second = run_turn(engine, messages, provider, max_retries=3)
        assert second.game_over
        assert second.reason == "won"
        assert engine.is_won()
        assert messages[-1]["role"] == "user"
        assert "Mutated vertex" in messages[-1]["content"]
