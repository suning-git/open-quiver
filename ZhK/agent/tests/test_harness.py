"""Unit tests for harness.py."""

from ZhK.agent.harness import (
    render_state,
    render_diff,
    parse_action_or_undo,
    parse_action,
    format_error,
    build_user_message,
)
from ZhK.agent.initial_prompts import get_system_prompt


# ── render_state ──────────────────────────────────────────────────


class TestRenderState:
    def test_initial_state(self):
        state = {
            "total_mutable": 3,
            "colors": {1: "green", 2: "green", 3: "green"},
            "edges": [(1, 2, 1), (2, 3, 1), (1, 4, 1), (2, 5, 1), (3, 6, 1)],
            "red_count": 0,
            "step": 0,
            "move_history": [],
        }
        text = render_state(state)
        assert "Step 0" in text
        assert "Red: 0/3" in text
        assert "1(G)" in text
        assert "2(G)" in text
        assert "3(G)" in text
        assert "1→2" in text
        assert "2→3" in text
        # Frozen edges are excluded from state rendering
        assert "1→4" not in text
        assert "2→5" not in text
        assert "3→6" not in text

    def test_partial_red(self):
        state = {
            "total_mutable": 3,
            "colors": {1: "green", 2: "red", 3: "green"},
            "edges": [(1, 3, 1), (3, 2, 1)],
            "red_count": 1,
            "step": 1,
            "move_history": [2],
        }
        text = render_state(state)
        assert "Red: 1/3" in text
        assert "2(R)" in text
        assert "1(G)" in text

    def test_multiple_edges(self):
        state = {
            "total_mutable": 2,
            "colors": {1: "green", 2: "green"},
            "edges": [(1, 2, 3)],
            "red_count": 0,
            "step": 0,
            "move_history": [],
        }
        text = render_state(state)
        assert "×3" in text

    def test_no_mutable_edges(self):
        state = {
            "total_mutable": 2,
            "colors": {1: "green", 2: "green"},
            "edges": [(1, 3, 1), (2, 4, 1)],
            "red_count": 0,
            "step": 0,
            "move_history": [],
        }
        text = render_state(state)
        assert "(none)" in text


# ── render_diff ───────────────────────────────────────────────────


class TestRenderDiff:
    def test_basic_diff(self):
        state = {
            "total_mutable": 3,
            "diff": {
                "mutated_vertex": 2,
                "color_changes": {2: ("green", "red")},
                "red_count_before": 0,
                "red_count_after": 1,
            },
            "colors": {1: "green", 2: "red", 3: "green"},
            "edges": [],
            "red_count": 1,
            "step": 1,
            "move_history": [2],
        }
        text = render_diff(state)
        assert "Mutated vertex 2" in text
        assert "Vertex 2: green → red" in text
        assert "0/3 → 1/3" in text

    def test_cycle_warning(self):
        state = {
            "total_mutable": 3,
            "diff": {
                "mutated_vertex": 2,
                "color_changes": {2: ("red", "green")},
                "red_count_before": 1,
                "red_count_after": 0,
                "cycle_warning": 0,
            },
            "colors": {1: "green", 2: "green", 3: "green"},
            "edges": [],
            "red_count": 0,
            "step": 2,
            "move_history": [2, 2],
        }
        text = render_diff(state)
        assert "step 0" in text
        assert "⚠" in text

    def test_no_color_change(self):
        state = {
            "total_mutable": 3,
            "diff": {
                "mutated_vertex": 1,
                "color_changes": {},
                "red_count_before": 1,
                "red_count_after": 1,
            },
            "colors": {1: "green", 2: "red", 3: "green"},
            "edges": [],
            "red_count": 1,
            "step": 2,
            "move_history": [2, 1],
        }
        text = render_diff(state)
        assert "Mutated vertex 1" in text
        assert "1/3 → 1/3" in text
        # No color change lines
        assert "green → red" not in text
        assert "red → green" not in text

    def test_undo_diff(self):
        state = {
            "total_mutable": 3,
            "diff": {
                "action_type": "undo",
                "undone_vertex": 2,
                "color_changes": {2: ("red", "green")},
                "red_count_before": 1,
                "red_count_after": 0,
            },
            "colors": {1: "green", 2: "green", 3: "green"},
            "edges": [],
            "red_count": 0,
            "step": 0,
            "move_history": [],
        }
        text = render_diff(state)
        assert "Undid last move on vertex 2" in text
        assert "1/3 → 0/3" in text


# ── parse_action ──────────────────────────────────────────────────


class TestParseAction:
    def test_bare_number(self):
        assert parse_action("3", 5) == 3

    def test_with_reasoning(self):
        assert parse_action("I think vertex 2 is best because...\n2", 5) == 2

    def test_mutate_format(self):
        assert parse_action("mutate(3)", 5) == 3

    def test_last_number_wins(self):
        assert parse_action("Vertices 1 and 3 are green, I'll pick 3", 5) == 3

    def test_out_of_range(self):
        assert parse_action("99", 5) is None

    def test_fallback_past_out_of_range(self):
        """Last number out of range, but earlier number is valid."""
        assert parse_action("I'll mutate vertex 3, which has 12 neighbors", 5) == 3

    def test_zero(self):
        assert parse_action("0", 5) is None

    def test_no_number(self):
        assert parse_action("I'm not sure what to do", 5) is None

    def test_empty(self):
        assert parse_action("", 5) is None


class TestParseActionOrUndo:
    def test_undo_word(self):
        assert parse_action_or_undo("undo", 5) == "undo"

    def test_undo_short(self):
        assert parse_action_or_undo("u", 5) == "undo"

    def test_number(self):
        assert parse_action_or_undo("3", 5) == 3

    def test_mixed_undo_then_number_takes_last_command(self):
        assert parse_action_or_undo("undo then 2", 5) == 2

    def test_mixed_number_then_undo_takes_last_command(self):
        assert parse_action_or_undo("2 ... undo", 5) == "undo"


# ── format_error ──────────────────────────────────────────────────


class TestFormatError:
    def test_contains_range(self):
        msg = format_error("blah", 5)
        assert "1" in msg
        assert "5" in msg

    def test_undo_hint_when_enabled(self):
        msg = format_error("blah", 5, allow_undo=True)
        assert "undo" in msg.lower()


# ── build_user_message ────────────────────────────────────────────


class TestBuildUserMessage:
    def test_first_turn(self):
        state = {
            "total_mutable": 2,
            "colors": {1: "green", 2: "green"},
            "edges": [(1, 2, 1)],
            "red_count": 0,
            "step": 0,
            "move_history": [],
        }
        msg = build_user_message(state)
        assert "Your move?" in msg
        assert "Step 0" in msg
        # No diff on first turn
        assert "Mutated" not in msg

    def test_with_diff(self):
        state = {
            "total_mutable": 2,
            "colors": {1: "red", 2: "green"},
            "edges": [(2, 1, 1)],
            "red_count": 1,
            "step": 1,
            "move_history": [1],
        }
        diff_text = "Mutated vertex 1.\n  Vertex 1: green → red\nRed: 0/2 → 1/2"
        msg = build_user_message(state, diff_text=diff_text)
        assert "Mutated vertex 1" in msg
        assert "Step 1" in msg
        assert "Your move?" in msg


# ── system prompt ─────────────────────────────────────────────────


class TestSystemPrompt:
    def test_contains_rules(self):
        assert "mutation" in get_system_prompt().lower()
        assert "GREEN" in get_system_prompt() or "green" in get_system_prompt().lower()
        assert "RED" in get_system_prompt() or "red" in get_system_prompt().lower()

    def test_contains_output_format(self):
        assert "integer" in get_system_prompt().lower()
