"""Unit tests for harness.py."""

from ning.agent.harness import (
    SYSTEM_PROMPT,
    render_state,
    render_diff,
    render_trajectory,
    parse_action,
    format_error,
    build_user_message,
)


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
        # Frozen edges are rendered as f1..fn
        assert "1→f1" in text
        assert "2→f2" in text
        assert "3→f3" in text

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

    def test_only_frozen_edges(self):
        state = {
            "total_mutable": 2,
            "colors": {1: "green", 2: "green"},
            "edges": [(1, 3, 1), (2, 4, 1)],
            "red_count": 0,
            "step": 0,
            "move_history": [],
        }
        text = render_state(state)
        assert "1→f1" in text
        assert "2→f2" in text
        assert "(none)" not in text

    def test_empty_edges(self):
        state = {
            "total_mutable": 2,
            "colors": {1: "green", 2: "green"},
            "edges": [],
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


# ── render_trajectory ─────────────────────────────────────────────


class TestRenderTrajectory:
    def test_basic(self):
        summary = {
            "red_history": [9, 10, 9, 10, 11, 12],
            "edge_total_history": [54, 66, 78, 92, 113, 92],
            "best_red": 12,
        }
        text = render_trajectory(summary)
        assert "Recent trajectory" in text
        assert "last 5 mutations" in text
        assert "9 → 10 → 9 → 10 → 11 → 12" in text
        assert "54 → 66 → 78 → 92 → 113 → 92" in text
        assert "Best red:   12" in text

    def test_short_history(self):
        """Two data points = one transition: still rendered."""
        summary = {
            "red_history": [0, 1],
            "edge_total_history": [33, 35],
            "best_red": 1,
        }
        text = render_trajectory(summary)
        assert "last 1 mutations" in text
        assert "0 → 1" in text

    def test_single_point_returns_empty(self):
        """At step 0 (no mutations) trajectory has nothing meaningful."""
        summary = {
            "red_history": [0],
            "edge_total_history": [33],
            "best_red": 0,
        }
        assert render_trajectory(summary) == ""


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

    def test_ignores_frozen_labels(self):
        """Digits inside f1, f2... should not be parsed as actions."""
        assert parse_action("I'll mutate 2 to break the f1 connection", 5) == 2
        assert parse_action("Cut f3, then mutate 1", 5) == 1
        assert parse_action("Just f2", 5) is None

    def test_zero(self):
        assert parse_action("0", 5) is None

    def test_no_number(self):
        assert parse_action("I'm not sure what to do", 5) is None

    def test_empty(self):
        assert parse_action("", 5) is None


# ── format_error ──────────────────────────────────────────────────


class TestFormatError:
    def test_contains_range(self):
        msg = format_error("blah", 5)
        assert "1" in msg
        assert "5" in msg


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

    def test_with_trajectory(self):
        state = {
            "total_mutable": 2,
            "colors": {1: "red", 2: "green"},
            "edges": [(2, 1, 1)],
            "red_count": 1,
            "step": 1,
            "move_history": [1],
        }
        traj = "Recent trajectory (last 1 mutations):\n  Red:        0 → 1"
        msg = build_user_message(state, diff_text="diff", trajectory_text=traj)
        assert "Recent trajectory" in msg
        # Trajectory should appear before "Your move?"
        assert msg.index("Recent trajectory") < msg.index("Your move?")

    def test_empty_trajectory_omitted(self):
        state = {
            "total_mutable": 2,
            "colors": {1: "green", 2: "green"},
            "edges": [],
            "red_count": 0,
            "step": 0,
            "move_history": [],
        }
        msg = build_user_message(state, trajectory_text="")
        assert "Recent trajectory" not in msg


# ── system prompt ─────────────────────────────────────────────────


class TestSystemPrompt:
    def test_contains_rules(self):
        assert "mutation" in SYSTEM_PROMPT.lower()
        assert "GREEN" in SYSTEM_PROMPT or "green" in SYSTEM_PROMPT.lower()
        assert "RED" in SYSTEM_PROMPT or "red" in SYSTEM_PROMPT.lower()

    def test_contains_output_format(self):
        assert "integer" in SYSTEM_PROMPT.lower()
