"""Streamlit app for the green-red mutation game."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import streamlit.components.v1 as components

from ning.agent.engine import QuiverEngine
from ning.agent.catalog import list_graphs, get_graph
from ning.agent.game_turn_runner import initialize_messages, run_turn
from ning.agent.provider_registry import (
    create_provider,
    get_provider_config,
    list_provider_names,
)
from ning.agent.graph_viz import render_graph

# ── Page config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Green-Red Mutation Game",
    layout="wide",
)

# ── Graph catalog ────────────────────────────────────────────────

GRAPH_LIST = list_graphs()
GRAPH_OPTIONS = {f"{g['name']} (n={g['n']})": g["name"] for g in GRAPH_LIST}

GAME_HISTORY_DIR = Path(__file__).parent / "game_history"

# ── Session state init ────────────────────────────────────────────


def init_session():
    if "engine" not in st.session_state:
        st.session_state.engine = QuiverEngine()
        st.session_state.messages = []
        st.session_state.game_started = False
        st.session_state.game_over = False
        st.session_state.auto_playing = False
        st.session_state.view_step = 0
        st.session_state.graph_name = ""
        st.session_state.last_export_path = ""
        st.session_state.loaded_from = ""


init_session()


def clamp_view_step(view_step: int, total_steps: int) -> int:
    """Keep history browsing within the valid inclusive range."""
    return max(0, min(view_step, total_steps))


def start_game(graph_label: str, provider_name: str):
    """Initialize a new game."""
    graph_name = GRAPH_OPTIONS[graph_label]
    graph_data = get_graph(graph_name)
    engine = QuiverEngine()
    engine.reset_from_matrix(graph_data["B_A"])
    messages = initialize_messages(engine)

    st.session_state.engine = engine
    st.session_state.messages = messages
    st.session_state.game_started = True
    st.session_state.game_over = False
    st.session_state.auto_playing = False
    st.session_state.view_step = 0
    st.session_state.provider_name = provider_name
    st.session_state.graph_name = graph_name
    st.session_state.last_export_path = ""
    st.session_state.loaded_from = ""


def load_game(filename: str) -> str | None:
    """Load a saved game JSON and replay it.

    Returns None on success, or an error message on failure.
    On failure, session_state is left untouched.
    """
    try:
        path = GAME_HISTORY_DIR / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph_name = payload["graph"]
        graph_data = get_graph(graph_name)
        engine = QuiverEngine()
        engine.reset_from_matrix(graph_data["B_A"])
        for k in payload["move_history"]:
            engine.mutate(k)
    except Exception as e:
        return f"Failed to load {filename}: {e}"

    # All computation succeeded — now commit to session_state.
    st.session_state.engine = engine
    st.session_state.messages = payload.get("messages", [])
    st.session_state.game_started = True
    st.session_state.game_over = bool(payload.get("game_over", False))
    st.session_state.auto_playing = False
    st.session_state.view_step = engine.total_steps
    st.session_state.provider_name = payload.get("provider", "")
    st.session_state.graph_name = graph_name
    st.session_state.last_export_path = ""
    st.session_state.loaded_from = filename
    return None


def export_chat_history() -> str:
    """Write current game history to a JSON file and return its path."""
    engine = st.session_state.engine
    state = engine.get_state()
    provider_name = st.session_state.get("provider_name", "")
    graph_name = st.session_state.get("graph_name", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{graph_name}_{provider_name}_{timestamp}.json"

    GAME_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = GAME_HISTORY_DIR / filename

    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "graph": graph_name,
        "provider": provider_name,
        "won": engine.is_won(),
        "game_over": st.session_state.game_over,
        "step": state["step"],
        "red_count": state["red_count"],
        "total_mutable": state["total_mutable"],
        "move_history": state["move_history"],
        "messages": st.session_state.messages,
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def get_provider():
    """Create provider from current selection."""
    name = st.session_state.get("provider_name", "deepseek-chat")
    try:
        return create_provider(name)
    except RuntimeError:
        cfg = get_provider_config(name)
        st.error(f"Set {cfg['api_key_env']} in .env file.")
        return None
    except ValueError as e:
        st.error(str(e))
        return None


def step_game():
    """Execute one step: LLM decides, engine mutates."""
    engine = st.session_state.engine
    messages = st.session_state.messages

    if engine.is_won():
        st.session_state.game_over = True
        return

    provider = get_provider()
    if provider is None:
        return

    # Keep the user's history cursor at the frontier when stepping.
    was_at_frontier = st.session_state.view_step == engine.total_steps

    turn = run_turn(engine, messages, provider, max_retries=3)

    if was_at_frontier and turn.reason != "parse_failure":
        st.session_state.view_step = engine.total_steps

    if turn.game_over:
        st.session_state.game_over = True


# ── Sidebar: Controls ─────────────────────────────────────────────

with st.sidebar:
    is_loaded = bool(st.session_state.get("loaded_from"))

    # ── Current Game (high-frequency controls, top) ──────────
    if st.session_state.game_started:
        st.header("Current Game")
        engine = st.session_state.engine
        total_steps = engine.total_steps
        current_state = engine.get_state()

        if is_loaded:
            st.caption(f"📂 {st.session_state.loaded_from}")

        if engine.is_won():
            st.success(f"WON in {total_steps} steps!")
        elif st.session_state.game_over:
            st.error("Game over (parse failure)")

        st.metric("Red", f"{current_state['red_count']}/{current_state['total_mutable']}")
        if current_state["move_history"]:
            moves_str = " -> ".join(f"u{k}" for k in current_state["move_history"])
            st.text(f"Moves: {moves_str}")

        if not st.session_state.game_over and not is_loaded:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Step", use_container_width=True):
                    st.session_state.auto_playing = False
                    step_game()
                    st.rerun()
            with col2:
                if st.session_state.auto_playing:
                    if st.button("Stop", use_container_width=True):
                        st.session_state.auto_playing = False
                        st.rerun()
                else:
                    if st.button("Auto-play", use_container_width=True):
                        st.session_state.auto_playing = True
                        st.rerun()

        if not is_loaded:
            if st.button("Export Chat History", use_container_width=True):
                st.session_state.last_export_path = export_chat_history()
                st.rerun()
            if st.session_state.last_export_path:
                st.caption(f"Saved: {st.session_state.last_export_path}")

        st.divider()

    # ── Setup (low-frequency, bottom, collapsible) ───────────
    st.header("Setup")

    with st.expander("🎮 New Game", expanded=not st.session_state.game_started):
        provider_name = st.selectbox("LLM Provider", list_provider_names())
        graph_label = st.selectbox("Graph", list(GRAPH_OPTIONS.keys()))
        if st.session_state.game_started and not is_loaded:
            st.caption("⚠ Will discard the current game")
        if st.button("Start New Game", use_container_width=True):
            start_game(graph_label, provider_name)
            st.rerun()

    with st.expander("📂 Load Saved", expanded=False):
        if GAME_HISTORY_DIR.exists():
            saved_files = sorted(
                (p.name for p in GAME_HISTORY_DIR.glob("*.json")),
                reverse=True,
            )
        else:
            saved_files = []
        if saved_files:
            selected_file = st.selectbox(
                "History file",
                saved_files,
                label_visibility="collapsed",
            )
            if st.session_state.game_started and not is_loaded:
                st.caption("⚠ Will discard the current game")
            if st.button("Load", use_container_width=True):
                err = load_game(selected_file)
                if err:
                    st.error(err)
                else:
                    st.rerun()
        else:
            st.caption("(no saved games)")

# ── Main area ─────────────────────────────────────────────────────

if not st.session_state.game_started:
    st.title("Green-Red Mutation Game")
    st.write("Select a graph and LLM provider from the sidebar, then click **Start New Game**.")
else:
    if st.session_state.get("loaded_from"):
        st.info(f"📂 Loaded: {st.session_state.loaded_from} (read-only replay)")
    engine = st.session_state.engine
    total_steps = engine.total_steps
    view_step = clamp_view_step(st.session_state.view_step, total_steps)

    # Clamp view_step in case session state drifts out of range.
    if view_step != st.session_state.view_step:
        st.session_state.view_step = view_step

    view_state = engine.get_state_at(view_step)
    mutated_vertex = view_state.get("last_move")

    left, right = st.columns([1, 1])

    # Left: Graph visualization with navigation
    with left:
        st.subheader("Graph")

        # Navigation controls
        nav_cols = st.columns([1, 1, 3, 1, 1])
        with nav_cols[0]:
            if st.button("<<", use_container_width=True, disabled=view_step == 0):
                st.session_state.view_step = 0
                st.rerun()
        with nav_cols[1]:
            if st.button("<", use_container_width=True, disabled=view_step == 0):
                st.session_state.view_step = view_step - 1
                st.rerun()
        with nav_cols[2]:
            st.markdown(
                f"<div style='text-align:center; padding:6px 0;'>"
                f"Step <b>{view_step}</b> / {total_steps}</div>",
                unsafe_allow_html=True,
            )
        with nav_cols[3]:
            if st.button(">", use_container_width=True, disabled=view_step == total_steps):
                st.session_state.view_step = view_step + 1
                st.rerun()
        with nav_cols[4]:
            if st.button(">>", use_container_width=True, disabled=view_step == total_steps):
                st.session_state.view_step = total_steps
                st.rerun()

        # Slider for quick navigation
        if total_steps > 0:
            new_step = st.slider(
                "Browse moves",
                min_value=0,
                max_value=total_steps,
                value=view_step,
                label_visibility="collapsed",
            )
            if new_step != view_step:
                st.session_state.view_step = new_step
                st.rerun()

        html = render_graph(view_state, mutated_vertex=mutated_vertex)
        components.html(html, height=550, scrolling=False)

    # Right: Chat log
    with right:
        st.subheader("LLM Conversation")
        chat_container = st.container(height=550)
        with chat_container:
            for msg in st.session_state.messages:
                role = msg["role"]
                if role == "system":
                    continue
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])

# ── Auto-play: execute one step per rerun cycle ──────────────────

if (
    st.session_state.game_started
    and st.session_state.auto_playing
    and not st.session_state.game_over
):
    import time

    time.sleep(0.5)  # brief pause so user can see each step
    step_game()
    if st.session_state.game_over:
        st.session_state.auto_playing = False
    st.rerun()
