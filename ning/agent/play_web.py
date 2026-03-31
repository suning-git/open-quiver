"""Streamlit app for the green-red mutation game."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import streamlit.components.v1 as components

from ning.agent.engine import QuiverEngine
from ning.agent.catalog import list_graphs, get_graph
from ning.agent.harness import (
    SYSTEM_PROMPT,
    build_user_message,
    format_error,
    parse_action,
    render_diff,
)
from ning.agent.llm_provider import OpenAICompatProvider
from ning.agent.graph_viz import render_graph

# ── Page config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Green-Red Mutation Game",
    layout="wide",
)

# ── Provider config ───────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "gpt-5.4-mini": {
        "model": "gpt-5.4-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-5.4": {
        "model": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-5.4-nano": {
        "model": "gpt-5.4-nano",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}

# ── Graph catalog ────────────────────────────────────────────────

GRAPH_LIST = list_graphs()  # [{"name": ..., "n": ...}, ...]
GRAPH_OPTIONS = {f"{g['name']} (n={g['n']})": g["name"] for g in GRAPH_LIST}

# ── Session state init ────────────────────────────────────────────


def init_session():
    if "engine" not in st.session_state:
        st.session_state.engine = QuiverEngine()
        st.session_state.messages = []
        st.session_state.game_started = False
        st.session_state.game_over = False
        st.session_state.auto_playing = False
        st.session_state.view_step = 0


init_session()


def start_game(graph_label: str, provider_name: str):
    """Initialize a new game."""
    graph_name = GRAPH_OPTIONS[graph_label]
    graph_data = get_graph(graph_name)
    engine = QuiverEngine()
    state = engine.reset_from_matrix(graph_data["B_A"])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_msg = build_user_message(state)
    messages.append({"role": "user", "content": user_msg})

    st.session_state.engine = engine
    st.session_state.messages = messages
    st.session_state.game_started = True
    st.session_state.game_over = False
    st.session_state.auto_playing = False
    st.session_state.view_step = 0
    st.session_state.provider_name = provider_name


def get_provider() -> OpenAICompatProvider | None:
    """Create provider from current selection."""
    name = st.session_state.get("provider_name", "deepseek")
    cfg = PROVIDERS[name]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        st.error(f"Set {cfg['api_key_env']} in .env file.")
        return None
    return OpenAICompatProvider(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=api_key,
    )


def step_game():
    """Execute one step: LLM decides, engine mutates."""
    engine = st.session_state.engine
    messages = st.session_state.messages
    n = engine.n

    provider = get_provider()
    if provider is None:
        return

    # Get LLM response
    response = provider.chat(messages)
    messages.append({"role": "assistant", "content": response})

    # Parse with retries
    k = parse_action(response, n)
    retries = 0
    max_retries = 3
    while k is None and retries < max_retries:
        error_msg = format_error(response, n)
        messages.append({"role": "user", "content": error_msg})
        response = provider.chat(messages)
        messages.append({"role": "assistant", "content": response})
        k = parse_action(response, n)
        retries += 1

    if k is None:
        st.session_state.game_over = True
        return

    # Execute mutation
    was_at_frontier = st.session_state.view_step == engine.total_steps
    state = engine.mutate(k)
    diff_text = render_diff(state)

    # Auto-advance view if user was watching the latest step
    if was_at_frontier:
        st.session_state.view_step = engine.total_steps

    if engine.is_won():
        messages.append({"role": "user", "content": diff_text})
        st.session_state.game_over = True
    else:
        user_msg = build_user_message(state, diff_text=diff_text)
        messages.append({"role": "user", "content": user_msg})


# ── Sidebar: Controls ─────────────────────────────────────────────

with st.sidebar:
    st.header("Game Controls")

    provider_name = st.selectbox("LLM Provider", list(PROVIDERS.keys()))
    graph_label = st.selectbox("Graph", list(GRAPH_OPTIONS.keys()))

    if st.button("Start New Game", use_container_width=True):
        start_game(graph_label, provider_name)
        st.rerun()

    if st.session_state.game_started and not st.session_state.game_over:
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

    if st.session_state.game_started:
        engine = st.session_state.engine
        total_steps = engine.total_steps
        current_state = engine.get_state()
        st.divider()
        st.subheader("Status")
        if engine.is_won():
            st.success(f"WON in {total_steps} steps!")
        elif st.session_state.game_over:
            st.error("Game over (parse failure)")
        st.metric("Red", f"{current_state['red_count']}/{current_state['total_mutable']}")
        if current_state["move_history"]:
            moves_str = " -> ".join(f"u{k}" for k in current_state["move_history"])
            st.text(f"Moves: {moves_str}")

# ── Main area ─────────────────────────────────────────────────────

if not st.session_state.game_started:
    st.title("Green-Red Mutation Game")
    st.write("Select a graph and LLM provider from the sidebar, then click **Start New Game**.")
else:
    engine = st.session_state.engine
    total_steps = engine.total_steps
    view_step = st.session_state.view_step

    # Clamp view_step in case of reset
    if view_step > total_steps:
        view_step = total_steps
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
