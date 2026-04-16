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

from ZhK.agent.catalog import get_graph, list_graphs
from ZhK.agent.dual_agent_runner import DualAgentSearchRunner, SearchConfig
from ZhK.agent.engine import QuiverEngine
from ZhK.agent.game_turn_runner import initialize_messages, run_turn
from ZhK.agent.graph_search_harness import (
    format_iteration_log,
    node_to_render_state,
    summarize_snapshot,
)
from ZhK.agent.graph_viz import render_graph
from ZhK.agent.provider_registry import (
    create_provider,
    get_provider_config,
    list_provider_names,
)

# -- Page config -----------------------------------------------------------

st.set_page_config(
    page_title="Green-Red Mutation Game",
    layout="wide",
)

# -- Graph catalog ---------------------------------------------------------

GRAPH_LIST = list_graphs()
GRAPH_OPTIONS = {f"{g['name']} (n={g['n']})": g["name"] for g in GRAPH_LIST}
GAME_HISTORY_DIR = Path(__file__).parent / "game_history"

LINEAR_MODE = "linear"
GRAPH_SEARCH_MODE = "graph_search"


# -- Session state init ----------------------------------------------------

def init_session() -> None:
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
        st.session_state.mode = LINEAR_MODE
        st.session_state.provider_name = "deepseek-chat"

        # Graph-search mode state
        st.session_state.search_runner = None
        st.session_state.search_snapshot = None
        st.session_state.search_last_iteration = None
        st.session_state.search_view_state_id = ""
        st.session_state.search_logs = []
        st.session_state.search_messages = []


init_session()


def clamp_view_step(view_step: int, total_steps: int) -> int:
    """Keep history browsing within the valid inclusive range."""
    return max(0, min(view_step, total_steps))


def get_provider(name: str | None = None):
    """Create provider from selection."""
    provider_name = name or st.session_state.get("provider_name", "deepseek-chat")
    try:
        return create_provider(provider_name)
    except RuntimeError:
        cfg = get_provider_config(provider_name)
        st.error(f"Set {cfg['api_key_env']} in .env file.")
        return None
    except ValueError as e:
        st.error(str(e))
        return None


def _reset_graph_search_state() -> None:
    st.session_state.search_runner = None
    st.session_state.search_snapshot = None
    st.session_state.search_last_iteration = None
    st.session_state.search_view_state_id = ""
    st.session_state.search_logs = []
    st.session_state.search_messages = []


def start_game(graph_label: str, provider_name: str, mode: str) -> str | None:
    """Initialize a new game.

    Returns None on success, or an error message on failure.
    """
    graph_name = GRAPH_OPTIONS[graph_label]
    graph_data = get_graph(graph_name)

    st.session_state.provider_name = provider_name
    st.session_state.graph_name = graph_name
    st.session_state.mode = mode
    st.session_state.last_export_path = ""
    st.session_state.loaded_from = ""
    st.session_state.auto_playing = False
    st.session_state.game_started = True

    if mode == LINEAR_MODE:
        engine = QuiverEngine()
        engine.reset_from_matrix(graph_data["B_A"])
        messages = initialize_messages(engine)

        st.session_state.engine = engine
        st.session_state.messages = messages
        st.session_state.game_over = False
        st.session_state.view_step = 0
        _reset_graph_search_state()
        return None

    provider = get_provider(provider_name)
    if provider is None:
        st.session_state.game_started = False
        return "Provider initialization failed."

    runner = DualAgentSearchRunner(provider, SearchConfig(max_iterations=50))
    root_id = runner.initialize_from_exchange_matrix(graph_data["B_A"])
    snapshot = runner.get_result_snapshot()

    st.session_state.search_runner = runner
    st.session_state.search_snapshot = snapshot
    st.session_state.search_last_iteration = None
    st.session_state.search_view_state_id = root_id
    st.session_state.search_logs = []
    st.session_state.search_messages = []

    st.session_state.messages = []
    st.session_state.game_over = snapshot.reason != "in_progress"
    st.session_state.view_step = 0

    # Keep an engine placeholder so old references remain safe.
    st.session_state.engine = QuiverEngine()
    return None


def load_game(filename: str) -> str | None:
    """Load a saved linear-game JSON and replay it.

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
    st.session_state.mode = LINEAR_MODE
    _reset_graph_search_state()
    return None


def export_chat_history() -> str:
    """Write current linear-game history to a JSON file and return its path."""
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
        "mode": LINEAR_MODE,
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def step_game_linear() -> None:
    """Execute one linear-mode step: LLM decides, engine mutates."""
    engine = st.session_state.engine
    messages = st.session_state.messages

    if engine.is_won():
        st.session_state.game_over = True
        return

    provider = get_provider()
    if provider is None:
        return

    was_at_frontier = st.session_state.view_step == engine.total_steps
    turn = run_turn(engine, messages, provider, max_retries=3)

    if was_at_frontier and turn.reason != "parse_failure":
        st.session_state.view_step = engine.total_steps

    if turn.game_over:
        st.session_state.game_over = True


def step_game_graph_search() -> None:
    """Execute one graph-search iteration."""
    runner = st.session_state.get("search_runner")
    if runner is None:
        st.session_state.game_over = True
        return

    if runner.is_done():
        st.session_state.search_snapshot = runner.get_result_snapshot()
        st.session_state.game_over = True
        return

    it = runner.run_one_iteration()
    st.session_state.search_last_iteration = it
    st.session_state.search_snapshot = runner.get_result_snapshot()

    if it.to_state_id:
        st.session_state.search_view_state_id = it.to_state_id
    elif it.selected_state_id:
        st.session_state.search_view_state_id = it.selected_state_id

    st.session_state.search_logs.append(format_iteration_log(it, runner.store))
    if len(st.session_state.search_logs) > 300:
        st.session_state.search_logs = st.session_state.search_logs[-300:]

    if it.llm_messages:
        for msg in it.llm_messages:
            if msg.get("role") == "system":
                continue
            st.session_state.search_messages.append(
                {"role": msg.get("role", ""), "content": msg.get("content", "")}
            )
        if len(st.session_state.search_messages) > 500:
            st.session_state.search_messages = st.session_state.search_messages[-500:]

    st.session_state.game_over = st.session_state.search_snapshot.reason != "in_progress"


def step_game() -> None:
    """Dispatch one step according to current mode."""
    mode = st.session_state.get("mode", LINEAR_MODE)
    if mode == GRAPH_SEARCH_MODE:
        step_game_graph_search()
    else:
        step_game_linear()


# -- Sidebar: Controls -----------------------------------------------------

with st.sidebar:
    mode = st.session_state.get("mode", LINEAR_MODE)
    is_loaded = bool(st.session_state.get("loaded_from"))

    # -- Current Game ------------------------------------------------------
    if st.session_state.game_started:
        st.header("Current Game")

        if mode == LINEAR_MODE:
            engine = st.session_state.engine
            total_steps = engine.total_steps
            current_state = engine.get_state()

            if is_loaded:
                st.caption(f"{st.session_state.loaded_from}")

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

        else:
            runner = st.session_state.get("search_runner")
            snapshot = st.session_state.get("search_snapshot")
            if runner is not None and snapshot is None:
                snapshot = runner.get_result_snapshot()
                st.session_state.search_snapshot = snapshot

            if runner is not None and snapshot is not None:
                summary = summarize_snapshot(snapshot, runner.store)

                if summary["won"]:
                    st.success(f"WON in {summary['iterations']} iterations!")
                elif st.session_state.game_over:
                    st.error(f"Search stopped ({summary['reason']})")

                st.metric("Best Red", f"{summary['best_red_count']}/{summary['best_total_mutable']}")
                st.caption(
                    f"Iter={summary['iterations']} | States={summary['state_count']} | "
                    f"Edges={summary['edge_count']}"
                )
                st.caption(
                    f"Expansions={summary['expansions']} | Parse failures={summary['parse_failures']}"
                )
                st.caption(
                    "Best state: "
                    f"{summary['best_state_created_iter']} "
                    f"{summary['best_state_created_path']}"
                )

            if not st.session_state.game_over:
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

            # if st.button("Export Search History", use_container_width=True):
            #     st.session_state.last_export_path = export_graph_search_history()
            #     st.rerun()
            if st.session_state.last_export_path:
                st.caption(f"Saved: {st.session_state.last_export_path}")

        st.divider()

    # -- Setup -------------------------------------------------------------
    st.header("Setup")

    mode_options = {
        "Linear (legacy)": LINEAR_MODE,
        "Graph Search (new)": GRAPH_SEARCH_MODE,
    }
    default_mode_label = (
        "Graph Search (new)" if mode == GRAPH_SEARCH_MODE else "Linear (legacy)"
    )

    with st.expander("New Game", expanded=not st.session_state.game_started):
        provider_name = st.selectbox("LLM Provider", list_provider_names())
        graph_label = st.selectbox("Graph", list(GRAPH_OPTIONS.keys()))
        selected_mode_label = st.selectbox(
            "Mode",
            list(mode_options.keys()),
            index=list(mode_options.keys()).index(default_mode_label),
        )
        next_mode = mode_options[selected_mode_label]

        if st.session_state.game_started and (mode != GRAPH_SEARCH_MODE and not is_loaded or mode == GRAPH_SEARCH_MODE):
            st.caption("Will discard the current game")

        if st.button("Start New Game", use_container_width=True):
            err = start_game(graph_label, provider_name, next_mode)
            if err:
                st.error(err)
            else:
                st.rerun()

    with st.expander("Load Saved", expanded=False):
        if GAME_HISTORY_DIR.exists():
            saved_files = sorted((p.name for p in GAME_HISTORY_DIR.glob("*.json")), reverse=True)
        else:
            saved_files = []
        if saved_files:
            selected_file = st.selectbox("History file", saved_files, label_visibility="collapsed")
            if st.session_state.game_started and not is_loaded:
                st.caption("Will discard the current game")
            if st.button("Load", use_container_width=True):
                err = load_game(selected_file)
                if err:
                    st.error(err)
                else:
                    st.rerun()
        else:
            st.caption("(no saved games)")


# -- Main area -------------------------------------------------------------

if not st.session_state.game_started:
    st.title("Green-Red Mutation Game")
    st.write(
        "Select a graph, provider, and mode from the sidebar, then click **Start New Game**."
    )
else:
    mode = st.session_state.get("mode", LINEAR_MODE)

    if mode == LINEAR_MODE:
        if st.session_state.get("loaded_from"):
            st.info(f"Loaded: {st.session_state.loaded_from} (read-only replay)")
        engine = st.session_state.engine
        total_steps = engine.total_steps
        view_step = clamp_view_step(st.session_state.view_step, total_steps)

        if view_step != st.session_state.view_step:
            st.session_state.view_step = view_step

        view_state = engine.get_state_at(view_step)
        mutated_vertex = view_state.get("last_move")

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Graph")

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

    else:
        runner = st.session_state.get("search_runner")
        if runner is None:
            st.error("Search runner missing. Please start a new game.")
        else:
            snapshot = st.session_state.get("search_snapshot") or runner.get_result_snapshot()
            st.session_state.search_snapshot = snapshot

            store = runner.store
            state_ids = store.list_state_ids()
            if not state_ids:
                st.warning("No state in graph store.")
            else:
                view_id = st.session_state.get("search_view_state_id", "")
                if view_id not in state_ids:
                    view_id = snapshot.best_state_id
                    st.session_state.search_view_state_id = view_id

                labels = []
                label_to_id = {}
                for node in store.list_states_by_created_iter():
                    sid = node.state_id
                    path = (
                        "[]"
                        if not node.shortest_path_actions
                        else "[" + " ".join(str(k) for k in node.shortest_path_actions) + "]"
                    )
                    label = (
                        f"{node.created_iter} | path {path} | "
                        f"red {node.red_count}/{node.n} | visits={node.visit_count}"
                    )
                    labels.append(label)
                    label_to_id[label] = sid

                current_label = next((k for k, v in label_to_id.items() if v == view_id), labels[0])
                selected_label = st.selectbox(
                    "View State",
                    labels,
                    index=labels.index(current_label),
                )
                selected_view_id = label_to_id[selected_label]
                if selected_view_id != view_id:
                    st.session_state.search_view_state_id = selected_view_id
                    st.rerun()

                view_node = store.get_state(selected_view_id)
                view_state = node_to_render_state(view_node)

                mutated_vertex = None
                last_it = st.session_state.get("search_last_iteration")
                if (
                    last_it is not None
                    and last_it.to_state_id == selected_view_id
                    and last_it.action is not None
                ):
                    mutated_vertex = last_it.action

                left, right = st.columns([1, 1])

                with left:
                    st.subheader("Graph")
                    html = render_graph(view_state, mutated_vertex=mutated_vertex)
                    components.html(html, height=550, scrolling=False)

                with right:
                    st.subheader("LLM Conversation")
                    chat_container = st.container(height=360)
                    with chat_container:
                        msgs = st.session_state.get("search_messages", [])
                        if not msgs:
                            st.caption("(no LLM messages yet)")
                        else:
                            for msg in msgs:
                                role = msg.get("role", "")
                                if role == "user":
                                    with st.chat_message("user"):
                                        st.markdown(msg.get("content", ""))
                                elif role == "assistant":
                                    with st.chat_message("assistant"):
                                        st.markdown(msg.get("content", ""))

                    st.subheader("Search Logs")
                    if snapshot.best_path:
                        display_best_path = " -> ".join(
                            str(store.get_state(sid).created_iter) for sid in snapshot.best_path
                        )
                        st.caption("Best path (created_iter): " + display_best_path)
                    log_container = st.container(height=170)
                    with log_container:
                        logs = st.session_state.get("search_logs", [])
                        if not logs:
                            st.caption("(no logs yet)")
                        else:
                            for line in reversed(logs[-120:]):
                                st.markdown(f"- {line}")


# -- Auto-play: execute one step per rerun cycle ---------------------------

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
