"""CLI entry point: run a green-red game in the terminal.

Usage:
    python -m ning.agent.play_cli                       # default: deepseek, linear_2
    python -m ning.agent.play_cli deepseek test1_07_n4  # specify provider and graph
    python -m ning.agent.play_cli --list                # list available graphs
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

from ZhK.agent.game_session_runner import run_game_from_matrix
from ZhK.agent.catalog import list_graphs, get_graph
from ZhK.agent.provider_registry import (
    create_provider,
    get_provider_config,
    is_known_provider,
    list_provider_names,
)


def main():
    if "--list" in sys.argv:
        print("Available graphs:")
        for g in list_graphs():
            print(f"  {g['name']}  (n={g['n']})")
        print(f"\nAvailable providers: {', '.join(list_provider_names())}")
        return

    provider_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    graph_name = sys.argv[2] if len(sys.argv) > 2 else "linear_2"

    if not is_known_provider(provider_name):
        print(f"Unknown provider: {provider_name}")
        print(f"Choose from: {', '.join(list_provider_names())}")
        sys.exit(1)

    try:
        graph_data = get_graph(graph_name)
    except FileNotFoundError:
        print(f"Unknown graph: {graph_name}")
        print("Use --list to see available graphs.")
        sys.exit(1)

    cfg = get_provider_config(provider_name)
    try:
        provider = create_provider(provider_name)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    print(f"Provider: {provider_name} ({cfg['model']})")
    print(f"Graph: {graph_name} (n={graph_data['n']})")
    print()

    result = run_game_from_matrix(graph_data["B_A"], provider, max_steps=50)

    print(f"Result: {'WON' if result.won else 'LOST'} ({result.reason})")
    print(f"Steps: {result.steps}")
    print(f"Moves: {result.move_history}")
    print()

    for msg in result.messages:
        role = msg["role"].upper()
        print(f"--- {role} ---")
        print(msg["content"])
        print()


if __name__ == "__main__":
    main()
