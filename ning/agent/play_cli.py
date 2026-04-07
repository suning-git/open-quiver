"""CLI entry point: run a green-red game in the terminal.

Usage:
    python -m ning.agent.play_cli                                       # default: deepseek-chat, linear_2
    python -m ning.agent.play_cli deepseek-chat test1_07_n4             # specify provider and graph
    python -m ning.agent.play_cli gpt-5.4-mini linear_2 --max-steps 2   # cap mutation steps
    python -m ning.agent.play_cli --list                                # list available graphs
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

from ning.agent.game_session_runner import run_game_from_matrix
from ning.agent.catalog import list_graphs, get_graph
from ning.agent.provider_registry import (
    create_provider,
    get_provider_config,
    is_known_provider,
    list_provider_names,
)


def _parse_max_steps(argv: list[str]) -> tuple[list[str], int]:
    """Pop --max-steps N from argv, return (remaining_argv, max_steps)."""
    max_steps = 50
    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] == "--max-steps":
            if i + 1 >= len(argv):
                raise SystemExit("--max-steps requires an integer value")
            try:
                max_steps = int(argv[i + 1])
            except ValueError:
                raise SystemExit(
                    f"--max-steps value must be an integer, got: {argv[i + 1]}"
                )
            i += 2
        else:
            remaining.append(argv[i])
            i += 1
    return remaining, max_steps


def main():
    if "--list" in sys.argv:
        print("Available graphs:")
        for g in list_graphs():
            print(f"  {g['name']}  (n={g['n']})")
        print(f"\nAvailable providers: {', '.join(list_provider_names())}")
        return

    argv, max_steps = _parse_max_steps(sys.argv[1:])

    provider_name = argv[0] if len(argv) > 0 else "deepseek-chat"
    graph_name = argv[1] if len(argv) > 1 else "linear_2"

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
    print(f"Max steps: {max_steps}")
    print()

    result = run_game_from_matrix(graph_data["B_A"], provider, max_steps=max_steps)

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
