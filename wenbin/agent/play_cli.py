"""CLI entry point: run a green-red game in the terminal.

Usage:
    python -m wenbin.agent.play_cli                       # default: deepseek, linear_2
    python -m wenbin.agent.play_cli deepseek test1_07_n4  # specify provider and graph
    python -m wenbin.agent.play_cli --list                # list available graphs
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

from wenbin.agent.agent import run_game_from_matrix
from wenbin.agent.catalog import list_graphs, get_graph
from wenbin.agent.llm_provider import OpenAICompatProvider

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


def main():
    if "--list" in sys.argv:
        print("Available graphs:")
        for g in list_graphs():
            print(f"  {g['name']}  (n={g['n']})")
        print(f"\nAvailable providers: {', '.join(PROVIDERS.keys())}")
        return

    provider_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    graph_name = sys.argv[2] if len(sys.argv) > 2 else "linear_2"

    if provider_name not in PROVIDERS:
        print(f"Unknown provider: {provider_name}")
        print(f"Choose from: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    try:
        graph_data = get_graph(graph_name)
    except FileNotFoundError:
        print(f"Unknown graph: {graph_name}")
        print("Use --list to see available graphs.")
        sys.exit(1)

    cfg = PROVIDERS[provider_name]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        print(f"Set {cfg['api_key_env']} in .env first.")
        sys.exit(1)

    provider = OpenAICompatProvider(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=api_key,
    )

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
