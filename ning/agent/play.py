"""Run a green-red game with a real LLM provider."""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from ning.agent.agent import run_game
from ning.agent.llm_provider import OpenAICompatProvider

PROVIDERS = {
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "gpt": {
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    if name not in PROVIDERS:
        print(f"Unknown provider: {name}. Choose from: {list(PROVIDERS.keys())}")
        sys.exit(1)

    cfg = PROVIDERS[name]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        print(f"Set {cfg['api_key_env']} in .env first.")
        sys.exit(1)

    provider = OpenAICompatProvider(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=api_key,
    )

    # Simple game: n=2, graph 1→2
    print(f"Running game with {name} ({cfg['model']})...")
    print(f"Graph: n=2, edges: 1→2")
    print()

    result = run_game(2, [(1, 2)], provider, max_steps=20)

    print(f"Result: {'WON' if result.won else 'LOST'} ({result.reason})")
    print(f"Steps: {result.steps}")
    print(f"Moves: {result.move_history}")
    print()

    # Print conversation
    for msg in result.messages:
        role = msg["role"].upper()
        print(f"--- {role} ---")
        print(msg["content"])
        print()


if __name__ == "__main__":
    main()
