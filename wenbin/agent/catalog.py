"""Graph catalog: discovers and loads game definitions from games/*.json.

Each JSON file contains:
  - "n": number of mutable vertices
  - "B_A": n×n antisymmetric exchange matrix (nested list)
  - "solution" (optional): mutation sequence that reaches all-red
"""

import json
from pathlib import Path

import numpy as np

GAMES_DIR = Path(__file__).parent / "games"


def list_graphs() -> list[dict]:
    """List all available graphs.

    Returns:
        List of {"name": str, "n": int} dicts, sorted by (n, name).
    """
    graphs = []
    for p in GAMES_DIR.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        graphs.append({"name": p.stem, "n": data["n"]})
    graphs.sort(key=lambda g: (g["n"], g["name"]))
    return graphs


def _load_raw(name: str) -> dict:
    """Load and parse a game JSON file."""
    path = GAMES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Graph '{name}' not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_graph(name: str) -> dict:
    """Load a graph definition (without solution).

    Args:
        name: Filename stem (e.g. "linear_3", "test1_03_n6").

    Returns:
        {"n": int, "B_A": np.ndarray}
    """
    data = _load_raw(name)
    return {
        "n": data["n"],
        "B_A": np.array(data["B_A"], dtype=np.int64),
    }


def get_solution(name: str) -> list[int] | None:
    """Load the solution sequence for a graph, if available.

    Returns:
        Mutation sequence as list of 1-indexed vertex IDs, or None.
    """
    data = _load_raw(name)
    return data.get("solution")
