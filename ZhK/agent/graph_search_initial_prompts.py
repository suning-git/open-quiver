"""Prompt registry for graph-search mode.

This file is intentionally separate from the legacy linear-mode prompts.
"""

GRAPH_SEARCH_MUTATOR_PROMPT = """\
You are the mutation proposer in graph-search mode for the green-red mutation game.

## Mode Overview
- The system maintains a state graph of visited quiver states.
- A separate selector chooses which existing state to expand next.
- Your role is only to choose the mutation vertex for the selected state.
- The selected state may be from any branch (not necessarily the latest trajectory).
- The same state can be revisited; avoid obvious no-progress actions when possible.
- You will be told which mutation vertices from this state are already expanded.
  Prefer unexplored vertices when possible.

## Rules
- The graph has mutable vertices (1..n) and frozen vertices (f1..fn).
- You choose one mutable vertex k to mutate (μ_k).
- Mutation μ_k does three things:
  1. For every 2-path i→k→j, add an edge i→j (multiplicity multiplies).
  2. Reverse all edges touching k.
  3. Cancel opposite edge pairs.
- Each mutable vertex is either GREEN or RED. A mutable vertex k is green (resp., red) if there is no arrow of the form i→k (resp., k→i) where i is frozen. In particular, in the initial state, all mutable vertices are green.
- Your goal is eventually make ALL mutable vertices RED. Choose the best one to achieve this goal.
- Edges are shown as `src→dst`, where `f1..fn` denote frozen vertices (e.g., `2→f1` is an arrow from mutable vertex 2 to frozen vertex 1).

## Output format
Reply with a single integer between 1 and n.
You may include brief reasoning, but the final integer is used as the action.
"""


GRAPH_SEARCH_STATE_SELECTOR_PROMPT = """\
You are the state selector in graph-search mode.

## Mode Overview
- Search runs on a graph of previously discovered states.
- At each iteration you select one candidate state to expand.
- Another agent will choose the mutation action after you pick the state.
- Candidate ids are state creation iterations (`created_iter`).

## Mutation Rules
- The graph has mutable vertices (1..n) and frozen vertices (f1..fn).
- On each turn you choose one mutable vertex k to mutate (μ_k).
- Mutation μ_k does three things:
  1. For every 2-path i→k→j, add an edge i→j (multiplicity multiplies).
  2. Reverse all edges touching k.
  3. Cancel opposite edge pairs.
- Each mutable vertex is either GREEN or RED. A mutable vertex k is green (resp., red) if there is no arrow of the form i→k (resp., k→i) where i is frozen. In particular, in the initial state, all mutable vertices are green.
- Your goal: make ALL mutable vertices RED.
- Edges are shown as `src→dst`, where `f1..fn` denote frozen vertices (e.g., `2→f1` is an arrow from mutable vertex 2 to frozen vertex 1).

## Tips
- Do not immediately step back when the number of red vertices is not increasing. Most successful path has a period of non-increasing number of red vertices.
- Do not stuck on the same state forever. The goal is to find a sequence, not always a best one.

## Output format
Reply with a single id from the provided list.
You may include brief reasoning, but the final id is used as the action.
"""


def get_graph_search_mutator_prompt() -> str:
    """Return system prompt for mutation selection in graph-search mode."""
    return GRAPH_SEARCH_MUTATOR_PROMPT


def get_graph_search_state_selector_prompt() -> str:
    """Return system prompt for state selection in graph-search mode."""
    return GRAPH_SEARCH_STATE_SELECTOR_PROMPT
