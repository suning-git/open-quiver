# Analysis: qwen3-14b on A2_square_E7 (n=14)

A multi-run case study showing that **prompt-only methods cannot get qwen3-14b
past ~12/14 red on this graph**. Three prompt versions tried with progressively
more direct hints. All failed in the same structural way.

## Runs

| Game file | Prompt | Result | Best red | Final edges | Immediate undos |
|---|---|---|---|---|---|
| `..._20260407_161704.json` | v4_ning | lost (38 moves) | 12 | 840 | 0 |
| `..._20260407_164651.json` | v4_ning | lost (20 moves) | 10 | 410 | 0 |
| `..._20260407_171026.json` | v5_ning | lost (37 moves) | 12 | 3897 | 0 |

For comparison, qwen-flash (a closed proprietary model) wins this graph in
~30 moves; qwen3-32b also wins. So the graph is solvable by LLMs in this size
class — qwen3-14b in particular is the wall.

## Prompt versions

- **v3_ning**: just rules + brief involution note ("mutating same vertex twice
  is wasted").
- **v4_ning**: adds three explicit hints — backtracking via repeated mutation,
  watch for double/triple-digit multiplicities, expect red count to trend
  upward.
- **v5_ning**: pure descriptive framing — introduces the "Recent trajectory"
  section, presents red/edge total/best red as soft signals, frames undo as one
  of n equally valid choices. Removes all imperative language and absolute
  thresholds.

The v5 design is principled (don't inject strategy, only information), and
should in theory let a capable model derive the right behavior on its own.
**It does not work for qwen3-14b.**

## What the model does

Two phases, identical across all three runs:

**Phase 1: mechanical sweep.** Mutate every vertex 1..n once (the order is
mostly sequential or near-sequential). Gets to ~9-10 red without much thought.

**Phase 2: thrashing.** A small set of ~5-7 "edge" vertices is identified as
"the problem area," and the model cycles through different permutations of
these, trying to fix things by re-ordering instead of stepping back. Red
oscillates in a narrow band; edge multiplicities grow into the hundreds and
then thousands.

**Zero immediate undos in any run.** Even after the v5 prompt explicitly
explains how to undo and points to the exact "Mutated vertex N" line for
reference, the pattern `μ_k μ_k` never appears.

## The pivotal moment in the v5 run

Between message 32 and message 33:

```
msg32: red=12 edges= 553 best=12       <- just achieved best ever
msg33: red=10 edges=1846 best=12       <- one move later
```

In a single move:
- Red dropped 2 (12 → 10)
- Edge total **tripled** (553 → 1846)
- Best red still equals current best (no further progress possible)

The model had complete information: trajectory section showed all three
indicators in plain text. v5 prompt had explicitly described what each signal
meant and how to undo. The model still picked another forward move, then
another, and the next four steps pushed edges to 4388 with no red recovery.

## What we ruled out

By the time we got to v5, we had tested every prompt-only lever:

1. **Information availability** ✅ — trajectory section provides exactly the
   cross-turn comparison the model can't do internally.
2. **Action reachability** ✅ — undo is one literal token away, and the prompt
   explicitly tells the model the vertex to use.
3. **Last-move locatability** ✅ — prompt points at the "Mutated vertex N"
   line.
4. **Phrasing clarity** ✅ — both imperative (v4) and descriptive (v5) framings
   tested.
5. **Threshold concreteness** ✅ — v4 used "double/triple digits"; v5 used pure
   data; neither moved the needle.

None of these are the bottleneck.

## The actual bottleneck

Three structural limitations of the model in this regime, none of them
patchable by rewording the system prompt:

### 1. Anti-undo behavioral prior from RLHF

Instruction-tuned LLMs are trained on a distribution where "redo my last
action" looks like failure or stalling. The default policy fights backtracking.
Telling the model "undo is allowed" in prose does not overcome a prior baked
in over millions of training examples — especially in smaller dense models
where the alignment phase dominates more strongly.

### 2. No cross-turn meta-reasoning

The trajectory section tells the model "edge total: 553 → 1846 → 1798 → 2866 →
4388." A human reads this and immediately thinks "the line is going up; I am
making things worse." The model reads each turn as a fresh local decision and
does not perform that comparison even when the data is laid out beside the
choice point. Providing the data is necessary but not sufficient — the model
also has to **use** it, which requires meta-reasoning about its own
trajectory.

### 3. No credit assignment

In the v4 run, steps 22-26 happen to be a *good* subsequence: edge count
drops, red count climbs. The model has no mechanism to notice this and "do
more of that." It treats the lucky stretch the same as every other turn and
abandons it on the next move. The same blindness that prevents detecting bad
trajectories also prevents reinforcing good ones.

## What would actually fix this

Two paths, both moving away from prompt-only methods:

### Path A: External search (MCTS / proof tree)

Take the meta-reasoning out of the LLM. An external searcher controls
backtracking, expansion, and pruning; the LLM is reduced to a local evaluator
("from this state, which child looks most promising?"). The anti-undo prior
becomes irrelevant because undo is a search-tree edge, not an LLM output. The
trajectory awareness becomes irrelevant because the search remembers state
visits explicitly.

This is the "agent-style" solution and matches the eventual research
direction: critic/performer separation, with the LLM playing performer only.

### Path B: RL fine-tuning

Take qwen3-4b/8b/14b base models and fine-tune them on this game directly,
either via SFT on winning trajectories or via on-policy RL. The reward signal
will naturally teach backtracking and trajectory awareness because **the game
itself rewards those behaviors** — something the prompt cannot communicate by
description alone.

This is more expensive but produces a model that has internalized the
strategy, rather than one that follows external scaffolding. The long-term
research target.

## Conclusion

**Prompt engineering on qwen3-14b for this graph is exhausted.** Three runs
with progressively more direct prompts produced the same failure mode: zero
backtracking, mechanical sweep followed by thrashing, ceiling at 10-12 red.
The information was there, the instructions were there, the model still
couldn't use them.

The bottleneck is structural — RLHF prior plus missing meta-reasoning — and
cannot be moved by changing the prompt. Next steps should be **MCTS** (Path A)
for short-term experiments, and **RL fine-tuning** (Path B) for the longer
research goal.
