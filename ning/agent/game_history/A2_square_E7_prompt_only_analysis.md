# Analysis: prompt-only methods on A2_square_E7 (n=14)

A multi-model, multi-prompt case study showing that **prompt engineering alone
cannot produce reliable play on this graph**, regardless of model size or how
explicit the hints are. Strengthening the hints does not just fail to help —
on stronger models it actively introduces a new failure mode.

## Runs

| Game file | Model | Prompt | Result | Best red | Final edges | Immediate undos |
|---|---|---|---|---|---|---|
| `..._qwen3-14b_20260407_161704.json` | qwen3-14b | v4_ning | lost (38 moves) | 12 | 840 | 0 |
| `..._qwen3-14b_20260407_164651.json` | qwen3-14b | v4_ning | lost (20 moves) | 10 | 410 | 0 |
| `..._qwen3-14b_20260407_171026.json` | qwen3-14b | v5_ning | lost (37 moves) | 12 | 3897 | 0 |
| `..._qwen-flash_20260407_173018.json` | qwen-flash | v5_ning | lost (29 moves) | 8 | — | 0 |
| `..._qwen-flash_20260408_003134.json` | qwen-flash | v5_undo_hint | lost (46 moves) | 12 | — | 12 |
| `..._qwen3-32b_20260407_235649.json` | qwen3-32b | v5_undo_hint | lost (133 moves) | 8 | — | 62 |

For comparison, both qwen-flash and qwen3-32b have **won** this graph in earlier
runs under different prompt versions / random seeds, so the graph is solvable
by these models — winning is just unreliable.

## Prompt versions

- **v3_ning**: rules + brief involution note ("mutating same vertex twice is
  wasted").
- **v4_ning**: imperative hints — backtracking via repeated mutation, watch
  for double/triple-digit multiplicities, expect red count to trend upward.
- **v5_ning**: pure descriptive framing — introduces the "Recent trajectory"
  section, presents red/edge total/best red as soft signals, frames undo as
  one of n equally valid choices. Removes all imperative language.
- **v5_undo_hint**: a later experiment that adds back one
  imperative paragraph instructing the model to undo when at a local maximum.
  See "The imperative-hint experiment" below.

## Phase 1 of the story: zero undos (qwen3-14b on v3/v4/v5)

Under v3, v4, and even v5, qwen3-14b shows the same failure pattern:

**Phase 1: mechanical sweep.** Mutate every vertex 1..n once, mostly in
sequential order. Reaches ~9-10 red.

**Phase 2: thrashing.** A small set of ~5-7 "edge" vertices is identified as
"the problem area." The model cycles through different permutations of these,
trying to fix things by re-ordering instead of stepping back. Red oscillates
in a narrow band; edge multiplicities grow into the hundreds and then
thousands.

**Zero immediate undos in any of the three runs.** Even after v5 explicitly
explains how to undo and points to the exact "Mutated vertex N" line for
reference, the pattern `μ_k μ_k` never appears.

### The pivotal moment in the v5 / qwen3-14b run

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
indicators in plain text. v5 prompt described what each signal meant and how
to undo. The model still picked another forward move, and the next four steps
pushed edges to 4388 with no red recovery.

## Phase 2 of the story: imperative-hint experiment

After the zero-undo failures, v5_ning was patched with an imperative paragraph:

> When **every** forward move you can think of would cause a net loss in red
> count (i.e., you are at a local maximum), the correct play is usually to
> **undo** the last move by mutating the same vertex again, and then try a
> different starting vertex on the next turn. Do NOT pick the "least bad"
> forward move just to avoid undoing — that wastes the involution property and
> digs you deeper into a losing branch.

The hope was that an explicit instruction would override the anti-undo prior.
What actually happened on stronger models was much worse than zero undos.

### qwen-flash run with the new hint (46 moves, 12 undos)

The model played reasonably for ~28 moves, climbing to red 12. Then it
discovered undo and entered a pathological terminal state:

```
... 14, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
```

**Eleven consecutive `1`s** at the end of the run. Each pair `(1,1)` is a
no-op; the engine state oscillates between two configurations and red count
stays flat. The model is stuck in a loop of "mutate vertex 1 → undo vertex 1
→ mutate vertex 1 → undo vertex 1" for the entire tail of the game.

It found undo. It cannot find anything else to do.

### qwen3-32b run with the new hint (133 moves, 62 undos)

Even more striking. Over 133 moves the model performs 62 immediate undos —
nearly half of all moves are followed by their own undo. Move pattern:

```
[3, 10, 10, 2, 1, 1, 10, 10, 4, 5, 5, 10, 10, 5, 5, 11, 12, 13, 13, ...]
```

Reading this: mutate 3 → mutate 10 → undo 10 → mutate 2 → mutate 1 → undo 1
→ mutate 10 → undo 10 → mutate 4 → mutate 5 → undo 5 → mutate 10 → undo 10 →
mutate 5 → undo 5 → ...

The model has converted undo into a **lookahead probe**: try X for one step,
see what happens, undo, try Y, see what happens, undo, try Z. It never
commits. Worse, it has **no memory of what it has already probed** — vertex
10 is tried and undone roughly 8 times across the run. Vertex 5 even more.

This is "pure exploration with zero commitment." The opposite extreme of the
14b run, and equally fatal.

## The two failure modes of prompt-only undo

| Failure mode | Example | Cause |
|---|---|---|
| Zero exploration, all commitment | qwen3-14b on v5 (no undos, blow-up to 3897 edges) | Anti-undo RLHF prior dominates; model never tries backtracking. |
| All exploration, zero commitment | qwen3-32b on v5 + imperative hint (62 undos in 133 moves) | Explicit "undo when bad" instruction; model treats every move as a probe and undoes everything. |

**Both failure modes have the same root cause**: the model lacks a value
function. It cannot judge whether a branch is "worth committing to" versus
"worth abandoning." Without that judgment, every prompt-level instruction
either pushes the model fully into commit mode (failure 1) or fully into
probe mode (failure 2). The healthy middle — explore briefly, evaluate, then
commit — requires a value function the model does not have.

## What we ruled out

By the end of the v5 + imperative experiments, we had tested every available
prompt-only lever:

1. **Information availability** — trajectory section provides exactly the
   cross-turn comparison the model can't do internally.
2. **Action reachability** — undo is one literal token away, and the prompt
   explicitly tells the model how to issue it.
3. **Last-move locatability** — prompt points at the "Mutated vertex N" line.
4. **Phrasing clarity** — both descriptive (v5) and imperative (v5 + hint)
   framings tested.
5. **Threshold concreteness** — v4 used "double/triple digits"; v5 used pure
   data; v5+hint used "every forward move is a net loss."
6. **Model capacity** — tried at 14b, 32b, and the closed qwen-flash.

None of these moved the bottleneck. Strengthening the prompt actively made
stronger models worse.

## The actual bottleneck

Three structural limitations, none patchable by rewording the system prompt:

### 1. Anti-undo behavioral prior from RLHF

Instruction-tuned LLMs are trained on a distribution where "redo my last
action" looks like failure or stalling. The default policy fights backtracking.
Telling the model "undo is allowed" in prose does not overcome a prior baked
in over millions of training examples — especially in smaller dense models
where the alignment phase dominates more strongly.

### 2. No cross-turn meta-reasoning

The trajectory section tells the model "edge total: 553 → 1846 → 1798 → 2866
→ 4388." A human reads this and immediately thinks "the line is going up; I
am making things worse." The model reads each turn as a fresh local decision
and does not perform that comparison even when the data is laid out beside
the choice point. Providing the data is necessary but not sufficient — the
model also has to **use** it, which requires meta-reasoning about its own
trajectory.

### 3. No value function over states

Even when a model is told "undo when stuck," it has no way to evaluate whether
the current state is genuinely stuck or merely hard. The qwen3-32b run shows
what happens when an explicit undo instruction meets a missing value function:
the model undoes everything because every state looks "potentially bad."
Reliable play requires comparing the value of "commit and continue" versus
"undo and try elsewhere," and that comparison is precisely what the model
cannot do without external scaffolding or training.

### 4. No credit assignment

In the v4 / 14b run, steps 22-26 happen to be a *good* subsequence: edge
count drops, red count climbs. The model has no mechanism to notice this and
"do more of that." It treats the lucky stretch the same as every other turn
and abandons it on the next move. The same blindness that prevents detecting
bad trajectories also prevents reinforcing good ones.

## What would actually fix this

Two paths, both moving away from prompt-only methods:

### Path A: External search (MCTS / proof tree)

Take the meta-reasoning out of the LLM. An external searcher controls
backtracking, expansion, and pruning; the LLM is reduced to a local evaluator
("from this state, which child looks most promising?"). The anti-undo prior
becomes irrelevant because undo is a search-tree edge, not an LLM output. The
trajectory awareness becomes irrelevant because the search remembers state
visits explicitly. The missing value function is replaced by either (a) the
LLM's local node evaluations rolled up via MCTS UCB, or (b) a separately
trained value head.

This is the "agent-style" solution and matches the eventual research
direction: critic/performer separation, with the LLM playing performer only.

### Path B: RL fine-tuning

Take qwen3-4b/8b/14b base models and fine-tune them on this game directly,
either via SFT on winning trajectories or via on-policy RL. The reward signal
will naturally teach backtracking, trajectory awareness, and value estimation
because **the game itself rewards those behaviors** — something the prompt
cannot communicate by description alone, and that imperative instructions
turn into pathology.

This is more expensive but produces a model that has internalized the
strategy, rather than one that follows external scaffolding. The long-term
research target.

## Conclusion

**Prompt engineering for this graph is exhausted across the entire size range
we have access to.** Six runs covering three model sizes and three prompt
philosophies (descriptive, imperative-soft, imperative-hard) produced two
failure modes, both fatal:

- Smaller / weaker prompts → no undos, mechanical thrashing, edge blow-up.
- Stronger prompts on capable models → undo addiction, oscillation, no
  commitment.

The bottleneck is structural: RLHF anti-undo prior plus missing meta-reasoning
plus missing value function. Strengthening the prompt cannot fix any of
these, and on stronger models can introduce **new** pathological behaviors.

The next steps should be **MCTS** (Path A) for short-term experiments, and
**RL fine-tuning** (Path B) for the longer research goal. Further prompt
iteration on this graph is unlikely to yield useful information.
