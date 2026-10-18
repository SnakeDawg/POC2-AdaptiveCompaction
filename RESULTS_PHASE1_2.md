# Phase 1–2 Results: What We Learned

## What Works

- ✅ **ReAct loop**: The agent calls tools, reads results, loops
- ✅ **Compaction fires**: Fixed policy compresses every 3 tool calls
- ✅ **Summarizer works**: Model generates summaries from its own context
- ✅ **Message reset works**: Context becomes [system, question, summary, continue]
- ✅ **Capacity guard**: Forces final answer when tokens exceed budget

## The Problem We Discovered

With `temperature=0.0` (deterministic), the model gets **stuck in search loops**:

```
Iteration 1: search("Mars moons Phobos Deimos size orbit fate NASA")
Iteration 2: search("Mars moons Phobos Deimos names sizes orbits ultimate fate")
Iteration 3: search("Mars moons Phobos Deimos sizes orbits fate")
Iteration 4: search("Mars moons Phobos Deimos sizes orbits ultimate fate")  ← same as 3!
...
```

**Why this matters:** This is exactly the problem SelfCompact's rubric solves.

When the model re-summarizes (compaction), the summary becomes the new context.
If the summary doesn't give the model enough to know it has enough info, it will:
1. Keep searching with similar queries
2. Generate more summary tokens (cost)
3. Never converge on an answer (utility loss)

## What the Rubric Solves

The SelfCompact rubric (C1/C2/C3/N1) directly addresses this:

| Rule | What it checks | How it helps |
|------|---------------|--------------|
| **C3: Progress** | "Did I learn anything new?" | Stops re-searching dead ends |
| **N1: Not Stuck** | "Have 3+ of my last 4 searches returned nothing new?" | Forces a different strategy instead of looping |

In our current code, C3 and N1 are **not yet implemented** (Phase 3 stub).
That's why the model loops forever — it has no meta-cognitive awareness.

## Next Step: Phase 3 — The Rubric

The rubric adds a probe call *before* compaction fires:
1. After every N iterations, append the rubric prompt to the conversation
2. Ask the model to answer C1/C2/C3/N1
3. Only compress if the rubric says so AND the model is making progress

This is the key insight from the paper: **the model can be guided to be smarter about when to compress** — it just needs the right prompt.
