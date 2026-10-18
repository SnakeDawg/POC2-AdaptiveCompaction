# POC 2: Adaptive Context Compaction

Learning project based on the [SelfCompact paper](https://arxiv.org/abs/2606.23525) and the spec at `02-Adaptive-Context-Compaction.md`.

## Goal

Build a self-contained compaction system from scratch to understand:
- How context management works in long-running agents
- The difference between fixed-interval and adaptive compaction
- Why the rubric matters (C1/C2/C3/N1)
- The tradeoffs between token savings and information loss

## What We're Building

A minimal ReAct-style agent that can run with four compaction modes:

| Mode | What triggers compaction | When it fires |
|------|--------------------------|---------------|
| `none` | Never | Baseline — no compaction |
| `fixed` | Every N tool calls | Blind clock, ignores trajectory state |
| `selfcompact` | Model's own rubric (C1-C3/N1) | Only when a sub-task is closed and safe |
| `adaptive` | Rubric + hard capacity guard | Rubric when possible, deterministic fallback at limit |

## Learning Phases

| Phase | What we build | Key concept learned |
|-------|---------------|---------------------|
| 1 | Bare ReAct loop | How agents call tools, parse results, loop |
| 2 | Fixed compaction | Token counting, when summaries fire, KV cache reuse |
| 3 | Rubric (C1/C2/C3/N1) | Meta-cognition: agent deciding *when* to compress |
| 4 | Summarizer | How summaries replace history, what must be preserved |
| 5 | Capacity guard | Hard limits, evidence retrieval, rollback |
| 6 | Evaluation harness | Comparing modes, measuring success vs cost |

## Model

- **Local**: Qwen3.6-35B-A3B (vLLM at 127.0.0.1:8000, 262k context, free)
- **Hosted** (optional): Nemotron-3-Super-120B via NVIDIA API

## Key Files

- `agent.py` — Core agent with all four compaction modes
- `prompts.py` — System prompt, rubric, and summarizer prompts
- `evaluation.py` — Synthetic task harness and scoring
- `run_poc.py` — CLI runner that exercises all modes
- `synthetic_tasks.py` — Test tasks (no internet required)

## References

- SelfCompact paper: https://arxiv.org/abs/2606.23525
- Reference implementation: https://github.com/tianjianl/selfcompact
- Spec: `02-Adaptive-Context-Compaction.md`
