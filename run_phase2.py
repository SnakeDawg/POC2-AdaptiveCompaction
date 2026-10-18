"""
Phase 2 Runner: Fixed-interval compaction test

Compare:
  1. No compaction (baseline)
  2. Fixed compaction (every 3 tool calls)
"""

import os
import sys
import json

PYTHON_PATH = os.path.join(
    os.path.expanduser("~"), "ailab/repos/gateway/.venv/lib/python3.12/site-packages"
)
sys.path.insert(0, PYTHON_PATH)

from agent import ReActAgent
from synthetic_tasks import get_tasks, evaluate_answer


def run_with_policy(policy: str, summarize_every: int, max_tool_calls: int = 8,
                    context_budget: int = 50000) -> list:
    """Run all tasks with a given compaction policy."""
    agent = ReActAgent(
        api_base="http://127.0.0.1:8000/v1",
        model="nvidia/Qwen3.6-35B-A3B-NVFP4",
        max_tool_calls=max_tool_calls,
        policy=policy,
        summarize_every_n_calls=summarize_every,
        context_budget=context_budget,
    )

    results = []
    tasks = get_tasks()

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"[Policy: {policy}] {task['id']}")
        print(f"{'='*60}")

        try:
            result = agent.run(task["question"])
            eval_result = evaluate_answer(task, result["answer"])

            print(f"\n  -> Iterations: {result['iterations']}")
            print(f"  -> Tool calls: {result['tool_calls']}")
            print(f"  -> Summarizations: {result['summarizations']}")
            print(f"  -> Score: {eval_result['score']}")

            results.append({
                "task_id": task["id"],
                "policy": policy,
                "iterations": result["iterations"],
                "tool_calls": result["tool_calls"],
                "summarizations": result["summarizations"],
                "token_counts": result["debug_data"]["token_counts"],
                "compaction_events": result["debug_data"].get("compaction_events", []),
                "answer": result["answer"],
                "evaluation": eval_result,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "task_id": task["id"],
                "policy": policy,
                "error": str(e),
            })

    return results


def main():
    print("=" * 60)
    print("Phase 2: Fixed-Interval Compaction")
    print("=" * 60)
    print("\nTesting: 'none' vs 'fixed' (every 3 calls)")
    print()

    all_results = []

    # Test 1: No compaction
    print("\n\n>>> TEST 1: No compaction (baseline)")
    r1 = run_with_policy("none", summarize_every=3, max_tool_calls=8)
    all_results.extend(r1)

    # Test 2: Fixed compaction (every 3 calls)
    print("\n\n>>> TEST 2: Fixed compaction (every 3 tool calls)")
    r2 = run_with_policy("fixed", summarize_every=3, max_tool_calls=8)
    all_results.extend(r2)

    # Print comparison
    print("\n\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    for policy in ["none", "fixed"]:
        print(f"\n--- Policy: {policy} ---")
        for r in all_results:
            if r["policy"] == policy:
                total_tokens = sum(r["token_counts"]) if r["token_counts"] else 0
                compactions = len(r.get("compaction_events", []))
                print(f"  {r['task_id']}: iter={r['iterations']} calls={r['tool_calls']} "
                      f"summarize={compactions} score={r['evaluation']['score']}")
                if compactions > 0:
                    for ce in r["compaction_events"]:
                        print(f"    → Compacted: {ce['tokens_before']} → {ce['tokens_after']} tokens")

    with open("output_phase2.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to output_phase2.json")


if __name__ == "__main__":
    main()
