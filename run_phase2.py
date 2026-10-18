"""
Phase 2 Runner: Test fixed-interval compaction

Compare:
  1. No compaction (baseline)
  2. Fixed compaction (every N tool calls)

The key question: does compaction help or hurt?
"""

import asyncio
import os
import sys

PYTHON_PATH = os.path.join(
    os.path.expanduser("~"), "ailab/repos/gateway/.venv/lib/python3.12/site-packages"
)
sys.path.insert(0, PYTHON_PATH)

from agent import ReActAgent
from synthetic_tasks import get_tasks, evaluate_answer


async def run_with_policy(policy: str, summarize_every: int, max_tool_calls: int,
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
        print(f"Question: {task['question'][:100]}...")
        print(f"{'='*60}")

        try:
            result = await asyncio.wait_for(
                agent.run(task["question"]),
                timeout=300
            )
            eval_result = evaluate_answer(task, result["answer"])

            print(f"\n--- Results ---")
            print(f"Iterations: {result['iterations']}")
            print(f"Tool calls: {result['tool_calls']}")
            print(f"Summarizations: {result['summarizations']}")
            print(f"Answer preview: {result['answer'][:200]}")
            print(f"Score: {eval_result['score']}")

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
            print(f"ERROR: {e}")
            results.append({
                "task_id": task["id"],
                "policy": policy,
                "error": str(e),
            })

    return results


async def main():
    print("=" * 60)
    print("Phase 2: Fixed-Interval Compaction")
    print("=" * 60)
    print("\nWe'll test TWO policies:")
    print("  1. 'none'   - No compaction (baseline)")
    print("  2. 'fixed'  - Compress every 3 tool calls")
    print()

    all_results = []

    # Test 1: No compaction
    print("\n\n>>> TEST 1: No compaction (baseline)")
    r1 = await run_with_policy("none", summarize_every=3, max_tool_calls=8, context_budget=50000)
    all_results.extend(r1)

    # Test 2: Fixed compaction (every 3 calls)
    print("\n\n>>> TEST 2: Fixed compaction (every 3 tool calls)")
    r2 = await run_with_policy("fixed", summarize_every=3, max_tool_calls=8, context_budget=50000)
    all_results.extend(r2)

    # Print summary
    print("\n\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for policy in ["none", "fixed"]:
        print(f"\n--- Policy: {policy} ---")
        for r in all_results:
            if r["policy"] == policy:
                total_tokens = sum(r["token_counts"]) if r["token_counts"] else 0
                compactions = len(r.get("compaction_events", []))
                print(f"  {r['task_id']}:")
                print(f"    Iterations: {r['iterations']}")
                print(f"    Tool calls: {r['tool_calls']}")
                print(f"    Compactions: {compactions}")
                print(f"    Total tokens (est): {total_tokens:,}")
                print(f"    Score: {r['evaluation']['score']}")
                if r.get("compaction_events"):
                    for ce in r["compaction_events"]:
                        print(f"    → Compacted at iter {ce['iteration']}: "
                              f"{ce['tokens_before']} → {ce['tokens_after']} tokens")

    # Save results
    with open("output_phase2.json", "w") as f:
        import json
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to output_phase2.json")

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
