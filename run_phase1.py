#!/usr/bin/env python3
"""
Phase 1 Runner: Test the bare ReAct agent (no compaction)

Uses the ailab gateway venv which has litellm installed.
"""

import asyncio
import json
import os
import sys

# Use the ailab gateway venv's Python for litellm access
VENV_PYTHON = os.path.join(os.path.expanduser("~"), "ailab/repos/gateway/.venv/bin/python3")
PYTHON_PATH = os.path.join(
    os.path.expanduser("~"), "ailab/repos/gateway/.venv/lib/python3.12/site-packages"
)
sys.path.insert(0, PYTHON_PATH)

from agent import ReActAgent
from synthetic_tasks import get_tasks, evaluate_answer


async def test_phase1():
    """Run Phase 1 tests."""
    agent = ReActAgent(
        api_base="http://127.0.0.1:8000/v1",
        model="nvidia/Qwen3.6-35B-A3B-NVFP4",
        max_tool_calls=8,  # Limit for testing
    )

    tasks = get_tasks()
    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"TASK {i+1}/{len(tasks)}: {task['id']}")
        print(f"Question: {task['question'][:100]}...")
        print(f"{'='*60}")

        try:
            result = await asyncio.wait_for(
                agent.run(task["question"]),
                timeout=300  # 5 minute timeout per task
            )

            # Evaluate
            eval_result = evaluate_answer(task, result["answer"])

            print(f"\n--- Results ---")
            print(f"Iterations: {result['iterations']}")
            print(f"Tool calls: {result['tool_calls']}")
            print(f"Answer preview: {result['answer'][:300]}")
            print(f"\nEvaluation:")
            print(f"  Keyword coverage: {eval_result['keyword_coverage']}")
            print(f"  Score: {eval_result['score']}")

            results.append({
                "task_id": task["id"],
                "iterations": result["iterations"],
                "tool_calls": result["tool_calls"],
                "token_counts": result["debug_data"]["token_counts"],
                "answer": result["answer"],
                "evaluation": eval_result,
            })

        except asyncio.TimeoutError:
            print(f"\n*** TIMEOUT after 300s ***")
            results.append({
                "task_id": task["id"],
                "error": "timeout",
            })
        except Exception as e:
            print(f"\n*** ERROR: {e} ***")
            import traceback
            traceback.print_exc()
            results.append({
                "task_id": task["id"],
                "error": str(e),
            })

    # Print summary
    print(f"\n\n{'='*60}")
    print(f"PHASE 1 SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\nCompleted: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}")

    for r in successful:
        total_tokens = sum(r["token_counts"]) if r["token_counts"] else 0
        print(f"\n  {r['task_id']}:")
        print(f"    Iterations: {r['iterations']}")
        print(f"    Tool calls: {r['tool_calls']}")
        print(f"    Total tokens (est): {total_tokens:,}")
        print(f"    Keyword coverage: {r['evaluation']['keyword_coverage']}")
        print(f"    Score: {r['evaluation']['score']}")

    return results


if __name__ == "__main__":
    print("Phase 1: Bare ReAct Agent (no compaction)")
    print("=" * 60)
    print("This tests the agent loop WITHOUT any compaction.")
    print("The context will grow with each tool call.")
    print()

    results = asyncio.run(test_phase1())

    # Save results
    with open("output_phase1.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to output_phase1.json")
