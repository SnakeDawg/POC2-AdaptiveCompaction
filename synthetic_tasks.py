"""
Synthetic test tasks for the POC.

These simulate real research questions that would require multi-step
searching and synthesis. In a real agent, these would be answered via
live web search or document retrieval.
"""


# ---------------------------------------------------------------------------
# Test tasks
# ---------------------------------------------------------------------------
# Each task has:
#   - question: The research question
#   - expected_answer: What a good answer would include
#   - expected_searches: How many searches are needed (for evaluation)
#   - notes: What concepts this task tests


TASKS = [
    {
        "id": "task_001",
        "question": (
            "What is the Great Barrier Reef, how large is it, and what marine "
            "life does it support? Provide specific numbers if available."
        ),
        "expected_answer": (
            "The Great Barrier Reef is the world's largest coral reef system, "
            "spanning over 2,300 km off Queensland, Australia. It is home to "
            "at least 1,500 species of fish and 400 types of coral."
        ),
        "expected_searches": 2,
        "notes": "Tests basic multi-step search and synthesis",
    },
    {
        "id": "task_002",
        "question": (
            "What is CRISPR-Cas9 and how does it work at a high level? "
            "What are its key applications?"
        ),
        "expected_answer": (
            "CRISPR-Cas9 is a gene-editing system adapted from bacterial "
            "defense mechanisms. It uses guide RNA and Cas9 nuclease to target "
            "and modify specific DNA sequences."
        ),
        "expected_searches": 2,
        "notes": "Tests technical concept explanation",
    },
    {
        "id": "task_003",
        "question": (
            "What are Mars's moons? What are their names, sizes, orbits, "
            "and what is their ultimate fate?"
        ),
        "expected_answer": (
            "Mars has two moons: Phobos (closer, larger, completes 3 orbits "
            "per Martian day) and Deimos (smaller, farther). Phobos is decaying "
            "and will crash or form a ring in ~50 million years."
        ),
        "expected_searches": 2,
        "notes": "Tests retrieval of multiple facts from different sources",
    },
]


def get_tasks() -> list:
    """Return all synthetic tasks."""
    return TASKS


def get_task_by_id(task_id: str) -> dict:
    """Return a specific task by ID."""
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise ValueError(f"Unknown task: {task_id}")


def evaluate_answer(task: dict, answer: str) -> dict:
    """
    Simple heuristic evaluation.

    In a real evaluation (like the SelfCompact paper's auto_judge),
    this would use an LLM to score correctness. Here we check for
    keyword overlap to keep it self-contained.
    """
    question_lower = task["question"].lower()
    answer_lower = answer.lower()

    # Extract key terms from expected answer
    key_terms = set()
    expected_words = task["expected_answer"].lower().split()
    for word in expected_words:
        cleaned = word.strip(".,!?\"'()[]")
        if len(cleaned) > 3:  # Skip short words
            key_terms.add(cleaned)

    # Check overlap
    answer_words = set(answer_lower.split())
    matches = key_terms & answer_words
    coverage = len(matches) / len(key_terms) if key_terms else 0

    return {
        "task_id": task["id"],
        "keyword_coverage": round(coverage, 3),
        "has_keyword": coverage > 0.2,
        "score": 1.0 if coverage > 0.3 else 0.0,
    }
