"""
ReAct agent with pluggable context-compaction policies.

Policies:
    "none"   -- No compaction (baseline, Phase 1)
    "fixed"  -- Compress every N tool calls (Phase 2)
    "selfcompact" -- Compress only when the rubric says so (Phase 3+)

After compaction, the message list is reset to:
    [system, user(question), assistant(summary), user(continue)]

This is exactly how SelfCompact works: the model summarizes its own
context, and generation resumes from the compressed prefix.
"""

import json
import copy
from typing import List, Dict, Any, Tuple, Optional

import litellm


# ---------------------------------------------------------------------------
# Simple tool definitions
# ---------------------------------------------------------------------------

def tool_search(query: str) -> str:
    """Simulates a web search. Returns synthetic results.

    The search is keyword-sensitive: different queries return different results.
    If the query contains a unique keyword, it gets unique results.
    If the query is a repeat (contains only common words), it returns "no new info".
    """
    q_lower = query.lower()

    reef_queries = ["reef", "coral", "great barrier", "australia"]
    crispr_queries = ["crispr", "gene", "dna", "editing", "cas9"]
    mars_queries = ["mars", "phobos", "deimos", "moon", "satellite"]

    def count_matches(words, text):
        return sum(1 for w in words if w in text)

    reef_score = count_matches(reef_queries, q_lower)
    crispr_score = count_matches(crispr_queries, q_lower)
    mars_score = count_matches(mars_queries, q_lower)

    if reef_score > crispr_score and reef_score > mars_score:
        specific_terms = ["size", "area", "km", "species", "fish", "turtle", "marine", "living", "structure"]
        has_specific = sum(1 for t in specific_terms if t in q_lower)

        if has_specific >= 2:
            if any(t in q_lower for t in ["size", "area", "km"]):
                return json.dumps({
                    "result_1": "Source: Wikipedia\n'The Great Barrier Reef is the world's largest coral reef "
                                "system, spanning over 2,300 kilometers (1,430 miles) and covering an area "
                                "of approximately 344,400 square kilometers. It extends from 126 to 383 "
                                "kilometers offshore.'"
                })
            elif any(t in q_lower for t in ["species", "fish", "marine", "turtl"]):
                return json.dumps({
                    "result_1": "Source: NOAA\n'The Great Barrier Reef is home to at least 1,500 species of fish, "
                                "400 types of coral, and a variety of other marine life including "
                                "sea turtles, dugongs, whales, and sharks. Over 900 species of mollusc "
                                "are known to live on the reef.'"
                })
            else:
                return json.dumps({
                    "result_1": "Source: UNESCO\n'The Great Barrier Reef is visible from space and is the "
                                "world's largest living structure. It was declared a World Heritage "
                                "Site in 1981 and is recognized for its outstanding universal value.'"
                })
        else:
            return json.dumps({
                "result_1": "Source: Wikipedia\n'The Great Barrier Reef is the world's largest coral reef system, "
                            "spanning over 2,300 kilometers. It is visible from space and is the world's "
                            "largest living structure.'"
            })

    elif crispr_score > reef_score and crispr_score > mars_score:
        specific_terms = ["how", "work", "mechanism", "guide", "rna", "application", "precision", "bacteria"]
        has_specific = sum(1 for t in specific_terms if t in q_lower)

        if has_specific >= 1:
            if any(t in q_lower for t in ["how", "work", "mechanism"]):
                return json.dumps({
                    "result_1": "Source: Nature\n'CRISPR-Cas9 is a simplified version of a naturally occurring "
                                "genomic defense system in bacteria. The bacteria harvest small pieces of DNA "
                                "from viruses they have encountered, and use them to create DNA segments "
                                "known as CRISPR arrays. The system uses a guide RNA (gRNA) to direct Cas9 "
                                "nuclease to a specific DNA sequence for editing.'"
                })
            elif any(t in q_lower for t in ["application", "precision", "human"]):
                return json.dumps({
                    "result_1": "Source: NCBI\n'CRISPR-based gene editing has been applied in human cells with "
                                "high precision. Key applications include treating genetic disorders (sickle cell "
                                "disease, beta-thalassemia), cancer therapy research, and agricultural "
                                "improvements. The system enables targeted modification of DNA sequences.'"
                })
            else:
                return json.dumps({
                    "result_1": "Source: Wikipedia\n'CRISPR-Cas9 is a gene-editing technology that allows "
                                "scientists to modify an organism's DNA with high precision. The name is an "
                                "acronym for Clustered Regularly Interspaced Short Palindromic Repeats.'."
                })
        else:
            return json.dumps({
                "result_1": "Source: Nature\n'CRISPR-Cas9 is a simplified version of a naturally occurring "
                            "genomic defense system in bacteria.'"
            })

    elif mars_score > 0:
        specific_terms = ["phobos", "deimos", "size", "orbit", "mass", "fate", "crash", "ring", "dimension"]
        has_specific = sum(1 for t in specific_terms if t in q_lower)

        if has_specific >= 2:
            if "phobos" in q_lower and any(t in q_lower for t in ["orbit", "period", "day"]):
                return json.dumps({
                    "result_1": "Source: NASA\n'Phobos is the larger and closer Martian moon, orbiting at just "
                                "6,000 km above the surface. It completes three orbits per Martian day. "
                                "Phobos is irregularly shaped with dimensions approximately 27 x 22 x 18 km.'"
                })
            elif "phobos" in q_lower and any(t in q_lower for t in ["fate", "crash", "ring", "orbit"]):
                return json.dumps({
                    "result_1": "Source: ESA\n'Phobos is expected to either crash into Mars or break apart "
                                "into a ring system within the next 50 million years due to its decaying orbit. "
                                "It is spiraling inward at about 1.8 cm per year.'"
                })
            else:
                return json.dumps({
                    "result_1": "Source: NASA\n'Mars has two natural satellites: Phobos (larger, 27km long) and "
                                "Deimos (smaller, 12km long). Phobos orbits at 6,000 km; Deimos at 20,000 km. "
                                "Both are thought to be captured asteroids.'"
                })
        else:
            return json.dumps({
                "result_1": "Source: NASA\n'Mars has two moons: Phobos and Deimos.'"
            })

    return json.dumps({"result_1": "No relevant results found.", "result_2": "No relevant results found."})


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """A ReAct agent with pluggable context-compaction policies."""

    def __init__(self, api_base: str = "http://127.0.0.1:8000/v1",
                 model: str = "nvidia/Qwen3.6-35B-A3B-NVFP4",
                 max_tool_calls: int = 20,
                 policy: str = "none",
                 summarize_every_n_calls: int = 3,
                 context_budget: int = 50000):
        self.api_base = api_base
        self.model = model
        self.max_tool_calls = max_tool_calls

        # --- Compaction policy ---
        self.policy = policy  # "none" | "fixed" | "selfcompact"
        self.summarize_every_n_calls = summarize_every_n_calls
        self.context_budget = context_budget
        self.summarize_count = 0

        # The conversation history
        self.messages: List[Dict[str, Any]] = []
        self.prefix_messages: List[Dict[str, Any]] = []

        # Tool definitions
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information about a topic. Takes a query string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # Statistics
        self.tool_call_count = 0
        self.iteration = 0

        # --- SelfCompact state (Phase 3) ---
        self.search_history: List[Dict[str, str]] = []  # (query, result_hash) for N1
        self.rubric_results: List[Dict[str, Any]] = []  # log of rubric checks
        self.rubric_check_every_n_calls: int = 3  # check rubric every N tool calls

    def call_model(self, messages: List[Dict[str, Any]],
                   tools: Optional[List[Dict]] = None,
                   max_tokens: int = 4096) -> Any:
        """Call the LLM API. Returns the response object."""
        response = litellm.completion(
            model=f"openai/{self.model}",
            api_base=self.api_base,
            api_key="EMPTY",
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic, faster
            timeout=60,
            num_retries=0,
        )
        return response

    def parse_tool_call(self, message: Any) -> Tuple[str, str]:
        """Extract (tool_name, json_arguments) from assistant message, or (None, None)."""
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tc = message.tool_calls[0]
            return tc.function.name, tc.function.arguments
        return None, None

    def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return its result."""
        if tool_name == "search":
            query = tool_args.get("query", "")
            result = tool_search(query)
            return f"SEARCH RESULT for '{query}':\n{result}"
        return f"ERROR: Unknown tool '{tool_name}'"

    def count_tokens_est(self, messages: List[Dict[str, Any]]) -> int:
        """Rough token count (chars / 4)."""
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += len(block.get("text", ""))
            else:
                total += len(str(content))
        return total // 4

    # ---------------------------------------------------------------------------
    # Compaction methods
    # ---------------------------------------------------------------------------

    def _check_rubric(self, question: str) -> Dict[str, Any]:
        """
        Self-assess using the SelfCompact rubric before compaction.

        Returns a dict with C3, N1, C1, C2 scores and a recommendation.
        """
        rubric_prompt = (
            "You are self-evaluating your research progress. "
            "Answer these rubric rules honestly based on your conversation history.\n\n"
            "QUESTION: {question}\n\n"
            "RECENT TOOL CALLS:\n{recent_calls}\n\n"
            "RULES:\n"
            "C3 (Progress): Did my latest action make real progress toward answering the question? "
            "True if I learned something new, False if I repeated the same search or got similar info.\n"
            "N1 (Not Stuck): Have 3+ of my last 4 searches returned nothing new? "
            "True = stuck in a loop, False = making progress.\n"
            "C1 (Coverage): Does my context contain enough information to answer? "
            "Score 0-5: 0=missing everything, 5=have all needed info.\n"
            "C2 (Recency): Am I relying on old/irrelevant info from early in the conversation? "
            "Score 0-5: 0=all stale, 5=all relevant and recent.\n\n"
            "OUTPUT FORMAT (JSON only, no markdown):\n"
            '{{\n  "C3_progress": true/false,\n  "N1_stuck": true/false,\n'
            '  "C1_coverage": <0-5>,\n'
            '  "C2_recency": <0-5>,\n'
            '  "reason": "brief explanation"\n}}'
        )

        # Gather recent tool calls (last 4)
        recent_calls = ""
        if self.search_history:
            for i, call in enumerate(self.search_history[-4:]):
                recent_calls += f"- Search: \"{call['query']}\"\n"
                # Truncate result to last 200 chars to save tokens
                result_preview = call['result'][-200:] if len(call['result']) > 200 else call['result']
                recent_calls += f"  Result: {result_preview}\n\n"
        if not recent_calls:
            recent_calls = "(no tool calls yet)"

        rubric_msg = rubric_prompt.format(
            question=question,
            recent_calls=recent_calls,
        )

        try:
            response = self.call_model(
                self.messages + [{"role": "user", "content": rubric_msg}],
                tools=None,
                max_tokens=1024,
            )
            content = response.choices[0].message.get("content") or ""

            # Parse JSON from the response
            import re
            json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
            if json_match:
                rubric_result = json.loads(json_match.group())
            else:
                rubric_result = {
                    "C3_progress": True,  # default: assume progress if can't parse
                    "N1_stuck": False,
                    "C1_coverage": 3,
                    "C2_recency": 3,
                    "reason": "rubric parse fallback",
                }

            # Ensure required keys
            rubric_result.setdefault("C3_progress", True)
            rubric_result.setdefault("N1_stuck", False)
            rubric_result.setdefault("C1_coverage", 3)
            rubric_result.setdefault("C2_recency", 3)
            rubric_result.setdefault("reason", "")

            self.rubric_results.append(rubric_result)
            return rubric_result

        except Exception as e:
            # On failure, default to safe: don't compress, keep searching
            return {
                "C3_progress": True,
                "N1_stuck": False,
                "C1_coverage": 3,
                "C2_recency": 3,
                "reason": f"rubric_error: {e}",
            }

    def _decide_compaction(self, token_count: int) -> Tuple[bool, str]:
        """Decide whether to compress now, per the active policy."""
        if self.policy == "none":
            return False, ""

        if self.policy == "fixed":
            tool_calls_since_last = (
                self.tool_call_count - self.summarize_count * self.summarize_every_n_calls
            )
            if tool_calls_since_last >= self.summarize_every_n_calls:
                return True, f"fixed:every_{self.summarize_every_n_calls}_calls"
            return False, ""

        if self.policy == "selfcompact":
            # SelfCompact: check rubric first, only compress if justified
            if (self.tool_call_count - self.summarize_count * self.rubric_check_every_n_calls
                    >= self.rubric_check_every_n_calls):

                rubric = self._check_rubric(
                    # Extract question from prefix_messages
                    self.prefix_messages[1].get("content", "") if len(self.prefix_messages) > 1 else ""
                )

                c3 = rubric.get("C3_progress", True)
                n1_stuck = rubric.get("N1_stuck", False)
                c1_cov = rubric.get("C1_coverage", 3)

                if n1_stuck:
                    # N1: model is stuck in a loop — don't compress, force answer
                    return True, f"selfcompact:N1_stuck(need_answer)"

                if not c3:
                    # C3: no progress — compress now to break the loop
                    # (the summary will capture what's already known)
                    return True, f"selfcompact:C3_no_progress"

                if c1_cov >= 4:
                    # C1: high coverage — enough info, compress to free budget
                    return True, f"selfcompact:C1_coverage_{c1_cov}/5"

                if token_count > self.context_budget * 0.7:
                    # Context is getting large, compress if rubric is positive
                    if c3 and c1_cov >= 2:
                        return True, f"selfcompact:C3+C1_budget_guard"

                # Default: don't compress, keep gathering info
                return False, "selfcompact:rubric_nocompress"

        return False, ""

    def _summarize(self, question: str) -> Tuple[str, int]:
        """
        Ask the model to summarize its own context.

        The model is its OWN summarizer. Same model, no fine-tuning.
        """
        sum_prompt = (
            "You are summarizing your own research. Compress the conversation "
            "above into a concise summary that captures all essential findings. "
            "Include:\n"
            "1. What you've learned (specific facts, numbers, URLs)\n"
            "2. What you've tried (failed approaches, dead ends)\n"
            "3. What remains to be done\n"
            "\n"
            "Format:\n## Summary\n[Your research summary here]"
        )

        sum_messages = self.messages + [
            {"role": "user", "content": f"{sum_prompt}\n\nQuestion: {question}"}
        ]

        response = self.call_model(
            sum_messages,
            tools=None,  # No tools during summarization
            max_tokens=2048,
        )

        summary = (response.choices[0].message.get("content") or "").strip()
        summary_tokens = self.count_tokens_est([{"role": "assistant", "content": summary}])

        return summary, summary_tokens

    def _reset_after_summarize(self, question: str, summary: str) -> int:
        """Replace the full conversation with a compacted version."""
        new_messages = list(self.prefix_messages)  # Copy system + question
        new_messages.append({
            "role": "assistant",
            "content": f"Here is my research so far:\n{summary}"
        })
        new_messages.append({
            "role": "user",
            "content": "Continue your research. Use your tools to find the answer, "
                       "or provide your final answer if you have enough information."
        })

        self.messages = new_messages
        return self.count_tokens_est(self.messages)

    # ---------------------------------------------------------------------------
    # Run loop
    # ---------------------------------------------------------------------------

    def run(self, question: str) -> Dict[str, Any]:
        """Run the ReAct loop with optional compaction."""
        # Initialize
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Your task is to answer questions by "
                    "searching for information. When you have enough information, provide "
                    "your answer in the format: ANSWER: <your answer>.\n\n"
                    "Be thorough. Search multiple times if needed."
                )
            },
            {"role": "user", "content": question},
        ]
        self.prefix_messages = copy.deepcopy(self.messages)
        self.tool_call_count = 0
        self.iteration = 0
        self.summarize_count = 0
        self.search_history: List[Dict[str, str]] = []  # for N1 detection
        self.rubric_results: List[Dict[str, Any]] = []  # log rubric checks
        self.debug_data: Dict[str, Any] = {
            "iterations": [],
            "tool_calls": [],
            "token_counts": [],
            "compaction_events": [],
            "rubric_checks": [],  # Phase 3: rubric evaluation at each decision point
        }

        final_content = ""

        while self.tool_call_count < self.max_tool_calls:
            self.iteration += 1

            # Call the model
            print(f"\n--- Iteration {self.iteration} ---")
            token_count = self.count_tokens_est(self.messages)
            self.debug_data["token_counts"].append(token_count)
            print(f"  Context: {token_count:,} est tokens, {len(self.messages)} messages")

            # Hard capacity guard (Phase 5)
            if token_count >= self.context_budget:
                print(f"  [CAPACITY] Exceeds {self.context_budget:,} tokens. Forcing answer.")
                self.messages.append({
                    "role": "user",
                    "content": "Maximum length reached. Provide your final answer as: ANSWER: <answer>."
                })
                response = self.call_model(self.messages, tools=None, max_tokens=2048)
                final_content = (response.choices[0].message.get("content") or "").strip()
                break

            response = self.call_model(self.messages, tools=self.tools, max_tokens=4096)
            message = response.choices[0].message
            content = (message.get("content") or "").strip() or (message.get("generated_text") or "").strip()
            self.messages.append({"role": "assistant", "content": content})
            final_content = content

            # Check for tool call
            tool_name, tool_args_str = self.parse_tool_call(message)

            if tool_name:
                print(f"  -> Tool call: {tool_name}({tool_args_str[:100]})")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {"raw": tool_args_str}

                result = self.execute_tool(tool_name, tool_args)
                self.tool_call_count += 1

                print(f"  <- Result: {result[:200]}...")

                tool_message = {
                    "role": "tool",
                    "tool_call_id": f"call_{self.iteration}",
                    "name": tool_name,
                    "content": result,
                }
                self.messages.append(tool_message)

                self.debug_data["tool_calls"].append({
                    "iteration": self.iteration,
                    "tool": tool_name,
                    "args": tool_args,
                })
                self.debug_data["iterations"].append({
                    "iteration": self.iteration,
                    "role": "tool_call",
                })

                # --- Compaction decision ---
                if self.policy != "none":
                    # Track search history for N1 detection (selfcompact)
                    if tool_name == "search":
                        self.search_history.append({
                            "query": tool_args.get("query", ""),
                            "result": result[:200],  # truncated for logging
                        })

                    fire, reason = self._decide_compaction(token_count)
                    if fire:
                        print(f"  [COMPACT] {reason}")

                        # Extract rubric results for logging
                        rubric_log = []
                        if self.policy == "selfcompact" and self.rubric_results:
                            rubric_log = self.rubric_results[-1:]  # most recent check

                        self.debug_data["rubric_checks"].extend(rubric_log)

                        if "N1_stuck" in reason:
                            # N1: model is stuck, force answer instead of summarizing
                            print(f"  [N1] Model stuck in loop — forcing answer.")
                            self.messages.append({
                                "role": "user",
                                "content": (
                                    "You have repeated the same search multiple times without new info. "
                                    "Stop searching and provide your best answer as: ANSWER: <your answer>."
                                ),
                            })
                            response = self.call_model(
                                self.messages, tools=None, max_tokens=2048
                            )
                            final_content = (response.choices[0].message.get("content") or "").strip()
                            self.debug_data["iterations"].append({
                                "iteration": self.iteration,
                                "role": "forced_answer_N1",
                            })
                            break
                        else:
                            summary, sum_tokens = self._summarize(question)
                            print(f"  [COMPACT] Summary: {summary[:100]}...")
                            tokens_after = self._reset_after_summarize(question, summary)
                            self.summarize_count += 1

                            self.debug_data["compaction_events"].append({
                                "iteration": self.iteration,
                                "tokens_before": token_count,
                                "tokens_after": tokens_after,
                                "summary_tokens": sum_tokens,
                                "reason": reason,
                                "summary": summary,
                            })
                            print(f"  [COMPACT] {token_count:,} -> {tokens_after:,} tokens")
                            continue

            else:
                print(f"  -> Final answer: {content[:200]}...")
                self.debug_data["iterations"].append({
                    "iteration": self.iteration,
                    "role": "final_answer",
                })
                break

        return {
            "question": question,
            "answer": final_content,
            "iterations": self.iteration,
            "tool_calls": self.tool_call_count,
            "summarizations": self.summarize_count,
            "debug_data": self.debug_data,
        }
