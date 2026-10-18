"""
Phase 1: Bare ReAct Agent Loop (no compaction)

This is the foundation. A ReAct agent runs in a loop:
  1. Send the current conversation to the model
  2. Model responds with either a tool call OR a final answer
  3. If tool call: execute it, append result, loop back
  4. If answer: done

The key data structure is `messages` -- a list of dicts with roles:
  - "system": The instructions (always at index 0)
  - "user": The question
  - "assistant": Model's response (may contain tool_calls)
  - "tool": Tool result (has tool_call_id to match the call)

We call this "ReAct" because the model alternates between
"Reasoning" (thinking about what to do next) and "Acting" (calling tools).

IMPORTANT: In the SelfCompact paper, the agent is *its own summarizer*.
The same model that does the research also decides when to compress.
No external judge, no fine-tuning. Just prompts.
"""

import asyncio
import json
import os
import copy
import re
from typing import List, Dict, Any, Optional, Tuple

import litellm


# ---------------------------------------------------------------------------
# Simple tool definitions
# ---------------------------------------------------------------------------
# These simulate what a real agent does: call a tool, get a result, loop.
# In SelfCompact, the tools are web search / visit. Here we use synthetic
# tools so we can run without internet.


def tool_search(query: str) -> str:
    """Simulates a web search. Returns synthetic results.

    The search is keyword-sensitive: different queries return different results.
    If the query contains a unique keyword, it gets unique results.
    If the query is a repeat (contains only common words), it returns "no new info".
    """
    q_lower = query.lower()

    # Check for unique keywords to determine which "page" to return
    reef_queries = ["reef", "coral", "great barrier", "australia"]
    crispr_queries = ["crispr", "gene", "dna", "editing", "cas9"]
    mars_queries = ["mars", "phobos", "deimos", "moon", "satellite"]

    def count_matches(words, text):
        return sum(1 for w in words if w in text)

    reef_score = count_matches(reef_queries, q_lower)
    crispr_score = count_matches(crispr_queries, q_lower)
    mars_score = count_matches(mars_queries, q_lower)

    if reef_score > crispr_score and reef_score > mars_score:
        # For reef, check if query seems "fresh" (has specific sub-terms)
        specific_terms = ["size", "area", "km", "species", "fish", "turtle", "marine", "living", "structure"]
        has_specific = sum(1 for t in specific_terms if t in q_lower)

        if has_specific >= 2:
            # Different aspects of the reef — return different results
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
            # Generic search — same result (simulating repeated searches)
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
                                "acronym for Clustered Regularly Interspaced Short Palindromic Repeats.'"
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

    # Default fallback
    return json.dumps({"result_1": "No relevant results found.", "result_2": "No relevant results found."})


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """A minimal ReAct agent -- the foundation we'll build compaction on."""

    def __init__(self, api_base: str = "http://127.0.0.1:8000/v1",
                 model: str = "nvidia/Qwen3.6-35B-A3B-NVFP4",
                 max_tool_calls: int = 20):
        self.api_base = api_base
        self.model = model
        self.max_tool_calls = max_tool_calls

        # The conversation history
        self.messages: List[Dict[str, Any]] = []
        self.prefix_messages: List[Dict[str, Any]] = []  # system + question (never changes)

        # Tool definitions (what the model knows about)
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

    async def call_model(self, messages: List[Dict[str, Any]],
                         tools: Optional[List[Dict]] = None,
                         max_tokens: int = 4096) -> Dict[str, Any]:
        """Call the LLM API. Returns the raw response."""
        # Pass a dummy api_key -- required by litellm even for local vLLM
        response = await litellm.acompletion(
            model=f"openai/{self.model}",
            api_base=self.api_base,
            api_key="EMPTY",
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.7,
            timeout=300,
            num_retries=0,
        )
        return response

    def parse_tool_call(self, message: Any) -> Optional[Tuple[str, str]]:
        """
        Extract a tool call from the assistant message.

        Returns (tool_name, json_string_arguments) or None if no tool call.

        This is where we learn how the model communicates its intent.
        When the model decides to act, it outputs a tool_call instead of text.
        """
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tc = message.tool_calls[0]
            return tc.function.name, tc.function.arguments
        return None

    async def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return its result as a string."""
        if tool_name == "search":
            query = tool_args.get("query", "")
            result = tool_search(query)
            return f"SEARCH RESULT for '{query}':\n{result}"
        else:
            return f"ERROR: Unknown tool '{tool_name}'"

    def count_tokens_est(self, messages: List[Dict[str, Any]]) -> int:
        """
        Rough token count (chars / 4).

        NOTE: In production, use the actual tokenizer.
        The SelfCompact code uses `transformers.AutoTokenizer` for accuracy.

        Why this matters:
        - The model has a context limit (Qwen3.6-35B: 262k tokens)
        - We need to know when we're getting close to trigger compaction
        - Token counting is one of the spec's risk areas ("inaccurate token counting")
        """
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, list):
                # Multimodal: count text blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += len(block.get("text", ""))
            else:
                total += len(str(content))
        return total // 4  # Rough estimate: 4 chars per token

    async def run(self, question: str) -> Dict[str, Any]:
        """
        Run the ReAct loop. This is the core of Phase 1.

        The loop:
          1. Send messages to the model
          2. Check if the response has a tool call or a final answer
          3. If tool call: execute it, add result to messages, repeat
          4. If answer: stop and return the result

        No compaction yet -- this is the "none" policy baseline.
        We'll add compaction in Phase 2.
        """
        # Initialize with system prompt + question
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
        self.debug_data: Dict[str, Any] = {
            "iterations": [],
            "tool_calls": [],
            "token_counts": [],
            "compaction_events": [],
        }

        final_content = ""

        while self.tool_call_count < self.max_tool_calls:
            self.iteration += 1

            # Step 1: Call the model with current context
            print(f"\n--- Iteration {self.iteration} ---")
            token_count = self.count_tokens_est(self.messages)
            self.debug_data["token_counts"].append(token_count)
            print(f"  Context: {token_count:,} estimated tokens, "
                  f"{len(self.messages)} messages")

            response = await self.call_model(
                self.messages, tools=self.tools, max_tokens=4096
            )

            message = response.choices[0].message

            # Step 2: Parse the response
            content = message.get("content", "") or ""
            self.messages.append({"role": "assistant", "content": content})
            final_content = content

            # Step 3: Check for tool call
            tool_result = self.parse_tool_call(message)

            if tool_result:
                tool_name, tool_args_str = tool_result
                print(f"  -> Tool call: {tool_name}({tool_args_str[:100]})")

                # Parse args
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {"raw": tool_args_str}

                # Step 4: Execute the tool
                result = await self.execute_tool(tool_name, tool_args)
                self.tool_call_count += 1

                print(f"  <- Result: {result[:200]}...")

                # Step 5: Add tool result to messages
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

            else:
                # No tool call -- the model is providing a final answer
                print(f"  -> Final answer: {content[:200]}...")
                self.debug_data["iterations"].append({
                    "iteration": self.iteration,
                    "role": "final_answer",
                })
                break

        # Return the results
        return {
            "question": question,
            "answer": final_content,
            "iterations": self.iteration,
            "tool_calls": self.tool_call_count,
            "debug_data": self.debug_data,
        }
