"""
graph/retrieval_validator.py

ReAG-style retrieval validator.

After hybrid retrieval + reranking, passes each chunk through an LLM
to verify it is clinically relevant to the query. Irrelevant chunks
(e.g., "No known drug allergies" for a medication query) are filtered out
before answer generation.

Batches all chunks into a single LLM call to minimize latency.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

OLLAMA_MODEL = "llama3"
# Max chunks to send for validation per call (to stay within context window)
MAX_CHUNKS_PER_VALIDATION = 8

_llm: Optional[ChatOllama] = None

_VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a clinical information retrieval validator.

You will be given:
1. A clinical question.
2. A numbered list of retrieved text chunks.

For EACH chunk, reply with its number followed by YES or NO:
- YES: the chunk directly helps answer the question
- NO: the chunk is irrelevant, contradictory, or about a different patient/topic

Output ONLY the numbered list, one per line, like:
1: YES
2: NO
3: YES

Do NOT explain. Do NOT add any other text.""",
    ),
    (
        "human",
        "Question: {question}\n\nChunks:\n{chunks_text}",
    ),
])


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=4096)
    return _llm


def _format_chunks_for_validation(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks as a numbered list for the validator prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        text = (chunk.get("chunk_text") or "").strip()
        section = chunk.get("section", "?")
        # Truncate long chunks to save context
        if len(text) > 400:
            text = text[:400] + "…"
        lines.append(f"{i}. [{section}] {text}")
    return "\n\n".join(lines)


def _parse_validation_response(response: str, num_chunks: int) -> List[bool]:
    """
    Parse the numbered YES/NO response from the LLM.
    Returns a list of booleans (True = keep, False = discard).
    Defaults to True (keep) if a line cannot be parsed — safe fallback.
    """
    decisions = [True] * num_chunks  # default: keep everything

    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Match patterns like "1: YES", "1. NO", "1 YES"
        match = re.match(r'^(\d+)\s*[:.]\s*(YES|NO)', line, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1  # 0-indexed
            verdict = match.group(2).upper()
            if 0 <= idx < num_chunks:
                decisions[idx] = (verdict == "YES")

    return decisions


def validate_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    min_chunks: int = 2,
) -> List[Dict[str, Any]]:
    """
    Filter retrieved chunks to those the LLM deems clinically relevant.

    Args:
        query: the user question
        chunks: list of reranked chunk dicts
        min_chunks: always keep at least this many chunks even if all fail validation
                    (safety net to avoid empty context)

    Returns:
        Filtered list of chunks. If LLM fails, returns all chunks (safe fallback).
    """
    if not chunks:
        return []

    # Only validate up to MAX_CHUNKS_PER_VALIDATION chunks
    chunks_to_validate = chunks[:MAX_CHUNKS_PER_VALIDATION]
    passthrough = chunks[MAX_CHUNKS_PER_VALIDATION:]  # rest always pass through

    try:
        llm = get_llm()
        chain = _VALIDATION_PROMPT | llm | StrOutputParser()

        chunks_text = _format_chunks_for_validation(chunks_to_validate)
        response = chain.invoke({
            "question": query,
            "chunks_text": chunks_text,
        })

        print(f"[Validator] Response:\n{response.strip()}")

        decisions = _parse_validation_response(response, len(chunks_to_validate))
        validated = [c for c, keep in zip(chunks_to_validate, decisions) if keep]

        print(
            f"[Validator] Kept {len(validated)}/{len(chunks_to_validate)} chunks "
            f"(+ {len(passthrough)} passthrough)"
        )

        # Safety net: never return fewer than min_chunks
        if len(validated) < min_chunks:
            print(f"[Validator] Too few chunks passed ({len(validated)}), keeping top {min_chunks}")
            validated = chunks_to_validate[:min_chunks]

        return validated + passthrough

    except Exception as e:
        print(f"[Validator ERROR]: {e} — returning all chunks unfiltered")
        return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ingestion.retriver import hybrid_search, expand_with_parent_chunks
    from graph.reranker import rerank_chunks, deduplicate_chunks
    from graph.intent_classifier import classify_intent

    query = "What medications was the allergy patient prescribed?"
    intent, priority_sections = classify_intent(query)

    chunks = hybrid_search(query, top_k=20, intent=intent, priority_sections=priority_sections)
    chunks = expand_with_parent_chunks(chunks, max_siblings=3)
    chunks = deduplicate_chunks(chunks)
    chunks = rerank_chunks(query, chunks, top_k=8, intent=intent, priority_sections=priority_sections)

    print(f"\nBefore validation: {len(chunks)} chunks")
    validated = validate_chunks(query, chunks)
    print(f"After validation: {len(validated)} chunks\n")

    for i, c in enumerate(validated, 1):
        print(f"  {i}. [{c.get('section')}] {c.get('chunk_text', '')[:100]}…")
