"""
graph/reranker.py

Cross-encoder reranker with:
- Intent-driven section boosting (via intent_classifier)
- Full NegEx generalized negation detection
- Prescription-language boosting
- Diversity filtering
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None

# ── NegEx patterns ────────────────────────────────────────────────────────────
# These are clinical negation trigger phrases. A chunk whose key term is
# immediately preceded (within 4 words) by one of these gets downranked.
_NEGATION_TRIGGERS = re.compile(
    r"\b("
    r"no\s+known|no\s+history\s+of|no\s+evidence\s+of|no\s+sign\s+of"
    r"|not\s+taking|not\s+prescribed|not\s+on"
    r"|denies|denied|without"
    r"|negative\s+for|nkda"
    r"|discontinued|stopped|held|not\s+started"
    r")\b",
    re.IGNORECASE,
)


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker: {MODEL_NAME}")
        _reranker = CrossEncoder(MODEL_NAME)
    return _reranker


def _diversify_by_section(chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Keep at most one chunk per section first, then fill remaining slots."""
    seen: set = set()
    diverse: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for c in chunks:
        section = (c.get("section") or "").strip().lower()
        if section and section not in seen:
            seen.add(section)
            diverse.append(c)
        else:
            leftovers.append(c)

        if len(diverse) >= top_k:
            return diverse

    for c in leftovers:
        if len(diverse) >= top_k:
            break
        diverse.append(c)

    return diverse


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by first 150 chars of chunk text."""
    seen: set = set()
    unique: List[Dict[str, Any]] = []

    for c in chunks:
        text = (c.get("chunk_text") or "").strip()[:150]
        if text not in seen:
            seen.add(text)
            unique.append(c)

    return unique


def _negation_penalty(text: str, query_terms: List[str]) -> float:
    """
    Generalized NegEx penalty.

    For each significant query term, check if it appears in the chunk text
    with a negation trigger within a 5-word window before it.
    Returns total penalty (negative float, 0.0 if no negations found).
    """
    total_penalty = 0.0
    text_lower = text.lower()

    for term in query_terms:
        stem = term[:5]
        # Pattern: negation trigger, then 0-4 words, then the term stem
        pattern = (
            r"(?:" + _NEGATION_TRIGGERS.pattern + r")"
            r"(?:\s+\w+){0,4}\s+" + re.escape(stem) + r"\w*"
        )
        if re.search(pattern, text_lower, re.IGNORECASE):
            total_penalty -= 5.0

    return total_penalty


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    intent: str = "general",
    priority_sections: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank chunks using cross-encoder scores plus clinical boosting.

    Args:
        query: user query string
        chunks: list of chunk dicts
        top_k: number of chunks to return
        intent: classified query intent (from intent_classifier)
        priority_sections: ordered list of high-priority section names
    """
    if not chunks:
        return []

    from graph.intent_classifier import section_boost

    if priority_sections is None:
        priority_sections = []

    reranker = get_reranker()

    # Step 1: deduplicate
    chunks = deduplicate_chunks(chunks)

    # Step 2: create (query, chunk_text) pairs for cross-encoder
    pairs = [(query, c.get("chunk_text", "")) for c in chunks]

    try:
        scores = reranker.predict(pairs, batch_size=16)
    except Exception as e:
        print(f"[Reranker ERROR]: {e}")
        return chunks[:top_k]

    # Step 3: attach raw cross-encoder scores
    reranked: List[Dict[str, Any]] = []
    for chunk, score in zip(chunks, scores):
        new_chunk = dict(chunk)
        new_chunk["reranker_score"] = float(score)
        reranked.append(new_chunk)

    # Step 4: sort by raw cross-encoder score first
    reranked = sorted(reranked, key=lambda x: x["reranker_score"], reverse=True)

    # Precompute query terms for NegEx (words ≥ 5 chars)
    query_terms = [w.lower() for w in re.findall(r'\b[a-zA-Z]{5,}\b', query)]

    # Step 5: apply clinical boosting/penalization per chunk
    for c in reranked:
        section = (c.get("section") or "").upper().strip()
        text = c.get("chunk_text", "")
        text_lower = text.lower()
        specialty = (c.get("specialty") or "").lower()

        # 5a. Section priority boost (intent-driven)
        boost = section_boost(section, priority_sections)
        c["reranker_score"] += boost

        # 5b. Prescription-language boost (for medication intent)
        if intent == "medication":
            PRESCRIPTION_KEYWORDS = [
                "given", "prescribed", "started on", "started",
                "samples", "written", "initiated", "administered",
                "dispensed", "ordered",
            ]
            if any(kw in text_lower for kw in PRESCRIPTION_KEYWORDS):
                c["reranker_score"] += 1.0

        # 5c. Specialty match boost
        query_words = set(w.strip("?.,") for w in query.lower().split())
        specialty_words = set(
            specialty.replace("/", " ").replace("-", " ").split()
        )
        if query_words & specialty_words:
            c["reranker_score"] += 1.5

        # 5d. Generalized NegEx penalty
        penalty = _negation_penalty(text, query_terms)
        c["reranker_score"] += penalty

    # Step 6: re-sort after all adjustments
    reranked = sorted(reranked, key=lambda x: x["reranker_score"], reverse=True)

    # Step 7: drop very low-score chunks (hard floor)
    reranked = [c for c in reranked if c["reranker_score"] > -5]

    if not reranked:
        # Fallback: just use original ordering
        reranked = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    # Step 8: diversity filter
    reranked = _diversify_by_section(reranked, top_k)

    return reranked[:top_k]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ingestion.retriver import hybrid_search, expand_with_parent_chunks
    from graph.intent_classifier import classify_intent

    query = "What medications was the allergy patient prescribed?"
    intent, priority_sections = classify_intent(query)
    print(f"Query: {query}")
    print(f"Intent: {intent} | Sections: {priority_sections}\n")

    chunks = hybrid_search(query, top_k=20)
    chunks = expand_with_parent_chunks(chunks, max_siblings=5)
    chunks = deduplicate_chunks(chunks)

    print(f"After hybrid search + expansion: {len(chunks)} chunks")

    reranked = rerank_chunks(query, chunks, top_k=5, intent=intent, priority_sections=priority_sections)
    print(f"\nAfter reranking ({intent} intent): {len(reranked)} chunks")
    for i, c in enumerate(reranked, 1):
        print(
            f"  {i}. [{c.get('section', '?')}] "
            f"[score={c['reranker_score']:.3f}] "
            f"{c.get('chunk_text', '')[:80]}..."
        )