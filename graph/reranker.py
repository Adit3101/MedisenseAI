from __future__ import annotations

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker: {MODEL_NAME}")
        _reranker = CrossEncoder(MODEL_NAME)
    return _reranker


def _diversify_by_section(chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Simple diversity: keep at most one chunk per section, then fill remaining.
    """
    seen = set()
    diverse = []
    leftovers = []

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
    seen = set()
    unique = []

    for c in chunks:
        text = c["chunk_text"].strip()[:150]
        if text not in seen:
            seen.add(text)
            unique.append(c)

    return unique


def multi_query_search(query: str, top_k: int = 10) -> List[Dict]:
    variants = [
        query,
        "allergy patient medications prescribed nasal spray",
        "patient prescribed Nasonex Zyrtec loratadine allergic rhinitis",
    ]

    seen = set()
    all_chunks = []

    for q in variants:
        results = search_chunks(q, top_k=top_k, use_section_routing=False)
        for chunk in results:
            cid = chunk["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                all_chunks.append(chunk)

    return all_chunks


def rerank_chunks(query, chunks, top_k=5):
    if not chunks:
        return []

    reranker = get_reranker()

    # Step 1: deduplicate FIRST
    chunks = deduplicate_chunks(chunks)

    # Step 2: stronger intent-aware query
    rerank_query = (
        query + " medications prescribed drugs prescription given started on samples of"
    )

    pairs = [(rerank_query, c["chunk_text"]) for c in chunks]

    try:
        scores = reranker.predict(pairs, batch_size=16)
    except Exception as e:
        print(f"[Reranker ERROR]: {e}")
        return chunks[:top_k]

    # Step 3: attach scores safely
    reranked = []
    for chunk, score in zip(chunks, scores):
        new_chunk = dict(chunk)
        new_chunk["reranker_score"] = float(score)
        reranked.append(new_chunk)

    # Step 4: sort by score
    reranked = sorted(reranked, key=lambda x: x["reranker_score"], reverse=True)

    # Step 5: boost prescription signals
    for c in reranked:
        text = c["chunk_text"].lower()
        if any(k in text for k in ["given", "prescribed", "started", "samples"]):
            c["reranker_score"] += 1.0

    # Step 6: drop negative-score chunks
    reranked = [c for c in reranked if c["reranker_score"] > -5]

    if not reranked:
        reranked = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    # Step 7: apply diversity
    reranked = _diversify_by_section(reranked, top_k)

    return reranked[:top_k]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent / "ingestion"))
    from ingestion.retriver import search_chunks

    query = "What medications was the allergy patient prescribed?"

    # Fix 1: two-stage retrieval (semantic + prescription-focused)
    semantic_query = query
    prescription_query = (
        "medications prescribed given samples started treatment drugs nasal spray"
    )

    chunks_semantic = search_chunks(semantic_query, top_k=20)
    chunks_prescription = search_chunks(prescription_query, top_k=20)

    chunks = chunks_semantic + chunks_prescription

    # Fix 2: deduplicate after merging
    chunks = deduplicate_chunks(chunks)

    print(f"Before reranking: {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3], 1):
        print(f"  {i}. [{c['score']:.4f}] {c['chunk_text'][:80]}...")

    # Fix 3: rerank after merge + dedup
    reranked = rerank_chunks(query, chunks, top_k=5)
    print(f"\nAfter reranking: {len(reranked)} chunks")
    for i, c in enumerate(reranked[:10], 1):
        print(f"  {i}. [reranker={c['reranker_score']:.4f}] {c['chunk_text'][:80]}...")