"""
ingestion/retriver.py

Intent-aware hybrid retrieval with:
- Negation pre-filtering at the Qdrant level (negation_flag field)
- Section-constrained parent chunk expansion
- BM25 keyword search + dense RRF fusion
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "medisense_chunks_v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Intents for which we should pre-filter negated chunks at the DB level.
NEGATION_FILTER_INTENTS = {"medication", "treatment_plan", "diagnosis"}

# Cache: whether the negation_flag payload index exists in the collection.
# None = not yet probed, True/False = result of probe.
_negation_index_available: Optional[bool] = None

_model: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading retrieval embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_client() -> QdrantClient:
    global _client
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing Qdrant credentials: set QDRANT_URL and QDRANT_API_KEY")
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def _probe_negation_index() -> bool:
    """
    Check once whether the negation_flag payload index exists in Qdrant.
    Caches the result so subsequent calls are free.
    """
    global _negation_index_available
    if _negation_index_available is not None:
        return _negation_index_available

    try:
        client = get_client()
        info = client.get_collection(COLLECTION_NAME)
        # payload_schema is a dict of field_name -> FieldType
        schema = info.payload_schema or {}
        _negation_index_available = "negation_flag" in schema
        if not _negation_index_available:
            print(
                "[Retriever] negation_flag index not found in collection. "
                "Negation pre-filtering disabled. "
                "Run ingestion/create_negation_index.py to enable it."
            )
    except Exception:
        _negation_index_available = False

    return _negation_index_available


def build_query_filter(
    section: Optional[str] = None,
    exclude_negated: bool = False,
) -> Optional[Filter]:
    """
    Build a Qdrant filter combining:
    - Section match (if provided)
    - Negation exclusion (only if negation_flag index exists in the collection)
    """
    conditions = []

    if section:
        conditions.append(
            FieldCondition(key="section", match=MatchValue(value=section))
        )

    if exclude_negated and _probe_negation_index():
        conditions.append(
            FieldCondition(key="negation_flag", match=MatchValue(value=False))
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def dedupe_results(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids: set = set()
    unique: List[Dict[str, Any]] = []
    for r in retrieved:
        cid = r.get("chunk_id")
        text = (r.get("chunk_text") or "").strip()
        if not cid or not text:
            continue
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        unique.append(r)
    return unique


def _points_to_dicts(points) -> List[Dict[str, Any]]:
    """Convert Qdrant ScoredPoint objects to standard chunk dict format."""
    results = []
    for p in points:
        payload = p.payload or {}
        results.append({
            "chunk_id": payload.get("chunk_id"),
            "chunk_text": payload.get("chunk_text"),
            "score": p.score if hasattr(p, "score") else 0.0,
            "specialty": payload.get("specialty"),
            "parent_doc_id": payload.get("parent_doc_id"),
            "section": payload.get("section"),
            "negation_flag": payload.get("negation_flag", False),
            "document_type": payload.get("document_type", ""),
        })
    return results


def search_chunks(
    query: str,
    top_k: int = 10,
    section: Optional[str] = None,
    intent: str = "general",
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Dense embedding search with optional section filter and negation pre-filter.

    Args:
        query: natural language query
        top_k: number of results to return
        section: if set, restrict to this section only
        intent: classified intent — controls whether negated chunks are pre-filtered
        score_threshold: drop results below this similarity score
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    model = get_model()
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].tolist()

    client = get_client()

    exclude_negated = intent in NEGATION_FILTER_INTENTS
    query_filter = build_query_filter(section=section, exclude_negated=exclude_negated)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

    # Fallback: if the filtered query returns nothing, try without filters
    if not results.points and query_filter is not None:
        print(f"[Retriever] Filter returned 0 results, retrying without filter")
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    retrieved = _points_to_dicts(results.points)
    retrieved = dedupe_results(retrieved)

    if score_threshold is not None:
        retrieved = [r for r in retrieved if float(r.get("score") or 0.0) >= score_threshold]

    return retrieved


def keyword_search(
    query: str,
    top_k: int = 10,
    intent: str = "general",
) -> List[Dict[str, Any]]:
    """
    Full-text keyword search using Qdrant's text index.
    Catches exact drug names and clinical terms that dense models miss.
    """
    client = get_client()
    exclude_negated = intent in NEGATION_FILTER_INTENTS

    filter_conditions = [
        FieldCondition(key="chunk_text", match=MatchText(text=query))
    ]
    if exclude_negated and _probe_negation_index():
        filter_conditions.append(
            FieldCondition(key="negation_flag", match=MatchValue(value=False))
        )

    try:
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=filter_conditions),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0]
    except Exception as e:
        print(f"[Keyword search ERROR]: {e}")
        return []

    retrieved = []
    for p in points:
        payload = p.payload or {}
        retrieved.append({
            "chunk_id": payload.get("chunk_id"),
            "chunk_text": payload.get("chunk_text"),
            "score": 0.5,
            "specialty": payload.get("specialty"),
            "parent_doc_id": payload.get("parent_doc_id"),
            "section": payload.get("section"),
            "negation_flag": payload.get("negation_flag", False),
            "document_type": payload.get("document_type", ""),
            "match_type": "keyword",
        })

    return retrieved


def reciprocal_rank_fusion(
    *result_lists: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).
    Each chunk gets score = sum(1 / (k + rank)) across all lists it appears in.
    """
    scores: Dict[str, float] = {}
    chunks_by_id: Dict[str, Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = chunk.get("chunk_id")
            if not cid:
                continue
            rrf_score = 1.0 / (k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + rrf_score
            if cid not in chunks_by_id:
                chunks_by_id[cid] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for cid in sorted_ids:
        chunk = dict(chunks_by_id[cid])
        chunk["rrf_score"] = scores[cid]
        fused.append(chunk)

    return fused


def hybrid_search(
    query: str,
    top_k: int = 10,
    intent: str = "general",
    priority_sections: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Combine dense (embedding) + sparse (keyword) retrieval using RRF.

    When priority_sections is provided (from intent classifier), run an
    additional dense search restricted to those sections and fuse results.
    This gives the retriever a section-aware bias without hard-filtering
    everything else out.
    """
    # Dense retrieval — wide net, no section filter
    dense_results = search_chunks(query, top_k=top_k * 2, intent=intent)

    # Section-targeted dense retrieval for top priority section
    section_results: List[Dict[str, Any]] = []
    if priority_sections:
        for section in priority_sections[:2]:  # top 2 priority sections
            sec_results = search_chunks(
                query,
                top_k=top_k,
                section=section,
                intent=intent,
            )
            section_results.extend(sec_results)

    # Keyword retrieval
    keyword_results = keyword_search(query, top_k=top_k * 2, intent=intent)

    # Fuse all result lists
    all_lists = [dense_results, keyword_results]
    if section_results:
        all_lists.append(section_results)

    fused = reciprocal_rank_fusion(*all_lists, k=60)
    return fused[:top_k]


def expand_with_parent_chunks(
    chunks: List[Dict[str, Any]],
    max_siblings: int = 5,
    same_section_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each retrieved chunk, fetch sibling chunks from the same parent document.

    Args:
        chunks: initial retrieved chunks
        max_siblings: max siblings per parent document
        same_section_only: if True (default), only expand within the same section.
            This prevents pulling unrelated sections into context.
    """
    client = get_client()
    seen = {c["chunk_id"] for c in chunks if c.get("chunk_id")}
    expanded = list(chunks)

    # Group chunks by (parent_doc_id, section) for section-constrained expansion
    parent_sections: Dict[str, set] = {}
    for c in chunks:
        pid = c.get("parent_doc_id")
        sec = (c.get("section") or "").upper().strip()
        if pid:
            parent_sections.setdefault(pid, set()).add(sec)

    for pid, sections in parent_sections.items():
        for section in sections:
            filter_conditions = [
                FieldCondition(key="parent_doc_id", match=MatchValue(value=pid))
            ]
            if same_section_only and section:
                filter_conditions.append(
                    FieldCondition(key="section", match=MatchValue(value=section))
                )

            try:
                siblings, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=filter_conditions),
                    limit=max_siblings,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in siblings:
                    payload = point.payload or {}
                    cid = payload.get("chunk_id")
                    if cid and cid not in seen:
                        seen.add(cid)
                        expanded.append({
                            "chunk_id": cid,
                            "chunk_text": payload.get("chunk_text"),
                            "score": 0.4,
                            "specialty": payload.get("specialty"),
                            "parent_doc_id": pid,
                            "section": payload.get("section"),
                            "negation_flag": payload.get("negation_flag", False),
                            "document_type": payload.get("document_type", ""),
                            "match_type": "parent_expansion",
                        })
            except Exception as e:
                print(f"[Parent expansion ERROR for {pid}/{section}]: {e}")

    return expanded