from __future__ import annotations

from typing import List, Dict, Any, Optional
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "medisense_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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


def infer_section(query: str) -> Optional[str]:
    q = query.lower()

    if "symptom" in q or "complaint" in q:
        return "SUBJECTIVE"
    if "diagnos" in q or "assessment" in q or "impression" in q:
        return "ASSESSMENT"
    if "treat" in q or "medication" in q or "drug" in q or "plan" in q:
        return "PLAN"

    return None


def build_section_filter(section: Optional[str]) -> Optional[Filter]:
    if not section:
        return None
    return Filter(
        must=[FieldCondition(key="section", match=MatchValue(value=section))]
    )


def dedupe_results(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Prefer stable IDs for dedupe
    seen_ids = set()
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


def search_chunks(
    query: str,
    top_k: int = 10,
    section: Optional[str] = None,
    *,
    use_section_routing: bool = True,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for chunks most similar to query.

    section behavior:
      - if section is not None: force that section filter (even if empty string -> no filter)
      - else if use_section_routing=True: infer_section(query)
      - else: no section filter

    score_threshold: drop results with score < threshold (optional)
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

    if section is not None:
        resolved_section = section or None  # allow passing "" to mean no filter
    else:
        resolved_section = infer_section(query) if use_section_routing else None

    query_filter = build_section_filter(resolved_section)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )
    if not results.points and query_filter is not None:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    retrieved: List[Dict[str, Any]] = []
    for p in results.points:
        payload = p.payload or {}
        retrieved.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "chunk_text": payload.get("chunk_text"),
                "score": p.score,
                "specialty": payload.get("specialty"),
                "parent_doc_id": payload.get("parent_doc_id"),
                "section": payload.get("section"),
            }
        )

    retrieved = dedupe_results(retrieved)

    if score_threshold is not None:
        retrieved = [r for r in retrieved if float(r.get("score") or 0.0) >= score_threshold]

    return retrieved