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
    """
    Very simple domain routing. Expand over time.
    """
    q = query.lower()

    # symptoms / complaints
    if "symptom" in q or "complaint" in q:
        return "SUBJECTIVE"

    # diagnosis / impression
    if "diagnos" in q or "assessment" in q or "impression" in q:
        return "ASSESSMENT"

    # treatment plan
    if "treat" in q or "medication" in q or "drug" in q or "plan" in q:
        return "PLAN"

    return None


def build_section_filter(section: Optional[str]) -> Optional[Filter]:
    if not section:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="section",
                match=MatchValue(value=section),
            )
        ]
    )


def dedupe_results(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []

    for r in retrieved:
        text = (r.get("chunk_text") or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        unique.append(r)

    return unique


def search_chunks(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search Qdrant for chunks most similar to the query.
    Uses optional section routing -> section filter -> dedupe.
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

    section = infer_section(query)
    query_filter = build_section_filter(section)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        query_filter=query_filter,
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

    return dedupe_results(retrieved)


if __name__ == "__main__":
    queries = [
        # Matches SUBJECTIVE sections
        "patient presents with nasal congestion and allergic rhinitis",

        # Matches ASSESSMENT sections
        "diagnosis of morbid obesity requiring surgical intervention",

        # Matches PLAN sections
        "prescribed medication and follow up instructions",

        # Matches cardiology notes
        "left ventricular ejection fraction systolic function",

        # Matches surgical notes
        "laparoscopic procedure anesthesia endotracheal intubation",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = search_chunks(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] {r['specialty']} — {r['section']}")
            print(f"      {r['chunk_text'][:120]}...")