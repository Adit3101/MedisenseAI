from __future__ import annotations

from typing import List, Dict, Any, Tuple
from pathlib import Path
import os
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "medisense_chunks"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment/.env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client: QdrantClient) -> None:
    """
    Create collection if missing; if it exists, do nothing.
    """
    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f" Collection '{COLLECTION_NAME}' already exists (points={info.points_count})")
        return
    except UnexpectedResponse:
        pass  # treat as "does not exist"

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(" Collection created")


def build_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    # Keep payload lean-ish. Storing chunk_text is OK for now.
    return {
        "chunk_id": chunk["chunk_id"],
        "chunk_text": chunk["chunk_text"],
        "parent_doc_id": chunk.get("parent_doc_id"),
        "specialty": chunk.get("specialty"),
        "sample_type": chunk.get("sample_type"),
        "section": chunk.get("section", "UNKNOWN"),
        "start_word": chunk.get("start_word", 0),
        "end_word": chunk.get("end_word", 0),
        "chunk_index": chunk.get("chunk_index"),
        "total_chunks": chunk.get("total_chunks"),
    }


def upload_chunks(
    client: QdrantClient,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    *,
    batch_size: int = 50,  # Reduced from 100
) -> None:
    """
    Upload chunks to Qdrant in smaller batches with retry.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

    if embeddings.ndim != 2 or embeddings.shape[1] != VECTOR_SIZE:
        raise ValueError(f"Expected shape (*, {VECTOR_SIZE}), got {embeddings.shape}")

    total = len(chunks)
    print(f"Uploading {total} points in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embs = embeddings[start:end]

        points = [
            PointStruct(
                id=idx + start,  # Stable numeric ID
                vector=emb.tolist(),
                payload=build_payload(ch),
            )
            for idx, (ch, emb) in enumerate(zip(batch_chunks, batch_embs))
        ]

        # Retry once on timeout
        retry = 0
        while retry < 2:
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"    Uploaded {end}/{total}")
                break
            except Exception as e:
                retry += 1
                if retry < 2:
                    print(f"     Batch {start}-{end} timeout, retrying...")
                    import time
                    time.sleep(2)
                else:
                    raise e

    print(f" Finished uploading {total} chunks")


if __name__ == "__main__":
    # robust path (independent of CWD)
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / "data" / "raw" / "mtsamples.csv"

    # adjust these imports based on your package layout
    from loader import load_mtsamples
    from chunker import chunk_documents  # or your sentence-aware chunker
    from embedder import embed_chunks

    docs = load_mtsamples(str(csv_path))
    chunks, stats = chunk_documents(docs)
    print(f"Chunk stats: {stats}")

    #  embedder API (new)
    valid_chunks, embeddings, skipped = embed_chunks(chunks, batch_size=64, normalize=True)

    print(f"Embeddings: {embeddings.shape}")
    print(f"Skipped chunks: {len(skipped)}")

    client = get_qdrant_client()
    ensure_collection(client)
    upload_chunks(client, valid_chunks, embeddings, batch_size=100)

    info = client.get_collection(COLLECTION_NAME)
    print("\nCollection info:")
    print(f"  Points: {info.points_count}")