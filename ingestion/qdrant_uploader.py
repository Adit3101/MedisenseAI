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

# V2 collection uses section-aware chunking + text index for hybrid search
COLLECTION_NAME = "medisense_chunks_v2"
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


def create_payload_indexes(client: QdrantClient) -> None:
    """Create indexes on filterable fields + full-text index for hybrid search."""
    from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType

    # Keyword indexes for filtering
    keyword_fields = ["section", "specialty", "parent_doc_id"]
    for field in keyword_fields:
        print(f"Creating keyword index on '{field}'...")
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"   Index created for '{field}'")
        except Exception as e:
            print(f"   Index for '{field}' may already exist: {e}")

    # Full-text index on chunk_text for BM25/keyword search (Fix 2.2)
    print("Creating full-text index on 'chunk_text' for hybrid search...")
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="chunk_text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=30,
                lowercase=True,
            ),
        )
        print("   Full-text index created for 'chunk_text'")
    except Exception as e:
        print(f"   Full-text index may already exist: {e}")


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
    from new_chunker import chunk_documents_by_section  # NEW: section-aware chunker
    from embedder import embed_chunks

    docs = load_mtsamples(str(csv_path))

    # Use the new section-aware chunker
    chunks, stats = chunk_documents_by_section(docs)
    print(f"Chunk stats: {stats}")

    # Show doc_0 sections to verify correct splitting
    doc0 = [c for c in chunks if c["parent_doc_id"] == "doc_0"]
    print(f"\ndoc_0 (allergy): {len(doc0)} section-chunks")
    for c in doc0:
        print(f"  [{c['section']}] {c['chunk_text'][:80]}...")

    # Embed all chunks
    valid_chunks, embeddings, skipped = embed_chunks(chunks, batch_size=64, normalize=True)

    print(f"\nEmbeddings: {embeddings.shape}")
    print(f"Skipped chunks: {len(skipped)}")

    client = get_qdrant_client()
    ensure_collection(client)
    create_payload_indexes(client)

    info = client.get_collection(COLLECTION_NAME)
    if info.points_count > 0:
        print(f"⚠️  Collection already has {info.points_count} points. Skipping upload.")
        print(f"   To re-ingest, delete the collection first:")
        print(f"   client.delete_collection('{COLLECTION_NAME}')")
    else:
        upload_chunks(client, valid_chunks, embeddings, batch_size=50)

    info = client.get_collection(COLLECTION_NAME)
    print("\nCollection info:")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Points: {info.points_count}")