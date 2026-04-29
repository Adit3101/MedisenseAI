from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIM = 384

_model: Optional[SentenceTransformer] = None
_device: Optional[str] = None


def get_device() -> str:
    #  Improvement 1: device control (CPU vs GPU)
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> SentenceTransformer:
    global _model, _device
    device = get_device()

    # reload only if not loaded yet OR device changed
    if _model is None or _device != device:
        _device = device
        print(f"Loading embedding model: {MODEL_NAME} on device={device}")
        _model = SentenceTransformer(MODEL_NAME, device=device)

    return _model


def embed_texts(
    texts: List[str],
    *,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = True,
    expected_dim: int = EXPECTED_DIM,
) -> np.ndarray:
    """
    Embed a list of texts -> np.ndarray shape (N, expected_dim).
    """
    #  Improvement 3: empty dataset guard
    if not texts:
        return np.empty((0, expected_dim), dtype=np.float32)

    print(f"Embedding {len(texts)} texts with batch_size={batch_size} normalize={normalize}")

    model = get_model()

    # Improvement 2: enforce numpy output
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)

    # Improvement 4: validate embedding dimension
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected embedding shape {embeddings.shape}; expected (*, {expected_dim}). "
            f"Model={MODEL_NAME}"
        )

    return embeddings


def embed_chunks(
    chunks: List[Dict[str, Any]],
    *,
    text_key: str = "chunk_text",
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[int]]:
    """
    Improvement 6: avoid embedding empty strings silently.

    Returns:
      - valid_chunks: chunks with non-empty text
      - embeddings: np.ndarray aligned with valid_chunks
      - skipped_indices: indices (from original chunks list) that were skipped
    """
    texts: List[str] = []
    valid_chunks: List[Dict[str, Any]] = []
    skipped_indices: List[int] = []

    for i, ch in enumerate(chunks):
        t = ch.get(text_key, None)

        if not isinstance(t, str):
            skipped_indices.append(i)
            continue

        t = t.strip()
        if not t:
            skipped_indices.append(i)
            continue

        valid_chunks.append(ch)
        texts.append(t)

    embeddings = embed_texts(
        texts,
        batch_size=batch_size,
        normalize=normalize,
        show_progress_bar=show_progress_bar,
    )

    return valid_chunks, embeddings, skipped_indices


def attach_embeddings_to_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    *,
    out_key: str = "embedding",
) -> List[Dict[str, Any]]:
    """
    Helper for debugging/printing. For production Qdrant upload,
    you can keep vectors separate.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")

    for ch, emb in zip(chunks, embeddings):
        ch[out_key] = emb.tolist()

    return chunks


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / "data" / "raw" / "mtsamples.csv"

    # adjust imports depending on how you run the script
    from loader import load_mtsamples
    from chunker import chunk_documents  # or your sentence-aware chunker

    docs = load_mtsamples(str(csv_path))
    chunks, stats = chunk_documents(docs)

    print("\nChunking stats:")
    print(stats)

    # Example: embed a sample
    sample = chunks[:100]

    valid_chunks, embs, skipped = embed_chunks(
        sample,
        batch_size=64,
        normalize=True,
        show_progress_bar=True,
    )

    print(f"\n Embedded {len(valid_chunks)} chunks")
    print(f"Skipped {len(skipped)} chunks (empty or invalid text)")
    print(f"Embedding shape: {embs.shape}")

    if len(valid_chunks) > 0:
        print(f"First chunk id: {valid_chunks[0].get('chunk_id')}")
        print(f"First embedding (first 5 dims): {embs[0][:5].tolist()}")

    # Optional: attach embeddings for quick inspection
    preview = attach_embeddings_to_chunks(valid_chunks[:3], embs[:3])
    print("\nPreview (first 3 with embeddings attached):")
    for p in preview:
        print(p["chunk_id"], p["embedding"][:5])