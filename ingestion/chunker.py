from __future__ import annotations

from typing import List, Dict, Any, Tuple
from pathlib import Path
import os
import json
import pprint


WORDS_PER_TOKEN = 1.3  # heuristic


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 50,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Split documents into overlapping chunks.

    chunk_size/overlap are in *approx tokens* (heuristic).
    Returns (chunks, stats).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    token_size_words = max(1, int(chunk_size / WORDS_PER_TOKEN))
    overlap_words = max(0, int(overlap / WORDS_PER_TOKEN))
    step = token_size_words - overlap_words
    if step <= 0:
        raise ValueError("Invalid step computed; decrease overlap or increase chunk_size.")

    chunks: List[Dict[str, Any]] = []
    skipped = 0

    for doc_idx, doc in enumerate(documents):
        text = doc.get("text", None)

        # Skip if text missing or not a usable string
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue

        words = text.split()
        if not words:
            skipped += 1
            continue

        doc_id = doc.get("id", f"doc_{doc_idx}")
        specialty = doc.get("medical_specialty", "Unknown")
        sample_type = doc.get("sample_type", "Unknown")

        chunk_num = 0
        for start in range(0, len(words), step):
            chunk_words = words[start : start + token_size_words]
            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words).strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "chunk_id": f"{doc_id}::chunk_{chunk_num}",
                    "chunk_text": chunk_text,
                    "parent_doc_id": doc_id,
                    "specialty": specialty,
                    "sample_type": sample_type,
                    "start_word": start,
                    "end_word": min(start + token_size_words, len(words)),
                }
            )
            chunk_num += 1

    stats = {"skipped_documents": skipped, "created_chunks": len(chunks)}
    return chunks, stats


if __name__ == "__main__":
    print("CWD:", os.getcwd())

    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / "data" / "raw" / "mtsamples.csv"
    print("Resolved path:", csv_path)

    from loader import load_mtsamples

    docs = load_mtsamples(str(csv_path))
    chunks, stats = chunk_documents(docs)

    print(f"Created {stats['created_chunks']} chunks from {len(docs)} documents")
    print(f"Skipped {stats['skipped_documents']} documents with missing text")

    if chunks:
        print("\nFirst chunk:")
        print(f"  Chunk ID: {chunks[0]['chunk_id']}")
        print(f"  Parent Doc: {chunks[0]['parent_doc_id']}")
        print(f"  Length: {len(chunks[0]['chunk_text'])} chars")

        output_path = "chunks_preview.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks[:20], f, indent=2, ensure_ascii=False)

        print(f"Saved first 20 chunks to {output_path}")
        pprint.pprint(chunks[0])