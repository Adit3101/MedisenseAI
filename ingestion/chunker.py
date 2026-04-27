from typing import List, Dict


def chunk_documents(documents: List[Dict], chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Split documents into overlapping chunks.

    Args:
        documents: List of dicts with 'text' key
        chunk_size: Target chunk size in tokens (approximate)
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of dicts with 'chunk_id', 'chunk_text', 'parent_doc_id'
    """
    chunks = []
    chunk_counter = 0

    for doc in documents:
        text = doc['text']
        words = text.split()

        # Rough token estimate: 1 token ≈ 1.3 words
        token_size = int(chunk_size / 1.3)
        overlap_tokens = int(overlap / 1.3)

        step = token_size - overlap_tokens

        for i in range(0, len(words), step):
            chunk_words = words[i:i + token_size]
            chunk_text = ' '.join(chunk_words)

            if chunk_text.strip():  # Skip empty chunks
                chunks.append({
                    'chunk_id': f"chunk_{chunk_counter}",
                    'chunk_text': chunk_text,
                    'parent_doc_id': doc['id'],
                    'specialty': doc['medical_specialty'],
                    'sample_type': doc['sample_type']
                })
                chunk_counter += 1

    return chunks

from pathlib import Path
import os

if __name__ == "__main__":
    print("CWD:", os.getcwd())

    # Anchor path to project root
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / "data" / "raw" / "mtsamples.csv"

    print("Resolved path:", csv_path)

    from loader import load_mtsamples

    docs = load_mtsamples(str(csv_path))
    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"\nFirst chunk:")
    print(f"  Chunk ID: {chunks[0]['chunk_id']}")
    print(f"  Parent Doc: {chunks[0]['parent_doc_id']}")
    print(f"  Length: {len(chunks[0]['chunk_text'])} chars")