from __future__ import annotations

from typing import List, Dict, Any, Tuple
import re

import nltk
from nltk.tokenize import sent_tokenize

WORDS_PER_TOKEN = 1.3  # heuristic


def ensure_nltk_punkt() -> None:
    """
    Ensure the NLTK punkt tokenizer is available.
    (Useful on fresh machines/venvs where the data isn't downloaded yet.)
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        # Some environments / NLTK versions may require this too
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass


def normalize_text_preserve_structure(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # NEW: turn inline ",HEADER:" markers into real section lines.
    # Example: ",MEDICATIONS:" -> "\nMEDICATIONS:"
    # This makes split_blocks_preserving_headers() actually separate them.
    text = re.sub(r"\s*,\s*([A-Z][A-Z0-9 /&\-]{2,40})\s*:\s*", r"\n\1:\n", text)

    text = text.strip()
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # NEW (optional): remove leading punctuation artifacts at the start of blocks/sentences
    text = re.sub(r"^[,\s]+", "", text)

    return text


def is_section_header(s: str) -> bool:
    """
    Clinical-ish heuristic:
    - All caps
    - Short (few words)
    Examples: "HISTORY:", "PLAN:", "ASSESSMENT"
    """
    s = s.strip()
    if not s:
        return False
    # allow trailing colon
    s2 = s[:-1] if s.endswith(":") else s
    return s2.isupper() and len(s2.split()) < 6


def split_blocks_preserving_headers(text: str) -> List[str]:
    """
    Split by newline blocks (structure preserved by normalization).
    Each block may be a header line or narrative text.
    """
    return [b.strip() for b in text.split("\n") if b.strip()]


def split_into_sentences(block: str) -> List[str]:
    """
    Sentence segmentation using NLTK (better with abbreviations like Dr., mg., etc.).
    """
    # sent_tokenize returns List[str]
    sents = sent_tokenize(block)
    return [s.strip() for s in sents if s and s.strip()]


def chunk_documents_sentence_aware(
    documents: List[Dict[str, Any]],
    chunk_size: int = 400,              # approx tokens
    overlap_sentences: int = 2,         # overlap last N sentences
    min_chunk_tokens: int = 80,         # drop chunks smaller than this (approx tokens)
    normalize: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ensure_nltk_punkt()  # NEW: make sure tokenizer data exists

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")
    if min_chunk_tokens < 0:
        raise ValueError("min_chunk_tokens must be >= 0")
    if min_chunk_tokens > chunk_size:
        raise ValueError("min_chunk_tokens must be <= chunk_size")

    max_words = max(1, int(chunk_size / WORDS_PER_TOKEN))
    min_words = max(0, int(min_chunk_tokens / WORDS_PER_TOKEN))

    chunks: List[Dict[str, Any]] = []
    skipped_docs = 0
    skipped_small_chunks = 0

    for doc_idx, doc in enumerate(documents):
        text = doc.get("text", None)
        if not isinstance(text, str):
            skipped_docs += 1
            continue

        if normalize:
            text = normalize_text_preserve_structure(text)

        if not text:
            skipped_docs += 1
            continue

        blocks = split_blocks_preserving_headers(text)
        if not blocks:
            skipped_docs += 1
            continue

        doc_id = doc.get("id", f"doc_{doc_idx}")
        specialty = doc.get("medical_specialty", "Unknown")
        sample_type = doc.get("sample_type", "Unknown")

        current_section = "UNKNOWN"

        sentence_global_index = 0
        approx_word_index = 0

        current_sentences: List[str] = []
        current_word_count = 0
        current_chunk_start_sentence = 0
        current_chunk_start_word = 0

        doc_chunk_indices: List[int] = []
        chunk_index = 0

        def flush_chunk(end_sentence_index_exclusive: int, end_word_index_exclusive: int):
            nonlocal chunk_index, current_sentences, current_word_count
            nonlocal current_chunk_start_sentence, current_chunk_start_word
            nonlocal skipped_small_chunks

            if not current_sentences:
                return

            chunk_text = " ".join(current_sentences).strip()
            word_count = len(chunk_text.split())

            if min_words and word_count < min_words:
                skipped_small_chunks += 1
            else:
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}::chunk_{chunk_index}",
                        "chunk_text": chunk_text,
                        "parent_doc_id": doc_id,
                        "specialty": specialty,
                        "sample_type": sample_type,
                        "section": current_section,
                        "chunk_index": chunk_index,
                        "total_chunks": None,
                        "start_sentence_index": current_chunk_start_sentence,
                        "end_sentence_index": end_sentence_index_exclusive,
                        "start_word": current_chunk_start_word,
                        "end_word": end_word_index_exclusive,
                        "sentence_count": len(current_sentences),
                        "word_count": word_count,
                    }
                )
                doc_chunk_indices.append(len(chunks) - 1)
                chunk_index += 1

            if overlap_sentences > 0:
                current_sentences = current_sentences[-overlap_sentences:]
                current_word_count = sum(len(s.split()) for s in current_sentences)

                current_chunk_start_sentence = max(
                    0, end_sentence_index_exclusive - len(current_sentences)
                )
                overlap_words = sum(len(s.split()) for s in current_sentences)
                current_chunk_start_word = max(0, end_word_index_exclusive - overlap_words)
            else:
                current_sentences = []
                current_word_count = 0
                current_chunk_start_sentence = end_sentence_index_exclusive
                current_chunk_start_word = end_word_index_exclusive

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Handle "HEADER: content..." on the same line (common in your data)
            if ":" in block:
                header_part, rest = block.split(":", 1)
                if is_section_header(header_part):
                    current_section = header_part.strip()
                    block = rest.strip()
                    if not block:
                        # header-only line like "PLAN:" -> nothing to chunk
                        continue

            # Handle header-only blocks (when the whole line is just the header)
            if is_section_header(block):
                current_section = block.rstrip(":").strip()
                continue

            sents = split_into_sentences(block)
            for sent in sents:
                sent_words = len(sent.split())

                if not current_sentences:
                    current_chunk_start_sentence = sentence_global_index
                    current_chunk_start_word = approx_word_index

                if not current_sentences and sent_words > max_words:
                    current_sentences = [sent]
                    current_word_count = sent_words
                    sentence_global_index += 1
                    approx_word_index += sent_words
                    flush_chunk(sentence_global_index, approx_word_index)
                    continue

                if current_sentences and (current_word_count + sent_words) > max_words:
                    flush_chunk(sentence_global_index, approx_word_index)
                    if not current_sentences:
                        current_chunk_start_sentence = sentence_global_index
                        current_chunk_start_word = approx_word_index

                current_sentences.append(sent)
                current_word_count += sent_words
                sentence_global_index += 1
                approx_word_index += sent_words

        flush_chunk(sentence_global_index, approx_word_index)

        total = chunk_index
        for global_chunk_pos in doc_chunk_indices:
            chunks[global_chunk_pos]["total_chunks"] = total

    if chunks:
        lengths = [c["word_count"] for c in chunks]
        avg_len = sum(lengths) // len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
    else:
        avg_len = min_len = max_len = 0

    stats: Dict[str, Any] = {
        "skipped_documents": skipped_docs,
        "skipped_small_chunks": skipped_small_chunks,
        "created_chunks": len(chunks),
        "avg_chunk_size_words": avg_len,
        "min_chunk_size_words": min_len,
        "max_chunk_size_words": max_len,
        "overlap_sentences": overlap_sentences,
        "max_words_target": max_words,
        "min_words_filter": min_words,
    }
    return chunks, stats
if __name__ == "__main__":
    import os
    import json
    from pathlib import Path

    print("CWD:", os.getcwd())

    # Anchor path to project root (repo root)
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / "data" / "raw" / "mtsamples.csv"
    print("Resolved path:", csv_path)

    from loader import load_mtsamples

    docs = load_mtsamples(str(csv_path))
    chunks, stats = chunk_documents_sentence_aware(docs)

    print("\nStats:")
    print(json.dumps(stats, indent=2))

    preview = chunks[:20]
    print("\nFirst 20 chunks (JSON):")
    print(json.dumps(preview, indent=2, ensure_ascii=False))

    # Optional: also write preview to disk
    output_path = BASE_DIR / "data" / "chunks_preview.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    print(f"\nSaved first 20 chunks to: {output_path}")