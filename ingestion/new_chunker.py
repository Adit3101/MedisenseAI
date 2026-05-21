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
    # Example: ",MEDICATIONS:" -> "\n\nMEDICATIONS:\n\n"
    # Uses DOUBLE newlines so they survive the single-newline collapse below.
    text = re.sub(r"\s*,\s*([A-Z][A-Z0-9 /&\-]{2,40})\s*:\s*", r"\n\n\1:\n\n", text)

    text = text.strip()
    # Preserve section boundaries first
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Then collapse single newlines inside sentences
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
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


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Section-aware chunker  (Fix 2.1 from RAG failure analysis)
# ──────────────────────────────────────────────────────────────────────────────

# Canonical clinical section headers (order roughly follows SOAP notes)
KNOWN_SECTIONS = {
    "SUBJECTIVE", "HISTORY OF PRESENT ILLNESS", "HPI", "CC", "CHIEF COMPLAINT",
    "PAST MEDICAL HISTORY", "PMH", "PAST SURGICAL HISTORY", "PSH",
    "SOCIAL HISTORY", "SHX", "FAMILY HISTORY", "FHX",
    "REVIEW OF SYSTEMS", "ROS",
    "MEDICATIONS", "CURRENT MEDICATIONS", "ALLERGIES",
    "OBJECTIVE", "PHYSICAL EXAMINATION", "EXAM", "VITALS",
    "HEENT", "NECK", "LUNGS", "HEART", "ABDOMEN", "EXTREMITIES", "NEUROLOGICAL",
    "ASSESSMENT", "IMPRESSION", "IMPRESSION/PLAN",
    "PLAN", "RECOMMENDATIONS", "DISPOSITION",
    "PROCEDURE", "PROCEDURE IN DETAIL", "OPERATIVE PROCEDURE",
    "PREOPERATIVE DIAGNOSIS", "PREOPERATIVE DIAGNOSES",
    "POSTOPERATIVE DIAGNOSIS", "POSTOPERATIVE DIAGNOSES",
    "ANESTHESIA", "INDICATION FOR PROCEDURE", "INDICATION FOR OPERATION",
    "INDICATIONS FOR PROCEDURE",
    "FINDINGS AND PROCEDURE", "FINDINGS",
    "DESCRIPTION", "2-D M-MODE", "2-D STUDY", "2-D ECHOCARDIOGRAM",
    "DOPPLER", "SUMMARY",
    "MISCELLANEOUS/EATING HISTORY",
    "INITIAL STUDIES", "COURSE",
}


def _is_known_header(text: str) -> bool:
    """Check if a line is a known clinical section header."""
    cleaned = text.strip().rstrip(":").strip().upper()
    if cleaned in KNOWN_SECTIONS:
        return True
    # Fallback: all-caps, short, looks like a header
    return is_section_header(text)


def split_into_clinical_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split normalized clinical text into (section_name, section_content) pairs.

    Returns a list of tuples. Each section_content is the text belonging
    to that section header. Content before the first header gets section
    name 'PREAMBLE'.
    """
    lines = text.split("\n")
    sections: List[Tuple[str, str]] = []
    current_header = "PREAMBLE"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line is a header (possibly with inline content)
        # Pattern: "HEADER: content..." or just "HEADER:"
        if ":" in stripped:
            header_part, rest = stripped.split(":", 1)
            if _is_known_header(header_part):
                # Flush previous section
                if current_lines:
                    sections.append((current_header, " ".join(current_lines).strip()))
                    current_lines = []
                current_header = header_part.strip().upper()
                rest = rest.strip()
                if rest:
                    current_lines.append(rest)
                continue

        # Check if the whole line is a standalone header (no colon)
        if _is_known_header(stripped):
            if current_lines:
                sections.append((current_header, " ".join(current_lines).strip()))
                current_lines = []
            current_header = stripped.rstrip(":").strip().upper()
            continue

        current_lines.append(stripped)

    # Flush last section
    if current_lines:
        sections.append((current_header, " ".join(current_lines).strip()))

    return sections


def chunk_documents_by_section(
    documents: List[Dict[str, Any]],
    max_chunk_tokens: int = 400,
    overlap_sentences: int = 1,
    min_chunk_tokens: int = 30,
    normalize: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Section-aware chunker: splits each document by clinical section headers,
    then sub-chunks large sections with sentence overlap.

    Each chunk gets the CORRECT section label (e.g. PLAN, MEDICATIONS).
    Small sections (< min_chunk_tokens) are merged with the next section
    to avoid tiny chunks.
    """
    ensure_nltk_punkt()

    max_words = max(1, int(max_chunk_tokens / WORDS_PER_TOKEN))
    min_words = max(0, int(min_chunk_tokens / WORDS_PER_TOKEN))

    chunks: List[Dict[str, Any]] = []
    skipped_docs = 0
    skipped_small = 0
    merged_sections = 0

    for doc_idx, doc in enumerate(documents):
        text = doc.get("text", None)
        if not isinstance(text, str) or not text.strip():
            skipped_docs += 1
            continue

        if normalize:
            text = normalize_text_preserve_structure(text)

        if not text:
            skipped_docs += 1
            continue

        doc_id = doc.get("id", f"doc_{doc_idx}")
        specialty = doc.get("medical_specialty", "Unknown")
        sample_type = doc.get("sample_type", "Unknown")

        # Split into clinical sections
        raw_sections = split_into_clinical_sections(text)

        if not raw_sections:
            skipped_docs += 1
            continue

        # Merge tiny sections with the next one
        # Sections that should NEVER be merged (too clinically important)
        PROTECTED_SECTIONS = {
            "PLAN", "ASSESSMENT", "IMPRESSION", "IMPRESSION/PLAN",
            "MEDICATIONS", "CURRENT MEDICATIONS",
        }

        merged: List[Tuple[str, str]] = []
        carry_text = ""
        carry_header = ""

        for header, content in raw_sections:
            header_upper = header.upper().strip()

            # If we're about to hit a protected section, flush any carry first
            if header_upper in PROTECTED_SECTIONS and carry_text:
                merged.append((carry_header or "PREAMBLE", carry_text))
                carry_text = ""
                carry_header = ""

            full_text = (carry_text + " " + content).strip() if carry_text else content
            word_count = len(full_text.split())

            if word_count < min_words and header_upper not in PROTECTED_SECTIONS:
                # Too small and not protected, carry forward
                carry_text = full_text
                carry_header = carry_header or header
                merged_sections += 1
            else:
                merged.append((carry_header or header, full_text))
                carry_text = ""
                carry_header = ""

        # Flush any remaining carry
        if carry_text:
            if merged:
                prev_header, prev_text = merged[-1]
                merged[-1] = (prev_header, prev_text + " " + carry_text)
            else:
                merged.append((carry_header or "UNKNOWN", carry_text))

        # Now create chunks from each section
        chunk_index = 0
        doc_chunk_positions: List[int] = []

        for section_header, section_text in merged:
            words = section_text.split()
            word_count = len(words)

            if word_count <= max_words:
                # Section fits in one chunk
                enriched_text = f"Patient Specialty: {specialty}\nSection: {section_header}\n\n{section_text}"
                chunk = {
                    "chunk_id": f"{doc_id}::chunk_{chunk_index}",
                    "chunk_text": enriched_text,
                    "parent_doc_id": doc_id,
                    "specialty": specialty,
                    "sample_type": sample_type,
                    "section": section_header,
                    "chunk_index": chunk_index,
                    "total_chunks": None,  # filled in later
                    "word_count": word_count,
                }
                doc_chunk_positions.append(len(chunks))
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Sub-chunk large section using sentence boundaries
                sentences = split_into_sentences(section_text)
                current_sents: List[str] = []
                current_wc = 0

                for sent in sentences:
                    sent_wc = len(sent.split())

                    if current_sents and (current_wc + sent_wc) > max_words:
                        # Flush current sub-chunk
                        raw_chunk_text = " ".join(current_sents).strip()
                        enriched_text = f"Patient Specialty: {specialty}\nSection: {section_header}\n\n{raw_chunk_text}"
                        chunk = {
                            "chunk_id": f"{doc_id}::chunk_{chunk_index}",
                            "chunk_text": enriched_text,
                            "parent_doc_id": doc_id,
                            "specialty": specialty,
                            "sample_type": sample_type,
                            "section": section_header,
                            "chunk_index": chunk_index,
                            "total_chunks": None,
                            "word_count": len(raw_chunk_text.split()),
                        }
                        doc_chunk_positions.append(len(chunks))
                        chunks.append(chunk)
                        chunk_index += 1

                        # Overlap: keep last N sentences
                        if overlap_sentences > 0:
                            current_sents = current_sents[-overlap_sentences:]
                            current_wc = sum(len(s.split()) for s in current_sents)
                        else:
                            current_sents = []
                            current_wc = 0

                    current_sents.append(sent)
                    current_wc += sent_wc

                    if current_sents:
                        raw_chunk_text = " ".join(current_sents).strip()
                        wc = len(raw_chunk_text.split())
                        if wc >= min_words:
                            enriched_text = f"Patient Specialty: {specialty}\nSection: {section_header}\n\n{raw_chunk_text}"
                            chunk = {
                                "chunk_id": f"{doc_id}::chunk_{chunk_index}",
                                "chunk_text": enriched_text,
                                "parent_doc_id": doc_id,
                                "specialty": specialty,
                                "sample_type": sample_type,
                                "section": section_header,
                                "chunk_index": chunk_index,
                                "total_chunks": None,
                                "word_count": wc,
                            }
                        doc_chunk_positions.append(len(chunks))
                        chunks.append(chunk)
                        chunk_index += 1
                    else:
                        skipped_small += 1

        # Fill in total_chunks for this document
        for pos in doc_chunk_positions:
            chunks[pos]["total_chunks"] = chunk_index

    # Compute stats
    if chunks:
        lengths = [c["word_count"] for c in chunks]
        avg_len = sum(lengths) // len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
    else:
        avg_len = min_len = max_len = 0

    stats: Dict[str, Any] = {
        "skipped_documents": skipped_docs,
        "skipped_small_chunks": skipped_small,
        "merged_tiny_sections": merged_sections,
        "created_chunks": len(chunks),
        "avg_chunk_size_words": avg_len,
        "min_chunk_size_words": min_len,
        "max_chunk_size_words": max_len,
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

    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from loader import load_mtsamples

    docs = load_mtsamples(str(csv_path))

    # ── Test NEW section-aware chunker ────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION-AWARE CHUNKER (NEW)")
    print("=" * 60)
    chunks_v2, stats_v2 = chunk_documents_by_section(docs)

    print("\nStats:")
    print(json.dumps(stats_v2, indent=2))

    # Show doc_0 chunks specifically (the allergy case)
    doc0_chunks = [c for c in chunks_v2 if c["parent_doc_id"] == "doc_0"]
    print(f"\n--- doc_0 (allergy case): {len(doc0_chunks)} chunks ---")
    for c in doc0_chunks:
        print(f"  [{c['section']}] ({c['word_count']} words) {c['chunk_text'][:100]}...")

    # Save preview
    preview = chunks_v2[:30]
    output_path = BASE_DIR / "data" / "chunks_preview_v2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)
    print(f"\nSaved first 30 section-aware chunks to: {output_path}")

    # ── Compare with old chunker ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SENTENCE-AWARE CHUNKER (OLD)")
    print("=" * 60)
    chunks_v1, stats_v1 = chunk_documents_sentence_aware(docs)
    print(f"\nStats: {stats_v1['created_chunks']} chunks")

    doc0_old = [c for c in chunks_v1 if c["parent_doc_id"] == "doc_0"]
    print(f"doc_0 old: {len(doc0_old)} chunks, section={doc0_old[0]['section'] if doc0_old else 'N/A'}")