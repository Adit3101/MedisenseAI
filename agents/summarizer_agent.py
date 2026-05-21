"""
agents/summarizer_agent.py

Extractive summarization — no model download required.

Replaces facebook/bart-large-cnn (which was trained on news and
produces fragmented output on clinical SOAP notes).

Algorithm:
1. Split context into sentences.
2. Score each sentence by:
   - TF-IDF overlap with the query
   - Section weight (PLAN/ASSESSMENT sentences score highest)
   - Position weight (early sentences in PLAN sections score slightly higher)
3. Return top-N sentences joined in document order.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Dict

# Sections whose sentences get a scoring bonus
SECTION_WEIGHTS: Dict[str, float] = {
    "PLAN":             2.0,
    "ASSESSMENT":       1.8,
    "IMPRESSION":       1.8,
    "IMPRESSION/PLAN":  2.0,
    "MEDICATIONS":      1.7,
    "CURRENT MEDICATIONS": 1.7,
    "SUBJECTIVE":       1.2,
    "HPI":              1.2,
    "HISTORY OF PRESENT ILLNESS": 1.2,
    "ALLERGIES":        1.5,
    "FINDINGS":         1.4,
}

# Maximum sentences to include in the summary
MAX_SUMMARY_SENTENCES = 6
MAX_SUMMARY_CHARS = 800


def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer — remove stop words."""
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "and", "or", "but", "if", "as",
        "this", "that", "these", "those", "it", "its", "he", "she", "they",
        "we", "you", "i", "his", "her", "their", "our", "your", "my",
        "not", "no", "nor", "so", "yet",
    }
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def _tfidf_score(sentence_tokens: List[str], query_tokens: List[str]) -> float:
    """Simple TF-IDF overlap between sentence and query."""
    if not sentence_tokens or not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    matches = sum(1 for t in sentence_tokens if t in query_set)
    # Normalize by sentence length (avoid rewarding very long sentences blindly)
    tf = matches / max(1, len(sentence_tokens))
    # IDF-like weighting: rare query terms matter more
    return tf * math.log(1 + len(query_tokens))


def _detect_section_from_text(text: str) -> str:
    """Detect section from the chunk text prefix (e.g. 'Section: PLAN')."""
    match = re.search(r"Section:\s*([A-Z /&\-]+)", text)
    if match:
        return match.group(1).strip().upper()
    return "UNKNOWN"


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling clinical list items."""
    # Handle numbered list items (e.g., "1. Furosemide 20mg daily")
    text = re.sub(r'(\d+\.\s)', r'\n\1', text)
    # Standard sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) >= 3:  # skip very short fragments
            result.append(s)
    return result


def summarize_text(
    text: str,
    query: Optional[str] = None,
    max_sentences: int = MAX_SUMMARY_SENTENCES,
) -> str:
    """
    Extractive summarization: score each sentence and return the top ones.

    Args:
        text: combined chunk texts
        query: the user query (used for TF-IDF scoring). If None, use pure
               section-based scoring.
        max_sentences: max number of sentences to include

    Returns:
        A summary string.
    """
    if not text or not text.strip():
        return "No content to summarize."

    # Strip all chunker-added metadata lines (can appear anywhere in combined text)
    clean = re.sub(r"Patient Specialty:[^\n]*\n?", "", text)
    clean = re.sub(r"Section:\s*[A-Z0-9 /&\-]+\n?", "", clean)
    # Strip residual comma-list artifacts (e.g. ",1. Drug" -> "1. Drug")
    clean = re.sub(r"^[,\s]+", "", clean, flags=re.MULTILINE)
    clean = clean.strip()

    if not clean:
        return "No content to summarize."

    # Detect section for bonus scoring
    section = _detect_section_from_text(text)
    section_bonus = SECTION_WEIGHTS.get(section, 1.0)

    query_tokens = _tokenize(query) if query else []
    sentences = _split_sentences(clean)

    if not sentences:
        return clean[:MAX_SUMMARY_CHARS]

    # Score each sentence
    scored: List[tuple] = []
    for idx, sent in enumerate(sentences):
        sent_tokens = _tokenize(sent)

        tfidf = _tfidf_score(sent_tokens, query_tokens) if query_tokens else 0.0
        position_bonus = 1.0 / (1 + idx * 0.1)  # earlier sentences slightly favored
        score = (tfidf + position_bonus) * section_bonus

        scored.append((score, idx, sent))

    # Sort by score descending, keep top N, then restore original order
    top = sorted(scored, key=lambda x: -x[0])[:max_sentences]
    top = sorted(top, key=lambda x: x[1])  # restore document order

    summary = " ".join(sent for _, _, sent in top)

    # Trim to max chars
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"

    return summary


def summarize_chunks(
    chunks: List[Dict],
    query: Optional[str] = None,
    max_sentences: int = MAX_SUMMARY_SENTENCES,
) -> str:
    """
    Summarize a list of chunk dicts directly, respecting section ordering.
    Used by the orchestrator summarizer_node.
    """
    if not chunks:
        return "No chunks to summarize."

    # Sort chunks by section priority so PLAN/ASSESSMENT text comes first
    def section_rank(c: Dict) -> float:
        sec = (c.get("section") or "UNKNOWN").upper()
        weight = SECTION_WEIGHTS.get(sec, 1.0)
        return -weight  # negate for ascending sort

    sorted_chunks = sorted(chunks, key=section_rank)

    combined = "\n\n".join(
        c.get("chunk_text", "") for c in sorted_chunks[:6]
    )
    return summarize_text(combined, query=query, max_sentences=max_sentences)


if __name__ == "__main__":
    sample = """
Patient Specialty: Allergy / Immunology
Section: MEDICATIONS

Her only medication currently is Ortho Tri-Cyclen and the Allegra.

Patient Specialty: Allergy / Immunology
Section: PLAN

She will try Zyrtec instead of Allegra again. Another option will be to use loratadine.
Samples of Nasonex two sprays in each nostril given for three weeks. A prescription was written as well.
    """
    query = "What medications was the allergy patient prescribed?"
    print(summarize_text(sample, query=query))