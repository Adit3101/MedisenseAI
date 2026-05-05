from __future__ import annotations

import re
from typing import Optional
from transformers import pipeline, AutoTokenizer

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 900
MIN_SUMMARY_LEN = 20

_summarizer = None
_tokenizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        print(f"Loading summarizer: {MODEL_NAME}")
        _summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            framework="pt",
        )
    return _summarizer


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def clean_text(text: str) -> str:
    """Strip common noisy clinical headers."""
    return re.sub(
        r"(HISTORY OF PRESENT ILLNESS:|PAST MEDICAL HISTORY:|CHIEF COMPLAINT:|ASSESSMENT:|PLAN:)",
        "",
        text,
        flags=re.I,
    ).strip()


def clean_generated(text: str) -> str:
    """
    - Remove instruction echoes
    - Remove repeated tokens/artifacts
    - Strip leading punctuation
    - Normalize whitespace
    - Capitalize first letter
    """
    text = re.sub(r"Summarize.*?:", "", text, flags=re.I)
    text = re.sub(r"\b(\w+)( \1){2,}\b", r"\1", text)  # collapse repeated words
    text = re.sub(r"^[\s,;:]+", "", text)  # leading punctuation
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def get_lengths(text):
    words = len(text.split())
    max_len = min(300, max(120, words // 2))   # 🔼 increase more
    min_len = max(60, max_len // 3)
    return min_len, max_len

def split_sections(text):
    return re.split(
        r"(HISTORY OF PRESENT ILLNESS:|ASSESSMENT:|PLAN:|PROCEDURE:)",
        text,
        flags=re.I
    )

def chunk_text_for_summary(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> list[str]:
    """Token-aware chunking."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, truncation=False)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks


def summarize_text(text: str, _depth: int = 0) -> str:
    """
    Map-reduce summarization with fixes applied.
    _depth: recursion guard for long reductions.
    """
    if _depth > 2:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text, truncation=True, max_length=MAX_INPUT_TOKENS)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        summarizer = get_summarizer()
        min_len, max_len = get_lengths(text)
        result = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )
        return clean_generated(result[0]["summary_text"])

    summarizer = get_summarizer()
    text = clean_text(text)
    chunks = chunk_text_for_summary(text)

    if not chunks:
        return "No text to summarize."

    results = []
    for chunk in chunks:
        min_len, max_len = get_lengths(chunk)
        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )
        results.append(clean_generated(result[0]["summary_text"]))

    if len(results) == 1:
        return results[0]

    combined = "\n".join(results)

    if len(combined.split()) > 800:
        print(f"  Combined summaries too long ({len(combined.split())} words), reducing recursively...")
        return summarize_text(combined, _depth=_depth + 1)

    # print(f"  Reducing {len(results)} summaries into final...")
    # min_len, max_len = get_lengths(combined)
    # combined = " ".join(results)
    #
    # final = summarizer(
    #     combined,
    #     max_length=max_len,
    #     min_length=min_len,
    #     do_sample=False,
    #     repetition_penalty=2.0,
    #     no_repeat_ngram_size=3,
    # )
    # return clean_generated(final[0]["summary_text"])
    combined = " ".join(results)

    template = f"""
    Summarize the clinical note with the following structure:

    Patient:
    Condition:
    History:
    Treatment/Plan:
    Risks:

    Text:
    {combined}
    """

    min_len, max_len = get_lengths(combined)

    final = summarizer(
        template,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
    )

    return clean_generated(final[0]["summary_text"])

if __name__ == "__main__":
    from pathlib import Path
    import sys

    BASE_DIR = Path(__file__).resolve().parent.parent
    sys.path.append(str(BASE_DIR))
    from ingestion.loader import load_mtsamples

    DATA_PATH = BASE_DIR / "data" / "raw" / "mtsamples.csv"

    print(f"Loading data from: {DATA_PATH}")
    print(f"Exists: {DATA_PATH.exists()}")

    docs = load_mtsamples(str(DATA_PATH))
    long_doc = next(d for d in docs if len((d["text"] or "").split()) > 400)

    print(f"\nDocument: {long_doc['id']}")
    print(f"Specialty: {long_doc['medical_specialty']}")
    print(f"Word count: {len(long_doc['text'].split())}")
    print(f"\nSummarizing...")

    summary = summarize_text(long_doc["text"])

    print(f"\nSummary:\n{summary}")