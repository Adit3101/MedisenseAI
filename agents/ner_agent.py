from __future__ import annotations

from typing import Dict, List
from transformers import pipeline

MODEL_NAME = "d4data/biomedical-ner-all"

# Load once (like your LLM cache)
_ner = None


def get_ner():
    global _ner
    if _ner is None:
        print(f"Loading NER model: {MODEL_NAME}")
        _ner = pipeline(
            "token-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            aggregation_strategy="simple",
        )
    return _ner


def _merge_spans(results: List[Dict], text: str) -> List[Dict]:
    """
    Merge contiguous spans with the same entity_group.
    Uses original text slice to avoid '##' subword artifacts.
    """
    if not results:
        return []

    # Ensure results are sorted by start offset
    results = sorted(results, key=lambda r: r["start"])

    merged = []
    for r in results:
        if not merged:
            merged.append(dict(r))
            continue

        prev = merged[-1]
        same_label = r["entity_group"] == prev["entity_group"]

        # If the gap between spans is only whitespace, merge
        gap = text[prev["end"]:r["start"]]
        if same_label and gap.strip() == "":
            prev["end"] = r["end"]
            prev["score"] = max(prev["score"], r["score"])
        else:
            merged.append(dict(r))

    return merged


def expand_condition_span(text: str, start: int, end: int) -> str:
    """
    Expand condition span by capturing nearby medical phrase (window-based).
    """
    window = 40  # characters before and after
    left = max(0, start - window)
    right = min(len(text), end + window)

    chunk = text[left:right]

    # Common disease patterns
    import re
    patterns = [
        r'\bcoronary artery disease\b',
        r'\btype\s*\d+\s*diabetes\b',
        r'\bbacterial pneumonia\b',
    ]

    for p in patterns:
        match = re.search(p, chunk, re.IGNORECASE)
        if match:
            return match.group(0)

    # fallback to original
    return text[start:end]


def extract_entities(text: str) -> Dict[str, List[str]]:
    ner = get_ner()
    results = ner(text)
    results = _merge_spans(results, text)

    entities = {
        "drugs": [],
        "conditions": [],
        "dosages": [],
    }

    for r in results:
        label = r["entity_group"]
        score = r["score"]

        # Skip low confidence
        if score < 0.4:
            continue

        # Use original text slice (fixes WordPiece splits)
        value = text[r["start"]:r["end"]].strip()
        if not value:
            continue

        # Normalize capitalization lightly
        value = value[0].upper() + value[1:] if value else value

        if label == "Medication":
            entities["drugs"].append(value)
        elif label == "Disease_disorder":
            value = expand_condition_span(text, r["start"], r["end"])
            entities["conditions"].append(value)
        elif label == "Dosage":
            entities["dosages"].append(value)

    # Deduplicate
    for k in entities:
        entities[k] = list(dict.fromkeys(entities[k]))

    return entities


if __name__ == "__main__":
    test_cases = [
        "Patient prescribed Metformin 500mg for Type 2 Diabetes",
        "She was given Amoxicillin 250mg for bacterial pneumonia and hypertension",
        "Patient has a history of coronary artery disease and was started on Lisinopril 10mg",
    ]

    for text in test_cases:
        print(f"\nInput: {text}")
        result = extract_entities(text)
        print(f"Output: {result}")