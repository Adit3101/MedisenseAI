"""
agents/ner_agent.py

Post-answer NER with:
- Entity cleaning (remove noise, short tokens, false positives)
- Hybrid dosage extraction (regex + NER)
- Medical abbreviation normalization
- Runs on the FINAL ANSWER TEXT only, not raw retrieved chunks
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Any
from transformers import pipeline

MODEL_NAME = "d4data/biomedical-ner-all"

_ner = None

# ── Abbreviation normalization table ─────────────────────────────────────────
ABBREVIATION_MAP: Dict[str, str] = {
    "CVA":   "Cerebrovascular Accident",
    "MI":    "Myocardial Infarction",
    "CHF":   "Congestive Heart Failure",
    "COPD":  "Chronic Obstructive Pulmonary Disease",
    "DM":    "Diabetes Mellitus",
    "HTN":   "Hypertension",
    "CAD":   "Coronary Artery Disease",
    "UTI":   "Urinary Tract Infection",
    "GERD":  "Gastroesophageal Reflux Disease",
    "BPH":   "Benign Prostatic Hyperplasia",
    "CKD":   "Chronic Kidney Disease",
    "DVT":   "Deep Vein Thrombosis",
    "PE":    "Pulmonary Embolism",
    "TIA":   "Transient Ischemic Attack",
    # Routes
    "PO":    "by mouth",
    "IV":    "intravenous",
    "IM":    "intramuscular",
    "SC":    "subcutaneous",
    "SQ":    "subcutaneous",
    "SL":    "sublingual",
    "TOP":   "topical",
    "INH":   "inhaled",
    "PR":    "per rectum",
    # Frequencies
    "QD":    "once daily",
    "BID":   "twice daily",
    "TID":   "three times daily",
    "QID":   "four times daily",
    "PRN":   "as needed",
    "QHS":   "every bedtime",
    "QOD":   "every other day",
    # Lab / clinical
    "WBC":   "White Blood Cell Count",
    "RBC":   "Red Blood Cell Count",
    "Hgb":   "Hemoglobin",
    "Hct":   "Hematocrit",
    "BUN":   "Blood Urea Nitrogen",
    "Cr":    "Creatinine",
    "K":     "Potassium",
    "Na":    "Sodium",
    "HbA1c": "Glycated Hemoglobin",
    "INR":   "International Normalized Ratio",
    "PT":    "Prothrombin Time",
    "PTT":   "Partial Thromboplastin Time",
    "NKDA":  "No Known Drug Allergies",
    "NKA":   "No Known Allergies",
}

# ── False positive entity filter ─────────────────────────────────────────────
# Known non-drug tokens that the NER model frequently hallucinates
_DRUG_BLACKLIST = {
    "p", "po", "iv", "im", "sc", "sq", "sl", "prn", "qd", "bid", "tid", "qid",
    "mg", "mcg", "ml", "units", "unit", "meq", "tab", "cap", "tabs", "caps",
    "patient", "medication", "drug", "dose", "daily", "twice", "three", "four",
    "times", "given", "prescribed", "started", "per", "the", "and", "or",
    "therapy", "treatment",
}

# Minimum score threshold for NER entities (raised from 0.4 to 0.6)
MIN_NER_SCORE = 0.60

# ── Dosage regex patterns ─────────────────────────────────────────────────────
_DOSE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|mL|units?|mEq|meq|iu|IU)\b',
    re.IGNORECASE,
)
_ROUTE_PATTERN = re.compile(
    r'\b(by\s+mouth|oral(?:ly)?|intravenous(?:ly)?|intramuscular(?:ly)?'
    r'|subcutaneous(?:ly)?|sublingual(?:ly)?|topical(?:ly)?|inhaled?|inhaler'
    r'|nebuliz(?:er|ed)|intranasal(?:ly)?|p\.o\.?|i\.v\.?|i\.m\.?|s\.c\.?'
    r'|p\.r\.n\.?|sprays?|drops?)\b',
    re.IGNORECASE,
)
_FREQ_PATTERN = re.compile(
    r'\b(once\s+daily|twice\s+daily|three\s+times\s+daily|four\s+times\s+daily'
    r'|every\s+\d+\s+hours?|every\s+other\s+day|every\s+bedtime'
    r'|daily|nightly|weekly|monthly|as\s+needed|p\.r\.n\.?'
    r'|q\.?d\.?|b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?|q\.?h\.?s\.?'
    r'|prn|qd|bid|tid|qid|qhs|qod)\b',
    re.IGNORECASE,
)

# ── Drug mention pattern for structured extraction ────────────────────────────
# Matches a drug name followed by optional dose and route/frequency info
_DRUG_WITH_DOSAGE_PATTERN = re.compile(
    r'([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)?)'  # drug name (capitalized)
    r'\s+'
    r'(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|units?|mEq|meq|iu|IU))'  # dose
    r'(?:\s+(' + _ROUTE_PATTERN.pattern + r'))?'   # optional route
    r'(?:\s+(' + _FREQ_PATTERN.pattern + r'))?',   # optional frequency
    re.IGNORECASE,
)


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
    """Merge overlapping or adjacent spans with the same entity_group."""
    if not results:
        return []
    results = sorted(results, key=lambda r: (r["start"], -r["end"]))
    merged = []
    for r in results:
        if not merged:
            merged.append(dict(r))
            continue
        prev = merged[-1]
        if r["start"] >= prev["start"] and r["end"] <= prev["end"]:
            prev["score"] = max(prev["score"], r["score"])
            continue
        same_label = r["entity_group"] == prev["entity_group"]
        if same_label and r["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], r["end"])
            prev["score"] = max(prev["score"], r["score"])
        else:
            merged.append(dict(r))
    return merged


def _run_ner_windowed(text: str, window_chars: int = 1200, overlap_chars: int = 200) -> List[Dict]:
    """Run NER in sliding windows to avoid truncation."""
    ner = get_ner()
    if len(text) <= window_chars:
        return ner(text)
    all_results = []
    start = 0
    while start < len(text):
        end = min(start + window_chars, len(text))
        chunk = text[start:end]
        results = ner(chunk)
        for r in results:
            r["start"] += start
            r["end"] += start
        all_results.extend(results)
        start += window_chars - overlap_chars
    return all_results


def expand_to_word_boundaries(text: str, start: int, end: int) -> str:
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return text[start:end]


def _clean_entity(value: str) -> Optional[str]:
    """
    Clean and validate an extracted entity.
    Returns None if the entity should be discarded.
    """
    value = value.strip().strip(".,;:()")
    if not value:
        return None
    # Drop single-character entities
    if len(value) <= 1:
        return None
    # Drop entries that are purely numeric
    if value.isdigit():
        return None
    # Drop known blacklisted tokens (case-insensitive)
    if value.lower() in _DRUG_BLACKLIST:
        return None
    # Normalize abbreviations
    upper = value.upper()
    if upper in ABBREVIATION_MAP:
        return ABBREVIATION_MAP[upper]
    # Capitalize first letter
    return value[0].upper() + value[1:]


def extract_dosages_regex(text: str) -> List[Dict[str, str]]:
    """
    Extract structured dosage information using regex.
    Returns list of dicts with keys: dose, route, frequency.
    """
    dosages = []
    for m in _DOSE_PATTERN.finditer(text):
        dose_str = m.group(0).strip()
        # Look ahead for route and frequency within 60 chars
        context = text[m.end():m.end() + 60]
        route_m = _ROUTE_PATTERN.search(context)
        freq_m = _FREQ_PATTERN.search(context)
        dosages.append({
            "dose": dose_str,
            "route": route_m.group(0).strip() if route_m else "",
            "frequency": freq_m.group(0).strip() if freq_m else "",
        })
    return dosages


def normalize_abbreviations_in_text(text: str) -> str:
    """Replace known medical abbreviations in text with full forms."""
    for abbr, full in ABBREVIATION_MAP.items():
        # Only replace standalone abbreviations (word boundaries)
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    return text


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract clinical entities from text (intended to run on the FINAL ANSWER).

    Returns:
        {
            "drugs": [...],
            "conditions": [...],
            "dosages": [{"dose": ..., "route": ..., "frequency": ...}, ...],
            "raw_dosage_strings": [...]
        }
    """
    if not text or not text.strip():
        return {"drugs": [], "conditions": [], "dosages": [], "raw_dosage_strings": []}

    results = _run_ner_windowed(text)
    results = _merge_spans(results, text)

    entities: Dict[str, Any] = {
        "drugs": [],
        "conditions": [],
        "dosages": [],
        "raw_dosage_strings": [],
    }

    for r in results:
        label = r["entity_group"]
        score = r["score"]

        if score < MIN_NER_SCORE:
            continue

        raw_value = expand_to_word_boundaries(text, r["start"], r["end"])
        value = _clean_entity(raw_value)
        if not value:
            continue

        if label == "Medication":
            entities["drugs"].append(value)
        elif label in ("Disease_disorder", "Sign_symptom"):
            entities["conditions"].append(value)
        elif label == "Dosage":
            entities["raw_dosage_strings"].append(value)

    # Hybrid dosage extraction: also run regex on the text
    regex_dosages = extract_dosages_regex(text)
    entities["dosages"] = regex_dosages

    # Deduplicate drugs and conditions
    entities["drugs"] = list(dict.fromkeys(entities["drugs"]))
    entities["conditions"] = list(dict.fromkeys(entities["conditions"]))

    return entities


if __name__ == "__main__":
    test_cases = [
        "According to [3] PLAN, the patient was prescribed Nasonex (two sprays in each nostril for three weeks).",
        "The patient was started on Furosemide 20 mg IV daily and Lisinopril 10 mg p.o. daily for CHF.",
        "Patient prescribed Metformin 500mg BID for Type 2 DM and Atorvastatin 40mg QHS.",
    ]
    for text in test_cases:
        print(f"\nInput: {text}")
        result = extract_entities(text)
        print(f"Drugs: {result['drugs']}")
        print(f"Conditions: {result['conditions']}")
        print(f"Dosages: {result['dosages']}")