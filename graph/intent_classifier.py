"""
graph/intent_classifier.py

Clinical intent classifier — pure rule-based, no model required.
Maps a user query to one of 7 clinical intents and a ranked list
of Qdrant section names to prioritize during retrieval and reranking.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ── Intent → Priority Sections ────────────────────────────────────────────────
# Ordered: first section = highest priority
INTENT_SECTION_MAP: Dict[str, List[str]] = {
    "medication":      ["PLAN", "MEDICATIONS", "CURRENT MEDICATIONS", "IMPRESSION/PLAN"],
    "diagnosis":       ["ASSESSMENT", "IMPRESSION", "IMPRESSION/PLAN"],
    "allergy_history": ["ALLERGIES"],
    "lab":             ["INITIAL STUDIES", "FINDINGS", "LABS", "DESCRIPTION"],
    "treatment_plan":  ["PLAN", "RECOMMENDATIONS", "IMPRESSION/PLAN", "DISPOSITION"],
    "symptom":         ["SUBJECTIVE", "HPI", "HISTORY OF PRESENT ILLNESS", "CHIEF COMPLAINT"],
    "general":         [],
}

# ── Keyword patterns per intent (order matters — first match wins) ─────────────
_INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Allergy history must come BEFORE medication so "allergy patient" doesn't
    # get routed to medication.
    ("allergy_history", re.compile(
        r"\b(allerg(?:y|ies|ic|en)|anaphyla|hives|urticaria|drug reaction|nkda"
        r"|no known.*allerg|allerg.*histor)\b",
        re.IGNORECASE,
    )),
    # Medication — ONLY when the question is about treatment/prescription
    ("medication", re.compile(
        r"\b(medic(?:ation|ine|ated)|prescri(?:bed|ption|be)|drug(?:s)?"
        r"|treatment|prescribed|started on|given|dose|dosage|rx\b|formulat)\b",
        re.IGNORECASE,
    )),
    ("diagnosis", re.compile(
        r"\b(diagnos(?:is|ed|e)|assessment|impression|condition|disorder"
        r"|disease|syndrome|patholog|finding)\b",
        re.IGNORECASE,
    )),
    ("lab", re.compile(
        r"\b(lab(?:s|oratory)?|test(?:s|ing|ed)?|result|blood|cbc|bmp|bun"
        r"|creatinine|glucose|x-ray|imaging|mri|ct scan|echo(?:cardiogram)?|ekg|ecg)\b",
        re.IGNORECASE,
    )),
    ("treatment_plan", re.compile(
        r"\b(plan|recommend(?:ation|ed)?|management|follow.?up|disposition"
        r"|discharge|instruction|advice|next step|refer)\b",
        re.IGNORECASE,
    )),
    ("symptom", re.compile(
        r"\b(symptom|complaint|present(?:ing|s|ed)?|chief|hpi|history of present"
        r"|pain|ache|dyspnea|nausea|vomit|fever|cough|fatigue|weakness)\b",
        re.IGNORECASE,
    )),
]


def classify_intent(query: str) -> Tuple[str, List[str]]:
    """
    Classify a clinical query into an intent and return the matching
    priority sections.

    Returns:
        (intent_str, priority_sections_list)

    Example:
        classify_intent("What medications was the allergy patient prescribed?")
        # → ("medication", ["PLAN", "MEDICATIONS", "CURRENT MEDICATIONS", ...])
        #
        # NOTE: This query contains BOTH "allergy" and "medication" keywords.
        # The allergy_history pattern fires first — but because the query asks
        # about *prescribed* medications, we detect co-occurrence and fall back
        # to "medication" intent while STILL noting the allergy context.
    """
    q = query.strip()

    # Detect all matching intents (not just the first)
    matched: List[str] = []
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(q):
            matched.append(intent)

    if not matched:
        return "general", []

    # Special case: query mentions BOTH allergy AND medication keywords
    # → the user is asking about medication FOR an allergy patient.
    # Route to "medication" intent but include ALLERGIES in priority sections
    # so the validator can confirm patient identity.
    if "allergy_history" in matched and "medication" in matched:
        priority = list(INTENT_SECTION_MAP["medication"])
        if "ALLERGIES" not in priority:
            priority.append("ALLERGIES")
        return "medication", priority

    # Otherwise take the first matched intent (patterns are priority-ordered)
    intent = matched[0]
    return intent, list(INTENT_SECTION_MAP[intent])


def section_boost(section: str, priority_sections: List[str]) -> float:
    """
    Return a score boost for a chunk whose section is in priority_sections.
    Higher boost for sections earlier in the priority list.
    """
    section_upper = (section or "").upper().strip()
    try:
        rank = priority_sections.index(section_upper)
        # rank 0 → +3.0, rank 1 → +2.5, rank 2 → +2.0, ...
        return max(1.0, 3.0 - rank * 0.5)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    test_queries = [
        "What medications was the allergy patient prescribed?",
        "What is the patient's diagnosis?",
        "Does the patient have any known drug allergies?",
        "What were the lab results for this patient?",
        "What is the treatment plan?",
        "What symptoms did the patient present with?",
        "Tell me about the patient.",
    ]
    for q in test_queries:
        intent, sections = classify_intent(q)
        print(f"Q: {q}")
        print(f"   Intent: {intent} | Sections: {sections}\n")
