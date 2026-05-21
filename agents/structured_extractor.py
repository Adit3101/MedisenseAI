"""
agents/structured_extractor.py

Structured condition-medication relationship extractor.

Takes the final answer text + NER entities and extracts:
- Condition → medication linkages
- Structured medication entries with dose/route/frequency

Output format:
{
  "findings": [
    {
      "condition": "allergic rhinitis",
      "medications": [
        {"name": "Nasonex", "dose": "two sprays", "route": "intranasal", "frequency": "three weeks"}
      ]
    }
  ],
  "medications": [
    {"name": "Furosemide", "dose": "20 mg", "route": "IV", "frequency": "daily"}
  ]
}
"""
from __future__ import annotations

import re
from typing import Dict, List, Any, Optional

# ── Regex patterns ────────────────────────────────────────────────────────────

# Drug name: capitalized word(s), optionally followed by dose
_DRUG_NAME_PATTERN = re.compile(
    r'\b([A-Z][a-zA-Z\-]{2,}(?:\s+[A-Z][a-zA-Z\-]{2,})?)\b'
)

_DOSE_UNIT_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|mL|units?|mEq|meq|iu|IU|sprays?|drops?|puffs?)\b',
    re.IGNORECASE,
)

_ROUTE_PATTERN = re.compile(
    r'\b(by\s+mouth|oral(?:ly)?|intravenous(?:ly)?|IV\b|intramuscular(?:ly)?|IM\b'
    r'|subcutaneous(?:ly)?|SC\b|SQ\b|sublingual(?:ly)?|SL\b|topical(?:ly)?'
    r'|inhaled?|inhaler|nebuliz(?:er|ed)|intranasal(?:ly)?|nasal'
    r'|p\.o\.?|i\.v\.?|i\.m\.?|s\.c\.?\b'
    r'|each\s+nostril|nostril|spray)\b',
    re.IGNORECASE,
)

_FREQ_PATTERN = re.compile(
    r'\b(once\s+daily|twice\s+daily|three\s+times\s+daily|four\s+times\s+daily'
    r'|every\s+\d+\s+hours?|every\s+other\s+day|every\s+bedtime'
    r'|for\s+\w+\s+weeks?|for\s+\w+\s+days?'
    r'|daily|nightly|weekly|monthly|as\s+needed|p\.r\.n\.?'
    r'|qd|bid|tid|qid|qhs|qod|prn)\b',
    re.IGNORECASE,
)

# Condition → medication relationship patterns
_COND_MED_PATTERNS = [
    # "Drug for Condition" / "Drug to treat Condition"
    re.compile(
        r'([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)?)'  # drug
        r'\s+(?:for|to\s+treat|to\s+manage|for\s+(?:the\s+)?(?:treatment|management)\s+of)\s+'
        r'([a-z][a-z\s\-]+(?:rhinitis|asthma|diabetes|hypertension|disease|disorder|syndrome|infection|failure|cancer|allerg\w+)?)',
        re.IGNORECASE,
    ),
    # "Condition treated with Drug" / "Condition managed with Drug"
    re.compile(
        r'([a-z][a-z\s\-]+(?:rhinitis|asthma|diabetes|hypertension|disease|disorder|syndrome|infection|failure|cancer|allerg\w+)?)'
        r'\s+(?:treated|managed|controlled)\s+with\s+'
        r'([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)?)',
        re.IGNORECASE,
    ),
]


def _extract_drug_entry(text: str, drug_name: str) -> Dict[str, str]:
    """
    Given a drug name, find dose/route/frequency near it in the text.
    Searches within a 100-char window after the drug mention.
    """
    # Find position of the drug name in text
    idx = text.lower().find(drug_name.lower())
    if idx == -1:
        return {"name": drug_name, "dose": "", "route": "", "frequency": ""}

    window = text[idx:idx + 120]

    dose_m = _DOSE_UNIT_PATTERN.search(window)
    route_m = _ROUTE_PATTERN.search(window)
    freq_m = _FREQ_PATTERN.search(window)

    return {
        "name": drug_name,
        "dose": dose_m.group(0).strip() if dose_m else "",
        "route": route_m.group(0).strip() if route_m else "",
        "frequency": freq_m.group(0).strip() if freq_m else "",
    }


def _extract_all_medications(answer_text: str, drug_names: List[str]) -> List[Dict[str, str]]:
    """
    Build a structured medication list from drug names + contextual dose/route/frequency.
    """
    medications = []
    for drug in drug_names:
        entry = _extract_drug_entry(answer_text, drug)
        medications.append(entry)
    return medications


def _extract_condition_medication_links(
    answer_text: str,
    drug_names: List[str],
    condition_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Attempt to link conditions to medications using pattern matching.
    Falls back to associating all drugs with all conditions if patterns fail.
    """
    findings: List[Dict[str, Any]] = []

    # Leading clinical verbs that might get captured as part of the drug match group
    _VERB_PREFIX = re.compile(
        r'^\s*(?:taking|prescribed|started\s+on|given|initiated|on|using|receiving|administered)\s+',
        re.IGNORECASE,
    )

    for pattern in _COND_MED_PATTERNS:
        for match in pattern.finditer(answer_text):
            groups = match.groups()
            if len(groups) >= 2:
                g1, g2 = groups[0].strip(), groups[1].strip()
                # Strip leading verbs from both groups before checking
                g1_clean = _VERB_PREFIX.sub("", g1).strip()
                g2_clean = _VERB_PREFIX.sub("", g2).strip()

                # Determine which is drug vs condition using NER drug list
                matched_drug = None
                matched_condition = None
                for d in drug_names:
                    if d.lower() in g1_clean.lower():
                        matched_drug, matched_condition = d, g2_clean
                        break
                    elif d.lower() in g2_clean.lower():
                        matched_drug, matched_condition = d, g1_clean
                        break

                if not matched_drug:
                    continue

                drug_entry = _extract_drug_entry(answer_text, matched_drug)
                findings.append({
                    "condition": matched_condition.lower().strip(),
                    "medications": [drug_entry],
                })

    # Deduplicate by condition
    seen_conditions: set = set()
    deduped = []
    for f in findings:
        if f["condition"] not in seen_conditions:
            seen_conditions.add(f["condition"])
            deduped.append(f)

    # If no pattern-based linkage found but we have both drugs and conditions,
    # create a generic association
    if not deduped and drug_names and condition_names:
        med_entries = _extract_all_medications(answer_text, drug_names)
        for condition in condition_names[:3]:  # limit to top 3
            deduped.append({
                "condition": condition.lower().strip(),
                "medications": med_entries,
            })

    return deduped


def extract_structured_findings(
    answer_text: str,
    entities: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build structured clinical extraction from the final answer + NER entities.

    Args:
        answer_text: the generated answer string
        entities: output from ner_agent.extract_entities()

    Returns:
        {
            "findings": [{"condition": str, "medications": [{"name", "dose", "route", "frequency"}]}],
            "medications": [{"name", "dose", "route", "frequency"}],
        }
    """
    drug_names: List[str] = entities.get("drugs", [])
    condition_names: List[str] = entities.get("conditions", [])

    medications = _extract_all_medications(answer_text, drug_names)
    findings = _extract_condition_medication_links(answer_text, drug_names, condition_names)

    return {
        "findings": findings,
        "medications": medications,
    }


if __name__ == "__main__":
    sample_answer = (
        "According to [3] PLAN, the allergy patient was prescribed Nasonex "
        "(two sprays in each nostril for three weeks) for allergic rhinitis. "
        "Additionally, she is currently taking Ortho Tri-Cyclen and Allegra daily."
    )
    sample_entities = {
        "drugs": ["Nasonex", "Ortho Tri-Cyclen", "Allegra"],
        "conditions": ["allergic rhinitis"],
        "dosages": [{"dose": "two sprays", "route": "intranasal", "frequency": "three weeks"}],
    }
    import json
    result = extract_structured_findings(sample_answer, sample_entities)
    print(json.dumps(result, indent=2))
