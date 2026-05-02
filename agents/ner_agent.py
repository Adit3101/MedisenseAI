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
            aggregation_strategy="first"  # Changed from "simple"
        )
    return _ner


def extract_entities(text: str) -> Dict[str, List[str]]:
    ner = get_ner()
    results = ner(text)

    entities = {
        "drugs": [],
        "conditions": [],
        "dosages": [],
    }

    for r in results:
        label = r["entity_group"]
        value = r["word"].strip()
        score = r["score"]

        # Skip low confidence and short fragments (likely subword artifacts)
        if score < 0.5 or len(value) < 3:
            continue

        if label == "Medication":
            entities["drugs"].append(value)
        elif label == "Disease_disorder":
            entities["conditions"].append(value)
        elif label == "Dosage":
            entities["dosages"].append(value)

    for k in entities:
        entities[k] = list(set(entities[k]))

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