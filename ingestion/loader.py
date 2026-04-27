import pandas as pd
from pathlib import Path
from typing import List, Dict


def load_mtsamples(csv_path: str) -> List[Dict]:
    """
    Load MTSamples CSV and return a list of documents.

    Args:
        csv_path: Path to mtsamples.csv

    Returns:
        List of dicts with keys: 'id', 'medical_specialty', 'sample_type', 'text'
    """
    df = pd.read_csv(csv_path)

    documents = []
    for idx, row in df.iterrows():
        doc = {
            'id': f"doc_{idx}",
            'medical_specialty': row.get('medical_specialty', 'Unknown'),
            'sample_type': row.get('sample_type', 'Unknown'),
            'text': row.get('transcription', '')
        }
        documents.append(doc)

    return documents

import os
print("CWD:", os.getcwd())

if __name__ == "__main__":
    # Test the loader
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / "data" / "raw" / "mtsamples.csv"

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        print("Resolved path:", csv_path.resolve())
        print("Make sure you've downloaded mtsamples.csv from Kaggle")
    else:
        docs = load_mtsamples(str(csv_path))
        print(f"Loaded {len(docs)} documents")
        print(f"\nFirst document:")
        print(f"  ID: {docs[0]['id']}")
        print(f"  Specialty: {docs[0]['medical_specialty']}")
        print(f"  Text length: {len(docs[0]['text'])} chars")