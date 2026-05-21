"""Quick test: show doc_0 chunks from the section-aware chunker."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from ingestion.new_chunker import chunk_documents_by_section
from ingestion.loader import load_mtsamples

BASE_DIR = Path(__file__).resolve().parent
docs = load_mtsamples(str(BASE_DIR / "data" / "raw" / "mtsamples.csv"))
chunks, stats = chunk_documents_by_section(docs)

doc0 = [c for c in chunks if c["parent_doc_id"] == "doc_0"]
print(f"doc_0: {len(doc0)} chunks\n")
for c in doc0:
    print(f"=== [{c['section']}] ({c['word_count']} words) ===")
    print(c["chunk_text"])
    print()

# Check if Nasonex is in any PLAN/ASSESSMENT chunk
for c in doc0:
    if "Nasonex" in c["chunk_text"]:
        print(f"*** NASONEX found in section: {c['section']} ***")
