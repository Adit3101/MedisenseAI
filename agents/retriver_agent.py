from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Temporary import approach; later convert repo to a package and remove sys.path hacking
sys.path.append(str(Path(__file__).resolve().parent.parent / "ingestion"))
from ingestion.retriver import search_chunks

load_dotenv()

OLLAMA_MODEL = "llama3"
MIN_SCORE = 0.40

# Fix 2: keep context budget aligned with ollama num_ctx=4096
MAX_CONTEXT_CHARS = 4000

# Fix 4: env-driven debug
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

_llm: Optional[ChatOllama] = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            num_ctx=4096,
            repeat_penalty=1.1,
        )
    return _llm


SYSTEM_PROMPT = """You are MediSense AI, a clinical document assistant.

STRICT RULES:
- Use ONLY the provided context.
- If the answer is not explicitly supported, reply exactly:
  "I cannot find relevant information in the provided documents."
- DO NOT use prior knowledge.
- DO NOT guess.
- Keep answers concise (2-5 sentences).
- Cite sources like [1], [2] after each claim.

Context:
{context}

Question:
{question}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def has_citations(text: str) -> bool:
    # Fix 1: flexible heuristic again (accept [2] etc.)
    return bool(re.search(r"\[\d+\]", text))


def dedupe_chunks_by_id(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for c in chunks:
        cid = c.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(c)
    return unique


def format_context(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts: List[str] = []
    total = 0

    for i, c in enumerate(chunks, 1):
        # Fix 3: skip empty chunk_text explicitly
        text = (c.get("chunk_text") or "").strip()
        if not text:
            continue

        block = f"[{i}] {c.get('section')}:\n{text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)


def answer_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    if DEBUG:
        print(f"[DEBUG] Query: {question}")

    fetch_k = max(top_k, 10)

    chunks_all = search_chunks(question, top_k=fetch_k, section=None)
    chunks_all = dedupe_chunks_by_id(chunks_all)

    # Never assume sorted ordering from retriever
    chunks_all = sorted(chunks_all, key=lambda x: float(x.get("score") or 0.0), reverse=True)

    filtered = [c for c in chunks_all if float(c.get("score") or 0.0) >= MIN_SCORE]

    # Adaptive fallback: if too few pass threshold, keep best available anyway
    if len(filtered) < 3:
        chunks = chunks_all[: min(10, len(chunks_all))]
    else:
        chunks = filtered

    if DEBUG:
        best = float(chunks[0].get("score") or 0.0) if chunks else 0.0
        print(f"[DEBUG] Retrieved {len(chunks)} chunks after filtering (fetch_k={fetch_k}, best_score={best:.4f})")

    if not chunks:
        return {
            "answer": "I cannot find relevant information in the provided documents.",
            "sources": [],
            "question": question,
        }

    context = format_context(chunks, max_chars=MAX_CONTEXT_CHARS)

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Stronger refusal check (still good): only enforce citations if it didn't refuse
    if not answer.lower().startswith("i cannot find") and not has_citations(answer):
        answer = chain.invoke(
            {
                "context": context,
                "question": question + " (Remember to cite sources like [1], [2] after each claim.)",
            }
        )

    return {
        "answer": answer,
        "sources": [
            {
                "chunk_id": c.get("chunk_id"),
                "specialty": c.get("specialty"),
                "section": c.get("section"),
                "score": round(float(c.get("score") or 0.0), 4),
            }
            for c in chunks
        ],
        "question": question,
        "fetch_k": fetch_k,
        "min_score": MIN_SCORE,
    }


if __name__ == "__main__":
    questions = [
        "What medications was the allergy patient prescribed?",
        "What were the findings of the echocardiogram?",
        "What is the treatment plan for the bariatric patient?",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print(f"{'=' * 60}")
        result = answer_question(q, top_k=5)
        print(f"A: {result['answer']}")
        print(f"\nSources ({len(result['sources'])} chunks):")
        for s in result["sources"]:
            print(f"  [{s['chunk_id']}] score={s['score']} | {s['section']}")