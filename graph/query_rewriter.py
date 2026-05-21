"""
graph/query_rewriter.py

LLM-based intent-preserving query rewriter.

Only fires on retry (not on first pass).
Uses a concise, focused prompt to rephrase the query while preserving
clinical intent — avoiding the previous naive "append medication details" hack.
"""
from __future__ import annotations

from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

OLLAMA_MODEL = "llama3"

_llm: Optional[ChatOllama] = None

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a clinical query reformulation assistant.

Your task: Rephrase the given medical query to improve document retrieval.

Rules:
- Preserve the EXACT clinical intent and patient context.
- Do NOT change which patient or condition is being asked about.
- Make the query more specific about the clinical ACTION being asked
  (e.g., "prescribed", "treated with", "administered").
- Keep the output to ONE concise sentence (≤ 20 words).
- Output ONLY the rewritten query. No explanation, no preamble.

Intent: {intent}
Original query: {query}""",
    ),
    ("human", "{query}"),
])


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=512)
    return _llm


def rewrite_query(query: str, intent: str) -> str:
    """
    Rewrite the query using the LLM, preserving clinical intent.

    Args:
        query: original user query
        intent: classified intent string (e.g., "medication", "diagnosis")

    Returns:
        Rewritten query string. Falls back to original if LLM fails.
    """
    try:
        llm = get_llm()
        chain = _REWRITE_PROMPT | llm | StrOutputParser()
        rewritten = chain.invoke({"query": query, "intent": intent}).strip()

        # Sanity check: if rewritten is empty or way too long, fall back
        if not rewritten or len(rewritten.split()) > 30:
            return query

        print(f"[QueryRewriter] '{query}' → '{rewritten}'")
        return rewritten

    except Exception as e:
        print(f"[QueryRewriter ERROR]: {e} — using original query")
        return query


if __name__ == "__main__":
    test_cases = [
        ("What medications was the allergy patient prescribed?", "medication"),
        ("What is the patient's diagnosis?", "diagnosis"),
        ("What symptoms did the patient come in with?", "symptom"),
    ]
    for q, intent in test_cases:
        rewritten = rewrite_query(q, intent)
        print(f"Original : {q}")
        print(f"Rewritten: {rewritten}\n")
