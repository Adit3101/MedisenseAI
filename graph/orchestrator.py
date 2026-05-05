from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from agents.ner_agent import extract_entities
from agents.summarizer_agent import summarize_text
from agents.retriver_agent import format_context, has_citations, get_llm, prompt
from ingestion.retriver import search_chunks
from langchain_core.output_parsers import StrOutputParser

# ── State ─────────────────────────────────────────────────────────────────────

class MediSenseState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    entities: Dict[str, List[str]]
    summary: str
    answer: str
    retry_count: int
    critique: str
    is_satisfactory: bool
    confidence: float
    error: Optional[str]

# ── Nodes ─────────────────────────────────────────────────────────────────────

def retriever_node(state: MediSenseState) -> MediSenseState:
    """Fetch relevant chunks. On retry, broaden the query."""
    query = state["query"]

    # Fix 4: targeted retry strategy
    if state["retry_count"] == 1:
        query = f"{state['query']} treatment medication details"
        print(f"[Retriever] Retry 1 with expanded query: {query}")
    elif state["retry_count"] == 2:
        query = f"{state['query']} clinical prescription drugs"
        print(f"[Retriever] Retry 2 with expanded query: {query}")
    else:
        print(f"[Retriever] Searching: {query}")

    try:
        chunks = search_chunks(query, top_k=10)
        return {**state, "retrieved_chunks": chunks, "error": None}
    except Exception as e:
        return {**state, "retrieved_chunks": [], "error": str(e)}


def ner_node(state: MediSenseState) -> MediSenseState:
    """Extract entities from top retrieved chunks."""
    print(f"[NER] Extracting entities")
    try:
        combined = " ".join(c["chunk_text"] for c in state["retrieved_chunks"][:3])
        entities = extract_entities(combined)
        return {**state, "entities": entities}
    except Exception:
        return {**state, "entities": {"drugs": [], "conditions": [], "dosages": []}}


def summarizer_node(state: MediSenseState) -> MediSenseState:
    """Summarize retrieved context."""
    print(f"[Summarizer] Summarizing context")
    try:
        combined = " ".join(c["chunk_text"] for c in state["retrieved_chunks"][:5])
        summary = summarize_text(combined)
        return {**state, "summary": summary}
    except Exception:
        return {**state, "summary": ""}


def answer_node(state: MediSenseState) -> MediSenseState:
    """
    Fix 1+2: Generate answer using retrieved chunks + summary + entities.
    Everything flows into the prompt — nothing is wasted.
    """
    print(f"[Answer] Generating grounded answer")

    chunks = state["retrieved_chunks"]
    if not chunks:
        return {**state, "answer": "I cannot find relevant information in the provided documents."}

    # Fix 1: smaller, focused context
    context = format_context(chunks[:5], max_chars=2500)

    # Enrich context with summary and entities (only if needed)
    enriched_context = context

    if state["summary"] and len(context) > 1500:
        enriched_context = f"{state['summary']}\n\n---\n\n{context}"

    if state["entities"]:
        e = state["entities"]
        entity_str = (
            "Key extracted clinical entities to consider in your answer:\n"
            f"- Drugs: {', '.join(e.get('drugs', [])) or 'none'}\n"
            f"- Conditions: {', '.join(e.get('conditions', [])) or 'none'}\n"
            f"- Dosages: {', '.join(e.get('dosages', [])) or 'none'}\n"
        )
        enriched_context = entity_str + "\n\n---\n\n" + enriched_context

    try:
        llm = get_llm()
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": enriched_context,
            "question": state["query"],
        })

        # Citation retry
        if "cannot find" not in answer.lower() and not has_citations(answer):
            answer = chain.invoke({
                "context": enriched_context,
                "question": state["query"] + " (Cite sources like [1], [2].)",
            })

        # # Fix 3: prefer summary if LLM deviates
        # if state["summary"] and state["summary"] not in answer:
        #     print("[Answer] LLM deviated — using summary fallback")
        #     answer = state["summary"]

        return {**state, "answer": answer}
    except Exception as e:
        print(f"[Answer ERROR]: {e}")  # ADD THIS
        return {**state, "answer": "Unable to generate answer.", "error": str(e)}


def critic_node(state: MediSenseState) -> MediSenseState:
    print(f"[Critic] Evaluating answer")

    answer = state["answer"]
    query = state["query"].lower()
    issues = []

    # 1. Basic sanity
    if not answer or len(answer.split()) < 8:
        issues.append("answer too short or empty")

    if not state["retrieved_chunks"]:
        issues.append("no chunks retrieved")

    if "unable to generate" in answer.lower():
        issues.append("answer generation failed")

    # 2. Intent-aware validation
    if "medication" in query or "prescribed" in query:
        has_med = any(
            word[0].isupper() and len(word) > 3
            for word in answer.split()
        )
        if not has_med:
            issues.append("no medications found in answer")

    context_text = " ".join(
        c["chunk_text"] for c in state["retrieved_chunks"][:5]
    ).lower()

    # check if answer tokens appear in context
    answer_tokens = [
        w.lower() for w in state["answer"].split()
        if len(w) > 4
    ]

    if not any(token in context_text for token in answer_tokens):
        issues.append("answer not grounded in retrieved context")

    legitimate_no_answer = "cannot find relevant information" in answer.lower()

    is_satisfactory = len(issues) == 0 or legitimate_no_answer
    critique = ", ".join(issues) if issues else "answer looks good"
    confidence = 1.0 if is_satisfactory else 0.5

    return {
        **state,
        "critique": critique,
        "is_satisfactory": is_satisfactory,
        "confidence": confidence,
        "retry_count": state["retry_count"] + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def should_retry(state: MediSenseState) -> str:
    if not state["is_satisfactory"] and state["retry_count"] < 2:
        print(f"[Router] Retrying — {state['critique']}")
        return "retry"
    print(f"[Router] Done — {state['critique']}")
    return "done"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(MediSenseState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("ner", ner_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("answer", answer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "ner")
    graph.add_edge("ner", "summarizer")
    graph.add_edge("summarizer", "answer")
    graph.add_edge("answer", "critic")

    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "retriever", "done": END}
    )

    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = build_graph()

    initial_state: MediSenseState = {
        "query": "What medications was the allergy patient prescribed?",
        "retrieved_chunks": [],
        "entities": {},
        "summary": "",
        "answer": "",
        "retry_count": 0,
        "critique": "",
        "is_satisfactory": False,
        "confidence": 0.0,
        "error": None,
    }

    print(f"Query: {initial_state['query']}\n")
    result = pipeline.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"FINAL ANSWER:\n{result['answer']}")
    print(f"\nENTITIES:\n{result['entities']}")
    print(f"\nSUMMARY:\n{result['summary']}")
    print(f"\nCRITIQUE: {result['critique']}")
    print(f"CONFIDENCE: {result['confidence']}")
    print(f"RETRIES: {result['retry_count'] - 1}")