"""
graph/orchestrator.py

MedisenseAI clinical RAG orchestrator — full architectural overhaul.

Pipeline order:
    health_check
    → intent_classifier_node
    → retriever_node
    → validator_node          (ReAG-style chunk filtering)
    → answer_node
    → summarizer_node         (extractive, query-aware)
    → ner_node                (post-answer, answer text only)
    → structured_extractor_node
    → critic_node             (evidence-based confidence)
    → [retry? → query_rewriter_node → retriever_node] | END

Key improvements vs previous version:
- Clinical intent classification before retrieval
- Negation-aware retrieval + reranking
- ReAG validation step
- NER runs on final answer only
- Structured condition→medication extraction
- Real confidence scoring (not heuristic 1.0)
- Failure-type-aware retry routing
- Ollama health check at startup
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from graph.intent_classifier import classify_intent
from graph.query_rewriter import rewrite_query
from graph.retrieval_validator import validate_chunks
from graph.reranker import rerank_chunks, deduplicate_chunks
from agents.retriver_agent import format_context, has_citations, get_llm, prompt
from agents.summarizer_agent import summarize_chunks
from agents.ner_agent import extract_entities
from agents.structured_extractor import extract_structured_findings
from ingestion.retriver import hybrid_search, expand_with_parent_chunks
from langchain_core.output_parsers import StrOutputParser


# ── State ──────────────────────────────────────────────────────────────────────

class MediSenseState(TypedDict):
    # Input
    query: str

    # Intent classification
    intent: str
    priority_sections: List[str]

    # Query rewriting (on retry)
    rewritten_query: str

    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]

    # Post-validator chunks (subset of retrieved)
    validated_chunks: List[Dict[str, Any]]

    # Generation
    answer: str
    summary: str

    # Post-answer extraction
    entities: Dict[str, Any]
    structured_findings: Dict[str, Any]

    # Evaluation
    retry_count: int
    critique: str
    is_satisfactory: bool
    confidence: float          # real evidence-based score (0.0–1.0)

    # Error tracking
    error: Optional[str]
    error_type: Optional[str]  # "connection" | "grounding" | "generation" | None


# ── Helpers ───────────────────────────────────────────────────────────────────

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434


def _check_ollama_alive() -> bool:
    """Quick TCP ping to check if Ollama is reachable."""
    try:
        with socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=2):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def _compute_confidence(state: MediSenseState) -> float:
    """
    Compute evidence-based confidence score (0.0–1.0).

    Based on:
    - Mean reranker score of validated chunks (normalized)
    - Whether the answer contains citations
    - Answer length (proxy for specificity)
    - Whether validated chunks are non-empty
    """
    chunks = state.get("validated_chunks") or state.get("retrieved_chunks") or []
    answer = state.get("answer", "")

    if not chunks or not answer:
        return 0.0

    # 1. Reranker score component (clamped to 0–1)
    reranker_scores = [
        float(c.get("reranker_score", 0))
        for c in chunks
        if "reranker_score" in c
    ]
    if reranker_scores:
        # Cross-encoder scores typically range from -10 to +10 after boosting
        mean_score = sum(reranker_scores) / len(reranker_scores)
        # Normalize: score of 5 → 1.0, score of -5 → 0.0
        retrieval_conf = min(1.0, max(0.0, (mean_score + 5) / 10))
    else:
        retrieval_conf = 0.3

    # 2. Citation component
    citation_conf = 0.2 if has_citations(answer) else 0.0

    # 3. Answer length component (specificity proxy)
    word_count = len(answer.split())
    length_conf = min(0.2, word_count / 100)  # caps at 0.2 for 20+ words

    # 4. Grounding check: answer words appear in chunk text
    context_text = " ".join(c.get("chunk_text", "") for c in chunks[:5]).lower()
    answer_tokens = [w.lower() for w in answer.split() if len(w) > 4]
    if answer_tokens:
        grounded = sum(1 for t in answer_tokens if t in context_text)
        grounding_conf = min(0.2, grounded / len(answer_tokens) * 0.2)
    else:
        grounding_conf = 0.0

    total = retrieval_conf * 0.4 + citation_conf + length_conf + grounding_conf
    return round(min(1.0, max(0.0, total)), 3)


def _classify_error(error_str: str) -> str:
    """Classify an error string into a routing category."""
    if not error_str:
        return "unknown"
    e = error_str.lower()
    if any(x in e for x in ["winerror 10061", "connection refused", "10061", "actively refused"]):
        return "connection"
    if any(x in e for x in ["timeout", "timed out"]):
        return "connection"
    if "unable to generate" in e or "generation" in e:
        return "generation"
    return "grounding"


# ── Nodes ─────────────────────────────────────────────────────────────────────

def intent_classifier_node(state: MediSenseState) -> MediSenseState:
    """Classify query intent and determine priority sections."""
    query = state["query"]
    print(f"[Intent] Classifying: {query}")

    intent, priority_sections = classify_intent(query)
    print(f"[Intent] -> {intent} | sections: {priority_sections}")

    return {
        **state,
        "intent": intent,
        "priority_sections": priority_sections,
        "rewritten_query": query,  # initialize with original
    }


def retriever_node(state: MediSenseState) -> MediSenseState:
    """Fetch relevant chunks using intent-aware hybrid search + reranking."""
    query = state.get("rewritten_query") or state["query"]
    intent = state.get("intent", "general")
    priority_sections = state.get("priority_sections", [])

    print(f"[Retriever] Query: {query} (intent={intent})")

    try:
        # Hybrid search: dense + keyword + section-targeted
        chunks = hybrid_search(
            query,
            top_k=20,
            intent=intent,
            priority_sections=priority_sections,
        )
        print(f"[Retriever] Hybrid search: {len(chunks)} chunks")

        # Section-constrained parent expansion
        chunks = expand_with_parent_chunks(chunks, max_siblings=4, same_section_only=True)
        print(f"[Retriever] After section-constrained expansion: {len(chunks)} chunks")

        # Deduplication + reranking with intent context
        chunks = deduplicate_chunks(chunks)
        chunks = rerank_chunks(
            query,
            chunks,
            top_k=10,
            intent=intent,
            priority_sections=priority_sections,
        )
        print(f"[Retriever] After reranking: {len(chunks)} chunks")

        return {**state, "retrieved_chunks": chunks, "error": None, "error_type": None}

    except Exception as e:
        print(f"[Retriever ERROR]: {e}")
        return {
            **state,
            "retrieved_chunks": [],
            "error": str(e),
            "error_type": _classify_error(str(e)),
        }


def validator_node(state: MediSenseState) -> MediSenseState:
    """
    ReAG-style validation: use LLM to filter irrelevant retrieved chunks.
    Only runs if Ollama is alive. Falls back gracefully if it fails.
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {**state, "validated_chunks": []}

    query = state.get("rewritten_query") or state["query"]
    print(f"[Validator] Validating {len(chunks)} chunks")

    validated = validate_chunks(query, chunks, min_chunks=2)
    print(f"[Validator] {len(validated)} chunks passed validation")

    return {**state, "validated_chunks": validated}


def answer_node(state: MediSenseState) -> MediSenseState:
    """Generate grounded answer from validated chunks."""
    print("[Answer] Generating grounded answer")

    # Use validated_chunks if available, else fall back to retrieved_chunks
    chunks = state.get("validated_chunks") or state.get("retrieved_chunks") or []

    if not chunks:
        return {
            **state,
            "answer": "I cannot find relevant information in the provided documents.",
            "error": None,
        }

    context = format_context(chunks[:6], max_chars=3000)

    try:
        llm = get_llm()
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": state["query"],
        })

        # If answer is missing citations and isn't a refusal, retry once with citation nudge
        if "cannot find" not in answer.lower() and not has_citations(answer):
            answer = chain.invoke({
                "context": context,
                "question": state["query"] + " (Cite sources like [1], [2] after each claim.)",
            })

        return {**state, "answer": answer, "error": None, "error_type": None}

    except Exception as e:
        print(f"[Answer ERROR]: {e}")
        error_type = _classify_error(str(e))
        return {
            **state,
            "answer": "Unable to generate answer.",
            "error": str(e),
            "error_type": error_type,
        }


def summarizer_node(state: MediSenseState) -> MediSenseState:
    """
    Extractive summarization — query-aware, section-weighted.
    No model download required (pure sentence scoring).
    """
    print("[Summarizer] Extracting summary")
    chunks = state.get("validated_chunks") or state.get("retrieved_chunks") or []

    if not chunks:
        return {**state, "summary": ""}

    try:
        summary = summarize_chunks(chunks, query=state["query"])
        return {**state, "summary": summary}
    except Exception as e:
        print(f"[Summarizer ERROR]: {e}")
        return {**state, "summary": ""}


def ner_node(state: MediSenseState) -> MediSenseState:
    """
    Post-answer NER: extracts entities from the FINAL ANSWER only.
    This prevents unrelated medications from contaminating entity output.
    """
    answer = state.get("answer", "")
    print("[NER] Extracting entities from final answer")

    if not answer or "cannot find" in answer.lower() or "unable to generate" in answer.lower():
        return {**state, "entities": {"drugs": [], "conditions": [], "dosages": [], "raw_dosage_strings": []}}

    try:
        entities = extract_entities(answer)
        print(
            f"[NER] Found: {len(entities.get('drugs', []))} drugs, "
            f"{len(entities.get('conditions', []))} conditions, "
            f"{len(entities.get('dosages', []))} dosages"
        )
        return {**state, "entities": entities}
    except Exception as e:
        print(f"[NER ERROR]: {e}")
        return {**state, "entities": {"drugs": [], "conditions": [], "dosages": [], "raw_dosage_strings": []}}


def structured_extractor_node(state: MediSenseState) -> MediSenseState:
    """Extract structured condition→medication linkages from answer + NER entities."""
    print("[Structured] Extracting structured findings")
    try:
        findings = extract_structured_findings(
            state.get("answer", ""),
            state.get("entities", {}),
        )
        return {**state, "structured_findings": findings}
    except Exception as e:
        print(f"[Structured ERROR]: {e}")
        return {**state, "structured_findings": {"findings": [], "medications": []}}


def query_rewriter_node(state: MediSenseState) -> MediSenseState:
    """Rewrite query for retry — only fires between retries."""
    query = state["query"]
    intent = state.get("intent", "general")
    print(f"[QueryRewriter] Rewriting for retry (intent={intent})")

    rewritten = rewrite_query(query, intent)
    return {**state, "rewritten_query": rewritten}


def critic_node(state: MediSenseState) -> MediSenseState:
    """
    Evidence-based answer critic.

    Evaluates:
    - Answer is non-empty and non-trivially short
    - Answer is grounded in retrieved chunks
    - Citations present
    - No connection/generation errors
    - Answer not flagged as "unable to generate"
    """
    print("[Critic] Evaluating answer")

    answer = state.get("answer", "")
    query = state["query"].lower()
    chunks = state.get("validated_chunks") or state.get("retrieved_chunks") or []
    issues = []

    # 1. Basic sanity checks
    if not answer or len(answer.split()) < 8:
        issues.append("answer too short or empty")

    if not chunks:
        issues.append("no chunks retrieved")

    if "unable to generate" in answer.lower():
        issues.append("answer generation failed")

    if state.get("error_type") == "connection":
        # Connection errors — don't retry retrieval, just report
        issues.append("ollama connection error")

    # 2. Citation check
    if not has_citations(answer) and "cannot find" not in answer.lower():
        issues.append("no citations")

    # 3. Grounding check
    context_text = " ".join(c.get("chunk_text", "") for c in chunks[:5]).lower()
    answer_tokens = [w.lower() for w in answer.split() if len(w) > 4]
    if answer_tokens and not any(t in context_text for t in answer_tokens):
        issues.append("answer not grounded in retrieved context")

    # 4. Medication intent — verify at least one drug-like token in answer
    if "medication" in query or "prescribed" in query or "drug" in query:
        has_drug = any(w[0].isupper() and len(w) > 3 for w in answer.split())
        if not has_drug:
            issues.append("no medication found in answer")

    legitimate_refusal = "cannot find relevant information" in answer.lower()
    is_satisfactory = (len(issues) == 0) or legitimate_refusal

    critique = ", ".join(issues) if issues else "answer looks good"
    confidence = _compute_confidence(state)

    print(f"[Critic] {critique} | confidence={confidence}")

    return {
        **state,
        "critique": critique,
        "is_satisfactory": is_satisfactory,
        "confidence": confidence,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

MAX_RETRIES = 2


def should_retry(state: MediSenseState) -> str:
    """
    Failure-type-aware routing:
    - Connection error → don't retry (just end with fallback message)
    - Max retries reached → end
    - Not satisfactory → rewrite query and retry retrieval
    - Satisfactory → end
    """
    error_type = state.get("error_type")
    retry_count = state.get("retry_count", 0)

    if error_type == "connection":
        print("[Router] Connection error — ending without retry")
        return "done"

    if not state.get("is_satisfactory") and retry_count < MAX_RETRIES:
        print(f"[Router] Retrying (attempt {retry_count}/{MAX_RETRIES}) — {state.get('critique')}")
        return "retry"

    print(f"[Router] Done — {state.get('critique')} | confidence={state.get('confidence')}")
    return "done"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(MediSenseState)

    # Register all nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("validator", validator_node)
    graph.add_node("answer", answer_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("ner", ner_node)
    graph.add_node("structured_extractor", structured_extractor_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("critic", critic_node)

    # Main pipeline
    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", "retriever")
    graph.add_edge("retriever", "validator")
    graph.add_edge("validator", "answer")
    graph.add_edge("answer", "summarizer")
    graph.add_edge("summarizer", "ner")
    graph.add_edge("ner", "structured_extractor")
    graph.add_edge("structured_extractor", "critic")

    # Retry loop: critic → router → (query_rewriter → retriever) | END
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "query_rewriter",
            "done": END,
        },
    )
    graph.add_edge("query_rewriter", "retriever")

    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Health check before running
    if not _check_ollama_alive():
        print("[ERROR] Ollama is not running. Start it with: ollama serve")
        print("        Then run: ollama pull llama3")
        sys.exit(1)

    pipeline = build_graph()

    initial_state: MediSenseState = {
        "query": "What medications was the allergy patient prescribed?",
        "intent": "",
        "priority_sections": [],
        "rewritten_query": "",
        "retrieved_chunks": [],
        "validated_chunks": [],
        "entities": {},
        "structured_findings": {},
        "summary": "",
        "answer": "",
        "retry_count": 0,
        "critique": "",
        "is_satisfactory": False,
        "confidence": 0.0,
        "error": None,
        "error_type": None,
    }

    print(f"\nQuery: {initial_state['query']}\n")
    result = pipeline.invoke(initial_state)

    print(f"\n{'=' * 60}")
    print(f"FINAL ANSWER:\n{result['answer']}")
    print(f"\nSUMMARY:\n{result['summary']}")
    print(f"\nENTITIES:")
    print(f"  Drugs:      {result['entities'].get('drugs', [])}")
    print(f"  Conditions: {result['entities'].get('conditions', [])}")
    print(f"  Dosages:    {result['entities'].get('dosages', [])}")
    print(f"\nSTRUCTURED FINDINGS:")
    print(json.dumps(result['structured_findings'], indent=2))
    print(f"\nCRITIQUE:   {result['critique']}")
    print(f"CONFIDENCE: {result['confidence']}")
    print(f"INTENT:     {result['intent']}")
    print(f"RETRIES:    {result['retry_count'] - 1}")