from __future__ import annotations

import logging
import re
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from mlf_assistant.config import get_settings
from mlf_assistant.grounding import verify_grounded_answer
from mlf_assistant.observability import langfuse_callbacks, workflow_span
from mlf_assistant.retrieval import retrieve
from mlf_assistant.schemas import AssistantAnswer, Evidence


logger = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    language: Literal["en", "sw"]
    intent: Literal["ministry", "livestock", "fisheries", "veterinary", "legal", "service", "other"]
    department: str = ""
    queries: list[str] = Field(min_length=1, max_length=3)


class EvidenceGrade(BaseModel):
    relevant: bool
    confidence: float = Field(ge=0, le=1)


class WorkflowState(TypedDict, total=False):
    question: str
    conversation: list[dict[str, str]]
    preferred_language: str
    language: str
    intent: str
    department: str
    queries: list[str]
    evidence: list[Evidence]
    answer: str
    citations: list[str]
    confidence: float
    needs_human: bool
    reason: str
    greeting: bool


@lru_cache(maxsize=1)
def build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node("plan", _plan)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("answer", _answer)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _route_plan, {"greeting": END, "retrieve": "retrieve"})
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_assistant(
    question: str, conversation: list[dict[str, str]] | None = None,
    preferred_language: str = "auto",
) -> dict[str, Any]:
    started = perf_counter()
    language = _detect_language(question, preferred_language)
    with workflow_span(language):
        state = build_workflow().invoke(
            {"question": question, "conversation": conversation or [], "preferred_language": preferred_language},
            config={"recursion_limit": 8, "callbacks": langfuse_callbacks()},
        )
    try:
        from mlf_assistant.db import record_usage_event

        if state.get("greeting") or state.get("reason") == "retrieval_unavailable":
            return state
        event_id = record_usage_event(
            language=state.get("language", "en"), category=state.get("intent", "general"),
            answered=bool(state.get("citations")), escalated=state.get("needs_human", False),
            latency_ms=int((perf_counter() - started) * 1000),
            input_tokens=state.get("input_tokens"), output_tokens=state.get("output_tokens"),
            cost_usd=state.get("cost_usd"),
        )
        state["usage_id"] = event_id
    except Exception:
        pass
    return state


def _plan(state: WorkflowState) -> WorkflowState:
    question = state["question"].strip()
    language = _detect_language(question, state.get("preferred_language", "auto"))
    if _is_greeting(question):
        greeting = (
            "Habari. Naweza kusaidia kupata taarifa za Wizara kutoka kwenye nyaraka zilizoidhinishwa."
            if language == "sw"
            else "Hello. I can help find Ministry information from uploaded documents."
        )
        return {"language": language, "greeting": True, "answer": greeting, "citations": [], "confidence": 1.0, "needs_human": False}

    fallback = _fallback_plan(question, language)
    model = _chat_model()
    if model:
        try:
            planner = model.with_structured_output(QueryPlan)
            history = state.get("conversation", [])[-8:]
            plan = planner.invoke(
                [
                    ("system", "Classify this Tanzania Ministry livestock/fisheries enquiry. "
                     "Preserve Kiswahili and legal terms. Rewrite follow-ups as standalone queries; "
                     "split a compound question into at most three focused retrieval queries. Use searchable "
                     "document terms, relevant synonyms, acronyms, and named entities when plain-language "
                     "wording may differ from the source. Search across documents without assuming a department. "
                     "Never answer the user. Return only the requested structured plan."),
                    ("human", f"Recent conversation: {history}\nQuestion: {question}\nDetected language: {language}"),
                ]
            )
            preferred = state.get("preferred_language", "auto")
            return {**plan.model_dump(), "language": preferred if preferred in {"en", "sw"} else plan.language}
        except Exception:
            pass
    return fallback


def _retrieve(state: WorkflowState) -> WorkflowState:
    question = state["question"].strip()
    queries = [*state.get("queries", []), question]
    history = state.get("conversation", [])
    previous_question = next(
        (
            message["content"].strip()
            for message in reversed(history)
            if message.get("role") == "user" and message.get("content", "").strip()
        ),
        "",
    )
    if previous_question and _is_contextual_followup(question):
        queries.append(f"{previous_question}\nFollow-up question: {question}")

    distinct_queries: list[str] = []
    seen_queries: set[str] = set()
    for query in queries:
        normalized = query.strip()
        key = normalized.casefold()
        if normalized and key not in seen_queries:
            distinct_queries.append(normalized)
            seen_queries.add(key)

    evidence_by_id: dict[str, Evidence] = {}
    completed_query = False
    any_query_failed = False
    for query in distinct_queries:
        try:
            results = retrieve(query)
            completed_query = True
            for item in results:
                current = evidence_by_id.get(item.chunk_id)
                if current is None or item.score > current.score:
                    evidence_by_id[item.chunk_id] = item
        except Exception:
            any_query_failed = True
            logger.exception("Document retrieval failed for one query; continuing with remaining queries.")
    if not completed_query or (not evidence_by_id and any_query_failed):
        return {"evidence": [], "reason": "retrieval_unavailable"}
    evidence = sorted(evidence_by_id.values(), key=lambda item: item.score, reverse=True)
    return {"evidence": evidence, "reason": "" if evidence else "no_current_evidence"}


def _is_contextual_followup(question: str) -> bool:
    lowered = question.strip().lower()
    return bool(
        re.search(
            r"\b(?:it|they|them|their|its|this|that|these|those|there|then|same|"
            r"hiyo|hizo|hizi|hivi|hivyo|yake|wao|hapo)\b",
            lowered,
        )
        or re.match(r"^(?:and|also|what about|how about|na kuhusu|vipi kuhusu)\b", lowered)
    )


def _answer(state: WorkflowState) -> WorkflowState:
    language = state.get("language", "en")
    evidence = state.get("evidence", [])
    model = _chat_model()
    if not evidence or not model:
        reason = state.get("reason") or "model_unavailable"
        return _abstain_state(state, reason)
    threshold = get_settings().min_relevance_score
    candidates = [item for item in evidence if item.score >= threshold][:12]
    if not candidates:
        return _abstain_state(state, "low_relevance")
    context = "\n\n".join(
        f"[source:{item.chunk_id}] {item.title} (v{item.version}, {item.language})\n{item.text}"
        for item in candidates
    )
    try:
        grader = model.with_structured_output(EvidenceGrade)
        grade = grader.invoke(
            [
                ("system", "Decide whether the supplied uploaded-document passages, considered together, contain "
                 "information that can answer the user's question. Accept paraphrases, synonyms, abbreviations, "
                 "and facts distributed across passages; do not require the document to use the question's exact "
                 "wording. Mark irrelevant only when the evidence is about a different subject or lacks the "
                 "requested facts. Do not answer the question."),
                ("human", f"Question: {state['question']}\nPassages:\n{context}"),
            ]
        )
        if not grade.relevant or grade.confidence < threshold:
            return _abstain_state(state, "evidence_not_relevant")
        responder = model.with_structured_output(AssistantAnswer, include_raw=True)
        response = responder.invoke(
            [
                ("system", "You are the information assistant for Tanzania's Ministry of Livestock and Fisheries. "
                 "Answer using the supplied uploaded-document passages only and respond in the user's language. "
                 "First identify what the user is asking for, then give a direct, complete, practically useful "
                 "response. Write in a polished, professional ministry style suitable for a real-world briefing note. "
                 "Start with a brief executive summary in 2-4 sentences that captures the main point. Then present the "
                 "most relevant detail using a structure that matches the task: concise prose for simple questions; "
                 "Markdown bullet list with one '- ' item per independent fact, option, or requirement; Markdown numbered "
                 "list with one '1. ' step per action when order matters; and a Markdown table for comparisons, "
                 "target tracking, or source data when the evidence includes targets, KPIs, timelines, owners, or "
                 "implementation periods. Use a Markdown table when the evidence includes targets, KPIs, timelines, "
                 "or ownership details. Use bold headings, short paragraphs, and a practical rationale that explains "
                 "why the result matters. Keep list items concise, "
                 "parallel in structure, and grounded in the evidence. Do not turn a simple answer into a list, invent "
                 "steps, or imply an order that the evidence does not establish. Keep citations beside the claims they "
                 "support, including inside list items and table cells where appropriate. For calculations, show the "
                 "formula, values used, result, and units; calculate only when the supplied evidence provides the necessary "
                 "inputs. Preserve source numbers, dates, ranges, currency, and units exactly. Preserve legal wording and "
                 "do not translate legal provisions unless supported. Explain uncertainty; do not infer missing fees, "
                 "deadlines, requirements, or facts. Cite every material claim using exact chunk_id values from the "
                 "sources, without adding a source: prefix or brackets. If the evidence is insufficient, return a brief "
                 "abstention with no citations."),
                ("human", f"Question: {state['question']}\nEvidence:\n{context}"),
            ]
        )
    except Exception:
        return _abstain_state(state, "generation_unavailable")
    generated = response.get("parsed")
    if not generated:
        return _abstain_state(state, "structured_response_invalid")
    checked = verify_grounded_answer(
        generated.answer, generated.citations, [item.model_dump() for item in candidates], language=language
    )
    if checked["abstained"] or generated.confidence < threshold:
        return _abstain_state(state, "citation_or_evidence_check_failed")
    raw = response.get("raw")
    usage = getattr(raw, "usage_metadata", {}) or {}
    settings = get_settings()
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cost = None
    if (
        input_tokens is not None and output_tokens is not None
        and settings.llm_input_cost_per_million is not None
        and settings.llm_output_cost_per_million is not None
    ):
        cost = (
            input_tokens * settings.llm_input_cost_per_million
            + output_tokens * settings.llm_output_cost_per_million
        ) / 1_000_000
    return {
        "answer": checked["answer"], "citations": checked["citations"],
        "confidence": generated.confidence, "needs_human": False, "reason": "",
        "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost,
    }


def _route_plan(state: WorkflowState) -> str:
    return "greeting" if state.get("greeting") else "retrieve"


@lru_cache(maxsize=1)
def _chat_model():
    settings = get_settings()
    if not settings.llm_provider or not settings.llm_model:
        return None
    try:
        if settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.llm_model, api_key=settings.llm_api_key or None,
                base_url=settings.llm_base_url or None, timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        if settings.llm_provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=settings.llm_model, base_url=settings.llm_base_url or "http://localhost:11434", temperature=0)
    except Exception:
        return None
    return None


def _detect_language(question: str, preferred: str) -> str:
    if preferred in {"en", "sw"}:
        return preferred
    sw_terms = re.findall(r"\b(?:na|ya|za|je|vipi|habari|kibali|leseni|ufugaji|uvuvi|samaki|afya|mfugo|mwongozo|sheria|omba|gharama|wapi|tafadhali)\b", question.lower())
    return "sw" if len(sw_terms) >= 1 else "en"


def _is_greeting(question: str) -> bool:
    normalized = re.sub(r"[^a-zA-Z\s]", "", question).strip().lower()
    return normalized in {"hi", "hello", "habari", "hujambo", "mambo", "asante", "thanks", "thank you", "kwaheri", "goodbye"}


def _fallback_plan(question: str, language: str) -> dict[str, Any]:
    lowered = question.lower()
    if any(term in lowered for term in ("fish", "fisher", "uvuvi", "samaki", "aquaculture")):
        intent, department = "fisheries", "Fisheries"
    elif any(term in lowered for term in ("law", "act", "regulation", "sheria", "kanuni", "section")):
        intent, department = "legal", "Legal"
    elif any(term in lowered for term in ("veterinary", "vet", "daktari wa mifugo", "animal health")):
        intent, department = "veterinary", "Livestock"
    elif any(term in lowered for term in ("permit", "licen", "kibali", "leseni", "application", "maombi")):
        intent, department = "service", ""
    elif any(term in lowered for term in ("livestock", "animal", "mifugo", "ng'ombe", "mbuzi", "vet")):
        intent, department = "livestock", "Livestock"
    else:
        intent, department = "ministry", ""
    queries = [part.strip() for part in re.split(r"\s+(?:and|na)\s+|[?;]", question) if part.strip()]
    return {"language": language, "intent": intent, "department": department, "queries": (queries or [question])[:3]}


def _abstain(language: str, reason: str) -> WorkflowState:
    if language == "sw":
        answer = (
            "Sina ushahidi wa sasa wa Wizara unaotosha kujibu kwa usahihi. "
            "Hakuna nyaraka zilizoidhinishwa za Wizara zilizopakiwa kwenye hifadhi ya ushahidi. "
            "Tafadhali pakia nyaraka rasmi au omba msaada wa afisa wa Wizara."
            if reason == "no_current_evidence"
            else "Sina ushahidi wa sasa na ulioidhinishwa wa kutosha kujibu kwa usahihi. "
            "Unaweza kuomba msaada wa afisa wa Wizara."
        )
    else:
        answer = (
            "I could not find sufficient current Ministry evidence to answer reliably. "
            "No approved Ministry documents are currently loaded in the evidence store. "
            "Please upload official documents or request help from a Ministry officer."
            if reason == "no_current_evidence"
            else "I could not find sufficient current Ministry evidence to answer reliably. "
            "You can request help from a Ministry officer."
        )
    return {"answer": answer, "citations": [], "confidence": 0.0, "needs_human": True, "reason": reason}


def _abstain_state(state: WorkflowState, reason: str) -> WorkflowState:
    return {
        **_abstain(state.get("language", "en"), reason),
        "intent": state.get("intent", "other"),
        "department": state.get("department", ""),
    }