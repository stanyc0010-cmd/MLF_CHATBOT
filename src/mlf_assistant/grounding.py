from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence


def eligible_evidence(
    evidence: Sequence[Mapping[str, Any]], *, today: date | None = None
) -> list[Mapping[str, Any]]:
    """Keep uploaded, non-withdrawn evidence that is currently in force."""
    current_date = today or date.today()
    eligible: list[Mapping[str, Any]] = []
    for item in evidence:
        if item.get("approval_status") in {"rejected", "withdrawn"} or item.get("superseded", False):
            continue
        effective = _date_value(item.get("effective_date"))
        review = _date_value(item.get("review_date"))
        if item.get("effective_date") and effective is None:
            continue
        if item.get("review_date") and review is None:
            continue
        if effective and effective > current_date:
            continue
        if review and review < current_date:
            continue
        if str(item.get("text", "")).strip():
            eligible.append(item)
    return eligible


def verify_grounded_answer(
    answer: str,
    citations: Sequence[str],
    evidence: Sequence[Mapping[str, Any]],
    *,
    language: str = "en",
    today: date | None = None,
) -> dict[str, Any]:
    """Return a cited answer only when every citation resolves to current evidence."""
    sources = eligible_evidence(evidence, today=today)
    valid_ids = {str(item.get("chunk_id", "")) for item in sources}
    valid_ids.update(f"{item.get('document_id', '')}:{item.get('chunk_id', '')}" for item in sources)
    requested = list(dict.fromkeys(_normalize_citation(str(ref)) for ref in citations))
    if not sources or not answer.strip() or not requested or any(ref not in valid_ids for ref in requested):
        return {
            "answer": _abstention(language),
            "citations": [],
            "confidence": 0.0,
            "abstained": True,
        }
    return {
        "answer": answer.strip(),
        "citations": requested,
        "confidence": 1.0,
        "abstained": False,
    }


def _date_value(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _normalize_citation(reference: str) -> str:
    reference = reference.strip()
    if reference.startswith("[source:") and reference.endswith("]"):
        return reference[len("[source:") : -1].strip()
    if reference.startswith("source:"):
        return reference[len("source:") :].strip()
    return reference


def _abstention(language: str) -> str:
    if language == "sw":
        return (
            "Sina ushahidi wa sasa na ulioidhinishwa wa kutosha kujibu kwa usahihi. "
            "Tafadhali wasiliana na Wizara au omba msaada wa afisa."
        )
    return (
        "I could not find sufficient current Ministry evidence to answer reliably. "
        "Please contact the Ministry or request assistance from an officer."
    )