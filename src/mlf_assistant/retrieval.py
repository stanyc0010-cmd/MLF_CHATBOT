from __future__ import annotations

import math
from typing import Any

from mlf_assistant.db import connection
from mlf_assistant.embeddings import embed_texts, rerank
from mlf_assistant.schemas import Evidence


def retrieve(question: str, *, department: str | None = None, limit: int = 12) -> list[Evidence]:
    query_vector = embed_texts([question])[0]
    vector_literal = "[" + ",".join(str(float(value)) for value in query_vector) + "]"
    filters = """d.approval_status IN ('pending', 'approved')
        AND d.ingestion_status = 'ready' AND NOT d.superseded
        AND (d.effective_date IS NULL OR d.effective_date <= CURRENT_DATE)
        AND (d.review_date IS NULL OR d.review_date >= CURRENT_DATE)"""
    params: list[Any] = []
    if department:
        filters += " AND (d.department ILIKE %s OR d.category ILIKE %s)"
        params.extend([f"%{department}%", f"%{department}%"])
    with connection() as conn:
        vector_rows = conn.execute(
            f"""SELECT c.id, c.document_id, c.text, d.title, d.department, d.category,
                d.language, d.publication_date, d.version, d.effective_date, d.review_date,
                d.approval_status, 1 - (c.embedding <=> %s::vector) AS vector_similarity
                FROM document_chunks c JOIN documents d ON d.id = c.document_id
                WHERE {filters}
                ORDER BY c.embedding <=> %s::vector LIMIT %s""",
            [vector_literal, *params, vector_literal, limit],
        ).fetchall()
        keyword_rows = conn.execute(
            f"""SELECT c.id, c.document_id, c.text, d.title, d.department, d.category,
                d.language, d.publication_date, d.version, d.effective_date, d.review_date,
                d.approval_status,
                ts_rank_cd(c.search_tsv, websearch_to_tsquery('simple', %s)) AS keyword_score
                FROM document_chunks c JOIN documents d ON d.id = c.document_id
                WHERE {filters} AND c.search_tsv @@ websearch_to_tsquery('simple', %s)
                ORDER BY ts_rank_cd(c.search_tsv, websearch_to_tsquery('simple', %s)) DESC
                LIMIT %s""",
            [question, *params, question, question, limit],
        ).fetchall()

    fused: dict[int, tuple[dict[str, Any], float]] = {}
    for rows in (vector_rows, keyword_rows):
        for rank, row in enumerate(rows, start=1):
            key = row["id"]
            current = fused.get(key, (dict(row), 0.0))
            merged = dict(current[0])
            merged.update({name: value for name, value in row.items() if name not in merged or merged[name] is None})
            fused[key] = (merged, current[1] + 1 / (60 + rank))
    candidates = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:limit]
    if not candidates:
        return []

    passages = [item[0]["text"] for item in candidates]
    rerank_scores = rerank(question, passages)
    results: list[Evidence] = []
    for index, (row, fusion_score) in enumerate(candidates):
        relevance = (
            1 / (1 + math.exp(-rerank_scores[index]))
            if rerank_scores is not None
            else 0.8 * max(0.0, min(1.0, float(row.get("vector_similarity", 0.0))))
            + 0.2 * max(0.0, min(1.0, float(row.get("keyword_score", 0.0)) * 8))
        )
        results.append(
            Evidence(
                chunk_id=str(row["id"]), document_id=str(row["document_id"]), title=row["title"],
                text=row["text"], department=row["department"], category=row["category"],
                language=row["language"], publication_date=_date_string(row["publication_date"]),
                effective_date=_date_string(row["effective_date"]),
                review_date=_date_string(row["review_date"]), version=row["version"],
                approval_status=row["approval_status"], score=relevance,
            )
        )
    if rerank_scores is not None:
        order = sorted(range(len(results)), key=lambda index: rerank_scores[index], reverse=True)
        results = [results[index] for index in order]
    return results


def _date_string(value: Any) -> str:
    return value.isoformat() if value else ""