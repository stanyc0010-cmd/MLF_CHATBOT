from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import date
from typing import Any, Iterator

from mlf_assistant.config import get_settings


class DatabaseUnavailable(RuntimeError):
    pass


@contextmanager
def connection() -> Iterator[Any]:
    import psycopg
    from psycopg.rows import dict_row

    url = get_settings().database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    try:
        conn = psycopg.connect(url, row_factory=dict_row, connect_timeout=3)
    except psycopg.Error as exc:
        raise DatabaseUnavailable("The Ministry information database is unavailable.") from exc
    try:
        with conn:
            yield conn
    except psycopg.Error as exc:
        raise DatabaseUnavailable("The Ministry information database operation failed.") from exc
    finally:
        conn.close()


def initialize() -> None:
    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY, sha256 CHAR(64) UNIQUE NOT NULL, title TEXT NOT NULL,
            department TEXT NOT NULL, category TEXT NOT NULL, language TEXT NOT NULL,
            issuing_authority TEXT NOT NULL, publication_date DATE, effective_date DATE,
            version TEXT NOT NULL, review_date DATE, approval_status TEXT NOT NULL
                CHECK (approval_status IN ('pending','approved','rejected','withdrawn')),
            ingestion_status TEXT NOT NULL DEFAULT 'processing'
                CHECK (ingestion_status IN ('processing','ready','failed')),
            ingestion_error TEXT,
            superseded BOOLEAN NOT NULL DEFAULT FALSE, original_filename TEXT NOT NULL,
            created_by TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            approved_by TEXT, approved_at TIMESTAMPTZ
        )""",
        """CREATE TABLE IF NOT EXISTS document_chunks (
            id BIGSERIAL PRIMARY KEY, document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
            embedding vector(1024) NOT NULL,
            search_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
            UNIQUE(document_id, chunk_index)
        )""",
        "CREATE INDEX IF NOT EXISTS document_chunks_search_idx ON document_chunks USING GIN(search_tsv)",
        "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx ON document_chunks USING hnsw (embedding vector_cosine_ops)",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingestion_status TEXT NOT NULL DEFAULT 'ready'",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingestion_error TEXT",
        """UPDATE documents SET approval_status = 'approved',
            approved_by = COALESCE(approved_by, created_by),
            approved_at = COALESCE(approved_at, now())
            WHERE approval_status = 'pending' AND ingestion_status = 'ready'""",
        """CREATE TABLE IF NOT EXISTS enquiries (
            id UUID PRIMARY KEY, reference TEXT UNIQUE NOT NULL, category TEXT NOT NULL,
            department TEXT NOT NULL, language TEXT NOT NULL, question TEXT NOT NULL,
            reason TEXT NOT NULL, contact_name TEXT, contact_email TEXT, contact_phone TEXT,
            consent BOOLEAN NOT NULL CHECK (consent = TRUE), status TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open','in_review','resolved','closed')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS audit_logs (
            id BIGSERIAL PRIMARY KEY, actor TEXT NOT NULL, action TEXT NOT NULL,
            target_id TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS auth_users (
            username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin','officer')),
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), created_by TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS usage_events (
            id BIGSERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            language TEXT NOT NULL, category TEXT NOT NULL, answered BOOLEAN NOT NULL,
            escalated BOOLEAN NOT NULL, latency_ms INTEGER NOT NULL,
            input_tokens INTEGER, output_tokens INTEGER, cost_usd NUMERIC(12,6),
            feedback SMALLINT CHECK (feedback IN (-1, 1))
        )""",
        "ALTER TABLE usage_events ADD COLUMN IF NOT EXISTS feedback SMALLINT",
    ]
    with connection() as conn:
        for statement in statements:
            conn.execute(statement)
    bootstrap_admin()


def bootstrap_admin() -> None:
    settings = get_settings()
    if not settings.admin_password and not settings.admin_password_hash:
        return
    from mlf_assistant.security import hash_admin_password

    password_hash = settings.admin_password_hash or hash_admin_password(settings.admin_password)
    with connection() as conn:
        conn.execute(
            "INSERT INTO auth_users (username, password_hash, role, created_by) VALUES (%s, %s, 'admin', 'environment') ON CONFLICT (username) DO NOTHING",
            (settings.admin_username, password_hash),
        )


def store_document(metadata: dict[str, Any], digest: str, filename: str, actor: str) -> str:
    document_id = str(uuid.uuid4())
    with connection() as conn:
        conn.execute(
            """INSERT INTO documents
            (id, sha256, title, department, category, language, issuing_authority,
             publication_date, effective_date, version, review_date, approval_status,
             original_filename, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)""",
            (
                document_id, digest, metadata["title"], metadata["department"],
                metadata["category"], metadata["language"], metadata["issuing_authority"],
                metadata.get("publication_date") or None, metadata.get("effective_date") or None,
                metadata["version"], metadata.get("review_date") or None, filename, actor,
            ),
        )
    return document_id


def retry_failed_document(
    metadata: dict[str, Any], digest: str, filename: str, actor: str
) -> str | None:
    with connection() as conn:
        document = conn.execute(
            """SELECT id FROM documents
            WHERE sha256 = %s AND approval_status = 'pending' AND ingestion_status = 'failed'
            FOR UPDATE""",
            (digest,),
        ).fetchone()
        if not document:
            return None
        conn.execute("DELETE FROM document_chunks WHERE document_id = %s", (document["id"],))
        conn.execute(
            """UPDATE documents SET title = %s, department = %s, category = %s, language = %s,
            issuing_authority = %s, publication_date = %s, effective_date = %s, version = %s,
            review_date = %s, ingestion_status = 'processing', ingestion_error = NULL,
            original_filename = %s, created_by = %s WHERE id = %s""",
            (
                metadata["title"], metadata["department"], metadata["category"],
                metadata["language"], metadata["issuing_authority"],
                metadata.get("publication_date") or None, metadata.get("effective_date") or None,
                metadata["version"], metadata.get("review_date") or None, filename, actor,
                document["id"],
            ),
        )
        return str(document["id"])


def document_exists(digest: str) -> bool:
    with connection() as conn:
        return conn.execute("SELECT 1 FROM documents WHERE sha256 = %s", (digest,)).fetchone() is not None


def set_ingestion_status(document_id: str, status: str, error: str = "") -> None:
    if status not in {"processing", "ready", "failed"}:
        raise ValueError("Invalid ingestion status.")
    with connection() as conn:
        conn.execute(
            "UPDATE documents SET ingestion_status = %s, ingestion_error = %s WHERE id = %s",
            (status, error[:1000] or None, document_id),
        )


def save_chunks(document_id: str, chunks: list[str], vectors: list[list[float]]) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("Every document chunk must have one embedding.")
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO document_chunks (document_id, chunk_index, text, embedding) VALUES (%s, %s, %s, %s::vector)",
                [
                    (document_id, index, text, _vector_literal(vector))
                    for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
        conn.execute(
            """UPDATE documents SET ingestion_status = 'ready', ingestion_error = NULL,
                approval_status = 'approved', approved_by = created_by, approved_at = now()
                WHERE id = %s""",
            (document_id,),
        )


def get_document_chunks(document_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        return list(
            conn.execute(
                "SELECT id, text FROM document_chunks WHERE document_id = %s ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
        )


def update_chunk_embedding(chunk_id: int, vector: list[float]) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE document_chunks SET embedding = %s::vector WHERE id = %s",
            (_vector_literal(vector), chunk_id),
        )


def approve_document(document_id: str, actor: str, replacement_id: str | None = None) -> None:
    try:
        document_uuid = uuid.UUID(document_id)
        replacement_uuid = uuid.UUID(replacement_id) if replacement_id else None
    except ValueError as exc:
        raise ValueError("Use a valid document ID.") from exc
    if replacement_uuid == document_uuid:
        raise ValueError("A document cannot replace itself. Leave the replacement field blank.")

    with connection() as conn:
        result = conn.execute(
            "UPDATE documents SET approval_status = 'approved', approved_by = %s, approved_at = now() WHERE id = %s AND approval_status = 'pending' AND ingestion_status = 'ready'",
            (actor, document_uuid),
        )
        if result.rowcount != 1:
            raise ValueError("Only a pending document can be approved.")
        if replacement_uuid:
            replacement_result = conn.execute(
                """UPDATE documents SET superseded = TRUE, approval_status = 'withdrawn'
                WHERE id = %s AND approval_status = 'approved' AND NOT superseded""",
                (replacement_uuid,),
            )
            if replacement_result.rowcount != 1:
                raise ValueError("Only an active approved document can be replaced.")
        conn.execute(
            "INSERT INTO audit_logs (actor, action, target_id) VALUES (%s, %s, %s)",
            (actor, "approve_document", str(document_uuid)),
        )


def reject_document(document_id: str, actor: str) -> None:
    with connection() as conn:
        result = conn.execute(
            "UPDATE documents SET approval_status = 'rejected' WHERE id = %s AND approval_status = 'pending'",
            (document_id,),
        )
        if result.rowcount != 1:
            raise ValueError("Only a pending document can be rejected.")
        conn.execute(
            "INSERT INTO audit_logs (actor, action, target_id) VALUES (%s, %s, %s)",
            (actor, "reject_document", document_id),
        )


def replace_document(document_id: str, replacement_id: str, actor: str) -> None:
    try:
        document_uuid = uuid.UUID(document_id)
        replacement_uuid = uuid.UUID(replacement_id)
    except ValueError as exc:
        raise ValueError("Use valid document IDs.") from exc
    if document_uuid == replacement_uuid:
        raise ValueError("A document cannot replace itself.")

    with connection() as conn:
        documents = conn.execute(
            """SELECT id, approval_status, ingestion_status, superseded FROM documents
            WHERE id IN (%s, %s) ORDER BY id FOR UPDATE""",
            (document_uuid, replacement_uuid),
        ).fetchall()
        by_id = {str(item["id"]): item for item in documents}
        current = by_id.get(str(document_uuid))
        replacement = by_id.get(str(replacement_uuid))
        if not current or current["approval_status"] != "approved" or current["superseded"]:
            raise ValueError("Only an active approved document can be replaced.")
        if (
            not replacement
            or replacement["approval_status"] != "approved"
            or replacement["ingestion_status"] != "ready"
            or replacement["superseded"]
        ):
            raise ValueError("The replacement must be successfully indexed and active.")

        conn.execute(
            "UPDATE documents SET superseded = TRUE, approval_status = 'withdrawn' WHERE id = %s",
            (document_uuid,),
        )
        conn.execute(
            "INSERT INTO audit_logs (actor, action, target_id) VALUES (%s, %s, %s)",
            (actor, f"replace_document:{replacement_uuid}", str(document_uuid)),
        )


def delete_document(document_id: str, actor: str) -> None:
    try:
        document_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise ValueError("Use a valid document ID.") from exc

    with connection() as conn:
        document = conn.execute(
            "SELECT id FROM documents WHERE id = %s FOR UPDATE", (document_uuid,)
        ).fetchone()
        if not document:
            raise ValueError("The document was not found or was already deleted.")
        conn.execute(
            "INSERT INTO audit_logs (actor, action, target_id) VALUES (%s, 'delete_document', %s)",
            (actor, str(document_uuid)),
        )
        conn.execute("DELETE FROM documents WHERE id = %s", (document_uuid,))


def update_document_dates(
    document_id: str, effective_date: date | None, review_date: date | None, actor: str
) -> None:
    try:
        document_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise ValueError("Use a valid document ID.") from exc
    if effective_date and review_date and review_date < effective_date:
        raise ValueError("The review date cannot be earlier than the effective date.")

    with connection() as conn:
        result = conn.execute(
            "UPDATE documents SET effective_date = %s, review_date = %s WHERE id = %s",
            (effective_date, review_date, document_uuid),
        )
        if result.rowcount != 1:
            raise ValueError("The document was not found.")
        conn.execute(
            "INSERT INTO audit_logs (actor, action, target_id) VALUES (%s, 'update_document_dates', %s)",
            (actor, str(document_uuid)),
        )


def submit_enquiry(
    *, category: str, department: str, language: str, question: str, reason: str,
    consent: bool, name: str = "", email: str = "", phone: str = "",
) -> str:
    if not consent:
        raise ValueError("Consent is required before contact information can be collected.")
    identifier = uuid.uuid4()
    reference = f"MLF-{identifier.hex[:10].upper()}"
    with connection() as conn:
        conn.execute(
            """INSERT INTO enquiries
            (id, reference, category, department, language, question, reason, consent,
             contact_name, contact_email, contact_phone)
            VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s, %s)""",
            (identifier, reference, category, department, language, question, reason,
             name or None, email or None, phone or None),
        )
    return reference


def list_documents(*, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    query = "SELECT * FROM documents"
    params: list[Any] = []
    if status:
        query += " WHERE approval_status = %s"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(min(max(limit, 1), 500))
    with connection() as conn:
        return list(conn.execute(query, params).fetchall())


def list_enquiries(*, limit: int = 100) -> list[dict[str, Any]]:
    with connection() as conn:
        return list(
            conn.execute(
                "SELECT * FROM enquiries ORDER BY created_at DESC LIMIT %s",
                (min(max(limit, 1), 500),),
            ).fetchall()
        )


def dashboard_metrics() -> dict[str, Any]:
    with connection() as conn:
        counts = conn.execute(
            """SELECT count(*) AS total,
            count(*) FILTER (WHERE status = 'open') AS open,
            count(*) FILTER (WHERE language = 'sw') AS kiswahili,
            count(*) FILTER (WHERE category = 'livestock') AS livestock,
            count(*) FILTER (WHERE category = 'fisheries') AS fisheries
            FROM enquiries"""
        ).fetchone()
        docs = conn.execute(
            "SELECT count(*) AS total, count(*) FILTER (WHERE approval_status = 'pending') AS pending FROM documents"
        ).fetchone()
        usage = conn.execute(
            """SELECT count(*) AS total, count(*) FILTER (WHERE answered) AS answered,
            count(*) FILTER (WHERE NOT answered) AS unanswered,
            count(*) FILTER (WHERE created_at >= CURRENT_DATE) AS today,
            count(*) FILTER (WHERE created_at >= date_trunc('month', CURRENT_DATE)) AS month,
            count(*) FILTER (WHERE escalated) AS escalated,
            count(*) FILTER (WHERE feedback = -1) AS negative_feedback,
            coalesce(round(avg(latency_ms)), 0) AS avg_latency_ms,
            coalesce(sum(input_tokens + output_tokens), 0) AS tokens,
            coalesce(sum(cost_usd), 0) AS cost_usd FROM usage_events"""
        ).fetchone()
        return {
            **counts,
            "documents": docs["total"],
            "pending_documents": docs["pending"],
            **usage,
        }


def dashboard_analytics() -> dict[str, list[dict[str, Any]]]:
    with connection() as conn:
        daily = conn.execute(
            """WITH days AS (
                SELECT ((now() AT TIME ZONE 'Africa/Dar_es_Salaam')::date - offsets.day_offset) AS day
                FROM generate_series(29, 0, -1) AS offsets(day_offset)
            )
            SELECT days.day, count(events.id) AS activity,
                round(100.0 * count(*) FILTER (WHERE events.feedback = 1)
                    / nullif(count(*) FILTER (WHERE events.feedback IN (-1, 1)), 0), 1)
                    AS answer_quality,
                round(avg(events.latency_ms)) AS response_time_ms
            FROM days
            LEFT JOIN usage_events AS events
                ON (events.created_at AT TIME ZONE 'Africa/Dar_es_Salaam')::date = days.day
            GROUP BY days.day ORDER BY days.day"""
        ).fetchall()
        categories = conn.execute(
            """SELECT category, count(*) AS questions
            FROM usage_events GROUP BY category ORDER BY questions DESC, category"""
        ).fetchall()
        hours = conn.execute(
            """WITH days AS (
                SELECT ((now() AT TIME ZONE 'Africa/Dar_es_Salaam')::date - offsets.day_offset) AS day
                FROM generate_series(29, 0, -1) AS offsets(day_offset)
            ), slots AS (
                SELECT day_of_week, hour
                FROM generate_series(0, 6) AS weekdays(day_of_week)
                CROSS JOIN generate_series(0, 23) AS hours(hour)
            )
            SELECT slots.day_of_week, slots.hour, count(events.id) AS questions
            FROM slots
            LEFT JOIN usage_events AS events
                ON extract(dow FROM events.created_at AT TIME ZONE 'Africa/Dar_es_Salaam') = slots.day_of_week
                AND extract(hour FROM events.created_at AT TIME ZONE 'Africa/Dar_es_Salaam') = slots.hour
                AND (events.created_at AT TIME ZONE 'Africa/Dar_es_Salaam')::date IN (SELECT day FROM days)
            GROUP BY slots.day_of_week, slots.hour
            ORDER BY slots.day_of_week, slots.hour"""
        ).fetchall()
        documents = conn.execute(
            """SELECT category, count(*) AS documents
            FROM documents GROUP BY category ORDER BY documents DESC, category"""
        ).fetchall()
        unanswered = conn.execute(
            """SELECT question, count(*) AS occurrences
            FROM enquiries WHERE btrim(question) <> ''
            GROUP BY question ORDER BY occurrences DESC, question LIMIT 10"""
        ).fetchall()

        return {
            "daily": [dict(row) for row in daily],
            "categories": [dict(row) for row in categories],
            "hours": [dict(row) for row in hours],
            "documents": [dict(row) for row in documents],
            "unanswered": [dict(row) for row in unanswered],
        }


def update_enquiry_status(reference: str, status: str, actor: str) -> None:
    if status not in {"in_review", "resolved", "closed"}:
        raise ValueError("Invalid enquiry status.")
    with connection() as conn:
        result = conn.execute(
            "UPDATE enquiries SET status = %s, updated_at = now() WHERE reference = %s",
            (status, reference),
        )
        if result.rowcount != 1:
            raise ValueError("Enquiry reference was not found.")
        conn.execute(
            "INSERT INTO audit_logs (actor, action, target_id) VALUES (%s, %s, %s)",
            (actor, f"update_enquiry:{status}", reference),
        )


def record_usage_event(
    *, language: str, category: str, answered: bool, escalated: bool, latency_ms: int,
    input_tokens: int | None = None, output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> int:
    with connection() as conn:
        result = conn.execute(
            """INSERT INTO usage_events
            (language, category, answered, escalated, latency_ms, input_tokens, output_tokens, cost_usd)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (language, category, answered, escalated, max(0, latency_ms), input_tokens, output_tokens, cost_usd),
        )
        return result.fetchone()["id"]


def record_feedback(event_id: int, helpful: bool) -> None:
    with connection() as conn:
        result = conn.execute(
            "UPDATE usage_events SET feedback = %s WHERE id = %s",
            (1 if helpful else -1, event_id),
        )
        if result.rowcount != 1:
            raise ValueError("Feedback reference was not found.")


def get_account_auth(username: str) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute(
            "SELECT username, password_hash, role, active FROM auth_users WHERE username = %s",
            (username,),
        ).fetchone()


def list_users() -> list[dict[str, Any]]:
    with connection() as conn:
        return list(
            conn.execute(
                "SELECT username, role, active, created_at, created_by FROM auth_users ORDER BY username"
            ).fetchall()
        )


def create_user(username: str, password_hash: str, role: str, actor: str) -> None:
    if role not in {"admin", "officer"}:
        raise ValueError("Invalid role.")
    with connection() as conn:
        if conn.execute("SELECT 1 FROM auth_users WHERE username = %s", (username,)).fetchone():
            raise ValueError("That username already exists.")
        conn.execute(
            "INSERT INTO auth_users (username, password_hash, role, created_by) VALUES (%s, %s, %s, %s)",
            (username, password_hash, role, actor),
        )
        conn.execute(
            "INSERT INTO audit_logs (actor, action, target_id) VALUES (%s, 'create_user', %s)",
            (actor, username),
        )


def set_user_active(username: str, active: bool, actor: str) -> None:
    with connection() as conn:
        account = conn.execute(
            "SELECT role, active FROM auth_users WHERE username = %s FOR UPDATE", (username,)
        ).fetchone()
        if not account:
            raise ValueError("User account was not found.")
        if account["role"] == "admin" and account["active"] and not active:
            active_admins = conn.execute(
                "SELECT count(*) AS total FROM auth_users WHERE role = 'admin' AND active"
            ).fetchone()["total"]
            if active_admins <= 1:
                raise ValueError("The last active administrator cannot be deactivated.")
        result = conn.execute(
            "UPDATE auth_users SET active = %s WHERE username = %s",
            (active, username),
        )
        if result.rowcount != 1:
            raise ValueError("User account was not found.")
        conn.execute(
            "INSERT INTO audit_logs (actor, action, target_id) VALUES (%s, %s, %s)",
            (actor, "activate_user" if active else "deactivate_user", username),
        )


def _vector_literal(values: list[float]) -> str:
    if len(values) != 1024:
        raise ValueError("The configured embedding model must return 1024 dimensions.")
    return "[" + ",".join(str(float(value)) for value in values) + "]"