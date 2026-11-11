import json
import os
import re
import uuid
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import conninfo, sql
from pydantic import BaseModel

from mlf_assistant import api, db
from mlf_assistant.config import get_settings
from mlf_assistant.retrieval import retrieve
from mlf_assistant.schemas import AssistantAnswer
from mlf_assistant.workflow import EvidenceGrade, QueryPlan


DATABASE_URL = os.environ.get("MLF_TEST_DATABASE_URL") or get_settings().test_database_url
POSTGRES_URL = (
    DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)
    if DATABASE_URL
    else None
)
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="Set MLF_TEST_DATABASE_URL to run PostgreSQL integration tests.",
)


@pytest.fixture
def isolated_postgres_schema(monkeypatch):
    schema_name = f"test_mlf_{uuid.uuid4().hex}"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))

    scoped_url = conninfo.make_conninfo(
        POSTGRES_URL,
        options=f"-csearch_path={schema_name},public",
    )
    settings = SimpleNamespace(
        database_url=scoped_url,
        admin_password="",
        admin_password_hash="",
        admin_username="integration-test",
    )
    monkeypatch.setattr(db, "get_settings", lambda: settings)
    try:
        db.initialize()
        yield
    finally:
        with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))


class FakeStructuredModel:
    def with_structured_output(self, schema: type[BaseModel], include_raw: bool = False):
        return FakeStructuredRunnable(schema, include_raw)


class FakeStructuredRunnable:
    def __init__(self, schema: type[BaseModel], include_raw: bool):
        self.schema = schema
        self.include_raw = include_raw

    def invoke(self, messages):
        if self.schema is QueryPlan:
            return QueryPlan(
                language="en",
                intent="fisheries",
                department="",
                queries=["fishing licence"],
            )
        if self.schema is EvidenceGrade:
            return EvidenceGrade(relevant=True, confidence=0.99)
        if self.schema is AssistantAnswer:
            prompt = messages[-1][1]
            citation = re.search(r"\[source:(\d+)\]", prompt).group(1)
            answer = AssistantAnswer(
                answer="Applicants can apply at the Fisheries licensing office.",
                citations=[citation],
                confidence=0.99,
            )
            return {"parsed": answer, "raw": SimpleNamespace(usage_metadata={})}
        raise AssertionError(f"Unexpected structured output schema: {self.schema}")


def test_upload_pgvector_retrieval_and_cited_answer(isolated_postgres_schema):
    api.app.dependency_overrides[api.authenticate_admin] = lambda: "integration-test"
    client = TestClient(api.app)
    metadata = {
        "title": "Fishing licence guidance",
        "department": "Fisheries",
        "category": "Government service",
        "language": "en",
        "issuing_authority": "Ministry",
        "version": "1",
    }
    source_text = (
        "Applicants can apply for a fishing licence at the Fisheries licensing office."
    )
    vector = [1.0] + [0.0] * 1023

    try:
        with patch(
            "mlf_assistant.documents.embed_texts",
            side_effect=RuntimeError("Simulated embedding service outage"),
        ):
            response = client.post(
                "/api/admin/documents",
                files={"upload": ("fishing.txt", source_text.encode(), "text/plain")},
                data={"metadata": json.dumps(metadata)},
            )
        assert response.status_code == 400
        with db.connection() as conn:
            failed_document = conn.execute(
                "SELECT id, ingestion_status FROM documents ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        assert failed_document["ingestion_status"] == "failed"

        with patch("mlf_assistant.documents.embed_texts", return_value=[vector]):
            response = client.post(
                "/api/admin/documents",
                files={"upload": ("fishing.txt", source_text.encode(), "text/plain")},
                data={"metadata": json.dumps(metadata)},
            )
        assert response.status_code == 201, response.text
        document_id = response.json()["document_id"]
        assert document_id == str(failed_document["id"])

        with patch("mlf_assistant.retrieval.embed_texts", return_value=[vector]), patch(
            "mlf_assistant.retrieval.rerank", return_value=None
        ):
            assert db.get_document_chunks(document_id)
            assert db.list_documents(status="approved")
            assert retrieve("fishing licence", department="Fisheries")

            with (
                patch("mlf_assistant.workflow._chat_model", return_value=FakeStructuredModel()),
                patch("mlf_assistant.workflow.langfuse_callbacks", return_value=[]),
            ):
                answer = client.post(
                    "/api/ask",
                    json={"question": "Where can I apply for a fishing licence?"},
                )

        assert answer.status_code == 200, answer.text
        result = answer.json()
        assert "Fisheries licensing office" in result["answer"]
        assert len(result["citations"]) == 1
        assert result["citations"][0]["document_id"] == document_id
        assert result["needs_human"] is False
    finally:
        api.app.dependency_overrides.clear()


def test_replace_and_delete_documents_preserve_audit_and_remove_chunks(isolated_postgres_schema):
    metadata = {
        "title": "Integration document",
        "department": "Fisheries",
        "category": "Government service",
        "language": "en",
        "issuing_authority": "Ministry",
        "version": "1",
    }
    original_id = db.store_document(metadata, "a" * 64, "original.txt", "integration-test")
    replacement_metadata = {**metadata, "version": "2"}
    replacement_id = db.store_document(
        replacement_metadata, "b" * 64, "replacement.txt", "integration-test"
    )
    vector_literal = "[" + ",".join(["1.0"] + ["0.0"] * 1023) + "]"

    with db.connection() as conn:
        conn.execute(
            "UPDATE documents SET approval_status = 'approved', ingestion_status = 'ready' WHERE id IN (%s, %s)",
            (original_id, replacement_id),
        )
        conn.execute(
            "INSERT INTO document_chunks (document_id, chunk_index, text, embedding) VALUES (%s, 0, 'old text', %s::vector), (%s, 0, 'new text', %s::vector)",
            (original_id, vector_literal, replacement_id, vector_literal),
        )

    db.update_document_dates(
        original_id, date(2026, 11, 10), date(2026, 12, 31), "integration-test"
    )
    with db.connection() as conn:
        updated_dates = conn.execute(
            "SELECT effective_date, review_date FROM documents WHERE id = %s",
            (original_id,),
        ).fetchone()
    assert updated_dates == {
        "effective_date": date(2026, 11, 10),
        "review_date": date(2026, 12, 31),
    }

    db.replace_document(original_id, replacement_id, "integration-test")
    with db.connection() as conn:
        original = conn.execute(
            "SELECT approval_status, superseded FROM documents WHERE id = %s",
            (original_id,),
        ).fetchone()
    assert original == {"approval_status": "withdrawn", "superseded": True}

    db.delete_document(replacement_id, "integration-test")
    with db.connection() as conn:
        assert conn.execute(
            "SELECT 1 FROM documents WHERE id = %s", (replacement_id,)
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM document_chunks WHERE document_id = %s", (replacement_id,)
        ).fetchone() is None
        audit = conn.execute(
            "SELECT actor, action, target_id FROM audit_logs WHERE target_id = %s",
            (replacement_id,),
        ).fetchone()
    assert audit == {
        "actor": "integration-test",
        "action": "delete_document",
        "target_id": replacement_id,
    }
