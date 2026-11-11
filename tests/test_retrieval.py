import unittest
from contextlib import contextmanager
from datetime import date
from unittest.mock import patch

from mlf_assistant.retrieval import retrieve


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, vector_rows, keyword_rows):
        self.results = [FakeResult(vector_rows), FakeResult(keyword_rows)]
        self.queries = []

    def execute(self, query, params):
        self.queries.append(query)
        return self.results.pop(0)


def row(identifier, title):
    return {
        "id": identifier,
        "document_id": f"doc-{identifier}",
        "text": f"Evidence for {title}",
        "title": title,
        "department": "Fisheries",
        "category": "Fisheries",
        "language": "en",
        "publication_date": date(2026, 1, 1),
        "version": "1",
        "effective_date": None,
        "review_date": None,
        "approval_status": "pending",
    }


class HybridRetrievalTests(unittest.TestCase):
    def test_hybrid_candidates_include_processed_uploads_without_approval(self):
        fake = FakeConnection([row(1, "Vector first"), row(2, "Vector second")], [row(1, "Vector first")])

        @contextmanager
        def fake_connection():
            yield fake

        with (
            patch("mlf_assistant.retrieval.connection", fake_connection),
            patch("mlf_assistant.retrieval.embed_texts", return_value=[[0.0] * 1024]),
            patch("mlf_assistant.retrieval.rerank", return_value=[-2.0, 3.0]),
        ):
            results = retrieve("fishing licence", department="Fisheries")

        self.assertEqual(results[0].title, "Vector second")
        self.assertEqual(len(results), 2)
        self.assertTrue(all("approval_status IN ('pending', 'approved')" in query for query in fake.queries))
        self.assertTrue(all("review_date" in query for query in fake.queries))
        self.assertEqual(results[0].score, 1 / (1 + __import__("math").exp(-3.0)))


if __name__ == "__main__":
    unittest.main()