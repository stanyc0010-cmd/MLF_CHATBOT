import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from mlf_assistant import api
from mlf_assistant.db import DatabaseUnavailable


class DocumentUploadApiTests(unittest.TestCase):
    def setUp(self):
        api.app.dependency_overrides[api.authenticate_admin] = lambda: "admin"
        self.client = TestClient(api.app)

    def tearDown(self):
        api.app.dependency_overrides.clear()

    def test_upload_indexes_document_and_returns_active_state(self):
        metadata = {
            "title": "Fishing regulations",
            "department": "Fisheries",
            "category": "Legal",
            "language": "en",
            "issuing_authority": "Ministry",
            "version": "1",
        }
        with patch("mlf_assistant.api.ingest_document", return_value="document-123") as ingest:
            response = self.client.post(
                "/api/admin/documents",
                files={"upload": ("regulations.txt", b"Fishing rules", "text/plain")},
                data={"metadata": json.dumps(metadata)},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.json(),
            {
                "document_id": "document-123",
                "approval_status": "approved",
                "ingestion_status": "ready",
            },
        )
        self.assertEqual(ingest.call_args.args[:2], ("regulations.txt", b"Fishing rules"))
        self.assertEqual(ingest.call_args.kwargs["actor"], "admin")

    def test_upload_rejects_files_over_configured_limit(self):
        metadata = {
            "title": "Fishing regulations",
            "department": "Fisheries",
            "category": "Legal",
            "language": "en",
            "issuing_authority": "Ministry",
            "version": "1",
        }
        with (
            patch("mlf_assistant.api.get_settings", return_value=SimpleNamespace(max_upload_mb=0)),
            patch("mlf_assistant.api.ingest_document") as ingest,
        ):
            response = self.client.post(
                "/api/admin/documents",
                files={"upload": ("regulations.txt", b"too large", "text/plain")},
                data={"metadata": json.dumps(metadata)},
            )

        self.assertEqual(response.status_code, 413)
        ingest.assert_not_called()

    def test_upload_validates_metadata(self):
        response = self.client.post(
            "/api/admin/documents",
            files={"upload": ("regulations.txt", b"Fishing rules", "text/plain")},
            data={"metadata": "{}"},
        )

        self.assertEqual(response.status_code, 422)

    def test_upload_requires_administrator_authentication(self):
        api.app.dependency_overrides.clear()
        response = self.client.post(
            "/api/admin/documents",
            files={"upload": ("regulations.txt", b"Fishing rules", "text/plain")},
            data={"metadata": "{}"},
        )

        self.assertEqual(response.status_code, 401)

    def test_upload_reports_account_database_unavailability(self):
        api.app.dependency_overrides.clear()
        with patch("mlf_assistant.api.verify_admin", side_effect=DatabaseUnavailable):
            response = self.client.post(
                "/api/admin/documents",
                files={"upload": ("regulations.txt", b"Fishing rules", "text/plain")},
                data={"metadata": "{}"},
                auth=("admin", "password"),
            )

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
