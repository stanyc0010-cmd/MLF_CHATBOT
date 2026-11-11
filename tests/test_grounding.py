import unittest
from datetime import date

from mlf_assistant.grounding import eligible_evidence, verify_grounded_answer


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "text": "Official source text",
            "approval_status": "approved",
            "effective_date": "2026-01-01",
            "review_date": "2027-01-01",
            "superseded": False,
        }

    def test_current_uploaded_documents_are_eligible_without_approval(self):
        pending = {**self.source, "approval_status": "pending"}
        rejected = {**self.source, "approval_status": "rejected"}
        withdrawn = {**self.source, "approval_status": "withdrawn"}
        expired = {**self.source, "review_date": "2026-11-08"}
        future = {**self.source, "effective_date": "2026-12-01"}
        self.assertEqual(
            eligible_evidence(
                [self.source, pending, rejected, withdrawn, expired, future],
                today=date(2026, 11, 9),
            ),
            [self.source, pending],
        )

    def test_missing_or_fabricated_citation_abstains(self):
        missing = verify_grounded_answer("Unsupported claim", [], [self.source])
        fabricated = verify_grounded_answer("Unsupported claim", ["made-up"], [self.source])
        self.assertTrue(missing["abstained"])
        self.assertTrue(fabricated["abstained"])
        self.assertEqual(fabricated["citations"], [])

    def test_malformed_freshness_metadata_is_not_authoritative(self):
        malformed = {**self.source, "review_date": "not-a-date"}
        result = verify_grounded_answer(
            "Unsupported claim", ["chunk-1"], [malformed], today=date(2026, 11, 9)
        )
        self.assertTrue(result["abstained"])

    def test_answer_requires_valid_source_reference(self):
        result = verify_grounded_answer(
            "A source-backed statement.", ["source:chunk-1"], [self.source]
        )
        self.assertFalse(result["abstained"])
        self.assertEqual(result["citations"], ["chunk-1"])

    def test_source_label_citation_must_still_resolve_to_supplied_evidence(self):
        fabricated = verify_grounded_answer(
            "An unsupported statement.", ["source:made-up"], [self.source]
        )
        self.assertTrue(fabricated["abstained"])


if __name__ == "__main__":
    unittest.main()