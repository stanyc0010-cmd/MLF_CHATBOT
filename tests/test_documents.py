import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from mlf_assistant.documents import (
    DocumentProcessingError,
    _chunk_document,
    _extract_text,
    reindex_document,
)


class DocumentChunkingTests(unittest.TestCase):
    def test_chunks_preserve_short_sections_and_respect_size(self):
        chunks = _chunk_document("Section 1\n\nRequirement A.\n\nSection 2\n\nRequirement B.", max_chars=35)
        self.assertIn("Section 1\n\nRequirement A.", chunks)
        self.assertIn("Section 2\n\nRequirement B.", chunks)
        self.assertTrue(all(len(chunk) <= 35 for chunk in chunks))

    def test_long_blocks_are_split_without_empty_chunks(self):
        chunks = _chunk_document("a" * 205, max_chars=80)
        self.assertEqual([len(chunk) for chunk in chunks], [80, 80, 45])

    def test_reindex_rejects_version_number_instead_of_document_uuid(self):
        with patch("mlf_assistant.documents.db.get_document_chunks") as get_chunks:
            with self.assertRaisesRegex(DocumentProcessingError, "version number"):
                reindex_document("1")
        get_chunks.assert_not_called()

    def test_pdf_extracts_text_without_docling(self):
        page = SimpleNamespace(extract_text=lambda: "PDF text")
        reader = SimpleNamespace(pages=[page])
        with patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=lambda _: reader)}):
            self.assertEqual(_extract_text(".pdf", b"pdf bytes"), "PDF text")


if __name__ == "__main__":
    unittest.main()