import unittest

from pydantic import ValidationError

from mlf_assistant.schemas import EscalationRequest


class EscalationSchemaTests(unittest.TestCase):
    def test_consent_requires_a_return_contact(self):
        with self.assertRaises(ValidationError):
            EscalationRequest(
                category="livestock", department="Livestock", language="en",
                question="Where do I apply?", reason="No current document found", consent=True,
            )

    def test_contact_is_accepted_when_consent_is_given(self):
        request = EscalationRequest(
            category="livestock", department="Livestock", language="en",
            question="Where do I apply?", reason="No current document found",
            consent=True, email="officer@example.org",
        )
        self.assertEqual(str(request.email), "officer@example.org")


if __name__ == "__main__":
    unittest.main()