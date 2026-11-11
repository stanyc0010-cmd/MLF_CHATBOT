import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mlf_assistant.schemas import AssistantAnswer, Evidence
from mlf_assistant.workflow import (
    EvidenceGrade,
    _answer,
    _detect_language,
    _fallback_plan,
    _retrieve,
    run_assistant,
)


class FakeAnswerModel:
    def __init__(self):
        self.messages = []

    def with_structured_output(self, schema, include_raw=False):
        owner = self

        class Runnable:
            def invoke(self, messages):
                owner.messages.append(messages)
                if schema is EvidenceGrade:
                    return EvidenceGrade(relevant=True, confidence=0.9)
                return {
                    "parsed": AssistantAnswer(
                        answer="The document describes animal traceability.",
                        citations=["chunk-1"],
                        confidence=0.9,
                    ),
                    "raw": SimpleNamespace(usage_metadata={}),
                }

        return Runnable()


class WorkflowTests(unittest.TestCase):
    def test_detects_kiswahili_and_respects_preference(self):
        self.assertEqual(_detect_language("Nahitaji kibali cha uvuvi", "auto"), "sw")
        self.assertEqual(_detect_language("Nahitaji kibali cha uvuvi", "en"), "en")

    def test_fallback_classifier_routes_fisheries_and_legal_queries(self):
        fisheries = _fallback_plan("Fishing licence for aquaculture", "en")
        legal = _fallback_plan("What does this regulation say?", "en")
        self.assertEqual((fisheries["intent"], fisheries["department"]), ("fisheries", "Fisheries"))
        self.assertEqual(legal["intent"], "legal")

    def test_greeting_skips_database_and_model(self):
        with patch("mlf_assistant.workflow.langfuse_callbacks", return_value=[]):
            result = run_assistant("Habari")
        self.assertTrue(result["greeting"])
        self.assertFalse(result["needs_human"])
        self.assertEqual(result["citations"], [])

    def test_no_evidence_message_explains_missing_uploaded_documents(self):
        state = {"question": "objectives", "language": "en", "intent": "ministry", "department": ""}
        result = _answer({**state, "evidence": [], "reason": "no_current_evidence"})
        self.assertIn("No approved Ministry documents are currently loaded", result["answer"])
        self.assertTrue(result["needs_human"])

    def test_relevant_typo_match_above_lowered_threshold_can_be_answered(self):
        evidence = Evidence(
            chunk_id="chunk-1",
            document_id="document-1",
            title="Livestock traceability",
            text="The document describes animal traceability.",
            approval_status="approved",
            score=0.378,
        )
        state = {
            "question": "tracebility",
            "language": "en",
            "evidence": [evidence],
        }
        settings = SimpleNamespace(
            min_relevance_score=0.35,
            llm_input_cost_per_million=None,
            llm_output_cost_per_million=None,
        )
        with (
            patch("mlf_assistant.workflow._chat_model", return_value=FakeAnswerModel()),
            patch("mlf_assistant.workflow.get_settings", return_value=settings),
        ):
            result = _answer(state)

        self.assertEqual(result["answer"], "The document describes animal traceability.")
        self.assertEqual(result["citations"], ["chunk-1"])

    def test_answer_prompt_matches_response_format_to_the_question(self):
        evidence = [
            Evidence(
                chunk_id=f"chunk-{index}",
                document_id="document-1",
                title="Livestock traceability",
                text=f"Evidence passage {index}.",
                approval_status="approved",
                score=0.8 - index / 100,
            )
            for index in range(1, 13)
        ]
        state = {
            "question": "Compare the options in a table",
            "language": "en",
            "evidence": evidence,
        }
        settings = SimpleNamespace(
            min_relevance_score=0.35,
            llm_input_cost_per_million=None,
            llm_output_cost_per_million=None,
        )
        model = FakeAnswerModel()
        with (
            patch("mlf_assistant.workflow._chat_model", return_value=model),
            patch("mlf_assistant.workflow.get_settings", return_value=settings),
        ):
            result = _answer(state)

        self.assertEqual(result["citations"], ["chunk-1"])
        generation_prompt = model.messages[1][0][1]
        self.assertIn("Markdown bullet list with one '- ' item per independent", generation_prompt)
        self.assertIn("Markdown numbered list with one '1. ' step per action", generation_prompt)
        self.assertIn("Do not turn a simple answer into a list", generation_prompt)
        self.assertIn("Keep citations beside the claims they support", generation_prompt)
        self.assertIn("a Markdown table for comparisons", generation_prompt)
        self.assertIn("For calculations, show the formula", generation_prompt)
        self.assertIn("Preserve source numbers, dates, ranges", generation_prompt)
        self.assertIn("Start with a brief executive summary", generation_prompt)
        self.assertIn("Use a Markdown table when the evidence includes targets", generation_prompt)
        self.assertIn("[source:chunk-12]", model.messages[1][1][1])

    def test_uses_unfiltered_search_when_planner_infers_department(self):
        evidence = Evidence(
            chunk_id="chunk-strategy",
            document_id="document-strategy",
            title="Strategic plan",
            text="The plan objective is to improve service delivery.",
            department="teh",
            approval_status="approved",
            score=0.64,
        )
        state = {
            "question": "strategic plan objectives",
            "queries": ["strategic plan objectives from uploaded ministry plans"],
            "department": "Ministry of Livestock and Fisheries",
        }
        with patch(
            "mlf_assistant.workflow.retrieve",
            side_effect=[[], [evidence]],
        ) as retrieve:
            result = _retrieve(state)

        self.assertEqual(result["evidence"], [evidence])
        self.assertEqual(result["reason"], "")
        self.assertEqual(
            [call.kwargs for call in retrieve.call_args_list],
            [{}, {}],
        )

    def test_searches_original_question_and_planned_queries_across_all_documents(self):
        state = {
            "question": "How long are animal records retained?",
            "queries": ["animal record retention"],
            "department": "Fisheries",
        }
        with patch("mlf_assistant.workflow.retrieve", return_value=[]) as retrieve:
            result = _retrieve(state)

        self.assertEqual(result["reason"], "no_current_evidence")
        self.assertEqual(
            [call.args[0] for call in retrieve.call_args_list],
            ["animal record retention", "How long are animal records retained?"],
        )
        self.assertTrue(all(not call.kwargs for call in retrieve.call_args_list))

    def test_retrieves_contextual_followup_using_previous_user_question(self):
        state = {
            "question": "How long does it last?",
            "queries": ["How long does the permit last?"],
            "conversation": [
                {"role": "user", "content": "What is the fishing permit?"},
                {"role": "assistant", "content": "The document describes the permit."},
            ],
        }
        with patch("mlf_assistant.workflow.retrieve", return_value=[]) as retrieve:
            result = _retrieve(state)

        self.assertEqual(result["reason"], "no_current_evidence")
        self.assertEqual(
            [call.args[0] for call in retrieve.call_args_list],
            [
                "How long does the permit last?",
                "How long does it last?",
                "What is the fishing permit?\nFollow-up question: How long does it last?",
            ],
        )

    def test_one_failed_query_does_not_discard_successful_retrieval(self):
        evidence = Evidence(
            chunk_id="chunk-1",
            document_id="document-1",
            title="Livestock records",
            text="Records are retained for ten years.",
            approval_status="approved",
            score=0.8,
        )
        state = {"question": "How long are records kept?", "queries": ["records retention"]}
        with patch(
            "mlf_assistant.workflow.retrieve",
            side_effect=[RuntimeError("temporary embedding failure"), [evidence]],
        ):
            result = _retrieve(state)

        self.assertEqual(result["evidence"], [evidence])
        self.assertEqual(result["reason"], "")

    def test_all_failed_queries_report_retrieval_unavailable(self):
        state = {"question": "Where can I apply?", "queries": ["application location"]}
        with patch(
            "mlf_assistant.workflow.retrieve",
            side_effect=RuntimeError("database unavailable"),
        ):
            result = _retrieve(state)

        self.assertEqual(result, {"evidence": [], "reason": "retrieval_unavailable"})

    def test_partial_retrieval_failure_without_evidence_reports_unavailable(self):
        state = {"question": "Where can I apply?", "queries": ["application location"]}
        with patch(
            "mlf_assistant.workflow.retrieve",
            side_effect=[RuntimeError("temporary embedding failure"), []],
        ):
            result = _retrieve(state)

        self.assertEqual(result, {"evidence": [], "reason": "retrieval_unavailable"})

    def test_keeps_best_score_when_multiple_query_variants_find_same_chunk(self):
        high = Evidence(
            chunk_id="chunk-1",
            document_id="document-1",
            title="Livestock records",
            text="Records are retained for ten years.",
            approval_status="approved",
            score=0.8,
        )
        low = high.model_copy(update={"score": 0.4})
        state = {"question": "How long are records kept?", "queries": ["records retention"]}
        with patch("mlf_assistant.workflow.retrieve", side_effect=[[high], [low]]) as retrieve:
            result = _retrieve(state)

        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0].score, 0.8)
        self.assertEqual(retrieve.call_count, 2)


if __name__ == "__main__":
    unittest.main()