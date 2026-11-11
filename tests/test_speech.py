import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from mlf_assistant.speech import SpeechServiceError, synthesize_speech, transcribe_audio


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            llm_provider="openai",
            llm_api_key="test-key",
            llm_base_url="https://api.example.test/v1/",
            llm_timeout_seconds=10,
            speech_transcription_model="test-transcriber",
            speech_synthesis_model="test-tts",
            speech_voice="alloy",
        )

    def test_transcribes_recording_using_selected_language(self):
        response = httpx.Response(
            200,
            json={"text": "Where can I get a fishing permit?"},
            request=httpx.Request("POST", "https://api.example.test/v1/audio/transcriptions"),
        )
        with (
            patch("mlf_assistant.speech.get_settings", return_value=self.settings),
            patch("mlf_assistant.speech.httpx.post", return_value=response) as post,
        ):
            transcript = transcribe_audio(b"recording", language="en")

        self.assertEqual(transcript, "Where can I get a fishing permit?")
        self.assertEqual(post.call_args.kwargs["data"], {"model": "test-transcriber", "language": "en"})
        self.assertIn("test-key", post.call_args.kwargs["headers"]["Authorization"])

    def test_generates_playable_audio(self):
        response = httpx.Response(
            200,
            content=b"mp3-data",
            request=httpx.Request("POST", "https://api.example.test/v1/audio/speech"),
        )
        with (
            patch("mlf_assistant.speech.get_settings", return_value=self.settings),
            patch("mlf_assistant.speech.httpx.post", return_value=response) as post,
        ):
            audio = synthesize_speech("Hello")

        self.assertEqual(audio, b"mp3-data")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "test-tts")
        self.assertEqual(post.call_args.kwargs["json"]["response_format"], "mp3")

    def test_voice_is_unavailable_without_openai_credentials(self):
        settings = SimpleNamespace(**{**vars(self.settings), "llm_provider": "ollama"})
        with patch("mlf_assistant.speech.get_settings", return_value=settings):
            with self.assertRaisesRegex(SpeechServiceError, "require the OpenAI provider"):
                transcribe_audio(b"recording")

    def test_rejects_audio_over_service_limit(self):
        with patch("mlf_assistant.speech.get_settings", return_value=self.settings):
            with self.assertRaisesRegex(SpeechServiceError, "25 MB"):
                transcribe_audio(b"x" * (25 * 1024 * 1024 + 1))


if __name__ == "__main__":
    unittest.main()