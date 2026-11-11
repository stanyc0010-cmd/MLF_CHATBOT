from __future__ import annotations

import httpx

from mlf_assistant.config import get_settings


MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_SPEECH_CHARACTERS = 4096


class SpeechServiceError(RuntimeError):
    pass


def transcribe_audio(audio: bytes, *, language: str = "auto") -> str:
    if not audio:
        raise SpeechServiceError("Record a question before requesting transcription.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise SpeechServiceError("The recording exceeds the 25 MB transcription limit.")

    settings = _openai_settings()
    data = {"model": settings.speech_transcription_model}
    if language in {"en", "sw"}:
        data["language"] = language
    try:
        response = httpx.post(
            f"{_base_url(settings)}/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            data=data,
            files={"file": ("question.wav", audio, "audio/wav")},
            timeout=max(settings.llm_timeout_seconds, 60),
        )
        response.raise_for_status()
        transcript = response.json().get("text", "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        raise SpeechServiceError("Transcription failed. Check the configured speech model and API.") from exc
    if not transcript:
        raise SpeechServiceError("No speech was detected. Record the question and try again.")
    return transcript


def synthesize_speech(text: str) -> bytes:
    if len(text) > MAX_SPEECH_CHARACTERS:
        raise SpeechServiceError("This answer is too long to generate as a single audio response.")
    if not text.strip():
        raise SpeechServiceError("There is no answer text to speak.")

    settings = _openai_settings()
    try:
        response = httpx.post(
            f"{_base_url(settings)}/audio/speech",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.speech_synthesis_model,
                "voice": settings.speech_voice,
                "input": text,
                "response_format": "mp3",
            },
            timeout=max(settings.llm_timeout_seconds, 60),
        )
        response.raise_for_status()
        return response.content
    except httpx.HTTPError as exc:
        raise SpeechServiceError("Speech generation failed. Check the configured speech model and API.") from exc


def _openai_settings():
    settings = get_settings()
    if settings.llm_provider != "openai" or not settings.llm_api_key:
        raise SpeechServiceError("Voice features require the OpenAI provider and a configured API key.")
    return settings


def _base_url(settings) -> str:
    return (settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")