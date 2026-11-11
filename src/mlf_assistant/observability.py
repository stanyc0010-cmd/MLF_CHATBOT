from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from mlf_assistant.config import get_settings


def langfuse_callbacks() -> list[Any]:
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except (ImportError, RuntimeError):
        return []


@contextmanager
def workflow_span(language: str) -> Iterator[None]:
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("mlf_assistant.workflow")
        with tracer.start_as_current_span("assistant.request", attributes={"mlf.language": language}):
            yield
    except ImportError:
        with nullcontext():
            yield