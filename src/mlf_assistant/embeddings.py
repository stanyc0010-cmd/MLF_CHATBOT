from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from mlf_assistant.config import get_settings


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _embedding_model().encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return vectors.tolist()


@lru_cache(maxsize=1)
def _reranker():
    model_name = get_settings().reranker_model
    if not model_name:
        return None
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query: str, passages: Sequence[str]) -> list[float] | None:
    model = _reranker()
    if model is None or not passages:
        return None
    scores = model.predict([(query, passage) for passage in passages])
    return [float(score) for score in scores]