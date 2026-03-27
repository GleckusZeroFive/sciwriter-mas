"""Local embedding model wrapper using sentence-transformers."""

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import settings

        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(
            settings.embedding_model,
            device="cpu",
        )
        logger.info("Embedding model loaded (dim=%d)", settings.embedding_dim)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Adds 'query: ' prefix for e5 models."""
    model = _get_model()
    prefixed = [f"query: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    return embed_texts([text])[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed documents (uses 'passage: ' prefix for e5 models)."""
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()
