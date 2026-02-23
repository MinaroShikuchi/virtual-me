"""
services/embedding_service.py — Embedding model loading + caching.

Replaces LazyEmbeddingFunction and @st.cache_resource from rag/resources.py.
"""
from __future__ import annotations

import os

# Suppress HuggingFace progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.utils import embedding_functions

from config import EMBEDDING_MODEL


class LazyEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deferred-loading embedding function for ChromaDB."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._ef = None

    def _get_ef(self):
        if self._ef is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name, device=device
            )
        return self._ef

    def __call__(self, input: Documents) -> Embeddings:
        return self._get_ef()(input)

    def name(self) -> str:
        return "sentence_transformer"


# Module-level cache: {model_name: LazyEmbeddingFunction}
_cache: dict[str, LazyEmbeddingFunction] = {}


def get_embedding_func(model_name: str | None = None) -> LazyEmbeddingFunction:
    """Return a cached LazyEmbeddingFunction for the given model."""
    model = model_name or EMBEDDING_MODEL
    if model not in _cache:
        _cache[model] = LazyEmbeddingFunction(model)
    return _cache[model]


def invalidate(model_name: str | None = None):
    """Clear cached embedding functions."""
    if model_name:
        _cache.pop(model_name, None)
    else:
        _cache.clear()
