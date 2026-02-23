"""
services/reranker_service.py — Cross-encoder reranker singleton.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Return the singleton cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker
