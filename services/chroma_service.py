"""
services/chroma_service.py — ChromaDB connection + collection management.

Replaces @st.cache_resource from rag/resources.py with module-level singletons.
"""
from __future__ import annotations

import os
import chromadb
from config import CHROMA_PATH, COLLECTION_NAME, EPISODIC_NAME


_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    """Return (or create) the singleton ChromaDB PersistentClient."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=os.path.expanduser(CHROMA_PATH))
    return _client


def get_collection(embedding_func=None):
    """Get or create the main knowledge collection."""
    client = get_client()
    kwargs = {"name": COLLECTION_NAME}
    if embedding_func:
        kwargs["embedding_function"] = embedding_func
    try:
        return client.get_or_create_collection(**kwargs)
    except Exception:
        client.delete_collection(COLLECTION_NAME)
        return client.get_or_create_collection(**kwargs)


def get_episodic(embedding_func=None):
    """Get or create the episodic memory collection."""
    client = get_client()
    kwargs = {"name": EPISODIC_NAME}
    if embedding_func:
        kwargs["embedding_function"] = embedding_func
    try:
        return client.get_or_create_collection(**kwargs)
    except Exception:
        return None


def invalidate():
    """Clear the cached client (forces reconnection on next call)."""
    global _client
    _client = None
