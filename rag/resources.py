"""
rag/resources.py — Cached resource loaders (embeddings, reranker, ChromaDB, BM25, mappings).

All functions use @st.cache_resource so they run once per Streamlit session.
"""
import json
import os
from pathlib import Path

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from config import CHROMA_PATH, COLLECTION_NAME, EPISODIC_NAME, NAME_MAPPING_FILE, EMBEDDING_MODEL


def _get_embedding_model() -> str:
    """Return the active embedding model name (session_state > config default)."""
    try:
        return st.session_state.get("embedding_model", EMBEDDING_MODEL)
    except Exception:
        return EMBEDDING_MODEL


@st.cache_resource(show_spinner="Loading embeddings…")
def load_embedding_func(_model_name: str | None = None):
    model = _model_name or _get_embedding_model()
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model
    )


@st.cache_resource(show_spinner="Loading reranker…")
def load_reranker():
    """Cross-encoder for second-pass reranking. Scores (query, doc) pairs."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def load_chroma(_embedding_model: str | None = None):
    ef = load_embedding_func(_embedding_model or _get_embedding_model())
    client = chromadb.PersistentClient(path=os.path.expanduser(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)
    try:
        episodic = client.get_or_create_collection(EPISODIC_NAME, embedding_function=ef)
    except Exception:
        episodic = None
    return collection, episodic


@st.cache_resource(show_spinner="Building BM25 keyword index…")
def load_bm25_corpus(collection):
    """
    Loads ALL documents from ChromaDB and builds a BM25 full-text index.
    Cached for the session — rebuilt only when the app restarts.

    Returns (bm25, corpus_docs) where corpus_docs mirrors retrieve() output schema.
    """
    raw = collection.get(include=["documents", "metadatas"])
    corpus_docs = []
    tokenized = []
    for d, m in zip(raw["documents"] or [], raw["metadatas"] or []):
        if not d:
            continue
        corpus_docs.append({
            "content":        d,
            "conversation_id": m.get("conversation", ""),
            "date":           m.get("date", ""),
            "message_count":  m.get("message_count", 1),
            "source":         m.get("source", ""),
            "rerank_score":   None,
        })
        tokenized.append(d.lower().split())
    bm25 = BM25Okapi(tokenized)
    return bm25, corpus_docs


@st.cache_resource(show_spinner="Loading name mappings…")
def load_mappings():
    id_to_name, name_to_id = {}, {}
    try:
        if Path(NAME_MAPPING_FILE).exists():
            with open(NAME_MAPPING_FILE, "r", encoding="utf-8") as f:
                id_to_name = json.load(f)
            for cid, name in id_to_name.items():
                if name:
                    name_to_id[name.lower()] = cid
    except Exception as e:
        st.warning(f"Name mapping: {e}")
    return id_to_name, name_to_id
