"""
services/bm25_service.py — BM25 keyword index singleton.
"""
from rank_bm25 import BM25Okapi

_bm25 = None
_corpus_docs = None


def get_bm25_corpus(collection):
    """Build (or return cached) BM25 index from a ChromaDB collection."""
    global _bm25, _corpus_docs
    if _bm25 is not None:
        return _bm25, _corpus_docs

    raw = collection.get(include=["documents", "metadatas"])
    corpus_docs = []
    tokenized = []
    for d, m in zip(raw["documents"] or [], raw["metadatas"] or []):
        if not d:
            continue
        corpus_docs.append({
            "content": d,
            "conversation_id": m.get("conversation", ""),
            "date": m.get("date", ""),
            "message_count": m.get("message_count", 1),
            "source": m.get("source", ""),
            "rerank_score": None,
        })
        tokenized.append(d.lower().split())

    _bm25 = BM25Okapi(tokenized)
    _corpus_docs = corpus_docs
    return _bm25, _corpus_docs


def invalidate():
    """Clear the cached BM25 index."""
    global _bm25, _corpus_docs
    _bm25 = None
    _corpus_docs = None
