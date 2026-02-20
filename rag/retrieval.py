"""
rag/retrieval.py — Hybrid search pipeline: semantic + BM25 + RRF + cross-encoder rerank.

Pipeline for a semantic query:
  1a. ChromaDB vector search  → top n_results semantic candidates
  1b. BM25 keyword search     → top n_results keyword candidates   (if hybrid=True)
  2.  Reciprocal Rank Fusion  → merged, deduplicated ranked list   (if hybrid=True)
  3.  Cross-encoder rerank    → keep top_k by relevance score      (if do_rerank=True)
"""
import re

import streamlit as st
from rank_bm25 import BM25Okapi

from rag.resources import load_bm25_corpus, load_reranker


# ── Constants ─────────────────────────────────────────────────────────────────
RRF_K = 60   # standard constant; higher = dampens top-rank advantage


# ── Smart filter detection ────────────────────────────────────────────────────
def detect_smart_filter(question: str, name_to_id: dict):
    """
    Parses the user question for a friend name and returns
    (base_filter, matched_name, strategy).

    Strategy values: "Strict (Conversation)" | "Global (Mention)" | "None"
    """
    q  = question.lower()
    qn = q.replace(" ", "")

    matched_name, matched_id = None, None

    # Exact match
    for name, cid in name_to_id.items():
        if name in q:
            matched_name, matched_id = name, cid
            break

    # Partial / normalised match
    if not matched_name:
        for name, cid in name_to_id.items():
            nn = name.replace(" ", "")
            if len(nn) >= 3 and nn in qn:
                matched_name, matched_id = name, cid
                break
            for part in name.split():
                if len(part) >= 3 and part in q:
                    matched_name, matched_id = name, cid
                    break
            if matched_name:
                break

    if matched_name and matched_id:
        if re.search(rf"\b(about|mentioning)\s+{re.escape(matched_name)}\b", q):
            return None, matched_name, "Global (Mention)"
        return {"conversation": matched_id}, matched_name, "Strict (Conversation)"

    return None, None, "None"


# ── Metadata filter builder ───────────────────────────────────────────────────
def build_where(base_filter: dict | None, metadata_filters: dict) -> dict | None:
    """
    Merge a base filter (e.g. {conversation: id}) with extra metadata
    conditions into a ChromaDB-compatible where clause.

    metadata_filters keys (all optional):
      source       : str  e.g. "facebook_windowed"
      date_from    : str  ISO e.g. "2015-01-01T00:00:00"
      date_to      : str  ISO e.g. "2020-12-31T23:59:59"
      min_messages : int  minimum message_count
    """
    conditions = []

    if base_filter:
        for k, v in base_filter.items():
            conditions.append({k: {"$eq": v}})

    if metadata_filters.get("source"):
        conditions.append({"source": {"$eq": metadata_filters["source"]}})

    if metadata_filters.get("date_from"):
        conditions.append({"date": {"$gte": metadata_filters["date_from"]}})

    if metadata_filters.get("date_to"):
        conditions.append({"date": {"$lte": metadata_filters["date_to"]}})

    if metadata_filters.get("min_messages", 1) > 1:
        conditions.append({"message_count": {"$gte": metadata_filters["min_messages"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Reranker ──────────────────────────────────────────────────────────────────
def rerank(query: str, docs: list, top_k: int) -> list:
    """
    Second-pass reranking using a cross-encoder model.

    Steps:
      1. For each doc, score the pair (query, content) with the cross-encoder.
      2. Sort docs by descending score.
      3. Return the top_k highest-scoring docs.
    """
    if not docs:
        return docs

    reranker = load_reranker()
    pairs  = [(query, doc["content"]) for doc in docs]
    scores = reranker.predict(pairs)

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)

    docs.sort(key=lambda d: d["rerank_score"], reverse=True)
    return docs[:top_k]


# ── BM25 keyword search ───────────────────────────────────────────────────────
def keyword_search(query: str, bm25: BM25Okapi, corpus_docs: list,
                   n: int, id_to_name: dict) -> list:
    """BM25 keyword search over the full corpus. Returns top-n docs."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    results = []
    for idx in top_indices:
        doc = dict(corpus_docs[idx])
        doc["bm25_score"] = float(scores[idx])
        doc["friend"]     = id_to_name.get(doc["conversation_id"], doc["conversation_id"])
        results.append(doc)
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def rrf_merge(semantic_docs: list, keyword_docs: list) -> list:
    """
    Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i(d))

    Documents are identified by their content string.
    Results are annotated with semantic_rank, bm25_rank, and rrf_score.
    """
    sem_rank = {d["content"]: i + 1 for i, d in enumerate(semantic_docs)}
    kw_rank  = {d["content"]: i + 1 for i, d in enumerate(keyword_docs)}

    seen, merged = set(), []
    for d in semantic_docs + keyword_docs:
        key = d["content"]
        if key not in seen:
            seen.add(key)
            doc = dict(d)
            sr = sem_rank.get(key)
            kr = kw_rank.get(key)
            doc["semantic_rank"] = sr
            doc["bm25_rank"]     = kr
            rrf = 0.0
            if sr is not None:
                rrf += 1.0 / (RRF_K + sr)
            if kr is not None:
                rrf += 1.0 / (RRF_K + kr)
            doc["rrf_score"] = rrf
            merged.append(doc)

    merged.sort(key=lambda d: d["rrf_score"], reverse=True)
    return merged


# ── Main retrieval pipeline ───────────────────────────────────────────────────
def retrieve(question: str, n_results: int, base_filter, strategy: str,
             collection, episodic, id_to_name: dict,
             metadata_filters: dict | None = None,
             top_k: int = 10,
             do_rerank: bool = True,
             hybrid: bool = True):
    """
    Returns (docs_list, episodes_list).

    Full pipeline for semantic queries:
      1a. ChromaDB vector search  → top n_results semantic candidates
      1b. BM25 keyword search     → top n_results keyword candidates   (if hybrid=True)
      2.  Reciprocal Rank Fusion  → merged, deduplicated ranked list   (if hybrid=True)
      3.  Cross-encoder rerank    → keep top_k by relevance score      (if do_rerank=True)

    Strict conversation loads skip all of the above (want full chronological context).
    """
    docs, episodes = [], []
    if metadata_filters is None:
        metadata_filters = {}

    where = build_where(base_filter, metadata_filters)

    # Episodic memory
    if episodic:
        try:
            ep = episodic.query(query_texts=[question], n_results=5)
            if ep["documents"] and ep["documents"][0]:
                for d, m in zip(ep["documents"][0], ep["metadatas"][0]):
                    emotion = m.get("emotion", "")
                    label = f"{d}{' (' + emotion + ')' if emotion and emotion != 'neutral' else ''}"
                    episodes.append({"date": m.get("date", ""), "content": label})
        except Exception:
            pass

    # Main memory
    try:
        if strategy == "Strict (Conversation)" and base_filter and not metadata_filters:
            # Load entire conversation chronologically — skip search pipeline
            raw = collection.get(where=where, include=["metadatas", "documents"])
            pairs = list(zip(raw["documents"], raw["metadatas"]))
            pairs.sort(key=lambda x: x[1].get("date", ""))
            for d, m in pairs:
                conv_id = m.get("conversation", "")
                docs.append({
                    "date":            m.get("date", ""),
                    "conversation_id": conv_id,
                    "friend":          id_to_name.get(conv_id, conv_id),
                    "message_count":   m.get("message_count", 1),
                    "source":          m.get("source", ""),
                    "rerank_score":    None,
                    "rrf_score":       None,
                    "semantic_rank":   None,
                    "bm25_rank":       None,
                    "content":         d,
                })
        else:
            # ── Step 1a: Semantic (vector) search ──
            semantic_docs = []
            raw = collection.query(
                query_texts=[question],
                n_results=n_results,
                where=where,
            )
            if raw["documents"] and raw["documents"][0]:
                for d, m in zip(raw["documents"][0], raw["metadatas"][0]):
                    conv_id = m.get("conversation", "")
                    semantic_docs.append({
                        "date":            m.get("date", ""),
                        "conversation_id": conv_id,
                        "friend":          id_to_name.get(conv_id, conv_id),
                        "message_count":   m.get("message_count", 1),
                        "source":          m.get("source", ""),
                        "rerank_score":    None,
                        "rrf_score":       None,
                        "semantic_rank":   None,
                        "bm25_rank":       None,
                        "bm25_score":      None,
                        "content":         d,
                    })

            if hybrid:
                # ── Step 1b: BM25 keyword search over full corpus ──
                bm25, corpus_docs = load_bm25_corpus(collection)
                kw_docs = keyword_search(question, bm25, corpus_docs, n_results, id_to_name)

                # Post-filter BM25 results by metadata (BM25 doesn't know about metadata)
                if metadata_filters:
                    def _passes(doc):
                        if metadata_filters.get("source") and doc.get("source") != metadata_filters["source"]:
                            return False
                        if metadata_filters.get("date_from") and doc["date"] < metadata_filters["date_from"]:
                            return False
                        if metadata_filters.get("date_to") and doc["date"] > metadata_filters["date_to"]:
                            return False
                        if metadata_filters.get("min_messages", 1) > 1 and doc["message_count"] < metadata_filters["min_messages"]:
                            return False
                        return True
                    kw_docs = [d for d in kw_docs if _passes(d)]

                # ── Step 2: Reciprocal Rank Fusion ──
                docs = rrf_merge(semantic_docs, kw_docs)
            else:
                docs = semantic_docs

            # ── Step 3: Cross-encoder rerank ──
            if do_rerank and docs:
                docs = rerank(question, docs, top_k)
            else:
                docs = docs[:top_k]

    except Exception as e:
        st.error(f"Retrieval error: {e}")

    return docs, episodes
