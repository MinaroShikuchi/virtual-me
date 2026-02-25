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


import json
import ollama

# ── Intent Router ─────────────────────────────────────────────────────────────
def analyze_intent(question: str, model: str, host: str, name_to_id: dict) -> dict:
    """
    Pre-flight LLM call to structure the user's intent.
    Extracts relevant people, locations, and timeframes.
    Output is strictly JSON:
      {
        "people": ["Gabija"],
        "locations": ["Vilnius", "Paris"],
        "time_periods": ["2025", "last summer"],
        "query_type": "factual" | "emotional" | "exploratory"
      }
    """
    prompt = f"""
You are an intent router for a personal history database.
Perform semantic analysis on the user query and extract exactly the people, places, and timeframes mentioned.
Also, transform the natural language question into a keyword-heavy search query (search_query) that removes conversational filler and focuses only on core entities, actions, and topics to optimize retrieval from a vector and keyword database.

Only output valid JSON matching this schema:
{{
  "people": [list of person names extracted from the query],
  "locations": [list of places/locations extracted],
  "time_periods": [list of dates/years/time expressions],
  "query_type": "factual" | "emotional" | "exploratory",
  "search_query": "optimized keyword string for search engines"
}}

User query: {question}

Return ONLY the JSON. No markdown formatting.
    """.strip()
    
    try:
        # print(f"\n{'='*50}\n[DEBUG: INPUT TO INTENT ROUTER ({model})]\n{'='*50}")
        # print(f"PROMPT:\n{prompt}\n{'='*50}\n")
        
        client = ollama.Client(host=host)
        res = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json", # Ensure valid json
            options={"temperature": 0.0}
        )
        content = res["message"]["content"]
        intent = json.loads(content)
        
        # Capitalize extracted names for graph lookups 
        intent["people"] = [p.title() for p in intent.get("people", [])]
        
        print(f"[DEBUG: INTENT OUTPUT]\n{json.dumps(intent, indent=2)}\n{'='*50}\n")
        return intent

    except Exception as e:
        print(f"Intent analysis failed: {e}")
        return {"people": [], "locations": [], "time_periods": [], "query_type": "exploratory"}



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
def retrieve(question: str, n_results: int,
             collection, episodic, id_to_name: dict, name_to_id: dict,
             model: str, ollama_host: str,
             metadata_filters: dict | None = None,
             top_k: int = 10,
             relevance_threshold: float = 0.0,
             do_rerank: bool = True,
             hybrid: bool = True):
    """
    Returns (docs_list, episodes_list, intent_dict).


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

    print(f"\n[RETRIEVE] Query: '{question}'")

    # 1. Intent Analysis
    intent = analyze_intent(question, model, ollama_host, name_to_id)
    
    search_query = intent.get("search_query", "")
    if not search_query or not search_query.strip():
        search_query = question
        
    print(f"  -> Optimized Search Query: '{search_query}'")
    
    base_filter = None
    if intent.get("people"):
        print(f"  -> Intent: Detected people {intent['people']}")
        for p in intent["people"]:
            matched_id = name_to_id.get(p.lower())
            if matched_id:
                base_filter = {"conversation": matched_id}
                print(f"  -> Filter: Locking to conversation with {p} ({matched_id})")
                break

    where = build_where(base_filter, metadata_filters)
    print(f"  -> Chroma 'where' clause: {where}")

    # 2. Episodic Retrieval
    if episodic:
        try:
            ep = episodic.query(query_texts=[search_query], n_results=3)
            if ep["documents"] and ep["documents"][0]:
                print(f"  -> Episodic: Found {len(ep['documents'][0])} episodes")
                for d, m in zip(ep["documents"][0], ep["metadatas"][0]):
                    episodes.append({"date": m.get("date", ""), "content": d, "emotion": m.get("emotion", "neutral")})
        except Exception as e:
            print(f"  !! Episodic Error: {e}")

    # 3. Main Memory Retrieval
    try:
        # Step 1a: Semantic Search
        semantic_docs = []
        raw_semantic = collection.query(query_texts=[search_query], n_results=n_results, where=where)
        
        if raw_semantic["documents"] and raw_semantic["documents"][0]:
            print(f"  -> Semantic: Chroma returned {len(raw_semantic['documents'][0])} candidates")
            for d, m in zip(raw_semantic["documents"][0], raw_semantic["metadatas"][0]):
                conv_id = m.get("conversation", "")
                semantic_docs.append({
                    "content": d,
                    "date": m.get("date", ""),
                    "friend": id_to_name.get(conv_id, conv_id),
                    "importance": m.get("importance", 1),
                    "message_count":   m.get("message_count", 1),
                    "source":          m.get("source", ""),
                    "conversation_id": conv_id,
                    "type":            "semantic"
                })

        # Step 1b: Hybrid Logic
        if hybrid:
            bm25, corpus_docs = load_bm25_corpus(collection)
            kw_candidates = keyword_search(search_query, bm25, corpus_docs, n_results * 2, id_to_name)
            print(f"  -> BM25: Raw keyword search found {len(kw_candidates)} candidates")
            
            # Apply metadata filters to BM25 results manually
            kw_docs = []
            for d in kw_candidates:
                if _manual_filter(d, metadata_filters, log=True):
                    kw_docs.append(d)
            
            print(f"  -> BM25: {len(kw_docs)} survived metadata filters")
            
            # Step 2: RRF Merge
            docs = rrf_merge(semantic_docs, kw_docs[:n_results])
            print(f"  -> RRF: Merged pool size: {len(docs)}")
        else:
            docs = semantic_docs

        # Step 3: Reranking
        if do_rerank and docs:
            print(f"  -> Rerank: Scoring {len(docs)} documents...")
            ranked_docs = rerank(search_query, docs, len(docs)) # Get all scores
            
            # Filter by relevance threshold
            docs = [d for d in ranked_docs if d.get('rerank_score', -99) >= relevance_threshold]
            
            dropped = len(ranked_docs) - len(docs)
            if dropped > 0:
                print(f"  -> Filter: Discarded {dropped} docs below threshold ({relevance_threshold})")
            
            # Limit to top_k after filtering
            docs = docs[:top_k]

            for i, d in enumerate(docs[:3]):
                print(f"     Top {i+1} Score: {d.get('rerank_score'):.4f} | Date: {d.get('date')}")
        else:
            docs = docs[:top_k]

    except Exception as e:
        print(f"  !! Retrieval Error: {e}")

    print(f"[RETRIEVE] Completed. Returning {len(docs)} docs and {len(episodes)} episodes.\n")
    return docs, episodes, intent
    
def _manual_filter(doc, filters, log=False):
    """Helper with optional logging for filter rejections."""
    if not filters: return True
    
    if filters.get("source") and doc.get("source") != filters["source"]:
        return False
    
    # Example for date filtering
    if filters.get("date_from") and doc.get("date", "") < filters["date_from"]:
        if log: print(f"     (Dropped BM25 doc from {doc.get('date')} - too old)")
        return False

    return True