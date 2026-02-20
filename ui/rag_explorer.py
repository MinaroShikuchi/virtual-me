"""
ui/rag_explorer.py — RAG Explorer tab: inspect retrieved documents without calling the LLM.
"""
import streamlit as st

from rag.retrieval import retrieve, detect_smart_filter, build_where


def render_rag_tab(collection, episodic, id_to_name, name_to_id,
                   n_results, top_k, do_rerank, hybrid):
    st.markdown("### 🔍 RAG Explorer")
    st.caption("Query ChromaDB directly to inspect retrieved documents — **no LLM call**. "
               "All filters are passed directly as ChromaDB metadata `where` clauses.")

    # ── Top row: query + friend dropdown ──
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Semantic query",
                              placeholder="e.g. vacances, boulot, amour…", key="rag_query")
    with col2:
        friend_options = ["All conversations"] + sorted(
            {v for v in id_to_name.values() if v}, key=lambda x: x.lower()
        )
        selected_friend = st.selectbox("Filter by friend", friend_options, key="rag_friend")

    # ── Metadata filter row ──
    with st.expander("🔧 Metadata filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
        with fc1:
            date_from = st.date_input("Date from", value=None, key="rag_date_from")
        with fc2:
            date_to   = st.date_input("Date to",   value=None, key="rag_date_to")
        with fc3:
            source_opts = ["Any source", "facebook_windowed"]
            source_sel  = st.selectbox("Source", source_opts, key="rag_source")
        with fc4:
            min_msgs = st.number_input("Min messages in chunk", min_value=1, value=1,
                                       step=1, key="rag_min_msgs")

    if not query:
        st.info("Enter a query above to inspect retrieved documents.")
        return

    # ── Build base filter (conversation ID) ──
    base_filter = None
    strategy    = "Semantic"
    friend_name = None

    if selected_friend != "All conversations":
        matched_id = name_to_id.get(selected_friend.lower())
        if matched_id:
            base_filter = {"conversation": matched_id}
            strategy    = "Strict (Conversation)"
            friend_name = selected_friend
    else:
        base_filter, friend_name, strategy = detect_smart_filter(query, name_to_id)

    # ── Build extra metadata filters ──
    metadata_filters: dict = {}
    if source_sel != "Any source":
        metadata_filters["source"] = source_sel
    if date_from:
        metadata_filters["date_from"] = date_from.strftime("%Y-%m-%dT00:00:00")
    if date_to:
        metadata_filters["date_to"] = date_to.strftime("%Y-%m-%dT23:59:59")
    if min_msgs > 1:
        metadata_filters["min_messages"] = int(min_msgs)

    # Show the actual where clause being sent to ChromaDB
    final_where = build_where(base_filter, metadata_filters)
    with st.expander("🗂 ChromaDB `where` clause (debug)", expanded=False):
        st.json(final_where if final_where else {"note": "no filter — full collection search"})

    with st.spinner("Querying ChromaDB…"):
        docs, episodes = retrieve(
            query, n_results, base_filter, strategy,
            collection, episodic, id_to_name,
            metadata_filters=metadata_filters,
            top_k=top_k,
            do_rerank=do_rerank,
            hybrid=hybrid,
        )

    # ── Summary metrics ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Documents retrieved", len(docs))
    m2.metric("Strategy", strategy)
    m3.metric("Matched friend", friend_name.title() if friend_name else "—")
    m4.metric("Active filters", len(metadata_filters) + (1 if base_filter else 0))
    m5.metric("Mode", ("Hybrid" if hybrid else "Semantic") + ("+Rerank" if do_rerank else ""))

    st.divider()

    # ── Episodic results ──
    if episodes:
        st.markdown(f"#### 🧠 Episodic Memory ({len(episodes)} results)")
        for ep in episodes:
            st.markdown(
                f'<div class="rag-card">'
                f'<div class="rag-card-header">📅 <span>{ep["date"]}</span></div>'
                f'{ep["content"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.divider()

    # ── Document results ──
    if docs:
        st.markdown(f"#### 💬 Conversation Chunks ({len(docs)} results)")
        for i, doc in enumerate(docs, 1):
            src_badge = f" `{doc.get('source', '')}`" if doc.get("source") else ""
            score = doc.get("rerank_score")
            rrf   = doc.get("rrf_score")
            sem_r = doc.get("semantic_rank")
            kw_r  = doc.get("bm25_rank")

            badges = []
            if score is not None: badges.append(f"🏆 rerank {score:.2f}")
            if rrf   is not None: badges.append(f"RRF {rrf:.4f}")
            if sem_r is not None: badges.append(f"🔍 #{sem_r}")
            if kw_r  is not None: badges.append(f"🔤 #{kw_r}")
            badge_str = "  " + "  ".join(badges) if badges else ""

            with st.expander(
                f"**{i}.**{badge_str} [{doc['date'][:10]}] {doc['friend']} — {doc['message_count']} msg(s){src_badge}",
                expanded=(i <= 3),
            ):
                score_line = ""
                if score is not None or rrf is not None:
                    parts = []
                    if score is not None: parts.append(f"🏆 Rerank: <strong>{score:.4f}</strong>")
                    if rrf   is not None: parts.append(f"RRF: <strong>{rrf:.4f}</strong>")
                    if sem_r is not None: parts.append(f"🔍 Semantic rank: <strong>#{sem_r}</strong>")
                    if kw_r  is not None: parts.append(f"🔤 BM25 rank: <strong>#{kw_r}</strong>")
                    score_line = (
                        f'<div style="margin-bottom:6px;font-size:0.75rem;color:#a5b4fc">'
                        + "  |  ".join(parts)
                        + "</div>"
                    )
                st.markdown(
                    f'<div class="rag-card">'
                    f'{score_line}'
                    f'<div class="rag-card-header">'
                    f'📅 <span>{doc["date"]}</span> &nbsp;|&nbsp; '
                    f'👤 <span>{doc["friend"]}</span> &nbsp;|&nbsp; '
                    f'💬 <span>{doc["message_count"]} messages</span> &nbsp;|&nbsp; '
                    f'🏷 <span>{doc.get("source", "—")}</span>'
                    f'</div>'
                    f'<pre style="white-space:pre-wrap;color:#e2e8f0;font-size:0.82rem">{doc["content"]}</pre>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    elif not episodes:
        st.warning("No results found.")
