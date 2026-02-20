"""
ui/sidebar.py — Streamlit sidebar: connections status, LLM settings, RAG settings.
"""
import streamlit as st
import ollama

from config import (
    DEFAULT_MODEL, DEFAULT_OLLAMA, DEFAULT_CTX,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
)


def render_sidebar(collection, episodic):
    """
    Renders the sidebar and returns:
        (model, ollama_host, num_ctx, n_results, top_k, do_rerank, hybrid,
         neo4j_uri, neo4j_user, neo4j_password)
    """
    with st.sidebar:
        st.markdown("## 🧠 Virtual Me")
        st.divider()

        # Connection status
        st.markdown("**Connections**")
        count = collection.count()
        st.markdown(f'<span class="status-ok">● ChromaDB</span> — {count:,} docs',
                    unsafe_allow_html=True)

        if episodic:
            ec = episodic.count()
            st.markdown(f'<span class="status-ok">● Episodic Memory</span> — {ec:,} episodes',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">○ Episodic Memory</span> — not found',
                        unsafe_allow_html=True)

        st.divider()

        # LLM settings
        st.markdown("**LLM Settings**")
        ollama_host = st.text_input("Ollama host", value=DEFAULT_OLLAMA, key="ollama_host")

        # Live Ollama connectivity check + model picker
        try:
            _client = ollama.Client(host=ollama_host)
            _models = [m["model"] for m in _client.list().get("models", [])]
            st.markdown(
                f'<span class="status-ok">● Ollama</span> — reachable '
                f'({len(_models)} model{"s" if len(_models) != 1 else ""})',
                unsafe_allow_html=True,
            )
            _default_idx = _models.index(DEFAULT_MODEL) if DEFAULT_MODEL in _models else 0
            model = st.selectbox("Model", _models, index=_default_idx, key="model")
        except Exception as _e:
            st.markdown('<span class="status-err">○ Ollama</span> — unreachable',
                        unsafe_allow_html=True)
            st.caption(f"Error: {_e}")
            model = st.text_input("Model (manual)", value=DEFAULT_MODEL, key="model")

        num_ctx = st.select_slider(
            "Context window",
            options=[8192, 16384, 32768, 65536],
            value=DEFAULT_CTX,
            format_func=lambda x: f"{x//1024}k",
            key="num_ctx",
        )

        st.divider()
        st.markdown("**RAG Settings**")
        n_results = st.slider("Retrieve (vector search)", 5, 100, 30, key="n_results",
                              help="How many candidates to fetch from ChromaDB before reranking")
        top_k = st.slider("Keep top-k (after reranking)", 1, 30, 10, key="top_k",
                          help="Cross-encoder reranker keeps the best k documents from the candidates")
        do_rerank = st.toggle("Enable reranking", value=True, key="do_rerank")
        hybrid    = st.toggle("Hybrid search (semantic + BM25)", value=True, key="hybrid",
                              help="Combines vector search with BM25 keyword search via Reciprocal Rank Fusion")

        st.divider()
        st.markdown("**Neo4j Settings**")
        neo4j_uri = st.text_input("Neo4j URI", value=NEO4J_URI, key="neo4j_uri")
        neo4j_user = st.text_input("Neo4j User", value=NEO4J_USER, key="neo4j_user")
        neo4j_password = st.text_input("Neo4j Password", value=NEO4J_PASSWORD, type="password", key="neo4j_password")

        return model, ollama_host, num_ctx, n_results, top_k, do_rerank, hybrid, \
               neo4j_uri, neo4j_user, neo4j_password
