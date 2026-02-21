"""
ui/settings.py — Settings modal (cog) for LLM, RAG, and Neo4j parameters.
"""
import streamlit as st
import ollama

from config import (
    DEFAULT_MODEL, DEFAULT_OLLAMA, DEFAULT_CTX,
    EMBEDDING_MODEL, EMBEDDING_MODELS,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)


def init_settings_defaults():
    """Ensure all settings exist in session_state with sensible defaults."""
    defaults = {
        "ollama_host": DEFAULT_OLLAMA,
        "model": DEFAULT_MODEL,
        "num_ctx": DEFAULT_CTX,
        "embedding_model": EMBEDDING_MODEL,
        "n_results": 30,
        "top_k": 10,
        "do_rerank": True,
        "hybrid": True,
        "neo4j_uri": NEO4J_URI,
        "neo4j_user": NEO4J_USER,
        "neo4j_password": NEO4J_PASSWORD,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.dialog("Settings", width="small")
def _settings_dialog():
    """Modal dialog containing all tuneable parameters."""

    # ── LLM Settings ─────────────────────────────────────────────────
    st.markdown("#### :material/smart_toy: LLM Settings")

    ollama_host = st.text_input(
        "Ollama host",
        value=st.session_state["ollama_host"],
        key="dlg_ollama_host",
    )

    # Model picker
    try:
        _client = ollama.Client(host=ollama_host)
        _models = [m["model"] for m in _client.list().get("models", [])]
        _default_idx = (
            _models.index(st.session_state["model"])
            if st.session_state["model"] in _models
            else 0
        )
        model = st.selectbox("Model", _models, index=_default_idx, key="dlg_model")
    except Exception:
        model = st.text_input(
            "Model (manual)",
            value=st.session_state["model"],
            key="dlg_model",
        )

    num_ctx = st.select_slider(
        "Context window",
        options=[8192, 16384, 32768, 65536],
        value=st.session_state["num_ctx"],
        format_func=lambda x: f"{x // 1024}k",
        key="dlg_num_ctx",
    )

    st.divider()

    # ── Embedding Settings ────────────────────────────────────────────
    st.markdown("#### :material/model_training: Embedding Model")

    # Build options list: known models + current value (if custom)
    _emb_options = list(EMBEDDING_MODELS)
    _current_emb = st.session_state["embedding_model"]
    if _current_emb not in _emb_options:
        _emb_options.insert(0, _current_emb)
    _emb_idx = _emb_options.index(_current_emb)

    embedding_model = st.selectbox(
        "Embedding model",
        _emb_options,
        index=_emb_idx,
        key="dlg_embedding_model",
        help="SentenceTransformer model used for vector embeddings. "
             "⚠️ Changing this requires re-ingesting all ChromaDB collections.",
    )

    st.divider()

    # ── RAG Settings ─────────────────────────────────────────────────
    st.markdown("#### :material/manage_search: RAG Settings")

    n_results = st.slider(
        "Retrieve (vector search)", 5, 100,
        value=st.session_state["n_results"],
        key="dlg_n_results",
        help="How many candidates to fetch from ChromaDB before reranking",
    )

    top_k = st.slider(
        "Keep top-k (after reranking)", 1, 30,
        value=st.session_state["top_k"],
        key="dlg_top_k",
        help="Cross-encoder reranker keeps the best k documents",
    )

    do_rerank = st.toggle(
        "Enable reranking",
        value=st.session_state["do_rerank"],
        key="dlg_do_rerank",
    )

    hybrid = st.toggle(
        "Hybrid search (semantic + BM25)",
        value=st.session_state["hybrid"],
        key="dlg_hybrid",
        help="Combines vector search with BM25 keyword search via RRF",
    )

    st.divider()

    # ── Neo4j Settings ───────────────────────────────────────────────
    st.markdown("#### :material/hub: Neo4j Settings")

    neo4j_uri = st.text_input(
        "Neo4j URI",
        value=st.session_state["neo4j_uri"],
        key="dlg_neo4j_uri",
    )

    neo4j_user = st.text_input(
        "Neo4j User",
        value=st.session_state["neo4j_user"],
        key="dlg_neo4j_user",
    )

    neo4j_password = st.text_input(
        "Neo4j Password",
        value=st.session_state["neo4j_password"],
        type="password",
        key="dlg_neo4j_password",
    )

    # ── Save button ──────────────────────────────────────────────────
    st.markdown("")
    if st.button("Save settings", type="primary", width="stretch", icon=":material/save:"):
        # Detect embedding model change → clear cached resources
        _emb_changed = (embedding_model != st.session_state.get("embedding_model"))

        st.session_state["ollama_host"] = ollama_host
        st.session_state["model"] = model
        st.session_state["num_ctx"] = num_ctx
        st.session_state["embedding_model"] = embedding_model
        st.session_state["n_results"] = n_results
        st.session_state["top_k"] = top_k
        st.session_state["do_rerank"] = do_rerank
        st.session_state["hybrid"] = hybrid
        st.session_state["neo4j_uri"] = neo4j_uri
        st.session_state["neo4j_user"] = neo4j_user
        st.session_state["neo4j_password"] = neo4j_password

        if _emb_changed:
            # Invalidate cached embedding function & ChromaDB collections
            from rag.resources import load_embedding_func, load_chroma
            load_embedding_func.clear()
            load_chroma.clear()
            st.toast(
                f"Embedding model changed to **{embedding_model}**. "
                "Cached resources cleared — re-ingest collections to use the new model.",
                icon="⚠️",
            )

        st.rerun()


def render_settings():
    """
    Initialises settings defaults and returns the current settings tuple:
        (model, ollama_host, num_ctx, n_results, top_k, do_rerank, hybrid,
         neo4j_uri, neo4j_user, neo4j_password)

    The settings button itself lives in the sidebar (see ui/sidebar.py).
    """
    init_settings_defaults()

    return (
        st.session_state["model"],
        st.session_state["ollama_host"],
        st.session_state["num_ctx"],
        st.session_state["n_results"],
        st.session_state["top_k"],
        st.session_state["do_rerank"],
        st.session_state["hybrid"],
        st.session_state["neo4j_uri"],
        st.session_state["neo4j_user"],
        st.session_state["neo4j_password"],
    )
