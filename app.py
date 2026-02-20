"""
Virtual-Me — Streamlit entry point.

Module layout:
  config.py              — constants, SOURCES registry, DATA_DIR
  rag/
    resources.py         — @st.cache_resource loaders (embeddings, reranker, ChromaDB, BM25, mappings)
    retrieval.py         — hybrid search pipeline (semantic + BM25 + RRF + rerank)
    llm.py               — Ollama LLM call with thinking-token support
  ui/
    sidebar.py           — sidebar: connections, LLM settings, RAG settings
    dashboard.py         — 📊 Dashboard tab
    chat.py              — 💬 Chat tab
    rag_explorer.py      — 🔍 RAG Explorer tab
    ingest.py            — ⚙️ Ingest tab
"""
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Virtual Me",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #12141f 100%);
        border-right: 1px solid #2d2f45;
    }

    /* Status badges */
    .status-ok   { color: #4ade80; font-weight: 600; }
    .status-warn { color: #facc15; font-weight: 600; }
    .status-err  { color: #f87171; font-weight: 600; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* RAG card */
    .rag-card {
        background: #1e2030;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .rag-card-header {
        font-size: 0.75rem;
        color: #8b9cb6;
        margin-bottom: 6px;
        display: flex;
        gap: 12px;
    }
    .rag-card-header span { font-weight: 600; color: #a5b4fc; }

    /* Thinking block */
    .think-block {
        background: #181b2e;
        border-left: 3px solid #6366f1;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #94a3b8;
        white-space: pre-wrap;
    }

    /* token bar label */
    .token-label { font-size: 0.78rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ── Imports (after page config) ───────────────────────────────────────────────
from rag.resources   import load_chroma, load_mappings
from ui.sidebar      import render_sidebar
from ui.dashboard    import render_dashboard_tab
from ui.chat         import render_chat_tab
from ui.rag_explorer import render_rag_tab
from ui.ingest       import render_ingest_tab
from ui.graph        import render_graph_tab


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    collection, episodic   = load_chroma()
    id_to_name, name_to_id = load_mappings()

    model, ollama_host, num_ctx, n_results, top_k, do_rerank, hybrid, \
    neo4j_uri, neo4j_user, neo4j_password = render_sidebar(collection, episodic)

    tab_dash, tab_chat, tab_rag, tab_ingest, tab_graph = st.tabs([
        "📊 Dashboard", "💬 Chat", "🔍 RAG Explorer", "⚙️ Ingest", "🕸️ Graph"
    ])

    with tab_dash:
        render_dashboard_tab(collection)

    with tab_chat:
        render_chat_tab(
            collection, episodic, id_to_name, name_to_id,
            model, ollama_host, num_ctx, n_results, top_k, do_rerank, hybrid,
        )

    with tab_rag:
        render_rag_tab(
            collection, episodic, id_to_name, name_to_id,
            n_results, top_k, do_rerank, hybrid,
        )

    with tab_ingest:
        render_ingest_tab(collection)

    with tab_graph:
        render_graph_tab(neo4j_uri, neo4j_user, neo4j_password)


if __name__ == "__main__":
    main()
