"""
Virtual-Me — Streamlit entry point.

Module layout:
  config.py              — constants, SOURCES registry, DATA_DIR
  rag/
    resources.py         — @st.cache_resource loaders (embeddings, reranker, ChromaDB, BM25, mappings)
    retrieval.py         — hybrid search pipeline (semantic + BM25 + RRF + rerank)
    llm.py               — Ollama LLM call with thinking-token support
  ui/
    sidebar.py           — sidebar: connection status
    settings.py          — settings modal (LLM, RAG, Neo4j)
    dashboard.py         — Dashboard page
    chat.py              — Chat page
    rag_explorer.py      — RAG Explorer page
    ingest.py            — Vector Store page (episodic memory)
    graph.py             — Graph page
  pages/
    dashboard.py         — page wrapper for Dashboard
    chat.py              — page wrapper for Chat
    rag_explorer.py      — page wrapper for RAG Explorer
    ingest.py            — page wrapper for Vector Store
    graph.py             — page wrapper for Graph
"""
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Virtual Me",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
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
from rag.resources import load_chroma
from ui.sidebar    import render_sidebar

from pages.dashboard    import page as dashboard_page
from pages.chat         import page as chat_page
from pages.rag_explorer import page as rag_explorer_page
from pages.ingest       import page as ingest_page
from pages.graph        import page as graph_page
from pages.entity_browser import page as entity_browser_page


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Sidebar: connection status
    collection, episodic = load_chroma()
    render_sidebar(collection, episodic)

    # Navigation (renders in sidebar automatically)
    pg = st.navigation(
        [
            st.Page(dashboard_page,    title="Dashboard",    url_path="dashboard",    default=True, icon=":material/dashboard:"),
            st.Page(chat_page,         title="Chat",         url_path="chat",         icon=":material/chat:"),
            st.Page(ingest_page,       title="Vector",       url_path="vector",       icon=":material/database:"),
            st.Page(graph_page,        title="Platform Extract", url_path="graph",        icon=":material/manufacturing:"),
            st.Page(rag_explorer_page, title="RAG Explorer", url_path="rag",          icon=":material/search:"),
            st.Page(entity_browser_page, title="Graph Explorer", url_path="browser",  icon=":material/travel_explore:"),
        ]
    )

    # Run the selected page
    pg.run()


if __name__ == "__main__":
    main()
