"""
ui/sidebar.py — Streamlit sidebar: connection status + settings button.

Navigation is handled by st.navigation() in app.py, which renders
page links in the sidebar automatically. This module adds connection
status indicators and a settings button below the navigation.
"""
import streamlit as st
import ollama

from config import DEFAULT_OLLAMA, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from ui.settings import _settings_dialog, init_settings_defaults


def render_sidebar(collection, episodic):
    """
    Renders connection status indicators and a settings button in the sidebar.
    Called after st.navigation() so the nav links appear first.
    """
    init_settings_defaults()

    with st.sidebar:
        st.markdown("**Connections**")

        # ── ChromaDB status ───────────────────────────────────────────────
        try:
            count = collection.count()
            st.markdown(f'<span class="status-ok">● ChromaDB</span> — {count:,} docs',
                        unsafe_allow_html=True)
        except Exception:
            st.markdown('<span class="status-err">○ ChromaDB</span> — collection stale (re-ingest or restart app)',
                        unsafe_allow_html=True)

        # ── Episodic Memory status ────────────────────────────────────────
        if episodic:
            try:
                ec = episodic.count()
                st.markdown(f'<span class="status-ok">● Episodic Memory</span> — {ec:,} episodes',
                            unsafe_allow_html=True)
            except Exception:
                st.markdown('<span class="status-warn">○ Episodic Memory</span> — stale',
                            unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">○ Episodic Memory</span> — not found',
                        unsafe_allow_html=True)

        # ── Ollama status ─────────────────────────────────────────────────
        ollama_host = st.session_state.get("ollama_host", DEFAULT_OLLAMA)
        try:
            _client = ollama.Client(host=ollama_host)
            _models = [m["model"] for m in _client.list().get("models", [])]
            st.markdown(
                f'<span class="status-ok">● Ollama</span> — reachable '
                f'({len(_models)} model{"s" if len(_models) != 1 else ""})',
                unsafe_allow_html=True,
            )
        except Exception as _e:
            st.markdown('<span class="status-err">○ Ollama</span> — unreachable',
                        unsafe_allow_html=True)

        # ── Neo4j status ──────────────────────────────────────────────────
        neo4j_uri = st.session_state.get("neo4j_uri", NEO4J_URI)
        try:
            from neo4j import GraphDatabase
            _driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(
                    st.session_state.get("neo4j_user", NEO4J_USER),
                    st.session_state.get("neo4j_password", NEO4J_PASSWORD),
                ),
            )
            _driver.verify_connectivity()
            _driver.close()
            st.markdown('<span class="status-ok">● Neo4j</span> — connected',
                        unsafe_allow_html=True)
        except Exception:
            st.markdown('<span class="status-warn">○ Neo4j</span> — not connected',
                        unsafe_allow_html=True)

        # ── Settings button ───────────────────────────────────────────────
        st.divider()
        if st.button("Settings", width="stretch", icon=":material/settings:"):
            _settings_dialog()
