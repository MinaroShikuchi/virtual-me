"""
virtual_me/state/app_state.py — Root application state.

All settings that were in st.session_state now live here.
Substates (ChatState, VectorState, etc.) inherit from this.
"""
import reflex as rx

from config import (
    DEFAULT_MODEL, DEFAULT_INTENT_MODEL, DEFAULT_OLLAMA, DEFAULT_CTX,
    DEFAULT_SYSTEM_PROMPT, DEFAULT_DELIBERATION_ROUNDS,
    DEFAULT_ACTIVE_PERSONAS, DEFAULT_ENABLE_THINKING,
    EMBEDDING_MODEL,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)


class AppState(rx.State):
    """Global application state — settings, connection status."""

    # ── LLM settings ──
    ollama_host: str = DEFAULT_OLLAMA
    model: str = DEFAULT_MODEL
    intent_model: str = DEFAULT_INTENT_MODEL
    num_ctx: int = DEFAULT_CTX
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    deliberation_rounds: int = DEFAULT_DELIBERATION_ROUNDS
    active_personas: list[str] = DEFAULT_ACTIVE_PERSONAS

    # ── Embedding ──
    embedding_model: str = EMBEDDING_MODEL

    # ── RAG settings ──
    n_results: int = 30
    top_k: int = 10
    do_rerank: bool = True
    hybrid: bool = True

    # ── Neo4j settings ──
    neo4j_uri: str = NEO4J_URI
    neo4j_user: str = NEO4J_USER
    neo4j_password: str = NEO4J_PASSWORD

    # ── Connection status (computed on load) ──
    chroma_doc_count: int = 0
    episodic_count: int = 0
    ollama_connected: bool = False
    ollama_model_count: int = 0
    neo4j_connected: bool = False

    # ── Settings dialog ──
    settings_open: bool = False

    def toggle_settings(self):
        """Open/close the settings dialog."""
        self.settings_open = not self.settings_open

    def open_settings(self):
        self.settings_open = True

    def close_settings(self):
        self.settings_open = False

    def check_connections(self):
        """Check all backend connections and update status flags."""
        # ChromaDB
        try:
            from services.chroma_service import get_collection
            from services.embedding_service import get_embedding_func
            ef = get_embedding_func(self.embedding_model)
            col = get_collection(ef)
            self.chroma_doc_count = col.count()
        except Exception:
            self.chroma_doc_count = 0

        # Episodic
        try:
            from services.chroma_service import get_episodic
            from services.embedding_service import get_embedding_func
            ef = get_embedding_func(self.embedding_model)
            ep = get_episodic(ef)
            self.episodic_count = ep.count() if ep else 0
        except Exception:
            self.episodic_count = 0

        # Ollama
        try:
            import ollama as _ollama
            client = _ollama.Client(host=self.ollama_host)
            resp = client.list()
            # ollama >= 0.4 returns ListResponse with .models attribute
            models_list = getattr(resp, "models", None)
            if models_list is None and isinstance(resp, dict):
                models_list = resp.get("models", [])
            self.ollama_connected = True
            self.ollama_model_count = len(models_list or [])
        except Exception:
            self.ollama_connected = False
            self.ollama_model_count = 0

        # Neo4j
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            driver.verify_connectivity()
            driver.close()
            self.neo4j_connected = True
        except Exception:
            self.neo4j_connected = False

    def save_settings(
        self,
        ollama_host: str,
        model: str,
        intent_model: str,
        num_ctx: int,
        system_prompt: str,
        enable_thinking: bool,
        deliberation_rounds: int,
        active_personas: list[str],
        embedding_model: str,
        n_results: int,
        top_k: int,
        do_rerank: bool,
        hybrid: bool,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
        """Commit settings and invalidate caches if needed."""
        emb_changed = embedding_model != self.embedding_model

        self.ollama_host = ollama_host
        self.model = model
        self.intent_model = intent_model
        self.num_ctx = num_ctx
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.deliberation_rounds = deliberation_rounds
        self.active_personas = active_personas
        self.embedding_model = embedding_model
        self.n_results = n_results
        self.top_k = top_k
        self.do_rerank = do_rerank
        self.hybrid = hybrid
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        if emb_changed:
            from services.embedding_service import invalidate as inv_emb
            from services.chroma_service import invalidate as inv_chroma
            from services.bm25_service import invalidate as inv_bm25
            inv_emb()
            inv_chroma()
            inv_bm25()

        self.settings_open = False
        self.check_connections()
