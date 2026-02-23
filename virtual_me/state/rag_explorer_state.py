"""
virtual_me/state/rag_explorer_state.py — RAG Explorer state.
"""
import reflex as rx
from virtual_me.state.app_state import AppState


class RAGExplorerState(AppState):
    """State for the RAG Explorer page."""

    query: str = ""
    selected_friend: str = "All conversations"
    friend_options: list[str] = ["All conversations"]

    # Metadata filters
    source_filter: str = "Any source"
    min_messages: int = 1

    # Results
    results: list[dict] = []
    episodes: list[dict] = []
    intent: dict = {}
    result_count: int = 0
    episode_count: int = 0
    strategy: str = "Semantic"
    matched_friend: str = ""
    active_filter_count: int = 0
    mode_label: str = ""
    where_clause: dict = {}

    is_searching: bool = False
    error_message: str = ""

    def load_friends(self):
        """Load friend names for the filter dropdown."""
        try:
            from services.mapping_service import get_mappings
            id_to_name, _ = get_mappings()
            names = sorted(
                {v for v in id_to_name.values() if v},
                key=lambda x: x.lower(),
            )
            self.friend_options = ["All conversations"] + names
        except Exception:
            self.friend_options = ["All conversations"]

    def set_min_messages_from_str(self, value: str):
        """Set min_messages from a string input value."""
        try:
            self.min_messages = int(value) if value.isdigit() else 1
        except (ValueError, TypeError):
            self.min_messages = 1

    def search(self):
        """Run the RAG retrieval pipeline."""
        if not self.query:
            self.error_message = "Enter a query to search."
            return

        self.is_searching = True
        self.error_message = ""

        try:
            from services.embedding_service import get_embedding_func
            from services.chroma_service import get_collection, get_episodic
            from services.mapping_service import get_mappings
            from rag.retrieval import retrieve, build_where, analyze_intent

            ef = get_embedding_func(self.embedding_model)
            collection = get_collection(ef)
            episodic = get_episodic(ef)
            id_to_name, name_to_id = get_mappings()

            # Build metadata filters
            metadata_filters: dict = {}
            if self.source_filter != "Any source":
                metadata_filters["source"] = self.source_filter
            if self.min_messages > 1:
                metadata_filters["min_messages"] = self.min_messages

            # Run retrieval
            docs, eps, intent = retrieve(
                self.query,
                self.n_results,
                collection,
                episodic,
                id_to_name,
                name_to_id,
                self.intent_model,
                self.ollama_host,
                metadata_filters=metadata_filters,
                top_k=self.top_k,
                do_rerank=self.do_rerank,
                hybrid=self.hybrid,
            )

            self.results = docs
            self.episodes = eps
            self.intent = intent
            self.result_count = len(docs)
            self.episode_count = len(eps)

            # Determine strategy
            if self.selected_friend != "All conversations":
                self.strategy = "Strict (Conversation)"
                self.matched_friend = self.selected_friend
            elif intent.get("people"):
                self.strategy = "Strict (Conversation)"
                self.matched_friend = (
                    intent["people"][0] if intent["people"] else ""
                )
            else:
                self.strategy = "Semantic"
                self.matched_friend = ""

            self.active_filter_count = len(metadata_filters)
            mode_parts = []
            if self.hybrid:
                mode_parts.append("Hybrid")
            else:
                mode_parts.append("Semantic")
            if self.do_rerank:
                mode_parts.append(f"Rerank→top-{self.top_k}")
            self.mode_label = " + ".join(mode_parts)

            # Build where clause for debug display
            base_filter = None
            if self.selected_friend != "All conversations":
                matched_id = name_to_id.get(self.selected_friend.lower())
                if matched_id:
                    base_filter = {"conversation": matched_id}
            self.where_clause = build_where(base_filter, metadata_filters) or {}

        except Exception as e:
            self.error_message = f"Search error: {e}"
            self.results = []
            self.episodes = []
        finally:
            self.is_searching = False
