"""
virtual_me/components/settings_dialog.py — Settings dialog (centered modal).
"""
import reflex as rx
from virtual_me.state.app_state import AppState
from config import EMBEDDING_MODELS


class SettingsState(rx.State):
    """Local state for the settings form (draft values before save)."""

    # Tab selection
    active_tab: str = "llm"

    # Draft LLM
    draft_ollama_host: str = ""
    draft_model: str = ""
    draft_intent_model: str = ""
    draft_num_ctx: int = 32768
    draft_system_prompt: str = ""
    draft_enable_thinking: bool = True
    draft_deliberation_rounds: int = 2
    draft_active_personas: list[str] = []

    # Draft Embedding
    draft_embedding_model: str = ""

    # Draft RAG
    draft_n_results: int = 30
    draft_top_k: int = 10
    draft_do_rerank: bool = True
    draft_hybrid: bool = True

    # Draft Neo4j
    draft_neo4j_uri: str = ""
    draft_neo4j_user: str = ""
    draft_neo4j_password: str = ""

    # Available Ollama models (fetched on open)
    available_models: list[str] = []

    async def load_drafts(self):
        """Copy current AppState values into draft fields."""
        app = await self.get_state(AppState)
        self.draft_ollama_host = app.ollama_host
        self.draft_model = app.model
        self.draft_intent_model = app.intent_model
        self.draft_num_ctx = app.num_ctx
        self.draft_system_prompt = app.system_prompt
        self.draft_enable_thinking = app.enable_thinking
        self.draft_deliberation_rounds = app.deliberation_rounds
        self.draft_active_personas = list(app.active_personas)
        self.draft_embedding_model = app.embedding_model
        self.draft_n_results = app.n_results
        self.draft_top_k = app.top_k
        self.draft_do_rerank = app.do_rerank
        self.draft_hybrid = app.hybrid
        self.draft_neo4j_uri = app.neo4j_uri
        self.draft_neo4j_user = app.neo4j_user
        self.draft_neo4j_password = app.neo4j_password

        # Try to fetch available Ollama models
        try:
            import ollama as _ollama
            client = _ollama.Client(host=self.draft_ollama_host)
            self.available_models = [
                m["model"] for m in client.list().get("models", [])
            ]
        except Exception:
            self.available_models = []

    async def save_and_close(self):
        """Commit draft values to AppState and close dialog."""
        app = await self.get_state(AppState)
        app.save_settings(
            ollama_host=self.draft_ollama_host,
            model=self.draft_model,
            intent_model=self.draft_intent_model,
            num_ctx=self.draft_num_ctx,
            system_prompt=self.draft_system_prompt,
            enable_thinking=self.draft_enable_thinking,
            deliberation_rounds=self.draft_deliberation_rounds,
            active_personas=self.draft_active_personas,
            embedding_model=self.draft_embedding_model,
            n_results=self.draft_n_results,
            top_k=self.draft_top_k,
            do_rerank=self.draft_do_rerank,
            hybrid=self.draft_hybrid,
            neo4j_uri=self.draft_neo4j_uri,
            neo4j_user=self.draft_neo4j_user,
            neo4j_password=self.draft_neo4j_password,
        )

    def set_tab(self, tab: str):
        """Switch the active settings tab."""
        self.active_tab = tab

    def set_num_ctx_from_str(self, val: str):
        """Set num_ctx from a string select value."""
        try:
            self.draft_num_ctx = int(val)
        except (ValueError, TypeError):
            pass

    def set_deliberation_from_str(self, val: str):
        """Set deliberation_rounds from a string input value."""
        try:
            self.draft_deliberation_rounds = int(val)
        except (ValueError, TypeError):
            pass

    def set_n_results_from_slider(self, val: list[float]):
        """Set n_results from slider value (list[float])."""
        if val:
            self.draft_n_results = int(val[0])

    def set_top_k_from_slider(self, val: list[float]):
        """Set top_k from slider value (list[float])."""
        if val:
            self.draft_top_k = int(val[0])

    @rx.event
    def set_draft_ollama_host(self, value: str):
        """Set draft Ollama host."""
        self.draft_ollama_host = value

    @rx.event
    def set_draft_model(self, value: str):
        """Set draft model."""
        self.draft_model = value

    @rx.event
    def set_draft_intent_model(self, value: str):
        """Set draft intent model."""
        self.draft_intent_model = value

    @rx.event
    def set_draft_enable_thinking(self, value: bool):
        """Set draft enable thinking."""
        self.draft_enable_thinking = value

    @rx.event
    def set_draft_system_prompt(self, value: str):
        """Set draft system prompt."""
        self.draft_system_prompt = value

    @rx.event
    def set_draft_embedding_model(self, value: str):
        """Set draft embedding model."""
        self.draft_embedding_model = value

    @rx.event
    def set_draft_do_rerank(self, value: bool):
        """Set draft do rerank."""
        self.draft_do_rerank = value

    @rx.event
    def set_draft_hybrid(self, value: bool):
        """Set draft hybrid search."""
        self.draft_hybrid = value

    @rx.event
    def set_draft_neo4j_uri(self, value: str):
        """Set draft Neo4j URI."""
        self.draft_neo4j_uri = value

    @rx.event
    def set_draft_neo4j_user(self, value: str):
        """Set draft Neo4j user."""
        self.draft_neo4j_user = value

    @rx.event
    def set_draft_neo4j_password(self, value: str):
        """Set draft Neo4j password."""
        self.draft_neo4j_password = value

    @rx.event
    async def handle_dialog_open_change(self, is_open: bool):
        """Handle dialog open/close state changes."""
        app = await self.get_state(AppState)
        if is_open:
            app.open_settings()
            await self.load_drafts()
        else:
            app.close_settings()


def _llm_tab() -> rx.Component:
    """LLM settings tab content."""
    return rx.vstack(
        rx.heading("LLM Settings", size="4"),
        rx.text("Ollama Host", size="2", weight="medium"),
        rx.input(
            value=SettingsState.draft_ollama_host,
            on_change=SettingsState.set_draft_ollama_host,
            width="100%",
        ),
        rx.text("Model", size="2", weight="medium"),
        rx.input(
            value=SettingsState.draft_model,
            on_change=SettingsState.set_draft_model,
            width="100%",
        ),
        rx.text("Intent Model", size="2", weight="medium"),
        rx.input(
            value=SettingsState.draft_intent_model,
            on_change=SettingsState.set_draft_intent_model,
            width="100%",
        ),
        rx.text("Context Window", size="2", weight="medium"),
        rx.select(
            ["8192", "16384", "32768", "65536"],
            value=SettingsState.draft_num_ctx.to(str),
            on_change=SettingsState.set_num_ctx_from_str,
            width="100%",
        ),
        rx.hstack(
            rx.switch(
                checked=SettingsState.draft_enable_thinking,
                on_change=SettingsState.set_draft_enable_thinking,
            ),
            rx.text("Enable Thinking Tokens", size="2"),
            spacing="2",
            align="center",
        ),
        rx.text("System Prompt", size="2", weight="medium"),
        rx.text_area(
            value=SettingsState.draft_system_prompt,
            on_change=SettingsState.set_draft_system_prompt,
            width="100%",
            height="150px",
        ),
        rx.text("Deliberation Rounds", size="2", weight="medium"),
        rx.input(
            value=SettingsState.draft_deliberation_rounds.to(str),
            on_change=SettingsState.set_deliberation_from_str,
            type="number",
            width="100%",
        ),
        spacing="3",
        width="100%",
    )


def _embedding_tab() -> rx.Component:
    """Embedding model settings tab."""
    return rx.vstack(
        rx.heading("Embedding Model", size="4"),
        rx.select(
            EMBEDDING_MODELS,
            value=SettingsState.draft_embedding_model,
            on_change=SettingsState.set_draft_embedding_model,
            width="100%",
        ),
        rx.callout(
            "Changing the embedding model requires re-ingesting all ChromaDB collections.",
            icon="triangle-alert",
            color_scheme="orange",
        ),
        spacing="3",
        width="100%",
    )


def _rag_tab() -> rx.Component:
    """RAG settings tab."""
    return rx.vstack(
        rx.heading("RAG Settings", size="4"),
        rx.text("Retrieve (vector search)", size="2", weight="medium"),
        rx.text(
            SettingsState.draft_n_results.to(str) + " results",
            size="1",
            color="#94a3b8",
        ),
        rx.slider(
            min=5,
            max=100,
            step=5,
            default_value=[30],
            value=[SettingsState.draft_n_results],
            on_value_commit=SettingsState.set_n_results_from_slider,
            width="100%",
        ),
        rx.text("Keep top-k (after reranking)", size="2", weight="medium"),
        rx.text(
            SettingsState.draft_top_k.to(str) + " results",
            size="1",
            color="#94a3b8",
        ),
        rx.slider(
            min=1,
            max=30,
            step=1,
            default_value=[10],
            value=[SettingsState.draft_top_k],
            on_value_commit=SettingsState.set_top_k_from_slider,
            width="100%",
        ),
        rx.hstack(
            rx.switch(
                checked=SettingsState.draft_do_rerank,
                on_change=SettingsState.set_draft_do_rerank,
            ),
            rx.text("Enable reranking", size="2"),
            spacing="2",
            align="center",
        ),
        rx.hstack(
            rx.switch(
                checked=SettingsState.draft_hybrid,
                on_change=SettingsState.set_draft_hybrid,
            ),
            rx.text("Hybrid search (BM25 + semantic)", size="2"),
            spacing="2",
            align="center",
        ),
        spacing="3",
        width="100%",
    )


def _neo4j_tab() -> rx.Component:
    """Neo4j settings tab."""
    return rx.vstack(
        rx.heading("Neo4j Settings", size="4"),
        rx.text("URI", size="2", weight="medium"),
        rx.input(
            value=SettingsState.draft_neo4j_uri,
            on_change=SettingsState.set_draft_neo4j_uri,
            width="100%",
        ),
        rx.text("User", size="2", weight="medium"),
        rx.input(
            value=SettingsState.draft_neo4j_user,
            on_change=SettingsState.set_draft_neo4j_user,
            width="100%",
        ),
        rx.text("Password", size="2", weight="medium"),
        rx.input(
            value=SettingsState.draft_neo4j_password,
            on_change=SettingsState.set_draft_neo4j_password,
            type="password",
            width="100%",
        ),
        spacing="3",
        width="100%",
    )


def _tab_button(label: str, tab_key: str, icon_name: str) -> rx.Component:
    """A sidebar tab button that highlights when active."""
    return rx.button(
        rx.icon(icon_name, size=16),
        label,
        variant=rx.cond(
            SettingsState.active_tab == tab_key, "solid", "ghost"
        ),
        on_click=SettingsState.set_tab(tab_key),
        size="2",
        width="100%",
    )


def settings_dialog() -> rx.Component:
    """The settings centered modal dialog."""
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Settings"),
            rx.dialog.description(
                "Configure LLM, embedding, RAG, and Neo4j settings.",
                size="2",
                color="#94a3b8",
            ),
            rx.divider(),

            # Main body: left sidebar + right content
            rx.hstack(
                # Left: tab navigation sidebar
                rx.vstack(
                    _tab_button("LLM", "llm", "bot"),
                    _tab_button("Embedding", "embedding", "brain"),
                    _tab_button("RAG", "rag", "search"),
                    _tab_button("Neo4j", "neo4j", "network"),
                    width="180px",
                    spacing="2",
                    padding_top="4px",
                    align_items="stretch",
                    flex_shrink="0",
                ),
                # Right: content for selected tab
                rx.box(
                    rx.cond(SettingsState.active_tab == "llm", _llm_tab()),
                    rx.cond(SettingsState.active_tab == "embedding", _embedding_tab()),
                    rx.cond(SettingsState.active_tab == "rag", _rag_tab()),
                    rx.cond(SettingsState.active_tab == "neo4j", _neo4j_tab()),
                    flex="1",
                    overflow_y="auto",
                    padding_left="16px",
                    max_height="60vh",
                ),
                width="100%",
                spacing="4",
                flex="1",
                overflow="hidden",
            ),

            rx.divider(),

            # Bottom: Save / Cancel buttons
            rx.hstack(
                rx.dialog.close(
                    rx.button(
                        "Cancel",
                        variant="outline",
                        color_scheme="gray",
                    ),
                ),
                rx.button(
                    "Save Settings",
                    color_scheme="iris",
                    on_click=SettingsState.save_and_close,
                ),
                spacing="3",
                justify="end",
                width="100%",
            ),

            max_width="700px",
            bg="#1a1d2e",
        ),
        open=AppState.settings_open,
        on_open_change=SettingsState.handle_dialog_open_change,
    )
