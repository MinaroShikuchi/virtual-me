"""
virtual_me/state/chat_state.py — Chat page state.
"""
import reflex as rx
from pydantic import BaseModel
from virtual_me.state.app_state import AppState


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = ""  # "user" or "assistant"
    content: str = ""
    thinking: str = ""
    filter_info: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    intent: dict = {}
    facts: list[str] = []
    deliberations: list[dict] = []


class ChatState(AppState):
    """State for the Chat page."""

    messages: list[ChatMessage] = []
    current_input: str = ""
    is_loading: bool = False
    loading_status: str = ""
    error_message: str = ""

    @rx.event
    def set_current_input(self, value: str):
        """Set the current input value."""
        self.current_input = value

    def clear_history(self):
        """Clear all chat messages."""
        self.messages = []

    @rx.event(background=True)
    async def send_message(self):
        """Send a message and get a response."""
        async with self:
            if not self.current_input.strip():
                return

            user_msg = ChatMessage(role="user", content=self.current_input)
            self.messages.append(user_msg)
            question = self.current_input
            self.current_input = ""
            self.is_loading = True
            self.loading_status = "Analyzing intent and retrieving memories…"
            self.error_message = ""

        try:
            # Get all needed services and settings
            async with self:
                embedding_model = self.embedding_model
                intent_model = self.intent_model
                ollama_host = self.ollama_host
                model = self.model
                num_ctx = self.num_ctx
                system_prompt = self.system_prompt
                enable_thinking = self.enable_thinking
                n_results = self.n_results
                top_k = self.top_k
                do_rerank = self.do_rerank
                hybrid = self.hybrid
                active_personas = list(self.active_personas)
                deliberation_rounds = self.deliberation_rounds

            from services.embedding_service import get_embedding_func
            from services.chroma_service import get_collection, get_episodic
            from services.mapping_service import get_mappings
            from rag.retrieval import retrieve
            from rag.graph_retrieval import retrieve_facts

            ef = get_embedding_func(embedding_model)
            collection = get_collection(ef)
            episodic = get_episodic(ef)
            id_to_name, name_to_id = get_mappings()

            # Retrieve
            docs, episodes, intent = retrieve(
                question,
                n_results,
                collection,
                episodic,
                id_to_name,
                name_to_id,
                intent_model,
                ollama_host,
                metadata_filters=None,
                top_k=top_k,
                do_rerank=do_rerank,
                hybrid=hybrid,
            )

            # Graph facts
            facts = retrieve_facts(intent)

            # Context purification
            async with self:
                self.loading_status = "Purifying context…"

            from rag.llm import filter_irrelevant_context

            docs = filter_irrelevant_context(
                question, docs, intent_model, ollama_host
            )

            if not docs and not episodes and not facts:
                async with self:
                    self.messages.append(
                        ChatMessage(
                            role="assistant",
                            content="No relevant memories found.",
                            intent=intent,
                            facts=facts,
                        )
                    )
                    self.is_loading = False
                    self.loading_status = ""
                return

            # LLM call
            if active_personas and deliberation_rounds > 0:
                async with self:
                    self.loading_status = (
                        "Inner Deliberation Committee deliberating…"
                    )

                from rag.llm import deliberate_and_synthesize

                thinking, answer, p_tok, c_tok, deliberations = (
                    deliberate_and_synthesize(
                        question,
                        docs,
                        episodes,
                        facts,
                        model,
                        ollama_host,
                        num_ctx,
                        active_personas,
                        deliberation_rounds,
                        enable_thinking,
                    )
                )
            else:
                async with self:
                    self.loading_status = "Asking " + model + "…"

                from rag.llm import call_llm

                thinking, answer, p_tok, c_tok = call_llm(
                    question,
                    docs,
                    episodes,
                    facts,
                    model,
                    ollama_host,
                    num_ctx,
                    system_prompt,
                    enable_thinking,
                )
                deliberations = []

            async with self:
                self.messages.append(
                    ChatMessage(
                        role="assistant",
                        content=answer,
                        thinking=thinking or "",
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        intent=intent,
                        facts=facts,
                        deliberations=deliberations or [],
                    )
                )

        except Exception as e:
            async with self:
                self.error_message = f"Error: {e}"
                self.messages.append(
                    ChatMessage(
                        role="assistant",
                        content=f"Error: {e}",
                    )
                )
        finally:
            async with self:
                self.is_loading = False
                self.loading_status = ""
