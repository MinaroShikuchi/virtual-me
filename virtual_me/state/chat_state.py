"""
virtual_me/state/chat_state.py — Chat page state.
"""
import logging

import reflex as rx
from pydantic import BaseModel
from virtual_me.state.app_state import AppState

log = logging.getLogger("rag.chat_state")


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
    _cancel_requested: bool = False

    @rx.event
    def set_current_input(self, value: str):
        """Set the current input value."""
        self.current_input = value

    def clear_history(self):
        """Clear all chat messages."""
        self.messages = []

    @rx.event
    def cancel_generation(self):
        """Request cancellation of the running background task."""
        self._cancel_requested = True
        self.loading_status = "Cancelling…"

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
            self._cancel_requested = False
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

            ef = get_embedding_func(embedding_model)
            collection = get_collection(ef)
            episodic = get_episodic(ef)
            id_to_name, name_to_id = get_mappings()

            # ── Committee mode: naive retrieval + agent loop ───────────
            if active_personas and deliberation_rounds > 0:
                from rag.retrieval import retrieve
                from rag.graph_retrieval import retrieve_facts

                # 1. Upfront naive retrieval — baseline context for all personas
                log.info("[1/5 INTENT] Analyzing intent…")
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
                facts_list = retrieve_facts(intent)

                log.info(
                    "[2/5 RETRIEVE] Baseline: %d docs, %d episodes, %d facts",
                    len(docs), len(episodes), len(facts_list),
                )

                async with self:
                    self.loading_status = (
                        "Inner Deliberation Committee deliberating…"
                    )

                # 2. Agent loop — personas get baseline + can call tools for more
                from rag.tools import ToolExecutor
                from rag.llm import deliberate_and_synthesize

                tool_executor = ToolExecutor(
                    collection=collection,
                    episodic=episodic,
                    id_to_name=id_to_name,
                    name_to_id=name_to_id,
                    intent_model=intent_model,
                    ollama_host=ollama_host,
                    n_results=n_results,
                    top_k=top_k,
                    do_rerank=do_rerank,
                    hybrid=hybrid,
                )

                # Build a cancel-check that reads the flag via self
                def _is_cancelled() -> bool:
                    return self._cancel_requested          # type: ignore[has-type]

                thinking, answer, p_tok, c_tok, deliberations = (
                    deliberate_and_synthesize(
                        question,
                        docs,
                        episodes,
                        facts_list,
                        model,
                        ollama_host,
                        num_ctx,
                        active_personas,
                        deliberation_rounds,
                        enable_thinking,
                        tool_executor=tool_executor,
                        cancel_check=_is_cancelled,
                    )
                )

            # ── Solo mode: upfront retrieval + single LLM call ─────────
            else:
                from rag.retrieval import retrieve
                from rag.graph_retrieval import retrieve_facts

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

                facts_list = retrieve_facts(intent)

                # Context purification
                async with self:
                    self.loading_status = "Purifying context…"

                from rag.llm import filter_irrelevant_context

                docs = filter_irrelevant_context(
                    question, docs, intent_model, ollama_host
                )

                if not docs and not episodes and not facts_list:
                    async with self:
                        self.messages.append(
                            ChatMessage(
                                role="assistant",
                                content="No relevant memories found.",
                                intent=intent,
                                facts=facts_list,
                            )
                        )
                        self.is_loading = False
                        self.loading_status = ""
                    return

                async with self:
                    self.loading_status = "Asking " + model + "…"

                from rag.llm import call_llm

                thinking, answer, p_tok, c_tok = call_llm(
                    question,
                    docs,
                    episodes,
                    facts_list,
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
                        facts=facts_list,
                        deliberations=deliberations or [],
                    )
                )

        except Exception as e:
            from rag.agent import CancelledError

            async with self:
                if isinstance(e, CancelledError):
                    self.messages.append(
                        ChatMessage(
                            role="assistant",
                            content="⏹ Generation cancelled.",
                        )
                    )
                else:
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
