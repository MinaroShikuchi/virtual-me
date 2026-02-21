"""
ui/chat.py — Chat tab: conversational interface with memory retrieval.
"""
import streamlit as st

from rag.retrieval import retrieve, detect_smart_filter
from rag.llm import call_llm


def _render_token_bar(prompt_tok: int, comp_tok: int, max_ctx: int):
    total = prompt_tok + comp_tok
    pct   = min(total / max_ctx, 1.0)
    st.progress(pct, text=f"{total:,} / {max_ctx:,} tokens  "
                          f"(prompt: {prompt_tok:,} | response: {comp_tok:,}) "
                          f"— {pct*100:.1f}%")


def render_chat_tab(collection, episodic, id_to_name, name_to_id,
                    model, ollama_host, num_ctx, n_results, top_k, do_rerank, hybrid):
    st.markdown("### :material/chat: Chat with your memories")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if msg.get("thinking"):
                    with st.expander("Thinking…", expanded=False, icon=":material/psychology_alt:"):
                        st.markdown(f'<div class="think-block">{msg["thinking"]}</div>',
                                    unsafe_allow_html=True)
                st.markdown(msg["content"])
                if msg.get("filter_info"):
                    st.caption(msg["filter_info"])
                if msg.get("prompt_tokens"):
                    _render_token_bar(msg["prompt_tokens"], msg["completion_tokens"], num_ctx)
            else:
                st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask anything about your memories…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            where_filter, friend_name, strategy = detect_smart_filter(prompt, name_to_id)

            # Filter info badge
            filter_info = ""
            if strategy == "Strict (Conversation)":
                filter_info = f"Loading full conversation with **{friend_name.title()}**"
                st.info(filter_info)
            elif strategy == "Global (Mention)":
                filter_info = f"Global search mentioning **{friend_name.title()}**"
                st.info(filter_info)

            with st.spinner("Retrieving memories…"):
                docs, episodes = retrieve(
                    prompt, n_results, where_filter, strategy,
                    collection, episodic, id_to_name,
                    top_k=top_k, do_rerank=do_rerank, hybrid=hybrid,
                )

            if not docs and not episodes:
                st.warning("No relevant memories found.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No relevant memories found.",
                    "filter_info": filter_info,
                })
                return

            # Show source count
            src_parts  = []
            if docs:     src_parts.append(f"{len(docs)} conversation chunk(s)")
            if episodes: src_parts.append(f"{len(episodes)} episode(s)")
            mode_parts = []
            if hybrid:    mode_parts.append("hybrid")
            if do_rerank: mode_parts.append(f"reranked → top-{top_k}")
            note = f" [{', '.join(mode_parts)} from {n_results}]" if mode_parts else ""
            st.caption(f"Context: {', '.join(src_parts)}{note}")

            with st.spinner(f"Asking {model}…"):
                try:
                    thinking, answer, p_tok, c_tok = call_llm(
                        prompt, docs, episodes, model, ollama_host, num_ctx
                    )
                except Exception as e:
                    st.error(f"Ollama error: {e}")
                    return

            if thinking:
                with st.expander("Thinking…", expanded=False, icon=":material/psychology_alt:"):
                    st.markdown(f'<div class="think-block">{thinking}</div>',
                                unsafe_allow_html=True)

            st.markdown(answer)
            _render_token_bar(p_tok, c_tok, num_ctx)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "thinking": thinking,
                "filter_info": filter_info,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
            })
