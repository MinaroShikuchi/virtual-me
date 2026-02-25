"""
ui/chat.py — Chat tab: conversational interface with memory retrieval.
"""
import streamlit as st

from rag.retrieval import retrieve
from rag.llm import call_llm
from rag.graph_retrieval import retrieve_facts


def _render_token_bar(prompt_tok: int, comp_tok: int, max_ctx: int):
    total = prompt_tok + comp_tok
    pct   = min(total / max_ctx, 1.0)
    st.progress(pct, text=f"{total:,} / {max_ctx:,} tokens  "
                          f"(prompt: {prompt_tok:,} | response: {comp_tok:,}) "
                          f"— {pct*100:.1f}%")


def render_chat_tab(collection, episodic, id_to_name, name_to_id,
                    model, intent_model, ollama_host, num_ctx, deliberation_rounds, active_personas, enable_thinking, system_prompt, n_results, top_k, do_rerank, hybrid):
    
    ctx_kb = f"{num_ctx/1024:.0f}k" if num_ctx >= 1024 else f"{num_ctx}"
    st.markdown(f"### :material/chat: Chat with your memories")
    
    if active_personas and deliberation_rounds > 0:
        st.caption(f"**Intent Model:** {intent_model} · **Model:** {model} ({ctx_kb}) · **Number of Results:** {n_results} · **Top K:** {top_k} · **Do Rerank:** {do_rerank} · **Hybrid:** {hybrid}")
        st.caption(f"**Committee Active:** {', '.join(active_personas)} ({deliberation_rounds} rounds)")
        # st.caption(f"**System Prompt:** {system_prompt}")
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
                if msg.get("intent"):
                    with st.expander(f"Intent & Facts ({len(msg.get('facts', []))} found)", expanded=False, icon=":material/hub:"):
                        st.json(msg["intent"])
                        if msg.get("facts"):
                            st.markdown("**Semantic Facts:**")
                            for f in msg["facts"]:
                                st.markdown(f"- {f}")
                if msg.get("deliberations"):
                    with st.expander("Inner Deliberation Committee", expanded=False, icon=":material/groups:"):
                        for d in msg["deliberations"]:
                            st.markdown(f"**{d['persona']} (Round {d['round']})**")
                            st.info(d['response'])
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
            filter_info = ""

            with st.spinner("Translating query to introspective thought..."):
                from rag.llm import translate_to_introspective
                prompt = translate_to_introspective(prompt, intent_model, ollama_host)
                st.caption(f"*Introspective Query: {prompt}*")

            with st.spinner("Analyzing intent and retrieving memories…"):
                docs, episodes, intent = retrieve(
                    prompt, n_results,
                    collection, episodic, id_to_name, name_to_id,
                    intent_model, ollama_host,
                    metadata_filters=None,
                    relevance_threshold=-2.0,
                    top_k=top_k, do_rerank=do_rerank, hybrid=hybrid,
                )
                
                # Fetch semantic facts from Neo4j based on intent
                facts = retrieve_facts(intent)
                
            # with st.spinner("Purifying context..."):
            #     from rag.llm import filter_irrelevant_context
            #     docs = filter_irrelevant_context(prompt, docs, intent_model, ollama_host)

            if not docs and not episodes and not facts:
                st.warning("No relevant memories found.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No relevant memories found.",
                    "filter_info": filter_info,
                    "intent": intent,
                    "facts": facts
                })
                return

            # Show source count
            src_parts  = []
            if docs:     src_parts.append(f"{len(docs)} conversation chunk(s)")
            if episodes: src_parts.append(f"{len(episodes)} episode(s)")
            if facts:    src_parts.append(f"{len(facts)} semantic fact(s)")
            
            # Show the intent and facts in the UI before generation
            with st.expander(f"Intent & Facts ({len(facts)} found)", expanded=False, icon=":material/hub:"):
                st.json(intent)
                if facts:
                    st.markdown("**Semantic Facts:**")
                    for f in facts:
                        st.markdown(f"- {f}")
            mode_parts = []
            if hybrid:    mode_parts.append("hybrid")
            if do_rerank: mode_parts.append(f"reranked → top-{top_k}")
            note = f" [{', '.join(mode_parts)} from {n_results}]" if mode_parts else ""
            st.caption(f"Context: {', '.join(src_parts)}{note}")

            if active_personas and deliberation_rounds > 0:
                with st.status("Inner Deliberation Committee", expanded=True) as status:
                    from rag.llm import deliberate_and_synthesize
                    
                    def ui_callback(persona, round_num, state, response_text):
                        if state == "working":
                            if persona == "The Self":
                                status.update(label="The Self is synthesizing...", state="running")
                            else:
                                status.update(label=f"Committee deliberating: {persona} is speaking (Round {round_num})...", state="running")
                        elif state == "done":
                            if persona != "The Self":
                                st.markdown(f"**{persona} (Round {round_num})**")
                                st.info(response_text)

                    try:
                        thinking, answer, p_tok, c_tok, deliberations = deliberate_and_synthesize(
                            prompt, docs, episodes, facts, model, ollama_host, num_ctx,
                            active_personas, deliberation_rounds, enable_thinking,
                            update_callback=ui_callback
                        )
                        status.update(label="Inner Deliberation Committee", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label=f"Ollama error: {e}", state="error")
                        return
            else:
                with st.spinner(f"Asking {model}…"):
                    try:
                        thinking, answer, p_tok, c_tok = call_llm(
                            prompt, docs, episodes, facts, model, ollama_host, num_ctx, system_prompt, enable_thinking
                        )
                        deliberations = None
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
                "deliberations": deliberations,
                "filter_info": filter_info,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "intent": intent,
                "facts": facts,
            })
