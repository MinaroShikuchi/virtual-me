"""
ui/chat.py — Chat tab: conversational interface with memory retrieval.
"""
import streamlit as st
import json
import os
from pathlib import Path

from config import CHAT_HISTORY_FILE
from rag.rag_retrieval import rag_retrieval
from rag.llm import call_llm
from rag.graph_retrieval import retrieve_facts


def save_history(messages):
    try:
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def load_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat history: {e}")
    return []


def _render_token_bar(prompt_tok: int, comp_tok: int, max_ctx: int):
    total = prompt_tok + comp_tok
    pct   = min(total / max_ctx, 1.0)
    st.progress(pct, text=f"{total:,} / {max_ctx:,} tokens  "
                          f"(prompt: {prompt_tok:,} | response: {comp_tok:,}) "
                          f"— {pct*100:.1f}%")


def render_chat_tab(collection, episodic, id_to_name, name_to_id,
                    model, intent_model, ollama_host, num_ctx, deliberation_rounds, active_personas, enable_thinking, num_predict, system_prompt, n_results, top_k, do_rerank, hybrid,
                    neo4j_uri, neo4j_user, neo4j_password, enable_condenser, condenser_threshold):
    
    ctx_kb = f"{num_ctx/1024:.0f}k" if num_ctx >= 1024 else f"{num_ctx}"
    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.markdown(f"### :material/chat: Chat with your memories")
    with col_clear:
        if st.button("Clear History", icon=":material/delete_sweep:", help="Deletes the local chat history file."):
            st.session_state.messages = []
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)
            st.rerun()
    # st.caption(f":material/settings_input_component: Context: **{ctx_kb}** | :material/model_training: Model: **{model}** | :material/timer: Predict Limit: **{num_predict}**")
    
    if active_personas and deliberation_rounds > 0:
        st.caption(f"**Intent Model:** {intent_model} · **Model:** {model} ({ctx_kb}) - max: {num_predict} · **Hybrid:** {hybrid} (Combining vector (semantic) and keyword (exact match) search for peak accuracy.)")
        st.caption(f"**Committee Active:** {', '.join(active_personas)} ({deliberation_rounds} rounds)")
        # st.caption(f"**System Prompt:** {system_prompt}")
    
    # Committee Toggle
    bypass_committee = st.toggle("Talk only to The Self (Disable Committee)", value=False, help="Bypasses multi-persona deliberation for an immediate direct response.")

    if "messages" not in st.session_state:
        st.session_state.messages = load_history()

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if msg.get("thinking"):
                    duration = msg.get("duration", 0)
                    with st.expander(f"Thinking… ({duration:.1f}s)", expanded=False, icon=":material/psychology_alt:"):
                        st.markdown(f'<div class="think-block">{msg["thinking"]}</div>',
                                    unsafe_allow_html=True)
                if msg.get("facts") or msg.get("docs") or msg.get("episodes"):
                    n_facts = len(msg.get("facts", []))
                    n_docs = len(msg.get("docs", []))
                    n_epis = len(msg.get("episodes", []))
                    with st.expander(f"Context: {n_facts} Facts | {n_docs} Docs | {n_epis} Episo", expanded=False, icon=":material/hub:"):
                        if msg.get("intent"):
                            st.markdown("**Intent Analysis:**")
                            st.json(msg["intent"])
                        
                        if msg.get("facts"):
                            st.markdown("**Semantic Knowledge (Graph):**")
                            for f in msg["facts"]:
                                st.markdown(f"- {f}")
                        
                        if msg.get("docs") or msg.get("episodes"):
                            st.markdown("**Memory Fragments:**")
                            t_docs, t_epis = st.tabs(["Fragments", "Episodes"]) if (msg.get("docs") and msg.get("episodes")) else (None, None)
                            
                            def render_doc(d):
                                # Determine source icon/color
                                from config import SOURCES
                                s_meta = next((s for s in SOURCES if s["id"] == d.get("source")), {"icon": "description", "color": "#64748b"})
                                label = f":material/{s_meta['icon']}: {d.get('date', 'Unknown Date')} · {d.get('friend', 'Unknown')}"
                                score = d.get("rerank_score", 0)
                                if score != 0:
                                    label += f" · Relevance: {score:.2f}"
                                
                                with st.container(border=True):
                                    st.caption(f'<span style="color: {s_meta["color"]}; font-weight: bold;">{label}</span>', unsafe_allow_html=True)
                                    st.markdown(d.get("content", ""))

                            if t_docs:
                                with t_docs:
                                    for d in msg["docs"]: render_doc(d)
                                with t_epis:
                                    for e in msg["episodes"]: st.info(e)
                            elif msg.get("docs"):
                                for d in msg["docs"]: render_doc(d)
                            elif msg.get("episodes"):
                                for e in msg["episodes"]: st.info(e)
                if msg.get("deliberations"):
                    with st.expander("Inner Deliberation Committee", expanded=False, icon=":material/groups:"):
                        for d in msg["deliberations"]:
                            st.markdown(f"**{d['persona']} (Round {d['round']})**")
                            st.info(d['response'])
                st.markdown(msg["content"])
                if msg.get("filter_info"):
                    st.caption(msg["filter_info"])
                if msg.get("debug_log"):
                    for entry in msg["debug_log"]:
                        st.caption(entry)
                if msg.get("prompt_tokens"):
                    _render_token_bar(msg["prompt_tokens"], msg["completion_tokens"], num_ctx)
                
                # Full Context Inspector
                with st.expander("Full Prompt Context (Raw)", expanded=False, icon=":material/visibility:"):
                    from rag.llm import _build_context_string
                    
                    # Use the condensed version if available
                    full_p = msg.get("full_prompt_text", "")
                    if not full_p:
                        # Fallback for old messages
                        ctx_str = _build_context_string(msg.get("docs", []), msg.get("episodes", []), msg.get("facts", []))
                        sys_p = msg.get("system_prompt", system_prompt)
                        full_p = (
                            f"{sys_p}\n\n"
                            f"CRITICAL PERSPECTIVE RULE:\n"
                            f"You are the core of our mind. When the context or deliberations mention 'Romain', it is talking about US. We are Romain. Always refer to 'Romain' as 'us' or 'we' (as the unified, balanced voice of this system). Never refer to Romain in the third person.\n\n"
                            f"=== RELATIONSHIP INTERPRETATION GUIDE ===\n"
                            f"Pay extremely close attention to the tense of semantic facts from the graph.\n"
                            f"- 'WAS' or '(PAST relationship)' means the state is HISTORICAL and NO LONGER TRUE.\n"
                            f"- 'IS' or '(CURRENT relationship)' means the state is TRUE RIGHT NOW.\n"
                            f"Do not confuse past relationships with current ones.\n\n"
                            f"CONTEXT:\n{ctx_str}"
                        )
                    
                    if msg.get("condenser_stats") and msg["condenser_stats"] != "Not triggered":
                        st.info(f"🧬 **Auto-Condenser**: {msg['condenser_stats']}")

                    if msg.get("deliberations"):
                        full_p += "\n\n--- COMMITTEE DELIBERATIONS ---\n"
                        for d in msg["deliberations"]:
                            full_p += f"\nPersona: {d['persona']} (Round {d['round']})\nResponse: {d['response']}\n"
                    
                    st.code(full_p, language="markdown")
            else:
                st.markdown(msg["content"])

    # Input area handling: swaps submit icon to stop icon when generating
    is_generating = st.session_state.get("generating", False)
    
    if is_generating:
        # Overlay a styled CSS that transforms the submit arrow to a stop icon
        st.html("""
            <style>
            /* When generating: change arrow icon to a stop square */
            [data-testid="stChatInput"] button[kind="primary"] svg path {
                d: path("M6 6h12v12H6z") !important;
            }
            [data-testid="stChatInput"] button[kind="primary"] {
                background-color: #f87171 !important;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.65; }
            }
            </style>
        """)
    
    if prompt := st.chat_input("Ask anything about your memories…", disabled=is_generating):
        st.session_state.generating = True
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_history(st.session_state.messages)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            filter_info = ""

            with st.spinner("Translating query to introspective thought..."):
                from rag.llm import translate_to_introspective
                prompt = translate_to_introspective(prompt, intent_model, ollama_host)
                st.caption(f"*Introspective Query: {prompt}*")

            from rag.rag_retrieval import analyze_intent, rag_retrieval

            debug_log = []
            with st.spinner("Analyzing intent..."):
                intent = analyze_intent(prompt, intent_model, ollama_host, name_to_id)
                # DEBUG: Show suggested skills immediately
                s_list = intent.get("skills", [])
                if s_list:
                    for s_id in s_list:
                        msg = f"🎯 Intent Router suggested skill: `{s_id}`"
                        st.caption(msg)
                        debug_log.append(msg)

            docs, episodes, facts = [], [], []
            if intent.get("needs_retrieval", True):
                with st.spinner("Retrieving memories…"):
                    docs, episodes, _ = rag_retrieval(
                        prompt, n_results,
                        collection, episodic, id_to_name, name_to_id,
                        intent_model, ollama_host,
                        metadata_filters=None,
                        relevance_threshold=-2.0,
                        top_k=top_k, do_rerank=do_rerank, hybrid=hybrid,
                        intent=intent
                    )
                
                # Fetch semantic facts from Neo4j based on intent
                with st.spinner("Accessing knowledge graph…"):
                    facts = retrieve_facts(intent)
                    # DEBUG: Confirm skills were executed
                    executed_skills = intent.get("skills", [])
                    if executed_skills:
                        msg = f"✅ Executed {len(executed_skills)} specialized skill(s)"
                        st.caption(msg)
                        debug_log.append(msg)
                
            # with st.spinner("Purifying context..."):
            #     from rag.llm import filter_irrelevant_context
            #     docs = filter_irrelevant_context(prompt, docs, intent_model, ollama_host)

            if not docs and not episodes and not facts:
                # Only show a note if the query actually needed retrieval
                # But DO NOT return early — the model can call tools to fetch facts on demand
                if intent.get("needs_retrieval", True):
                    st.caption("ℹ️ No pre-fetched memories — the model may use tools to retrieve data.")

            # Show source count
            src_parts  = []
            if docs:     src_parts.append(f"{len(docs)} conversation chunk(s)")
            if episodes: src_parts.append(f"{len(episodes)} episode(s)")
            if facts:    src_parts.append(f"{len(facts)} semantic fact(s)")
            
            # 1. Show User Intent (always)
            with st.expander("User Intent", expanded=False, icon=":material/psychology:"):
                st.json(intent)
            
            # 2. Show Semantic Facts (only if found)
            if facts:
                with st.expander(f"Semantic Facts ({len(facts)} found)", expanded=False, icon=":material/hub:"):
                    for f in facts:
                        st.markdown(f"- {f}")

            mode_parts = []
            if hybrid:    mode_parts.append("hybrid")
            if do_rerank: mode_parts.append(f"reranked → top-{top_k}")
            note = f" [{', '.join(mode_parts)} from {n_results}]" if mode_parts else ""
            st.caption(f"Context: {', '.join(src_parts)}{note}")

            # Dynamic num_predict overwrite based on query type
            eff_num_predict = num_predict
            if intent.get("query_type") == "conversational":
                # For greetings/confirmations, provide a generous limit to ensure natural flow
                # (Overwriting the global limit if it was set too low for conversational filler)
                eff_num_predict = 1024

            def tool_callback(name, args):
                msg = f"🛠️ Tool called: `{name}({args})`"
                st.toast(msg, icon="⚙️")
                debug_log.append(msg)

            if active_personas and deliberation_rounds > 0 and not bypass_committee:
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
                        from rag.llm import _build_context_string
                        raw_ctx = _build_context_string(docs, episodes, facts)
                        history_text = "\n".join([m.get("content", "") for m in st.session_state.messages[:-1]])
                        # Holistic estimate for the UI feedback
                        total_est = (len(raw_ctx) + len(history_text) + 1000) // 3
                        
                        if enable_condenser and total_est > (num_ctx * condenser_threshold / 100):
                            with st.spinner("Condensing memory context (Intent Model)..."):
                                pass # Backend handles it, we just show the spinner

                        thinking, answer, p_tok, c_tok, deliberations, duration, ctx_used, condenser_stats = deliberate_and_synthesize(
                            prompt, docs, episodes, facts, model, ollama_host, num_ctx,
                            active_personas, deliberation_rounds, enable_thinking, eff_num_predict,
                            update_callback=ui_callback,
                            conversation_history=st.session_state.messages[:-1],
                            discovered_identities=st.session_state.get("discovered_identities"),
                            enable_condenser=enable_condenser,
                            condenser_threshold=condenser_threshold,
                            intent_model=intent_model
                        )
                        if condenser_stats != "Not triggered":
                            debug_log.append(f"🧬 Auto-Condenser: {condenser_stats}")
                        
                        status.update(label="Inner Deliberation Committee", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label=f"Ollama error: {e}", state="error")
                        return
            else:
                with st.spinner(f"Asking {model}…"):
                    try:
                        from rag.llm import _build_context_string
                        raw_ctx = _build_context_string(docs, episodes, facts)
                        history_text = "\n".join([m.get("content", "") for m in st.session_state.messages[:-1]])
                        total_est = (len(raw_ctx) + len(history_text) + len(system_prompt)) // 3
                        
                        if enable_condenser and total_est > (num_ctx * condenser_threshold / 100):
                            st.caption("🪄 *Condensing context to fit memory window...*")
                        
                        thinking, answer, p_tok, c_tok, duration, ctx_used, condenser_stats = call_llm(
                            prompt, docs, episodes, facts, model, ollama_host, num_ctx, system_prompt, enable_thinking, eff_num_predict,
                            conversation_history=st.session_state.messages[:-1],
                            tool_callback=tool_callback,
                            enable_condenser=enable_condenser,
                            condenser_threshold=condenser_threshold,
                            intent_model=intent_model
                        )
                        if condenser_stats != "Not triggered":
                            debug_log.append(f"🧬 Auto-Condenser: {condenser_stats}")
                        
                        deliberations = None
                    except Exception as e:
                        st.error(f"Ollama error: {e}")
                        return

                    if thinking:
                        with st.expander(f"Thinking… ({duration:.1f}s)", expanded=False, icon=":material/psychology_alt:"):
                            st.markdown(f'<div class="think-block">{thinking}</div>',
                                        unsafe_allow_html=True)

            # Final answer rendering before state commitment
            st.markdown(answer)
            _render_token_bar(p_tok, c_tok, num_ctx)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "thinking": thinking,
                "deliberations": deliberations,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "duration": duration,
                "filter_info": filter_info,
                "intent": intent,
                "facts": facts,
                "docs": docs,
                "episodes": episodes,
                "debug_log": debug_log,
                "system_prompt": system_prompt,
                "full_prompt_text": f"{system_prompt}\n\nCONTEXT:\n{ctx_used}",
                "condenser_stats": condenser_stats
            })
            save_history(st.session_state.messages)
            
            # Reset state and cleanup UI
            st.session_state.generating = False
            st.rerun()
