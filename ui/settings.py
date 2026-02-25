"""
ui/settings.py — Settings modal (cog) for LLM, RAG, and Neo4j parameters.
"""
import streamlit as st
import ollama

from config import (
    DEFAULT_MODEL, DEFAULT_INTENT_MODEL, DEFAULT_OLLAMA, DEFAULT_CTX, DEFAULT_SYSTEM_PROMPT,
    DEFAULT_DELIBERATION_ROUNDS, DEFAULT_ACTIVE_PERSONAS, DEFAULT_ENABLE_THINKING,
    IDENTITIES,
    EMBEDDING_MODEL, EMBEDDING_MODELS,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)


def init_settings_defaults():
    """Ensure all settings exist in session_state with sensible defaults."""
    defaults = {
        "ollama_host": DEFAULT_OLLAMA,
        "model": DEFAULT_MODEL,
        "intent_model": DEFAULT_INTENT_MODEL,
        "num_ctx": DEFAULT_CTX,
        "deliberation_rounds": DEFAULT_DELIBERATION_ROUNDS,
        "active_personas": DEFAULT_ACTIVE_PERSONAS,
        "enable_thinking": DEFAULT_ENABLE_THINKING,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "embedding_model": EMBEDDING_MODEL,
        "n_results": 500,
        "top_k": 100,
        "do_rerank": True,
        "hybrid": True,
        "neo4j_uri": NEO4J_URI,
        "neo4j_user": NEO4J_USER,
        "neo4j_password": NEO4J_PASSWORD,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
        # Also initialize draft state for the dialog menu
        draft_key = f"draft_{key}"
        if draft_key not in st.session_state:
            st.session_state[draft_key] = val


@st.dialog("Settings", width="large")
def _settings_dialog():
    """Modal dialog containing all tuneable parameters."""
    
    c_nav, c_content = st.columns([1, 2.5])
    
    with c_nav:
        st.markdown("### Menu")
        selected = st.radio(
            "Navigation",
            ["🤖 LLM", "🧠 Embedding", "🔎 RAG", "🕸️ Neo4j"],
            label_visibility="collapsed",
            key="dlg_nav"
        )
        st.markdown("<br>" * 10, unsafe_allow_html=True) 

    with c_content:
        if selected == "🤖 LLM":
            st.markdown("#### :material/smart_toy: LLM Settings")

            _host = st.text_input("Ollama host", value=st.session_state["draft_ollama_host"], key="dlg_ollama_host")
            st.session_state["draft_ollama_host"] = _host

            try:
                _client = ollama.Client(host=_host)
                _models = [m["model"] for m in _client.list().get("models", [])]
                _default_idx = _models.index(st.session_state["draft_model"]) if st.session_state["draft_model"] in _models else 0
                _mod = st.selectbox("Model", _models, index=_default_idx, key="dlg_model")
                st.session_state["draft_model"] = _mod

                _intent_idx = _models.index(st.session_state["draft_intent_model"]) if st.session_state["draft_intent_model"] in _models else 0
                _int_mod = st.selectbox("Pre-flight Intent Model", _models, index=_intent_idx, key="dlg_intent_model", help="Small, fast models like llama3.2:3b are recommended for intent routing.")
                st.session_state["draft_intent_model"] = _int_mod

            except Exception:
                _mod = st.text_input("Model (manual)", value=st.session_state["draft_model"], key="dlg_model")
                st.session_state["draft_model"] = _mod
                
                _int_mod = st.text_input("Pre-flight Intent Model (manual)", value=st.session_state["draft_intent_model"], key="dlg_intent_model")
                st.session_state["draft_intent_model"] = _int_mod

            _sys_col1, _sys_col2 = st.columns([1, 2])
            with _sys_col1:
                # Determine current selected identity based on text match; fallback to "Custom"
                _current_prompt = st.session_state["draft_system_prompt"].strip().lower()
                _matched_identity = "Custom"
                for name, prompt_text in IDENTITIES.items():
                    if _current_prompt == prompt_text.strip().lower():
                        _matched_identity = name
                        break
                        
                _id_opts = list(IDENTITIES.keys()) + ["Custom"]
                _id_idx = _id_opts.index(_matched_identity)
                
                def _on_identity_change():
                    new_id = st.session_state["dlg_identity_selector"]
                    if new_id != "Custom":
                        st.session_state["dlg_system_prompt"] = IDENTITIES[new_id]
                        st.session_state["draft_system_prompt"] = IDENTITIES[new_id]

                _id_sel = st.selectbox(
                    "Identity Preset", 
                    options=_id_opts, 
                    index=_id_idx, 
                    key="dlg_identity_selector",
                    on_change=_on_identity_change,
                    help="Select a psychological part to instantly change the system prompt."
                )

            with _sys_col2:
                _ctx = st.select_slider(
                    "Context window",
                    options=[8192, 16384, 32768, 65536],
                    value=st.session_state["draft_num_ctx"],
                    format_func=lambda x: f"{x // 1024}k",
                    key="dlg_num_ctx"
                )
                st.session_state["draft_num_ctx"] = _ctx
                
                _think = st.toggle("Enable Thinking Tokens", value=st.session_state["draft_enable_thinking"], key="dlg_enable_thinking")
                st.session_state["draft_enable_thinking"] = _think

            _sys = st.text_area(
                "System prompt",
                value=st.session_state["draft_system_prompt"],
                height=180,
                key="dlg_system_prompt",
                help="Instructions for how the AI should behave and answer questions."
            )
            st.session_state["draft_system_prompt"] = _sys

            st.markdown("#### :material/diversity_3: Inner Deliberation Committee")
            st.caption("Select personas to debate the answer before 'The Self' responds.")
            _personas = [p for p in IDENTITIES.keys() if p != "The Self"]
            
            _act_pers = st.multiselect(
                "Active Personas",
                options=_personas,
                default=[p for p in st.session_state["draft_active_personas"] if p in _personas],
                key="dlg_active_personas"
            )
            st.session_state["draft_active_personas"] = _act_pers

            _rounds = st.number_input(
                "Deliberation Rounds",
                min_value=1,
                max_value=5,
                value=st.session_state["draft_deliberation_rounds"],
                key="dlg_deliberation_rounds",
                help="How many turns the personas will pass the conversation between themselves to refine the context."
            )
            st.session_state["draft_deliberation_rounds"] = _rounds

        elif selected == "🧠 Embedding":
            st.markdown("#### :material/model_training: Embedding Model")

            _emb_options = list(EMBEDDING_MODELS)
            _current_emb = st.session_state["draft_embedding_model"]
            if _current_emb not in _emb_options:
                _emb_options.insert(0, _current_emb)
            _emb_idx = _emb_options.index(_current_emb)

            _emb = st.selectbox(
                "Embedding model",
                _emb_options,
                index=_emb_idx,
                key="dlg_embedding_model",
                help="SentenceTransformer model used for vector embeddings. ⚠️ Changing this requires re-ingesting all ChromaDB collections."
            )
            st.session_state["draft_embedding_model"] = _emb

        elif selected == "🔎 RAG":
            st.markdown("#### :material/manage_search: RAG Settings")

            # RAG limits are now removed as requested (hardcoded to 500/100)

            r3, r4 = st.columns(2)
            with r3:
                _drr = st.toggle("Enable reranking", value=st.session_state["draft_do_rerank"], key="dlg_do_rerank")
                st.session_state["draft_do_rerank"] = _drr
            with r4:
                _hyb = st.toggle("Hybrid search", value=st.session_state["draft_hybrid"], key="dlg_hybrid")
                st.session_state["draft_hybrid"] = _hyb

        elif selected == "🕸️ Neo4j":
            st.markdown("#### :material/hub: Neo4j Settings")
            
            _n_uri = st.text_input("Neo4j URI", value=st.session_state["draft_neo4j_uri"], key="dlg_neo4j_uri")
            st.session_state["draft_neo4j_uri"] = _n_uri
            
            _n_usr = st.text_input("Neo4j User", value=st.session_state["draft_neo4j_user"], key="dlg_neo4j_user")
            st.session_state["draft_neo4j_user"] = _n_usr
            
            _n_pass = st.text_input("Neo4j Password", value=st.session_state["draft_neo4j_password"], type="password", key="dlg_neo4j_password")
            st.session_state["draft_neo4j_password"] = _n_pass

    st.divider()

    # ── Save button ──────────────────────────────────────────────────
    if st.button("Save settings", type="primary", width="stretch", icon=":material/save:"):
        _emb_changed = (st.session_state["draft_embedding_model"] != st.session_state["embedding_model"])

        # Commit all drafts to live variables
        st.session_state["ollama_host"] = st.session_state["draft_ollama_host"]
        st.session_state["model"] = st.session_state["draft_model"]
        st.session_state["intent_model"] = st.session_state["draft_intent_model"]
        st.session_state["num_ctx"] = st.session_state["draft_num_ctx"]
        st.session_state["deliberation_rounds"] = st.session_state["draft_deliberation_rounds"]
        st.session_state["active_personas"] = st.session_state["draft_active_personas"]
        st.session_state["enable_thinking"] = st.session_state["draft_enable_thinking"]
        st.session_state["system_prompt"] = st.session_state["draft_system_prompt"]
        st.session_state["embedding_model"] = st.session_state["draft_embedding_model"]
        st.session_state["n_results"] = st.session_state["draft_n_results"]
        st.session_state["top_k"] = st.session_state["draft_top_k"]
        st.session_state["do_rerank"] = st.session_state["draft_do_rerank"]
        st.session_state["hybrid"] = st.session_state["draft_hybrid"]
        st.session_state["neo4j_uri"] = st.session_state["draft_neo4j_uri"]
        st.session_state["neo4j_user"] = st.session_state["draft_neo4j_user"]
        st.session_state["neo4j_password"] = st.session_state["draft_neo4j_password"]

        if _emb_changed:
            from rag.resources import load_embedding_func, load_chroma
            load_embedding_func.clear()
            load_chroma.clear()
            st.toast("Embedding model changed. Cached resources cleared.", icon="⚠️")

        st.rerun()

def render_settings():
    """
    Initialises settings defaults and returns the current settings tuple:
        (model, intent_model, ollama_host, num_ctx, 
         deliberation_rounds, active_personas, enable_thinking, system_prompt, 
         n_results, top_k, do_rerank, hybrid,
         neo4j_uri, neo4j_user, neo4j_password)

    The settings button itself lives in the sidebar (see ui/sidebar.py).
    """
    init_settings_defaults()
    with st.sidebar:
        st.markdown("**Connections**")
    return (
        st.session_state["model"],
        st.session_state["intent_model"],
        st.session_state["ollama_host"],
        st.session_state["num_ctx"],
        st.session_state["deliberation_rounds"],
        st.session_state["active_personas"],
        st.session_state["enable_thinking"],
        st.session_state["system_prompt"],
        st.session_state["n_results"],
        st.session_state["top_k"],
        st.session_state["do_rerank"],
        st.session_state["hybrid"],
        st.session_state["neo4j_uri"],
        st.session_state["neo4j_user"],
        st.session_state["neo4j_password"],
    )
