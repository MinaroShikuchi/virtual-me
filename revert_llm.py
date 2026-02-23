import re
from pathlib import Path

# 1. Update pages/chat.py
p_chat_path = Path('pages/chat.py')
p_chat_content = p_chat_path.read_text()
p_chat_content = p_chat_content.replace(
    "settings = render_settings()\n    n_results, top_k, do_rerank, hybrid = settings[4:8]",
    "model, ollama_host, num_ctx, system_prompt, n_results, top_k, do_rerank, hybrid, \\\n        *_ = render_settings()"
)
p_chat_path.write_text(p_chat_content)

# 2. Update ui/chat.py
chat_path = Path('ui/chat.py')
chat_lines = chat_path.read_text().split('\n')
for i, line in enumerate(chat_lines):
    if line.startswith("def render_chat_tab("):
        chat_lines[i] = "def render_chat_tab(collection, episodic, id_to_name, name_to_id,"
        chat_lines[i+1] = "                    model, ollama_host, num_ctx, system_prompt, n_results, top_k, do_rerank, hybrid):"
        break

# Find and remove sidebar block
start_idx = -1
end_idx = -1
for i, line in enumerate(chat_lines):
    if "    # ── LLM Settings (Sidebar) ───────────────────────────────────────" in line:
        start_idx = i
    if start_idx != -1 and line.strip() == 'system_prompt = st.session_state["system_prompt"]':
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    del chat_lines[start_idx:end_idx+1]

chat_path.write_text('\n'.join(chat_lines))


# 3. Rewrite _settings_dialog in ui/settings.py
set_path = Path('ui/settings.py')
set_content = set_path.read_text()

new_dialog_code = """
@st.dialog("Settings", width="large")
def _settings_dialog():
    \"\"\"Modal dialog containing all tuneable parameters.\"\"\"
    
    c_left, c_right = st.columns([1, 1.5])
    
    with c_left:
        # ── LLM Settings ─────────────────────────────────────────────────
        st.markdown("#### :material/smart_toy: LLM Settings")

        ollama_host = st.text_input(
            "Ollama host",
            value=st.session_state["ollama_host"],
            key="dlg_ollama_host",
        )

        try:
            _client = ollama.Client(host=ollama_host)
            _models = [m["model"] for m in _client.list().get("models", [])]
            _default_idx = (
                _models.index(st.session_state["model"])
                if st.session_state["model"] in _models
                else 0
            )
            model = st.selectbox("Model", _models, index=_default_idx, key="dlg_model")
        except Exception:
            model = st.text_input(
                "Model (manual)",
                value=st.session_state["model"],
                key="dlg_model",
            )

        num_ctx = st.select_slider(
            "Context window",
            options=[8192, 16384, 32768, 65536],
            value=st.session_state["num_ctx"],
            format_func=lambda x: f"{x // 1024}k",
            key="dlg_num_ctx",
        )

        system_prompt = st.text_area(
            "System prompt",
            value=st.session_state["system_prompt"],
            height=200,
            key="dlg_system_prompt",
            help="Instructions for how the AI should behave and answer questions.",
        )

    with c_right:
        # ── Embedding Settings ────────────────────────────────────────────
        st.markdown("#### :material/model_training: Embedding Model")

        _emb_options = list(EMBEDDING_MODELS)
        _current_emb = st.session_state["embedding_model"]
        if _current_emb not in _emb_options:
            _emb_options.insert(0, _current_emb)
        _emb_idx = _emb_options.index(_current_emb)

        embedding_model = st.selectbox(
            "Embedding model",
            _emb_options,
            index=_emb_idx,
            key="dlg_embedding_model",
            help="SentenceTransformer model used for vector embeddings. "
                 "⚠️ Changing this requires re-ingesting all ChromaDB collections.",
        )

        st.divider()

        # ── RAG Settings ─────────────────────────────────────────────────
        st.markdown("#### :material/manage_search: RAG Settings")

        r1, r2 = st.columns(2)
        with r1:
            n_results = st.slider(
                "Retrieve (vector search)", 5, 100,
                value=st.session_state["n_results"],
                key="dlg_n_results",
            )
        with r2:
            top_k = st.slider(
                "Keep top-k (after reranking)", 1, 30,
                value=st.session_state["top_k"],
                key="dlg_top_k",
            )

        r3, r4 = st.columns(2)
        with r3:
            do_rerank = st.toggle(
                "Enable reranking",
                value=st.session_state["do_rerank"],
                key="dlg_do_rerank",
            )
        with r4:
            hybrid = st.toggle(
                "Hybrid search",
                value=st.session_state["hybrid"],
                key="dlg_hybrid",
            )

        st.divider()

        # ── Neo4j Settings ───────────────────────────────────────────────
        st.markdown("#### :material/hub: Neo4j Settings")
        
        n1, n2, n3 = st.columns(3)
        with n1:
            neo4j_uri = st.text_input("Neo4j URI", value=st.session_state["neo4j_uri"], key="dlg_neo4j_uri")
        with n2:
            neo4j_user = st.text_input("Neo4j User", value=st.session_state["neo4j_user"], key="dlg_neo4j_user")
        with n3:
            neo4j_password = st.text_input("Neo4j Password", value=st.session_state["neo4j_password"], type="password", key="dlg_neo4j_password")

    # ── Save button ──────────────────────────────────────────────────
    st.markdown("")
    if st.button("Save settings", type="primary", width="stretch", icon=":material/save:"):
        _emb_changed = (embedding_model != st.session_state.get("embedding_model"))

        st.session_state["ollama_host"] = ollama_host
        st.session_state["model"] = model
        st.session_state["num_ctx"] = num_ctx
        st.session_state["system_prompt"] = system_prompt
        st.session_state["embedding_model"] = embedding_model
        st.session_state["n_results"] = n_results
        st.session_state["top_k"] = top_k
        st.session_state["do_rerank"] = do_rerank
        st.session_state["hybrid"] = hybrid
        st.session_state["neo4j_uri"] = neo4j_uri
        st.session_state["neo4j_user"] = neo4j_user
        st.session_state["neo4j_password"] = neo4j_password

        if _emb_changed:
            from rag.resources import load_embedding_func, load_chroma
            load_embedding_func.clear()
            load_chroma.clear()
            st.toast("Embedding model changed. Cached resources cleared.", icon="⚠️")

        st.rerun()

def render_settings():
"""

# Extract the old dialog out and replace it with new
start_idx = set_content.find('@st.dialog("Settings", width="small")')
end_idx = set_content.find('def render_settings():')

if start_idx != -1 and end_idx != -1:
    set_content = set_content[:start_idx] + new_dialog_code + set_content[end_idx + len('def render_settings():'):]

set_path.write_text(set_content)

