import re
from pathlib import Path

# 1. Update ui/chat.py
chat_path = Path('ui/chat.py')
chat_lines = chat_path.read_text().split('\n')

# Find where render_chat_tab starts
start_idx = -1
for i, line in enumerate(chat_lines):
    if line.startswith("def render_chat_tab("):
        start_idx = i
        break

# Insert the sidebar LLM settings right after standard markdown title
insert_idx = start_idx
while insert_idx < len(chat_lines) and 'st.markdown("### :material/chat: Chat with your memories")' not in chat_lines[insert_idx]:
    insert_idx += 1

sidebar_injection = """
    # ── LLM Settings (Sidebar) ───────────────────────────────────────
    with st.sidebar:
        with st.expander("🛠️ LLM Settings", expanded=False):
            # Ollama Host
            st.session_state["ollama_host"] = st.text_input(
                "Ollama host",
                value=st.session_state.get("ollama_host", "http://localhost:11434"),
                key="chat_ollama_host",
            )
            ollama_host = st.session_state["ollama_host"]
            
            # Model Picker
            import ollama
            try:
                _client = ollama.Client(host=ollama_host)
                _models = [m["model"] for m in _client.list().get("models", [])]
                _default_model = st.session_state.get("model", "deepseek-r1:14b")
                _default_idx = _models.index(_default_model) if _default_model in _models else 0
                st.session_state["model"] = st.selectbox("Model", _models, index=_default_idx, key="chat_model")
            except Exception:
                st.session_state["model"] = st.text_input(
                    "Model (manual)",
                    value=st.session_state.get("model", "deepseek-r1:14b"),
                    key="chat_model",
                )
            model = st.session_state["model"]
            
            # Context Window
            st.session_state["num_ctx"] = st.select_slider(
                "Context window",
                options=[8192, 16384, 32768, 65536],
                value=st.session_state.get("num_ctx", 32768),
                format_func=lambda x: f"{x // 1024}k",
                key="chat_num_ctx",
            )
            num_ctx = st.session_state["num_ctx"]
            
            # System Prompt
            st.session_state["system_prompt"] = st.text_area(
                "System prompt",
                value=st.session_state.get("system_prompt", "You are an AI assistant analysing personal Facebook history. Answer in the same language as the user's question. Be direct and specific."),
                height=150,
                key="chat_system_prompt",
                help="Instructions for how the AI should behave and answer questions.",
            )
            system_prompt = st.session_state["system_prompt"]
"""

chat_lines.insert(insert_idx + 1, sidebar_injection)
# We also need to rewrite the function signature to remove the llm kwargs since we now read/maintain them locally in the sidebar
new_def = "def render_chat_tab(collection, episodic, id_to_name, name_to_id,\n                    n_results, top_k, do_rerank, hybrid):"
# Replace the old def (lines start_idx to start_idx+1 usually)
chat_lines[start_idx] = "def render_chat_tab(collection, episodic, id_to_name, name_to_id,"
chat_lines[start_idx+1] = "                    n_results, top_k, do_rerank, hybrid):"

chat_path.write_text('\n'.join(chat_lines))


# 2. Update pages/chat.py
p_chat_path = Path('pages/chat.py')
p_chat_content = p_chat_path.read_text()

# We need to drop model, ollama_host, num_ctx, system_prompt from the render_chat_tab call
p_chat_content = p_chat_content.replace(
    "model, ollama_host, num_ctx, system_prompt, n_results, top_k, do_rerank, hybrid, \\",
    "*_, n_results, top_k, do_rerank, hybrid, \\"
)
p_chat_content = p_chat_content.replace(
    "model, ollama_host, num_ctx, system_prompt, n_results, top_k, do_rerank, hybrid,",
    "n_results, top_k, do_rerank, hybrid,"
)
p_chat_path.write_text(p_chat_content)


# 3. Update ui/settings.py
set_path = Path('ui/settings.py')
set_content = set_path.read_text()

# Delete LLM section from settings modal
llm_match = re.search(r'    # ── LLM Settings ─────────────────────────────────────────────────\n.*?    st\.divider\(\)\n', set_content, re.DOTALL)
if llm_match:
    set_content = set_content.replace(llm_match.group(0), '')
    
# Delete local save references in save button
set_content = set_content.replace('        st.session_state["ollama_host"] = ollama_host\n', '')
set_content = set_content.replace('        st.session_state["model"] = model\n', '')
set_content = set_content.replace('        st.session_state["num_ctx"] = num_ctx\n', '')
set_content = set_content.replace('        st.session_state["system_prompt"] = system_prompt\n', '')

set_path.write_text(set_content)
