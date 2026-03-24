"""
ui/models.py — Manage LoRA adapters, Fine-Tuning, and Ollama compilation
"""
import sys
import os
import subprocess
import threading
import re
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import DEFAULT_OLLAMA
from ui.components.log_viewer import scrollable_log

# Unsloth MUST be imported before transformers/peft to avoid warnings and speed regressions
try:
    import unsloth
    from unsloth import get_chat_template
except ImportError:
    unsloth = None
    get_chat_template = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None



def get_available_adapters():
    """Find all folders in ./models that contain an adapter_config.json"""
    models_dir = Path("models")
    if not models_dir.exists() or not models_dir.is_dir():
        return []
        
    adapters = []
    for d in models_dir.iterdir():
        if d.is_dir() and (d / "adapter_config.json").exists():
            adapters.append(d.name)
            
    return sorted(adapters)

def render_finetune_tab():
    st.markdown("### :material/neurology: Fine-Tune LoRA Adapter")
    st.write(
        "Fine-tune a base model on your conversation data using **LoRA** "
        "(Low-Rank Adaptation). Requires `transformers`, `peft`, `trl`, "
        "`datasets`, and optionally `bitsandbytes` for 4-bit quantization."
    )

    # ── Ollama host for model selection ────────────────────────────────
    _lora_ollama_host = st.session_state.get("ollama_host", DEFAULT_OLLAMA)

    # Fetch models from the specified Ollama host
    _lora_models: list[str] = []
    _lora_conn_error: str | None = None
    try:
        import ollama as _ollama_mod
        _lora_client = _ollama_mod.Client(host=_lora_ollama_host)
        _lora_model_list = _lora_client.list()
        _lora_models = sorted(
            m.get("name", m.get("model", ""))
            for m in _lora_model_list.get("models", [])
            if m.get("name") or m.get("model")
        )
    except Exception as _e:
        _lora_conn_error = str(_e)
        _lora_models = []

    if _lora_conn_error:
        st.warning(f"⚠️ Could not connect to Ollama at `{_lora_ollama_host}`: {_lora_conn_error}")

    col_lora1, col_lora2 = st.columns(2)
    with col_lora1:
        model_source = st.radio(
            "Model Source",
            options=["Ollama", "HuggingFace"],
            horizontal=True,
            index=0 if _lora_models else 1,
            help="Choose whether to select a model from your local Ollama library or specify a repo ID/URL from HuggingFace Hub."
        )

        if model_source == "Ollama" and _lora_models:
            lora_base_model = st.selectbox(
                "Base model (Ollama)",
                options=_lora_models,
                index=0,
                key="lora_base_model_select",
                help=f"Models from Ollama at {_lora_ollama_host}",
            )
        else:
            lora_base_model = st.text_input(
                "Base model (HuggingFace ID or URL)",
                value="meta-llama/Llama-3.2-3B-Instruct",
                key="lora_base_model_hf",
                help="Enter a HuggingFace model ID (e.g. 'unsloth/llama-3-8b-bnb-4bit') or a full HuggingFace URL.",
            )
        lora_data = st.text_input(
            "Training data (JSONL)",
            value="./data/facebook/finetune_data.jsonl",
            key="lora_data",
        )
        # Calculate dynamic auto-export name based on selected base model
        base_clean = getattr(lora_base_model, "name", lora_base_model) if not isinstance(lora_base_model, str) else lora_base_model
        base_clean = base_clean.split(":")[0].split("/")[-1].lower() # e.g. meta-llama/Llama-3.2-3B-Instruct -> llama-3.2-3b-instruct
        
        # Try finding parameter count like 3b, 14b, 7b in the original string
        params_match = re.search(r'(\d+(?:\.\d+)?[bmBM])', lora_base_model)
        params_str = params_match.group(1).lower() if params_match else "unknown"
        
        # Build string e.g my-llama-3.2-3b-v1.0 without duplicating the params if already present
        if params_str != "unknown" and params_str in base_clean:
            default_export_name = f"my-{base_clean}-v1.0"
        else:
            default_export_name = f"my-{base_clean}-{params_str}-v1.0"
            
        lora_output = st.text_input(
            "Output directory",
            value=f"./models/{default_export_name}",
            help="Folder to save your raw HuggingFace format LoRA adapter."
        )
    with col_lora2:
        lora_epochs = st.number_input(
            "Epochs", min_value=1, max_value=20, value=3, step=1,
            key="lora_epochs",
        )
        lora_batch = st.number_input(
            "Batch size", min_value=1, max_value=32, value=2, step=1,
            key="lora_batch",
            help="Per-device batch size. Keep small for limited GPU memory.",
        )
        lora_lr = st.select_slider(
            "Learning rate",
            options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3],
            value=2e-4,
            key="lora_lr",
            format_func=lambda x: f"{x:.0e}",
            help="2e-4 is typical for LoRA fine-tuning.",
        )
        lora_rank = st.selectbox(
            "LoRA rank",
            options=[4, 8, 16, 32, 64],
            index=1,  # default 8
            key="lora_rank",
            help="Higher rank = more parameters = more expressive but slower. "
                 "8 is a good starting point.",
        )

    col_lora3, col_lora4 = st.columns(2)
    with col_lora3:
        lora_alpha = st.number_input(
            "LoRA alpha", min_value=4, max_value=128, value=16, step=4,
            key="lora_alpha",
            help="Scaling factor. Typically 2× the rank.",
        )
        lora_max_seq = st.number_input(
            "Max sequence length", min_value=128, max_value=4096, value=512, step=128,
            key="lora_max_seq",
        )
    with col_lora4:
        lora_grad_accum = st.number_input(
            "Gradient accumulation steps",
            min_value=1, max_value=32, value=4, step=1,
            key="lora_grad_accum",
            help="Effective batch = batch_size × this.",
        )
        lora_4bit = st.checkbox(
            "Use 4-bit quantization (QLoRA)",
            value=True,
            key="lora_4bit",
            help="Reduces memory usage significantly. Requires bitsandbytes + CUDA GPU.",
        )
        lora_resume = st.checkbox(
            "Resume from checkpoint",
            value=False,
            key="lora_resume",
            help="If interrupted, resume training from the latest checkpoint saved in your output directory."
        )
        lora_auto_export = st.checkbox(
            "Auto-compile to Ollama",
            value=True,
            key="lora_auto_export",
            help="If checked, automatically converts the raw weights at the end of training into a GGUF model registered in Ollama."
        )

    st.caption(
        f"**Effective batch size:** {lora_batch * lora_grad_accum} "
        f"(batch {lora_batch} × accum {lora_grad_accum})"
    )

    with st.expander("👀 Preview formatted dataset"):
        st.caption("See how your JSONL data will be physically formatted for your selected base model's specific prompt template.")
        if st.button("Generate Preview", key="btn_format_preview", type="secondary"):
            if not Path(lora_data).exists():
                st.error(f"Dataset not found at {lora_data}")
            else:
                with st.spinner("Fetching tokenizer template (may take a moment)..."):
                    try:
                        if AutoTokenizer is None:
                            st.error("Transformers library not found. Please install it to use this feature.")
                            st.stop()
                            
                        # Try resolving the ID dynamically using HuggingFace Hub if it's an Ollama shorthand
                        hf_id = lora_base_model
                        if ":" in lora_base_model and "/" not in lora_base_model:
                            from tools.finetune_lora import _resolve_model_name
                            hf_id = _resolve_model_name(lora_base_model)
                            
                        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                        
                        # Read first line of JSONL
                        with open(lora_data, "r", encoding="utf-8") as f:
                            first_line = json.loads(f.readline())
                            
                        # Convert {"conversations": [...]} to standard HuggingFace {"messages": [...]} if needed
                        messages = first_line.get("conversations", first_line.get("messages", []))
                        if not messages:
                            st.warning("No 'conversations' or 'messages' array found in the first JSON record.")
                        else:
                            # Normalize content to plain strings for chat template rendering
                            sanitized_preview = []
                            for m in messages:
                                content = m.get("content", "")
                                if isinstance(content, list):
                                    # Extract text from content blocks (e.g. [{"type": "text", "text": "..."}])
                                    content = "".join(
                                        block.get("text", "") for block in content
                                        if isinstance(block, dict) and block.get("type") == "text"
                                    )
                                sanitized_preview.append({"role": m["role"], "content": content})

                            # Robust chat template check for Base models
                            if tokenizer.chat_template is None:
                                if get_chat_template is not None:
                                    try:
                                        tokenizer = get_chat_template(
                                            tokenizer,
                                            chat_template="llama-3", # Defaulting to Llama-3 style as requested (<|user|>)
                                        )
                                    except Exception as ex_templ:
                                        st.warning(f"Unsloth template injection failed: {ex_templ}")
                                        # Very basic fallback
                                        tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n'}}{% endfor %}"
                                else:
                                    # Very basic fallback if unsloth is missing
                                    tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n'}}{% endfor %}"
                            
                            formatted_text = tokenizer.apply_chat_template(sanitized_preview, tokenize=False, add_generation_prompt=False)
                            st.markdown("### 📄 First Entry Preview")
                            st.code(formatted_text, language="jinja2")

                            # ── Dataset Length Distribution & Outlier Analysis ─────────────────────
                            st.markdown("### 📊 Sequence Length Distribution")
                            st.info("Scanning full dataset to analyze token counts...")
                            
                            all_lengths = []
                            outliers = [] # List of (line_num, tokens, text_snippet)
                            
                            with open(lora_data, "r", encoding="utf-8") as f_dist:
                                for i, line in enumerate(f_dist, 1):
                                    if not line.strip(): continue
                                    try:
                                        obj = json.loads(line)
                                        msgs = obj.get("conversations", obj.get("messages", []))
                                        if msgs:
                                            # Sanitize for loop analysis too
                                            sanitized_loop = []
                                            for m in msgs:
                                                content = m.get("content", "")
                                                if isinstance(content, str):
                                                    sanitized_loop.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
                                                else:
                                                    sanitized_loop.append(m)

                                            # Estimate length via template + exact tokenization
                                            t_text = tokenizer.apply_chat_template(sanitized_loop, tokenize=False, add_generation_prompt=False)
                                            _tok_obj = getattr(tokenizer, "tokenizer", tokenizer)
                                            tokens = len(_tok_obj.encode(t_text, add_special_tokens=False))
                                            all_lengths.append(tokens)
                                            # Track for outliers (we will sort later)
                                            outliers.append((i, tokens, t_text[:100] + "..."))
                                    except: continue
                            
                            if all_lengths:
                                df_len = pd.DataFrame(all_lengths, columns=["Tokens"])
                                # Create bins for histogram (e.g. increments of 128)
                                max_len = max(all_lengths)
                                bin_size = 128
                                bins = list(range(0, max_len + bin_size, bin_size))
                                df_len["Range"] = pd.cut(df_len["Tokens"], bins=bins, labels=[f"{b}-{b+bin_size}" for b in bins[:-1]])
                                
                                # Plot
                                chart_data = df_len["Range"].value_counts().sort_index()
                                st.bar_chart(chart_data, x_label="Token Range", y_label="Number of Examples")
                                
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1: st.metric("Total Examples", len(all_lengths))
                                with col_stat2: st.metric("Average Tokens", int(sum(all_lengths)/len(all_lengths)))
                                with col_stat3: st.metric("Max Tokens", max_len)
                                
                                # 🔍 Outlier Drill-down
                                sorted_outliers = sorted(outliers, key=lambda x: x[1], reverse=True)
                                
                                col_out1, col_out2 = st.columns(2)
                                with col_out1:
                                    st.markdown("#### ⚠️ Longest Examples")
                                    for ln, tok, snip in sorted_outliers[:5]:
                                        with st.expander(f"Line {ln}: **{tok} tokens**"):
                                            st.write(f"_{snip}_")
                                with col_out2:
                                    st.markdown("#### 🔍 Shortest Examples")
                                    for ln, tok, snip in sorted_outliers[-5:][::-1]:
                                        with st.expander(f"Line {ln}: **{tok} tokens**"):
                                            st.write(f"_{snip}_")

                                # Highlight warnings
                                if max_len > lora_max_seq:
                                    st.error(f"⚠️ **OOM Risk Warning:** Some examples ({max_len} tokens) exceed your 'Max sequence length' ({lora_max_seq}). These will be truncated during training, which can lead to poor model quality. Consider pruning these long entries at the line numbers mentioned above.")
                            else:
                                st.warning("Could not analyze sequence lengths.")

                    except Exception as e:
                        st.error(f"Failed to generate preview and analysis: {e}")

    _LORA_PROC = "lora_proc"
    _LORA_LINES = "vec_log_lora"

    # Retrieve running process (if any) and check if it's still alive
    lora_proc: subprocess.Popen | None = st.session_state.get(_LORA_PROC)
    is_training = lora_proc is not None and lora_proc.poll() is None

    chart_box_lora = st.empty()
    log_box_lora = st.empty()
    
    # Extract and plot training metrics from the log stream if they exist
    _lines = st.session_state.get(_LORA_LINES, [])
    if _lines:
        _metrics = []
        for _ln in _lines:
            if "'loss':" in _ln and "'epoch':" in _ln:
                try:
                    _loss = float(re.search(r"'loss':\s*'?([0-9.]+)'?", _ln).group(1))
                    _epoch = float(re.search(r"'epoch':\s*'?([0-9.]+)'?", _ln).group(1))
                    _metrics.append({"Epoch": _epoch, "Loss": _loss})
                except AttributeError:
                    pass
        if _metrics:
            _df = pd.DataFrame(_metrics).set_index("Epoch")
            chart_box_lora.line_chart(_df, y="Loss", use_container_width=True)

    if _lines:
        scrollable_log(log_box_lora, _lines[-2000:], title="LoRA Training")

    run_col_lora, cancel_col_lora, _ = st.columns([1, 1, 2])
    with run_col_lora:
        if st.button("🚀 Start Training", key="btn_lora_train",
                     type="primary", width="stretch", disabled=is_training):
            lora_script = Path("tools/finetune_lora.py").resolve()
            if not lora_script.exists():
                st.error(f"Script not found: {lora_script}")
            else:
                cmd = [
                    sys.executable, str(lora_script),
                    "--base-model", lora_base_model,
                    "--data", lora_data,
                    "--output", lora_output,
                    "--epochs", str(lora_epochs),
                    "--batch-size", str(lora_batch),
                    "--lr", str(lora_lr),
                    "--lora-rank", str(lora_rank),
                    "--lora-alpha", str(lora_alpha),
                    "--max-seq-length", str(lora_max_seq),
                    "--grad-accum", str(lora_grad_accum),
                ]
                if not lora_4bit:
                    cmd.append("--no-4bit")
                if lora_resume:
                    cmd.append("--resume")
                if lora_auto_export:
                    # Derives the Ollama name from the final folder component of the output directory
                    raw_out = lora_output.strip().split("/")[-1]
                    cmd.extend(["--ollama-name", raw_out])

                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    env=env,
                )
                lines: list[str] = []
                st.session_state[_LORA_PROC] = proc
                st.session_state[_LORA_LINES] = lines

                # Background thread pipes stdout into session state list
                def _read(p: subprocess.Popen, out: list[str]):
                    for ln in p.stdout:
                        out.append(ln.rstrip())

                threading.Thread(target=_read, args=(proc, lines), daemon=True).start()
                st.rerun()

    with cancel_col_lora:
        if is_training and st.button("⏹ Cancel", key="btn_lora_cancel",
                                     type="secondary", width="stretch"):
            lora_proc.terminate()
            st.session_state.pop(_LORA_PROC, None)
            st.rerun()

    # Poll while training: refresh every second to stream new log lines
    if is_training:
        lines = st.session_state.get(_LORA_LINES, [])
        st.spinner("Training — this may take a while…")
        time.sleep(1); st.rerun()
    elif lora_proc is not None:
        # Process just finished
        st.session_state.pop(_LORA_PROC, None)
        if lora_proc.returncode == 0:
            st.success(f"✅ Training complete! LoRA adapter saved to `{lora_output}`")
        elif lora_proc.returncode == -15:  # SIGTERM from cancel
            st.warning("Training cancelled.")
        else:
            st.error(f"Training failed (exit {lora_proc.returncode})")


def render_export_tab():
    st.markdown("### :material/inventory_2: Export & Compile Adapters")
    st.write("Manage your generated LoRA adapters and effortlessly compile them into native Ollama models.")
    
    adapters = get_available_adapters()
    
    if not adapters:
        st.info("No trained models found yet. Go to the Train tab to generate your first adapter!")
        return

    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.markdown("#### 1. Select Adapter")
        selected_adapter = st.selectbox("Trained LoRA Adapter", adapters, help="These are the fine-tuned adapters located in the ./models folder.")
        
        st.markdown("#### 2. Target Ollama Name")
        ollama_name = st.text_input("Model Name", value=f"my-{selected_adapter}", help="The name this model will have inside Ollama.")
        
        quant_method = st.selectbox("Quantization", ["q4_k_m", "q8_0", "f16"], index=0, help="4-bit (q4_k_m) is standard for local GPUs.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Compile to Ollama", type="primary", use_container_width=True):
            try:
                from tools.export_to_ollama import export_model
            except ImportError:
                st.error("Could not load export script (tools/export_to_ollama.py).")
                st.stop()
            
            adapter_path = f"./models/{selected_adapter}"
            out_path = f"./models/{selected_adapter}-gguf"
                
            with st.status(f"Compiling '{selected_adapter}' to '{ollama_name}'...", expanded=True) as status:
                st.write("⏳ Merging weights and converting to GGUF format (this will take a few minutes)...")
                try:
                    # Streamlit blocks while generating
                    export_model(
                        adapter_path=adapter_path,
                        out_path=out_path,
                        quant_method=quant_method,
                        ollama_name=ollama_name
                    )
                    status.update(label="✅ Compilation Complete!", state="complete", expanded=False)
                    st.success(f"Successfully added `{ollama_name}` to Ollama! You can now select it in the LLM Settings.")
                    st.balloons()
                except Exception as e:
                    status.update(label="❌ Compilation Failed", state="error", expanded=True)
                    st.error(f"Error during export: {e}")

    with c2:
        st.markdown("#### Available Adapters")
        for adapter in adapters:
            adapter_path = Path(f"models/{adapter}")
            # Get size in MB
            size_mb = sum(f.stat().st_size for f in adapter_path.glob('**/*') if f.is_file()) / (1024*1024)
            
            st.markdown(
                f"""
                <div style="
                    background: #1e2030;
                    border: 1px solid #2d3250;
                    border-left: 4px solid #8b5cf6;
                    border-radius: 8px;
                    padding: 12px 16px;
                    margin-bottom: 8px;
                ">
                    <div style="font-weight:600; font-size:1.05rem; color:#f1f5f9;">📦 {adapter}</div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Adapter Size: <b>{size_mb:.1f} MB</b></div>
                    <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Path: <code>./models/{adapter}</code></div>
                </div>
                """,
                unsafe_allow_html=True
            )

def render_models_tab():
    st.markdown("### :material/model_training: Models & Adapters")
    st.write("Train new LoRA adapters and compile them into native Ollama models.")
    st.divider()
    
    tab_train, tab_export = st.tabs(["🚀 LoRA Fine-tuning", "📦 Ollama Export"])
    
    with tab_train:
        render_finetune_tab()
        
    with tab_export:
        render_export_tab()
