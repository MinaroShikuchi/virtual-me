"""
ui/models.py — Manage LoRA adapters, Fine-Tuning, and Ollama compilation

Directory layout:
    models/
    ├── adapters/   # LoRA adapters (local training output + downloaded from HF)
    ├── base/       # Base models (downloaded once, reused across exports)
    └── gguf/       # Merged GGUF output (for Ollama import)
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
    import warnings as _w
    with _w.catch_warnings():
        _w.filterwarnings("ignore", message=".*Unsloth should be imported before.*")
        import unsloth
        from unsloth import get_chat_template
except (ImportError, NotImplementedError):
    unsloth = None
    get_chat_template = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# ── Directory constants ───────────────────────────────────────────────────────

ADAPTERS_DIR = Path("models/adapters")
BASE_DIR = Path("models/base")
GGUF_DIR = Path("models/gguf")

# Legacy paths for backward compatibility
_LEGACY_MODELS_DIR = Path("models")
_LEGACY_HF_CACHE = Path("models/_hf_cache")


def _is_adapter_dir(d: Path) -> bool:
    """Check if a directory is a valid adapter (has adapter_config.json at root or in checkpoints)."""
    if not d.is_dir():
        return False
    # Check root level
    if (d / "adapter_config.json").exists():
        return True
    # Check inside checkpoint-* subdirectories
    for sub in d.iterdir():
        if sub.is_dir() and sub.name.startswith("checkpoint-"):
            if (sub / "adapter_config.json").exists():
                return True
    return False


def get_available_adapters():
    """Find all folders in models/adapters/ that contain an adapter_config.json.

    Checks both the root level and checkpoint-* subdirectories.
    Also checks the legacy flat models/ directory for backward compatibility.
    """
    adapters = []

    # Primary location: models/adapters/
    if ADAPTERS_DIR.exists() and ADAPTERS_DIR.is_dir():
        for d in ADAPTERS_DIR.iterdir():
            if _is_adapter_dir(d):
                adapters.append(d.name)

    # Backward compatibility: check legacy models/ (flat layout)
    if _LEGACY_MODELS_DIR.exists() and _LEGACY_MODELS_DIR.is_dir():
        for d in _LEGACY_MODELS_DIR.iterdir():
            if (d.name not in ("adapters", "base", "gguf", "_hf_cache")
                    and _is_adapter_dir(d)
                    and d.name not in adapters):
                adapters.append(d.name)

    return sorted(adapters)


def _resolve_adapter_path(adapter_name: str) -> Path:
    """Resolve an adapter name to its full path.

    Checks models/adapters/ first, then falls back to legacy models/ layout.
    """
    new_path = ADAPTERS_DIR / adapter_name
    if new_path.exists():
        return new_path

    # Legacy fallback
    legacy_path = _LEGACY_MODELS_DIR / adapter_name
    if legacy_path.exists():
        return legacy_path

    return new_path  # default to new path even if it doesn't exist


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
            value=f"./models/adapters/{default_export_name}",
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
        lora_save_steps = st.number_input(
            "Save checkpoint every N steps",
            min_value=10, max_value=1000, value=100, step=10,
            key="lora_save_steps",
            help="Save a checkpoint every N training steps. Lower values = more checkpoints "
                 "to compare, but uses more disk space.",
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
        _DEFAULT_SYSTEM_PROMPT = (
            "You are Romain. You write casually, mix French and English naturally, "
            "and prefer direct pragmatic answers. Reply as yourself — not as an AI."
        )
        preview_system_prompt = st.text_area(
            "System prompt (clear to use the model's default)",
            value=_DEFAULT_SYSTEM_PROMPT,
            key="preview_system_prompt",
            height=80,
            help="Clear the field to fall back to the model's built-in default system prompt.",
        )
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
                                if m["role"] == "system":
                                    continue  # will be re-injected below
                                content = m.get("content", "")
                                if isinstance(content, list):
                                    content = "".join(
                                        block.get("text", "") for block in content
                                        if isinstance(block, dict) and block.get("type") == "text"
                                    )
                                sanitized_preview.append({"role": m["role"], "content": content})

                            # Determine system prompt: UI input > JSONL > none
                            sys_from_jsonl = next((m["content"] for m in messages if m["role"] == "system"), None)
                            active_system = preview_system_prompt.strip() or sys_from_jsonl or None
                            if active_system:
                                sanitized_preview = [{"role": "system", "content": active_system}] + sanitized_preview

                            # Robust chat template check for Base models
                            if tokenizer.chat_template is None:
                                if get_chat_template is not None:
                                    try:
                                        tokenizer = get_chat_template(
                                            tokenizer,
                                            chat_template="llama-3",
                                        )
                                    except Exception as ex_templ:
                                        st.warning(f"Unsloth template injection failed: {ex_templ}")
                                        tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n'}}{% endfor %}"
                                else:
                                    tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n'}}{% endfor %}"

                            st.markdown("### System prompt")
                            if active_system:
                                st.code(active_system, language=None)
                            else:
                                st.caption("*(none — model default will be used)*")

                            st.markdown("### Messages")
                            for m in sanitized_preview:
                                if m["role"] == "system":
                                    continue
                                with st.chat_message(m["role"]):
                                    st.write(m["content"])

                            formatted_text = tokenizer.apply_chat_template(sanitized_preview, tokenize=False, add_generation_prompt=False)
                            st.markdown("### Formatted token string")
                            st.code(formatted_text, language=None)

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
                                            sanitized_loop = []
                                            for m in msgs:
                                                if m["role"] == "system":
                                                    continue
                                                content = m.get("content", "")
                                                if isinstance(content, list):
                                                    content = "".join(
                                                        b.get("text", "") for b in content
                                                        if isinstance(b, dict) and b.get("type") == "text"
                                                    )
                                                sanitized_loop.append({"role": m["role"], "content": content})
                                            if active_system:
                                                sanitized_loop = [{"role": "system", "content": active_system}] + sanitized_loop

                                            # Estimate length via template + exact tokenization
                                            t_text = tokenizer.apply_chat_template(sanitized_loop, tokenize=False, add_generation_prompt=False)
                                            _tok_obj = getattr(tokenizer, "tokenizer", tokenizer)
                                            tokens = len(_tok_obj.encode(t_text, add_special_tokens=False))
                                            all_lengths.append(tokens)
                                            msg_snippet = " / ".join(
                                                f"[{m['role']}] {m['content'][:80]}"
                                                for m in sanitized_loop
                                                if m["role"] != "system"
                                            )
                                            outliers.append((i, tokens, msg_snippet))
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
            chart_box_lora.line_chart(_df, y="Loss", width="stretch")

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
                    "--save-steps", str(lora_save_steps),
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


def _get_available_gguf():
    """Find available GGUF/safetensors models in models/gguf/ with a Modelfile."""
    if not GGUF_DIR.exists():
        return []
    models = []
    for d in sorted(GGUF_DIR.iterdir()):
        if d.is_dir() and (d / "Modelfile").exists():
            # Get size
            size_mb = sum(
                f.stat().st_size for f in d.glob("**/*") if f.is_file()
            ) / (1024 * 1024)
            models.append({"name": d.name, "path": d, "size_mb": size_mb})
    return models


def render_export_tab():
    st.markdown("### :material/inventory_2: Export & Compile Adapters")
    st.write("Download adapters, merge with base models, and register in Ollama.")

    adapters = get_available_adapters()

    # ── Section 1: Model Source ────────────────────────────────────────────
    st.markdown("#### 1. Model Source")

    source_options = ["Local Adapter", "HuggingFace Model"]
    default_idx = 0 if adapters else 1

    source_mode = st.radio(
        "Source",
        source_options,
        index=default_idx,
        horizontal=True,
        help="Local adapters (from models/adapters/) or download from HuggingFace.",
    )

    hf_model_id = None
    selected_adapter = None
    checkpoint = None

    if source_mode == "HuggingFace Model":
        hf_model_id = st.text_input(
            "HuggingFace Model ID",
            placeholder="e.g. Minar0/my-ministral-3-8B-Instruct-2512-bnb-4bit-v1.0",
            help="The adapter will be downloaded to models/adapters/.",
        )
        checkpoint = st.text_input(
            "Checkpoint (optional)",
            placeholder="e.g. checkpoint-400",
            help="Download a specific checkpoint subfolder from the repo.",
        ) or None

        # Show adapter info if entered
        if hf_model_id:
            st.caption(f"Will download to: `models/adapters/{hf_model_id.replace('/', '_')}/`")

    elif not adapters:
        st.info("No trained adapters found in `models/adapters/`. "
                "Train a model first or download from HuggingFace.")
    else:
        selected_adapter = st.selectbox(
            "Trained LoRA Adapter", adapters,
            help="Fine-tuned adapters in models/adapters/.",
        )

        # List available checkpoints
        adapter_dir = _resolve_adapter_path(selected_adapter)
        checkpoints = sorted(
            [d.name for d in adapter_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
        ) if adapter_dir.exists() else []

        if checkpoints:
            checkpoint_options = ["Latest (final weights)"] + checkpoints
            selected_cp = st.selectbox(
                "Checkpoint", checkpoint_options,
                help="Select a specific training checkpoint.",
            )
            if selected_cp != "Latest (final weights)":
                checkpoint = selected_cp

        # Show adapter info card
        if adapter_dir.exists():
            size_mb = sum(
                f.stat().st_size for f in adapter_dir.glob("**/*") if f.is_file()
            ) / (1024 * 1024)
            # Try to read base model from adapter_config
            base_model_name = "unknown"
            for cfg_path in [adapter_dir / "adapter_config.json"] + list(
                adapter_dir.glob("checkpoint-*/adapter_config.json")
            ):
                if cfg_path.exists():
                    import json as _json
                    cfg = _json.loads(cfg_path.read_text())
                    base_model_name = cfg.get("base_model_name_or_path", "unknown")
                    break
            st.caption(
                f"📦 **{selected_adapter}** — {size_mb:.0f} MB — "
                f"Base: `{base_model_name}` — "
                f"{len(checkpoints)} checkpoint(s)"
            )

    # Download button
    can_download = bool(hf_model_id) if source_mode == "HuggingFace Model" else bool(selected_adapter)
    if source_mode == "HuggingFace Model" and hf_model_id:
        if st.button("⬇️ Download Adapter + Base Model", type="secondary",
                     width="stretch"):
            with st.status("Downloading…", expanded=True) as status:
                try:
                    from tools.export_to_ollama import export_model
                    adapter_path = str(ADAPTERS_DIR / hf_model_id.replace("/", "_"))
                    out_path = str(GGUF_DIR / hf_model_id.split("/")[-1])
                    export_model(
                        adapter_path=adapter_path,
                        out_path=out_path,
                        quant_method="q4_k_m",
                        ollama_name=None,  # Don't register yet
                        base_model=hf_model_id,
                        checkpoint=checkpoint,
                    )
                    status.update(label="✅ Download & merge complete!", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Download failed", state="error")
                    st.error(f"Error: {e}")
    elif source_mode == "Local Adapter" and selected_adapter:
        if st.button("🔄 Merge Adapter + Base Model → GGUF", type="secondary",
                     width="stretch"):
            with st.status("Merging…", expanded=True) as status:
                try:
                    from tools.export_to_ollama import export_model
                    adapter_dir = _resolve_adapter_path(selected_adapter)
                    out_path = str(GGUF_DIR / selected_adapter)
                    export_model(
                        adapter_path=str(adapter_dir),
                        out_path=out_path,
                        quant_method="q4_k_m",
                        ollama_name=None,  # Don't register yet
                        checkpoint=checkpoint,
                    )
                    status.update(label="✅ Merge complete!", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Merge failed", state="error")
                    st.error(f"Error: {e}")

    st.divider()

    # ── Section 2: Export to Ollama ────────────────────────────────────────
    st.markdown("#### 2. Export to Ollama")

    available_gguf = _get_available_gguf()

    if not available_gguf:
        st.info("No merged models found in `models/gguf/`. "
                "Use Section 1 to download and merge an adapter first.")
    else:
        gguf_names = [m["name"] for m in available_gguf]
        selected_gguf = st.selectbox(
            "Merged Model", gguf_names,
            help="Select a merged model from models/gguf/.",
        )

        # Show info
        gguf_info = next(m for m in available_gguf if m["name"] == selected_gguf)
        st.caption(f"📦 {gguf_info['size_mb']:.0f} MB — `models/gguf/{selected_gguf}/`")

        col_name, col_quant = st.columns(2)
        with col_name:
            ollama_name = st.text_input(
                "Ollama Model Name",
                value=f"my-{selected_gguf}",
                help="The name this model will have in Ollama.",
            )
        with col_quant:
            quant_method = st.selectbox(
                "Quantization", ["q4_K_M", "q8_0", "f16"], index=0,
                help="4-bit (q4_K_M) is standard. f16 = no quantization.",
            )

        if st.button("🚀 Register in Ollama", type="primary", width="stretch",
                     disabled=not ollama_name):
            import subprocess
            gguf_path = str(gguf_info["path"])
            cmd = ["ollama", "create", ollama_name, "-f", "Modelfile"]
            if quant_method != "f16":
                cmd.extend(["--quantize", quant_method])

            with st.status(f"Creating '{ollama_name}' in Ollama…", expanded=True) as status:
                st.write(f"Running: `{' '.join(cmd)}`")
                st.write(f"Working directory: `{gguf_path}`")
                if quant_method != "f16":
                    st.write(f"⏳ Quantizing to {quant_method} — this may take 10-15 minutes…")
                try:
                    subprocess.run(cmd, cwd=gguf_path, check=True)
                    status.update(label="✅ Model registered!", state="complete",
                                  expanded=False)
                    st.success(f"Successfully added `{ollama_name}` to Ollama! "
                               f"Select it in LLM Settings.")
                    st.balloons()
                except Exception as e:
                    status.update(label="❌ Failed", state="error", expanded=True)
                    st.error(f"Error: {e}")
                    st.code(f"cd {gguf_path} && {' '.join(cmd)}", language="bash")

def render_models_tab():
    st.markdown("### :material/model_training: Models & Adapters")
    st.write("Train new LoRA adapters and compile them into native Ollama models.")
    st.divider()
    
    tab_train, tab_export = st.tabs(["🚀 LoRA Fine-tuning", "📦 Ollama Export"])
    
    with tab_train:
        render_finetune_tab()
        
    with tab_export:
        render_export_tab()
