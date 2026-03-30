"""
ui/models.py — Manage LoRA adapters, Fine-Tuning, and Ollama compilation

Directory layout:
    models/
    ├── adapters/   # LoRA adapters (local training output + downloaded from HF)
    └── merged/     # Merged safetensors output (adapter + base, fp16/bfloat16)

Base models are stored in the standard HuggingFace cache
(~/.cache/huggingface/hub/) and resolved via huggingface_hub APIs.
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
MERGED_DIR = Path("models/merged")

# Legacy flat paths (old layout without namespace subfolders)
_LEGACY_MODELS_DIR = Path("models")
_LEGACY_HF_CACHE = Path("models/_hf_cache")

_DEFAULT_SYSTEM_PROMPT = (
    "You are Romain. You write casually, mix French and English naturally, "
    "and prefer direct pragmatic answers. Reply as yourself — not as an AI."
)


def _is_adapter_dir(d: Path) -> bool:
    """True if d contains adapter_config.json at root or inside a checkpoint-* subfolder."""
    if not d.is_dir():
        return False
    if (d / "adapter_config.json").exists():
        return True
    for sub in d.iterdir():
        if sub.is_dir() and sub.name.startswith("checkpoint-"):
            if (sub / "adapter_config.json").exists():
                return True
    return False


def get_available_adapters() -> list[str]:
    """Return adapter names relative to models/adapters/.

    Supports both flat layout (models/adapters/my-model) and
    namespace layout (models/adapters/Namespace/my-model → 'Namespace/my-model').
    """
    adapters: list[str] = []

    if ADAPTERS_DIR.exists():
        for d in ADAPTERS_DIR.iterdir():
            if _is_adapter_dir(d):
                adapters.append(d.name)
            elif d.is_dir():
                # Namespace subfolder: models/adapters/Namespace/ModelName
                for sub in d.iterdir():
                    if _is_adapter_dir(sub):
                        adapters.append(f"{d.name}/{sub.name}")

    return sorted(adapters)


def _resolve_adapter_path(adapter_name: str) -> Path:
    """Resolve adapter name (may include namespace, e.g. 'Minar0/my-model') to full path."""
    return ADAPTERS_DIR / adapter_name


def _find_base_model(hf_id: str) -> Path | None:
    """Find a base model in the HuggingFace cache.

    Always prefers the non-BNB (fp16/bfloat16) variant, since BNB weights
    are uint8 and cannot be merged or imported by llama.cpp.

    Uses ``huggingface_hub`` to locate the cached snapshot.
    Returns the Path if found with safetensors, else None.
    """
    import re as _re
    clean_id = _re.sub(r"-bnb-\d+bit$", "", hf_id, flags=_re.IGNORECASE)

    try:
        from huggingface_hub import try_to_load_from_cache, scan_cache_dir
    except ImportError:
        return None

    for candidate_id in ([clean_id, hf_id] if clean_id != hf_id else [hf_id]):
        try:
            # Check if the repo exists in the HF cache
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == candidate_id:
                    # Find the latest revision snapshot path
                    for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
                        p = rev.snapshot_path
                        if any(p.glob("*.safetensors")):
                            return p
        except Exception:
            continue
    return None


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
                value="unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit",
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
        base_clean = base_clean.split(":")[0].split("/")[-1].lower() # e.g. unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit -> llama-3.2-3b-instruct
        
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
            "Epochs", min_value=1, max_value=20, value=2, step=1,
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
            index=2,  # default 8
            key="lora_rank",
            help="Higher rank = more parameters = more expressive but slower. "
                 "8 is a good starting point.",
        )

    col_lora3, col_lora4 = st.columns(2)
    with col_lora3:
        lora_alpha = st.number_input(
            "LoRA alpha", min_value=4, max_value=128, value=32, step=4,
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
            min_value=10, max_value=1000, value=50, step=10,
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

    st.caption(
        f"**Effective batch size:** {lora_batch * lora_grad_accum} "
        f"(batch {lora_batch} × accum {lora_grad_accum})"
    )

    with st.expander("👀 Preview formatted dataset"):
        st.caption("See how your JSONL data will be physically formatted for your selected base model's specific prompt template.")
        preview_system_prompt = st.text_area(
            "System prompt (clear to use the model's default)",
            value=_DEFAULT_SYSTEM_PROMPT,
            key="preview_system_prompt",
            height=80,
            help="Clear the field to fall back to the model's built-in default system prompt.",
        )
        st.info(
            "💡 **Instruct models:** It is not recommended to override the default system prompt "
            "during fine-tuning — instruct-tuned base models already have one baked in. "
            "You can safely customise the system prompt at **inference time** instead "
            "(e.g. in the Ollama `Modelfile`)."
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
                            st.info("Scanning full dataset — auto-splitting multi-turn examples that exceed max sequence length…")

                            _tok_obj = getattr(tokenizer, "tokenizer", tokenizer)

                            def _tokenize_msgs(msgs_list):
                                """Format messages with chat template and return token count."""
                                t = tokenizer.apply_chat_template(msgs_list, tokenize=False, add_generation_prompt=False)
                                return len(_tok_obj.encode(t, add_special_tokens=False))

                            def _sanitize_msgs(msgs):
                                """Sanitize message content, inject system prompt."""
                                sanitized = []
                                for m in msgs:
                                    if m["role"] == "system":
                                        continue
                                    content = m.get("content", "")
                                    if isinstance(content, list):
                                        content = "".join(
                                            b.get("text", "") for b in content
                                            if isinstance(b, dict) and b.get("type") == "text"
                                        )
                                    sanitized.append({"role": m["role"], "content": content})
                                if active_system:
                                    sanitized = [{"role": "system", "content": active_system}] + sanitized
                                return sanitized

                            def _split_single_turns(msgs_list):
                                """Split multi-turn into individual user/assistant pairs."""
                                sys_msgs = [m for m in msgs_list if m["role"] == "system"]
                                non_sys = [m for m in msgs_list if m["role"] != "system"]
                                pairs = []
                                i = 0
                                while i < len(non_sys) - 1:
                                    if non_sys[i]["role"] == "user" and non_sys[i+1]["role"] == "assistant":
                                        pairs.append(list(sys_msgs) + [non_sys[i], non_sys[i+1]])
                                        i += 2
                                    else:
                                        i += 1
                                return pairs

                            all_lengths = []
                            still_too_long = []  # (line_num, tokens, snippet, was_split)
                            num_auto_split = 0

                            with open(lora_data, "r", encoding="utf-8") as f_dist:
                                for i, line in enumerate(f_dist, 1):
                                    if not line.strip(): continue
                                    try:
                                        obj = json.loads(line)
                                        msgs = obj.get("conversations", obj.get("messages", []))
                                        if not msgs:
                                            continue
                                        sanitized_loop = _sanitize_msgs(msgs)
                                        tokens = _tokenize_msgs(sanitized_loop)

                                        if tokens <= lora_max_seq:
                                            all_lengths.append(tokens)
                                            continue

                                        # Too long — try auto-splitting multi-turn
                                        non_sys = [m for m in sanitized_loop if m["role"] != "system"]
                                        ua_pairs = sum(
                                            1 for j in range(len(non_sys) - 1)
                                            if non_sys[j]["role"] == "user" and non_sys[j+1]["role"] == "assistant"
                                        )

                                        if ua_pairs > 1:
                                            num_auto_split += 1
                                            for pair in _split_single_turns(sanitized_loop):
                                                try:
                                                    pair_tokens = _tokenize_msgs(pair)
                                                except Exception:
                                                    continue
                                                if pair_tokens <= lora_max_seq:
                                                    all_lengths.append(pair_tokens)
                                                else:
                                                    all_lengths.append(pair_tokens)
                                                    snip = " / ".join(
                                                        f"[{m['role']}] {m['content'][:80]}"
                                                        for m in pair if m["role"] != "system"
                                                    )
                                                    still_too_long.append((i, pair_tokens, snip, True))
                                        else:
                                            all_lengths.append(tokens)
                                            snip = " / ".join(
                                                f"[{m['role']}] {m['content'][:80]}"
                                                for m in sanitized_loop if m["role"] != "system"
                                            )
                                            still_too_long.append((i, tokens, snip, False))
                                    except:
                                        continue

                            if all_lengths:
                                df_len = pd.DataFrame(all_lengths, columns=["Tokens"])
                                max_len = max(all_lengths)
                                bin_size = 128
                                bins = list(range(0, max_len + bin_size, bin_size))
                                df_len["Range"] = pd.cut(df_len["Tokens"], bins=bins, labels=[f"{b}-{b+bin_size}" for b in bins[:-1]])

                                chart_data = df_len["Range"].value_counts().sort_index()
                                st.bar_chart(chart_data, x_label="Token Range", y_label="Number of Examples")

                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                with col_stat1: st.metric("Total Examples", f"{len(all_lengths):,}")
                                with col_stat2: st.metric("Average Tokens", int(sum(all_lengths)/len(all_lengths)))
                                with col_stat3: st.metric("Max Tokens", f"{max_len:,}")
                                with col_stat4: st.metric("Auto-split", f"{num_auto_split:,}")

                                # Show ALL examples that exceed max_seq_length
                                if still_too_long:
                                    sorted_too_long = sorted(still_too_long, key=lambda x: x[1], reverse=True)
                                    st.markdown(f"#### ⚠️ All examples exceeding {lora_max_seq} tokens ({len(sorted_too_long):,} total)")
                                    st.caption(
                                        "These examples will be **filtered out** during training. "
                                        "Multi-turn examples are auto-split into single-turn pairs first."
                                    )
                                    for ln, tok, snip, was_split in sorted_too_long:
                                        label = f"Line {ln}: **{tok:,} tokens**"
                                        if was_split:
                                            label += " _(split from multi-turn)_"
                                        with st.expander(label):
                                            st.write(snip)

                                    st.error(
                                        f"⚠️ **{len(sorted_too_long):,} examples** exceed max sequence length "
                                        f"({lora_max_seq} tokens) even after auto-splitting. "
                                        f"Consider increasing max sequence length or shortening these examples."
                                    )
                                else:
                                    st.success(f"✅ All examples fit within {lora_max_seq} tokens (after auto-splitting).")
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


def _get_merged_models() -> list[dict]:
    """Find merged models in models/merged/ that have a Modelfile.

    Supports both flat (merged/my-model) and namespace (merged/Minar0/my-model) layouts.
    Returns name as 'Minar0/my-model' for namespace entries.
    """
    if not MERGED_DIR.exists():
        return []
    models = []
    for d in sorted(MERGED_DIR.iterdir()):
        if not d.is_dir():
            continue
        if (d / "Modelfile").exists():
            size_mb = sum(f.stat().st_size for f in d.glob("**/*") if f.is_file()) / 1e6
            models.append({"name": d.name, "path": d, "size_mb": size_mb})
        else:
            # Namespace subfolder: merged/Minar0/my-model
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and (sub / "Modelfile").exists():
                    size_mb = sum(f.stat().st_size for f in sub.glob("**/*") if f.is_file()) / 1e6
                    models.append({"name": f"{d.name}/{sub.name}", "path": sub, "size_mb": size_mb})
    return models


def _read_adapter_base_model(adapter_path: Path) -> str | None:
    """Read base_model_name_or_path from adapter_config.json (root or first checkpoint)."""
    for cfg_p in [adapter_path / "adapter_config.json"] + sorted(
        adapter_path.glob("checkpoint-*/adapter_config.json")
    ):
        if cfg_p.exists():
            return json.loads(cfg_p.read_text()).get("base_model_name_or_path")
    return None


def render_export_tab():
    st.markdown("### :material/inventory_2: Export & Compile Adapters")

    # ── Step 1: Download adapter from HuggingFace (optional) ──────────────
    st.markdown("#### Step 1 — Download adapter from HuggingFace (optional)")
    st.caption("Skip if you already have the adapter locally in `models/adapters/`.")

    with st.expander("⬇️ Download adapter from HuggingFace"):
        dl_hf_id = st.text_input(
            "HuggingFace Adapter ID",
            placeholder="e.g. Minar0/my-ministral-3-8B-Instruct-2512-bnb-4bit-v1.0",
            key="dl_hf_id",
        )
        dl_checkpoint = st.text_input(
            "Checkpoint subfolder (optional)",
            placeholder="e.g. checkpoint-150",
            key="dl_checkpoint",
            help="Leave empty to download the full repo.",
        ) or None

        if dl_hf_id:
            dl_local = ADAPTERS_DIR / dl_hf_id   # keeps namespace: adapters/Minar0/my-model
            cp_flag = f"--include '{dl_checkpoint}/*' " if dl_checkpoint else ""
            st.code(
                f"huggingface-cli download {dl_hf_id} {cp_flag}"
                f"--local-dir {dl_local}",
                language="bash",
            )

            if st.button("⬇️ Download now", key="btn_dl_adapter"):
                with st.spinner(f"Downloading {dl_hf_id}…"):
                    try:
                        from huggingface_hub import snapshot_download
                        kwargs: dict = {"repo_id": dl_hf_id, "local_dir": str(dl_local)}
                        if dl_checkpoint:
                            kwargs["allow_patterns"] = [f"{dl_checkpoint}/*"]
                        snapshot_download(**kwargs)
                        st.success(f"✅ Downloaded to `{dl_local}`")
                    except Exception as e:
                        st.error(f"Download failed: {e}")

    st.divider()

    # ── Step 2: Select adapter + merge with base model ─────────────────────
    st.markdown("#### Step 2 — Merge adapter with base model")

    adapters = get_available_adapters()

    if not adapters:
        st.info("No adapters found in `models/adapters/`. Use Step 1 to download one.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        selected_adapter = st.selectbox(
            "LoRA Adapter",
            adapters,
            help="Adapters in models/adapters/ (supports Namespace/ModelName layout).",
        )
    adapter_dir_path = _resolve_adapter_path(selected_adapter)

    with col_b:
        checkpoints = sorted(
            [d.name for d in adapter_dir_path.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
        ) if adapter_dir_path.exists() else []

        checkpoint: str | None = None
        if checkpoints:
            cp_options = ["Latest (final weights)"] + checkpoints
            selected_cp = st.selectbox("Checkpoint", cp_options)
            if selected_cp != "Latest (final weights)":
                checkpoint = selected_cp
        else:
            st.caption("No checkpoints found — using final weights.")

    # Resolve base model
    base_model_hf_id = _read_adapter_base_model(adapter_dir_path)
    base_model_path: Path | None = None
    base_status_msg = ""

    if base_model_hf_id:
        clean_id = re.sub(r"-bnb-\d+bit$", "", base_model_hf_id, flags=re.IGNORECASE)
        base_model_path = _find_base_model(base_model_hf_id)
        if base_model_path:
            base_status_msg = f"✅ Base model: `{base_model_path}`"
        else:
            # Need to download the fp16 version into the HF cache
            st.warning(
                f"Base model not found in HuggingFace cache. The adapter requires `{base_model_hf_id}`. "
                f"The fp16 version (`{clean_id}`) will be downloaded to the HF cache.",
            )
            if st.button("⬇️ Download base model", key="btn_dl_base"):
                with st.spinner(f"Downloading {clean_id} to HF cache…"):
                    try:
                        from huggingface_hub import snapshot_download
                        cached_path = snapshot_download(repo_id=clean_id)
                        st.success(f"✅ Downloaded to HF cache: `{cached_path}`")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Download failed: {e}")

    if base_status_msg:
        st.caption(base_status_msg)

    # Build output path
    # Keep namespace tree: merged/Minar0/my-model-checkpoint-150
    _adapter_ns, _adapter_model = (
        selected_adapter.rsplit("/", 1) if "/" in selected_adapter
        else (None, selected_adapter)
    )
    # Strip BNB quantization suffix — the merged model uses fp16/bfloat16 weights
    _adapter_model_clean = _adapter_model.replace("-bnb-4bit", "")
    cp_suffix = f"-{checkpoint}" if checkpoint else ""
    merge_out_name = f"{_adapter_model_clean}{cp_suffix}"
    merge_out_path = (MERGED_DIR / _adapter_ns / merge_out_name) if _adapter_ns else (MERGED_DIR / merge_out_name)

    st.caption(f"Output: `{merge_out_path}`")

    # Merge process state
    _MERGE_PROC = "merge_proc"
    _MERGE_LINES = "merge_lines"
    merge_proc: subprocess.Popen | None = st.session_state.get(_MERGE_PROC)
    is_merging = merge_proc is not None and merge_proc.poll() is None

    merge_log_box = st.empty()
    merge_lines: list[str] = st.session_state.get(_MERGE_LINES, [])
    if merge_lines:
        scrollable_log(merge_log_box, merge_lines[-500:], title="Merge Progress")

    merge_col, cancel_col, _ = st.columns([2, 1, 1])
    with merge_col:
        if st.button(
            "🔀 Merge LoRA into Base Model",
            type="primary", width="stretch",
            disabled=base_model_path is None or is_merging,
            key="btn_merge_lora",
        ):
            script = Path("tools/merge_lora_safetensors.py").resolve()
            adapter_cp_path = (
                adapter_dir_path / checkpoint if checkpoint else adapter_dir_path
            )
            cmd = [
                sys.executable, str(script),
                "--adapter", str(adapter_cp_path),
                "--base", str(base_model_path),
                "--output", str(merge_out_path),
                "--name", merge_out_name,
            ]
            new_lines: list[str] = []
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            st.session_state[_MERGE_PROC] = proc
            st.session_state[_MERGE_LINES] = new_lines
            threading.Thread(
                target=lambda p, out: [out.append(ln.rstrip()) for ln in p.stdout],
                args=(proc, new_lines), daemon=True,
            ).start()
            st.rerun()

    with cancel_col:
        if is_merging and st.button("⏹ Cancel", key="btn_merge_cancel",
                                    type="secondary", width="stretch"):
            merge_proc.terminate()
            st.session_state.pop(_MERGE_PROC, None)
            st.rerun()

    if is_merging:
        time.sleep(1); st.rerun()
    elif merge_proc is not None:
        st.session_state.pop(_MERGE_PROC, None)
        if merge_proc.returncode == 0:
            st.success(f"✅ Merged model saved to `{merge_out_path}`. Proceed to Step 3.")
            st.session_state.pop(_MERGE_LINES, None)
        elif merge_proc.returncode != -15:
            st.error(f"Merge failed (exit {merge_proc.returncode})")

    st.divider()

    # ── Step 3: Export to Ollama ───────────────────────────────────────────
    st.markdown("#### Step 3 — Export to Ollama")

    merged_models = _get_merged_models()

    if not merged_models:
        st.info("No merged models found in `models/merged/`. Complete Step 2 first.")
        return

    merged_names = [m["name"] for m in merged_models]
    # Pre-select the model just merged (if any)
    _full_merge_name = f"{_adapter_ns}/{merge_out_name}" if _adapter_ns else merge_out_name
    default_idx = merged_names.index(_full_merge_name) if _full_merge_name in merged_names else 0
    selected_merged = st.selectbox(
        "Merged Model", merged_names, index=default_idx,
    )
    merged_info = next(m for m in merged_models if m["name"] == selected_merged)
    st.caption(f"📦 {merged_info['size_mb']:.0f} MB — `{merged_info['path']}`")

    _default_name = re.sub(r"[^a-z0-9\-:.]", "-", selected_merged.lower())
    _default_name = re.sub(r"-{2,}", "-", _default_name).strip("-")

    col_name, col_quant = st.columns(2)
    with col_name:
        ollama_name = st.text_input(
            "Ollama Model Name",
            value=_default_name,
            help="Lowercase, hyphens only.",
        )
    with col_quant:
        quant_method = st.selectbox(
            "Quantization", ["Q4_K_M", "Q8_0", "f16"], index=0,
            help="4-bit is standard. f16 = no quantization (large file).",
        )

    if st.button("🚀 Register in Ollama", type="primary", width="stretch",
                 disabled=not ollama_name):
        merged_path = str(merged_info["path"])
        cmd = ["ollama", "create", ollama_name, "-f", "Modelfile"]
        if quant_method != "f16":
            cmd.extend(["--quantize", quant_method])

        with st.status(f"Creating '{ollama_name}' in Ollama…", expanded=True) as status:
            st.write(f"`{' '.join(cmd)}`")
            st.write(f"Directory: `{merged_path}`")
            if quant_method != "f16":
                st.write(f"⏳ Quantizing to {quant_method} — this may take 10-15 minutes…")
            try:
                subprocess.run(cmd, cwd=merged_path, check=True)
                status.update(label="✅ Model registered!", state="complete", expanded=False)
                st.success(f"Added `{ollama_name}` to Ollama. Select it in LLM Settings.")
                st.balloons()
            except Exception as e:
                status.update(label="❌ Failed", state="error", expanded=True)
                st.error(f"Error: {e}")
                st.code(f"cd {merged_path} && {' '.join(cmd)}", language="bash")

def render_training_data_tab():
    st.markdown("### :material/model_training: Training Data")
    st.markdown(
        "Filter conversations (last N years, English only, no system messages) "
        "and export as **JSONL** for LLM fine-tuning. Each example is a "
        "`user → assistant` exchange pair."
    )

    col_ft1, col_ft2 = st.columns(2)
    with col_ft1:
        ft_json = st.text_input(
            "Source JSON",
            value="./data/facebook/facebook_messages.json",
            key="ft_json_input",
        )
        ft_years = st.number_input(
            "Years to include",
            min_value=1, max_value=20, value=3, step=1,
            key="ft_years",
            help="Only include messages from the last N years.",
        )
        ft_min_words = st.number_input(
            "Min words per message",
            min_value=1, max_value=20, value=2, step=1,
            key="ft_min_words",
            help="Skip messages shorter than this.",
        )
    with col_ft2:
        ft_output = st.text_input(
            "Output JSONL",
            value="./data/facebook/finetune_data.jsonl",
            key="ft_output",
        )
        ft_language = st.selectbox(
            "Language",
            options=["en", "fr"],
            index=0,
            key="ft_language",
            help="Keep only messages in this language.",
        )
        ft_max_words = st.number_input(
            "Max words (assistant reply)",
            min_value=10, max_value=1000, value=200, step=10,
            key="ft_max_words",
            help="Skip pairs where your reply exceeds this word count. "
                 "This is conversational fine-tuning, not long-form memory.",
        )
        ft_max_turns = st.number_input(
            "Max turns per example",
            min_value=1, max_value=10, value=1, step=1,
            key="ft_max_turns",
            help="1 = single user→assistant pair. "
                 "2+ = multi-turn conversations per training example.",
        )
        ft_max_reply_gap = st.number_input(
            "Max gap between discussions (min)",
            min_value=1, max_value=1440, value=30, step=5,
            key="ft_max_reply_gap",
            help="If the gap between the last user message and your first reply exceeds this, "
                 "the exchange is considered a different discussion and skipped.",
        )
        ft_filter_reactions = st.checkbox(
            "Filter out reactions", value=True, key="ft_filter_reactions",
            help="If checked, standalone Facebook reactions are excluded from the training data."
        )

    with st.expander("🔍 Artifact filter patterns", expanded=False):
        from tools.export_finetune import DEFAULT_ARTIFACT_PATTERNS
        st.caption(
            "Messages matching any of these regex patterns are considered low-quality "
            "artifacts and excluded from the training data. One pattern per line."
        )
        _default_patterns_text = "\n".join(DEFAULT_ARTIFACT_PATTERNS)
        ft_artifact_patterns_text = st.text_area(
            "Artifact patterns (regex, one per line)",
            value=_default_patterns_text,
            key="ft_artifact_patterns",
            height=200,
            help="Each line is a regex pattern. Messages matching any pattern are excluded.",
        )

    st.caption(
        "**Format:** `{\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, "
        "{\"role\": \"assistant\", \"content\": \"...\"}]}`\n\n"
        "• **user** = the other person's message\n"
        "• **assistant** = your reply"
    )

    log_key_ft = "vec_log_finetune"

    run_col_ft, _ = st.columns([1, 3])
    with run_col_ft:
        do_export = st.button("Export Fine-tune Data", key="btn_finetune",
                     icon=":material/play_arrow:", width="stretch")

    log_box_ft = st.empty()
    if do_export:
        from tools.export_finetune import export_finetune_data

        lines_ft: list[str] = []
        progress_ft = st.progress(0, text="Exporting fine-tune data…")

        def _ft_cb(current: int, total: int):
            pct = current / total if total > 0 else 0
            progress_ft.progress(pct, text=f"Processing conversations… {current:,}/{total:,}")

        try:
            # Parse artifact patterns from text area
            _custom_patterns = [
                line.strip() for line in ft_artifact_patterns_text.splitlines()
                if line.strip()
            ]
            stats = export_finetune_data(
                json_path=ft_json,
                output_path=ft_output,
                years=ft_years,
                language=ft_language,
                min_words=ft_min_words,
                max_words=ft_max_words,
                max_turns=ft_max_turns,
                max_reply_gap_min=ft_max_reply_gap,
                filter_reactions=ft_filter_reactions,
                artifact_patterns=_custom_patterns if _custom_patterns != DEFAULT_ARTIFACT_PATTERNS else None,
                progress_callback=_ft_cb,
            )
            progress_ft.empty()

            lines_ft.append(f"Self name: {stats['self_name']}")
            lines_ft.append(f"Total messages: {stats['total_messages']:,}")
            lines_ft.append(f"After filtering ({stats['years']}y, {stats['language']}): "
                            f"{stats['filtered_messages']:,}")
            lines_ft.append(f"Training examples exported: {stats['pairs_exported']:,}")
            if stats.get('skipped_too_long'):
                lines_ft.append(f"Skipped (reply > {stats['max_words']} words): "
                                f"{stats['skipped_too_long']:,}")
            lines_ft.append(f"Conversations used: {stats['conversations_used']:,}")
            lines_ft.append(f"Output: {stats['output_file']}")

            st.session_state[log_key_ft] = lines_ft
            scrollable_log(log_box_ft, lines_ft, follow=False, title="Fine-tune Export")

            st.success(
                f"Exported **{stats['pairs_exported']:,}** training examples "
                f"from **{stats['conversations_used']:,}** conversations → "
                f"`{stats['output_file']}`"
            )

        except FileNotFoundError as e:
            progress_ft.empty()
            st.error(str(e))
        except Exception as e:
            progress_ft.empty()
            st.error(f"Export failed: {e}")

    # Show previous log if available
    if not do_export and st.session_state.get(log_key_ft):
        scrollable_log(log_box_ft, st.session_state[log_key_ft], follow=False, title="Fine-tune Export")

    # Interactive dataset preview (persistent)
    output_p = Path(ft_output)
    if output_p.exists():
        st.divider()

        st.markdown("### Interactive Dataset Preview")
        st.caption("*(Note: To see the effects of changing settings like Max Turns, you must click **Export** again to regenerate the file!)*")

        @st.cache_data(show_spinner=False)
        def _load_preview_data(filepath, mtime):
            examples_by_conv = {}
            try:
                with open(filepath, "r", encoding="utf-8") as pf:
                    for line in pf:
                        if not line.strip(): continue
                        ex = json.loads(line)
                        conv = ex.get("conversation") or "Unknown Conversation"
                        if conv not in examples_by_conv:
                            examples_by_conv[conv] = []
                        examples_by_conv[conv].append(ex)
            except Exception:
                pass
            return examples_by_conv

        preview_data = _load_preview_data(str(output_p), os.path.getmtime(output_p))
        if preview_data:
            if "preview_conv" not in st.session_state or st.session_state.preview_conv not in preview_data:
                st.session_state.preview_conv = list(preview_data.keys())[0]
            if "preview_idx" not in st.session_state:
                st.session_state.preview_idx = 0

            convs = list(preview_data.keys())

            p_col1, p_col2, p_col3 = st.columns([3, 1, 1])
            with p_col1:
                selected_conv = st.selectbox(
                    "Conversation",
                    options=convs,
                    index=convs.index(st.session_state.preview_conv) if st.session_state.preview_conv in convs else 0,
                    key="preview_conv_select"
                )
            with p_col2:
                st.markdown("<div style='margin-top: 28px'></div>", unsafe_allow_html=True)
                if st.button("⬅ Previous", key="btn_prev_ex", width="stretch"):
                    st.session_state.preview_idx -= 1
            with p_col3:
                st.markdown("<div style='margin-top: 28px'></div>", unsafe_allow_html=True)
                if st.button("Next ➡", key="btn_next_ex", width="stretch"):
                    st.session_state.preview_idx += 1

            if selected_conv != st.session_state.preview_conv:
                st.session_state.preview_conv = selected_conv
                st.session_state.preview_idx = 0
                st.rerun()

            conv_examples = preview_data[st.session_state.preview_conv]
            idx = st.session_state.preview_idx % len(conv_examples)
            example = conv_examples[idx]

            st.caption(f"**Showing example {idx + 1} of {len(conv_examples)}** (from {st.session_state.preview_conv})")

            msgs = example.get("messages", [])
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                with st.chat_message(role):
                    st.markdown(content)

            # Shuffle button — after the preview
            if st.button("🔀 Shuffle to a random example", key="btn_shuffle_ex"):
                import random
                st.session_state.preview_conv = random.choice(convs)
                conv_examples_for_shuffle = preview_data[st.session_state.preview_conv]
                st.session_state.preview_idx = random.randint(0, len(conv_examples_for_shuffle) - 1)
                st.rerun()

        # ── Token distribution chart (after preview) ──────────────────
        @st.cache_data(show_spinner=False)
        def _load_token_stats(filepath, mtime):
            rows = []
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        ex = json.loads(line)
                        tokens = sum(
                            len(m.get("content", "")) // 4
                            for m in ex.get("messages", [])
                        )
                        rows.append((ex.get("conversation") or "Unknown", tokens))
            except Exception:
                pass
            return rows

        token_rows = _load_token_stats(str(output_p), os.path.getmtime(output_p))
        if token_rows:
            import random as _rng
            import plotly.graph_objects as go

            # Shuffle with fixed seed to simulate training order
            shuffled_rows = list(token_rows)
            _rng.Random(42).shuffle(shuffled_rows)
            y_tokens = [r[1] for r in shuffled_rows]
            hover_convs = [r[0] for r in shuffled_rows]

            fig_tok = go.Figure(data=[
                go.Bar(
                    x=list(range(len(y_tokens))),
                    y=y_tokens,
                    marker_color="#6366f1",
                    hovertemplate="Row %{x}<br><b>%{customdata}</b><br>~%{y:,} tokens<extra></extra>",
                    customdata=hover_convs,
                )
            ])
            fig_tok.update_layout(
                title=f"Tokens per dataset row (shuffled, seed=42) — {len(y_tokens):,} rows",
                title_font=dict(size=14, color="#e2e8f0"),
                xaxis=dict(title="Dataset rows (shuffled training order)", gridcolor="#333", showticklabels=False),
                yaxis=dict(title="Approx. tokens", gridcolor="#333"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=20, t=40, b=20),
                height=300,
            )
            st.plotly_chart(fig_tok, use_container_width=True)


def render_models_tab():
    st.markdown("### :material/model_training: Models & Adapters")
    st.write("Train new LoRA adapters and compile them into native Ollama models.")
    st.divider()
    
    tab_data, tab_train, tab_export = st.tabs(["📊 Training Data", "🚀 LoRA Fine-tuning", "📦 Ollama Export"])
    
    with tab_data:
        render_training_data_tab()

    with tab_train:
        render_finetune_tab()
        
    with tab_export:
        render_export_tab()
