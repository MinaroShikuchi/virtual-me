"""
ui/ingest.py — Data page (episodic memory & ingestion).

Sections:
  1. Connection status + collection stats (per-source document counts)
  2. Run ingestors (extract → ingest pipeline with live log streaming)
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import streamlit as st

from config import CHROMA_PATH, SOURCES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scrollable_log(container, lines: list[str], max_height: int = 300):
    """Render log lines inside a scrollable container with reversed layout so newest is at the top."""
    escaped = "\n".join(lines[::-1]).replace("&", "&").replace("<", "<").replace(">", ">")
    container.markdown(
        f'<div style="max-height:{max_height}px;overflow-y:auto;'
        f'background:#0e1117;border:1px solid #333;border-radius:6px;'
        f'padding:10px;font-family:monospace;font-size:13px;'
        f'white-space:pre-wrap;color:#ccc;">{escaped}</div>',
        unsafe_allow_html=True,
    )


# ── Main render function ─────────────────────────────────────────────────────

def render_vector_tab(collection, episodic=None):
    st.markdown("### :material/database: Data")
    st.caption("Episodic memory stored in ChromaDB — extract your data exports and embed them as searchable documents.")

    # ── Connection banner ──
    try:
        total_count = collection.count()
        st.markdown(
            f'<span class="status-ok">● ChromaDB</span> — connected at '
            f'<code>{CHROMA_PATH}</code> · <b>{total_count:,}</b> documents',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.markdown(
            f'<span class="status-err">○ ChromaDB</span> — unreachable: {e}',
            unsafe_allow_html=True,
        )
        st.stop()
        return

    if episodic:
        try:
            ep_count = episodic.count()
            st.markdown(
                f'<span class="status-ok">● Episodic Memory</span> — <b>{ep_count:,}</b> episodes',
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                '<span class="status-warn">○ Episodic Memory</span> — collection not available',
                unsafe_allow_html=True,
            )

    # ── Collection stats ──
    st.divider()
    st.markdown(
        "#### :material/bar_chart: Collection Statistics",
    )

    # Query per-source counts from ChromaDB metadata
    source_counts: dict[str, int] = {}
    try:
        all_meta = collection.get(include=["metadatas"])
        metas = all_meta.get("metadatas") or []
        for m in metas:
            src = (m or {}).get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
    except Exception:
        pass  # fall back to showing zeros

    # Display per-source stat buttons (matching Graph page entity buttons)
    src_cols = st.columns(max(len(SOURCES), 1))
    for col, src_cfg in zip(src_cols, SOURCES):
        count = source_counts.get(src_cfg["chroma_source"], 0)
        with col:
            st.button(
                f"{src_cfg['label']}  {count:,}",
                key=f"vec_stat_{src_cfg['id']}",
                help=f"{src_cfg['description']}",
                width="stretch",
                type="secondary",
            )

    # Source breakdown pills
    pills_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'background:#1e293b;border:1px solid #334155;'
        f'border-radius:20px;padding:4px 12px;margin:3px 4px;font-size:0.78rem;'
        f'color:#cbd5e1;white-space:nowrap;">'
        f'<span class="material-symbols-outlined" style="font-size:16px;color:{src["color"]}">'
        f'{src["icon"]}</span>'
        f' <b style="color:#e2e8f0">{src["label"]}</b>'
        f' <span style="color:{src["color"]};font-weight:700">'
        f'{source_counts.get(src["chroma_source"], 0):,}</span>'
        f' {src["stat_label"]}'
        f'</span>'
        for src in SOURCES
    )
    # Add total pill
    pills_html += (
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'background:#1e293b;border:1px solid #6366f1;'
        f'border-radius:20px;padding:4px 12px;margin:3px 4px;font-size:0.78rem;'
        f'color:#cbd5e1;white-space:nowrap;">'
        f'<span class="material-symbols-outlined" style="font-size:16px;color:#6366f1">'
        f'database</span>'
        f' <b style="color:#e2e8f0">Total</b>'
        f' <span style="color:#6366f1;font-weight:700">{total_count:,}</span>'
        f' documents'
        f'</span>'
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:8px;">'
        f'{pills_html}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Ingestor pipeline ──
    _render_ingestor_section(collection, total_count)


# ── Ingestor pipeline section ─────────────────────────────────────────────────

def _render_ingestor_section(collection, current_count: int):
    st.markdown("#### :material/manufacturing: Run Ingestors")

    # ── STEP 1: EXTRACT ──────────────────────────────────────────────────────
    with st.expander("Step 1 — Extract Facebook export → JSON", expanded=False, icon=":material/description:"):
        st.markdown(
            "Point at your Facebook **HTML export** folder (the one containing `messages/inbox/`). "
            "This runs `tools/extract_facebook.py` and writes `facebook_messages.json`."
        )
        export_dir = st.text_input(
            "Facebook export folder",
            value="./data/facebook",
            key="export_dir",
            help="Absolute or relative path to the unzipped Facebook export",
        )
        out_json = st.text_input(
            "Output JSON file",
            value="./data/facebook/facebook_messages.json",
            key="extract_out_json",
        )

        log_key_extract = "vec_log_extract"
        log_box_extract = st.empty()
        if st.session_state.get(log_key_extract):
            _scrollable_log(log_box_extract, st.session_state[log_key_extract])

        run_col, _ = st.columns([1, 3])
        with run_col:
            if st.button("Run Extract", key="btn_extract", icon=":material/play_arrow:", width="stretch"):
                extract_script = Path("tools/extract_facebook.py").resolve()
                if not extract_script.exists():
                    st.error(f"Script not found: {extract_script}")
                else:
                    lines: list[str] = []
                    with st.spinner("Extracting…"):
                        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                        proc = subprocess.Popen(
                            [sys.executable, str(extract_script),
                             "--input", export_dir, "--output", out_json],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            env=env,
                        )
                        for line in proc.stdout:
                            lines.append(line.rstrip())
                            _scrollable_log(log_box_extract, lines[-40:])
                        proc.wait()
                    st.session_state[log_key_extract] = lines
                    if lines:
                        _scrollable_log(log_box_extract, lines)
                    if proc.returncode == 0:
                        st.success(f"Extract complete — written to `{out_json}`")
                    else:
                        st.error(f"Extract failed (exit {proc.returncode})")

    # ── STEP 2: FINE-TUNE EXPORT ──────────────────────────────────────────────
    with st.expander("Step 2 — Data Cleaning & Fine-tune Export", expanded=False, icon=":material/model_training:"):
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

        st.caption(
            "**Format:** `{\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, "
            "{\"role\": \"assistant\", \"content\": \"...\"}]}`\n\n"
            "• **user** = the other person's message\n"
            "• **assistant** = your reply"
        )

        log_key_ft = "vec_log_finetune"
        log_box_ft = st.empty()
        if st.session_state.get(log_key_ft):
            _scrollable_log(log_box_ft, st.session_state[log_key_ft])

        run_col_ft, _ = st.columns([1, 3])
        with run_col_ft:
            if st.button("Export Fine-tune Data", key="btn_finetune",
                         icon=":material/play_arrow:", width="stretch"):
                from tools.export_finetune import export_finetune_data

                lines_ft: list[str] = []
                progress_ft = st.progress(0, text="Exporting fine-tune data…")

                def _ft_cb(current: int, total: int):
                    pct = current / total if total > 0 else 0
                    progress_ft.progress(pct, text=f"Processing conversations… {current:,}/{total:,}")

                try:
                    stats = export_finetune_data(
                        json_path=ft_json,
                        output_path=ft_output,
                        years=ft_years,
                        language=ft_language,
                        min_words=ft_min_words,
                        max_words=ft_max_words,
                        max_turns=ft_max_turns,
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
                    _scrollable_log(log_box_ft, lines_ft)

                    st.success(
                        f"Exported **{stats['pairs_exported']:,}** training examples "
                        f"from **{stats['conversations_used']:,}** conversations → "
                        f"`{stats['output_file']}`"
                    )

                    # Show a preview of the first few examples
                    output_p = Path(ft_output)
                    if output_p.exists():
                        with open(output_p, encoding="utf-8") as pf:
                            preview_lines = [pf.readline() for _ in range(3)]
                        preview_lines = [l for l in preview_lines if l.strip()]
                        if preview_lines:
                            st.markdown("**Preview (first 3 examples):**")
                            for pl in preview_lines:
                                try:
                                    example = json.loads(pl)
                                    msgs = example.get("messages", [])
                                    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                                    asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                                    st.markdown(
                                        f'<div style="padding:6px 10px;margin:3px 0;'
                                        f'border:1px solid rgba(128,128,128,0.25);'
                                        f'border-radius:6px;font-size:0.85em">'
                                        f'👤 <b>User:</b> {user_msg[:150]}<br>'
                                        f'🤖 <b>Assistant:</b> {asst_msg[:150]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                except Exception:
                                    pass

                except FileNotFoundError as e:
                    progress_ft.empty()
                    st.error(str(e))
                except Exception as e:
                    progress_ft.empty()
                    st.error(f"Export failed: {e}")

    # ── STEP 3: LORA FINE-TUNING ──────────────────────────────────────────────
    with st.expander("Step 3 — LoRA Fine-tuning", expanded=False, icon=":material/neurology:"):
        st.markdown(
            "Fine-tune a base model on your conversation data using **LoRA** "
            "(Low-Rank Adaptation). Requires `transformers`, `peft`, `trl`, "
            "`datasets`, and optionally `bitsandbytes` for 4-bit quantization."
        )

        # Ollama host selector for model discovery
        from config import DEFAULT_OLLAMA
        _lora_ollama_host = st.text_input(
            "Ollama host",
            value=st.session_state.get("ollama_host", DEFAULT_OLLAMA),
            key="lora_ollama_host",
            help="Remote or local Ollama instance to browse available models.",
        )

        _lora_models: list[str] = []
        _lora_conn_error: str = ""
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(
                f"{_lora_ollama_host.rstrip('/')}/api/tags", timeout=5
            ) as _r:
                _lora_data = _json.loads(_r.read())
            _lora_models = sorted(
                m.get("model", m.get("name", ""))
                for m in _lora_data.get("models", [])
                if m.get("model") or m.get("name")
            )
        except Exception as _e:
            _lora_conn_error = str(_e)

        if _lora_conn_error:
            st.warning(f"Cannot reach Ollama at `{_lora_ollama_host}`: {_lora_conn_error}")

        col_lora1, col_lora2 = st.columns(2)
        with col_lora1:
            if _lora_models:
                lora_base_model = st.selectbox(
                    "Base model (Ollama)",
                    options=_lora_models,
                    index=0,
                    key="lora_base_model_select",
                    help="Select from available Ollama models on the chosen host.",
                )
            else:
                lora_base_model = st.text_input(
                    "Base model (HuggingFace ID)",
                    value="meta-llama/Llama-3.2-3B-Instruct",
                    key="lora_base_model",
                    help="No Ollama models found. Enter a HuggingFace model ID.",
                )
            lora_data = st.text_input(
                "Training data (JSONL)",
                value="./data/facebook/finetune_data.jsonl",
                key="lora_data",
            )
            lora_output = st.text_input(
                "Output directory",
                value="./models/my-lora",
                key="lora_output",
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

        st.caption(
            f"**Effective batch size:** {lora_batch * lora_grad_accum} "
            f"(batch {lora_batch} × accum {lora_grad_accum})"
        )

        _LORA_PROC = "lora_proc"
        _LORA_LINES = "vec_log_lora"

        # Retrieve running process (if any) and check if it's still alive
        lora_proc: subprocess.Popen | None = st.session_state.get(_LORA_PROC)
        is_training = lora_proc is not None and lora_proc.poll() is None

        log_box_lora = st.empty()
        if st.session_state.get(_LORA_LINES):
            _scrollable_log(log_box_lora, st.session_state[_LORA_LINES])

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
            _scrollable_log(log_box_lora, lines[-50:])
            st.spinner("Training — this may take a while…")
            import time; time.sleep(1); st.rerun()
        elif lora_proc is not None:
            # Process just finished
            st.session_state.pop(_LORA_PROC, None)
            if lora_proc.returncode == 0:
                st.success(f"✅ Training complete! LoRA adapter saved to `{lora_output}`")
            elif lora_proc.returncode == -15:  # SIGTERM from cancel
                st.warning("Training cancelled.")
            else:
                st.error(f"Training failed (exit {lora_proc.returncode})")

    # ── STEP 4: INGEST ───────────────────────────────────────────────────────
    with st.expander("Step 4 — Ingest JSON → ChromaDB", expanded=True, icon=":material/psychology:"):
        st.markdown(
            "Reads the extracted JSON and embeds it into ChromaDB. "
            "Runs `tools/ingest_facebook_messages.py`."
        )

        # ── Pre-ingest Stats Graph ──
        @st.cache_data(show_spinner=False)
        def _load_conversation_stats(json_path: str) -> dict[str, int]:
            import json, os
            from collections import defaultdict
            if not os.path.isfile(json_path):
                return {}
            try:
                with open(json_path) as f:
                    data = json.load(f)
                counts = defaultdict(int)
                for msg in data:
                    if msg.get("text"):
                        counts[msg.get("conversation", "Unknown")] += 1
                return dict(counts)
            except Exception:
                return {}

        import plotly.graph_objects as go
        json_default_path = "./data/facebook/facebook_messages.json"
        conv_stats = _load_conversation_stats(json_default_path)
        
        if conv_stats:
            # Group into logarithmic-style buckets
            buckets = {
                "1-10": 0,
                "11-50": 0,
                "51-100": 0,
                "101-500": 0,
                "501-1k": 0,
                "1k-5k": 0,
                "5k-10k": 0,
                "10k+": 0
            }
            for count in conv_stats.values():
                if count <= 10: buckets["1-10"] += 1
                elif count <= 50: buckets["11-50"] += 1
                elif count <= 100: buckets["51-100"] += 1
                elif count <= 500: buckets["101-500"] += 1
                elif count <= 1000: buckets["501-1k"] += 1
                elif count <= 5000: buckets["1k-5k"] += 1
                elif count <= 10000: buckets["5k-10k"] += 1
                else: buckets["10k+"] += 1
            
            # Remove empty trailing buckets for a cleaner chart (but keep the shape continuous)
            x_vals = list(buckets.keys())
            y_vals = list(buckets.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=x_vals, 
                    y=y_vals,
                    marker_color="#6366f1",
                    text=y_vals,
                    textposition="auto",
                    hovertemplate="Message count: %{x}<br>Conversations: %{y}<extra></extra>"
                )
            ])
            fig.update_layout(
                title="Conversation Length Distribution",
                title_font=dict(size=14, color="#e2e8f0"),
                xaxis=dict(title="Messages in Conversation", gridcolor="#333", tickfont=dict(size=11)),
                yaxis=dict(title="Number of Conversations", gridcolor="#333"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=20, t=40, b=20),
                height=280
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Run Step 1 to extract stats.")

        col_a, col_b = st.columns(2)
        with col_a:
            json_file = st.text_input(
                "JSON file",
                value="./data/facebook/facebook_messages.json",
                key="ingest_json",
            )
            batch_size = st.number_input(
                "Batch size", min_value=4, max_value=5000,
                value=32, step=4, key="batch_size",
            )
        with col_b:
            session_gap_h = st.slider(
                "Session gap (hours)", min_value=1, max_value=48, value=8, key="session_gap",
                help="Conversations with gaps longer than this are split into separate chunks",
            )
            max_msgs = st.number_input(
                "Max msgs per chunk", min_value=10, max_value=500,
                value=150, step=10, key="max_msgs",
            )

        reset_col = st.checkbox(
            "Reset collection before ingesting (deletes all existing data)",
            value=True, key="reset_collection",
        )

        st.caption(f"Current ChromaDB document count: **{current_count:,}**")

        log_key_ingest = "vec_log_ingest"
        log_box_ingest = st.empty()
        if st.session_state.get(log_key_ingest):
            _scrollable_log(log_box_ingest, st.session_state[log_key_ingest])

        run_col2, _ = st.columns([1, 3])
        with run_col2:
            if st.button("Run Ingest", key="btn_ingest", type="primary", icon=":material/play_arrow:", width="stretch"):
                ingest_script = Path("tools/ingest_facebook_messages.py").resolve()
                if not ingest_script.exists():
                    st.error(f"Script not found: {ingest_script}")
                else:
                    lines: list[str] = []
                    cmd = [
                        sys.executable, str(ingest_script),
                        "--json-file",    json_file,
                        "--session-gap",  str(session_gap_h * 3600),
                        "--max-msgs",     str(max_msgs),
                        "--batch-size",   str(batch_size),
                    ]
                    if reset_col:
                        cmd.append("--reset")
                    with st.spinner("Ingesting — this may take several minutes for large datasets…"):
                        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            env=env,
                        )
                        for line in proc.stdout:
                            lines.append(line.rstrip())
                            _scrollable_log(log_box_ingest, lines[-50:])
                        proc.wait()
                    st.session_state[log_key_ingest] = lines
                    if lines:
                        _scrollable_log(log_box_ingest, lines)
                    if proc.returncode == 0:
                        if reset_col:
                            # Clear Streamlit caches so they don't hold the old deleted collection UUID
                            from rag.resources import load_chroma, load_bm25_corpus
                            load_chroma.clear()
                            load_bm25_corpus.clear()
                            
                            import chromadb
                            temp_c = chromadb.PersistentClient(path=os.path.expanduser(CHROMA_PATH)).get_collection("virtual_me_knowledge")
                            new_count = temp_c.count()
                        else:
                            new_count = collection.count()
                            
                        st.success(
                            f"Ingest complete! "
                            f"Documents: {current_count:,} → **{new_count:,}** "
                            f"(+{new_count - current_count:,})"
                        )
                        if st.button("Refresh stats", icon=":material/refresh:",
                                     key="vec_refresh_stats"):
                            st.rerun()
                    else:
                        st.error(f"Ingest failed (exit {proc.returncode})")
