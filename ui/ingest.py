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
from ui.components.log_viewer import scrollable_log


# ── Helpers ───────────────────────────────────────────────────────────────────



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
            scrollable_log(log_box_extract, st.session_state[log_key_extract], follow=False, title="Extract")

        run_col, _ = st.columns([1, 2])
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
                            scrollable_log(log_box_extract, lines[-2000:], title="Extract")
                        proc.wait()
                    st.session_state[log_key_extract] = lines
                    if lines:
                        scrollable_log(log_box_extract, lines, follow=False, title="Extract")
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

        _DEFAULT_SYSTEM_PROMPT = (
            "You are Romain. You write casually, mix French and English naturally, "
            "and prefer direct pragmatic answers. Reply as yourself — not as an AI."
        )
        ft_system_prompt = st.text_area(
            "System prompt (clear to use the model's default)",
            value=_DEFAULT_SYSTEM_PROMPT,
            key="ft_system_prompt",
            height=100,
            help="Prepended as a system message to every training example. "
                 "Clear the field to train without a system prompt (model default will be used).",
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
            scrollable_log(log_box_ft, st.session_state[log_key_ft], follow=False, title="Fine-tune Export")

        run_col_ft, _ = st.columns([1, 3])
        with run_col_ft:
            do_export = st.button("Export Fine-tune Data", key="btn_finetune",
                         icon=":material/play_arrow:", width="stretch")
        if do_export:
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
                    max_reply_gap_min=ft_max_reply_gap,
                    filter_reactions=ft_filter_reactions,
                    system_prompt=ft_system_prompt.strip() or None,
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

        # Interactive dataset preview (persistent)
        output_p = Path(ft_output)
        if output_p.exists():
            st.divider()

            # ── Token distribution chart ──────────────────────────────────
            @st.cache_data(show_spinner=False)
            def _load_token_stats(filepath, mtime):
                import json
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
                import plotly.graph_objects as go
                sorted_rows = sorted(token_rows, key=lambda x: x[1], reverse=True)
                y_tokens = [r[1] for r in sorted_rows]
                hover_convs = [r[0] for r in sorted_rows]
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
                    title=f"Tokens per dataset row (estimated) — {len(y_tokens):,} rows",
                    title_font=dict(size=14, color="#e2e8f0"),
                    xaxis=dict(title="Dataset rows (sorted by tokens)", gridcolor="#333", showticklabels=False),
                    yaxis=dict(title="Approx. tokens", gridcolor="#333"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=20, t=40, b=20),
                    height=300,
                )
                st.plotly_chart(fig_tok, width="stretch")

            st.markdown("### Interactive Dataset Preview")
            st.caption("*(Note: To see the effects of changing settings like Max Turns, you must click **Export** again to regenerate the file!)*")

            @st.cache_data(show_spinner=False)
            def _load_preview_data(filepath, mtime):
                import json
                examples_by_conv = {}
                try:
                    with open(filepath, "r", encoding="utf-8") as pf:
                        for line in pf:
                            if not line.strip(): continue
                            ex = json.loads(line)
                            # Handle missing or empty conversation labels
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
                
                p_col1, p_col2 = st.columns([3, 1])
                with p_col1:
                    selected_conv = st.selectbox(
                        "Conversation", 
                        options=convs, 
                        index=convs.index(st.session_state.preview_conv) if st.session_state.preview_conv in convs else 0,
                        key="preview_conv_select"
                    )
                with p_col2:
                    st.markdown("<div style='margin-top: 28px'></div>", unsafe_allow_html=True)
                    if st.button("Next Example", key="btn_next_ex", width="stretch"):
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

    # ── STEP 3: INGEST ───────────────────────────────────────────────────────
    with st.expander("Step 3 — Ingest JSON → ChromaDB", expanded=True, icon=":material/psychology:"):
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
        ingest_filter_reactions = st.checkbox(
            "Filter out reactions", value=True, key="ingest_filter_reactions",
            help="If checked, standalone Facebook reactions are excluded from the vector database context."
        )

        st.caption(f"Current ChromaDB document count: **{current_count:,}**")

        log_key_ingest = "vec_log_ingest"
        log_box_ingest = st.empty()
        if st.session_state.get(log_key_ingest):
            scrollable_log(log_box_ingest, st.session_state[log_key_ingest], follow=False, title="Ingest")

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
                    if ingest_filter_reactions:
                        cmd.append("--filter-reactions")
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
                            scrollable_log(log_box_ingest, lines[-2000:], title="Ingest")
                        proc.wait()
                    st.session_state[log_key_ingest] = lines
                    if lines:
                        scrollable_log(log_box_ingest, lines, follow=False, title="Ingest")
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
