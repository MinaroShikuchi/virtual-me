"""
ui/ingest.py — Vector Store tab (semantic memory).

Sections:
  1. Connection status + collection stats (per-source document counts)
  2. Run ingestors (extract → ingest pipeline with live log streaming)
"""
import subprocess
import sys
from pathlib import Path

import streamlit as st

from config import CHROMA_PATH, SOURCES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scrollable_log(container, lines: list[str], max_height: int = 300):
    """Render log lines inside a scrollable container with monospace font."""
    escaped = "\n".join(lines).replace("&", "&").replace("<", "<").replace(">", ">")
    container.markdown(
        f'<div style="max-height:{max_height}px;overflow-y:auto;'
        f'background:#0e1117;border:1px solid #333;border-radius:6px;'
        f'padding:10px;font-family:monospace;font-size:13px;'
        f'white-space:pre-wrap;color:#ccc;">{escaped}</div>',
        unsafe_allow_html=True,
    )


# ── Main render function ─────────────────────────────────────────────────────

def render_vector_tab(collection, episodic=None):
    st.markdown("### :material/database: Vector Store")
    st.caption("Semantic memory stored in ChromaDB — extract your data exports and embed them as searchable documents.")

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
                use_container_width=True,
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
                        proc = subprocess.Popen(
                            [sys.executable, str(extract_script),
                             "--input", export_dir, "--output", out_json],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
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

    # ── STEP 2: INGEST ───────────────────────────────────────────────────────
    with st.expander("Step 2 — Ingest JSON → ChromaDB", expanded=True, icon=":material/psychology:"):
        st.markdown(
            "Reads the extracted JSON and embeds it into ChromaDB. "
            "Runs `tools/ingest_facebook_messages.py`."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            json_file = st.text_input(
                "JSON file",
                value="./data/facebook/facebook_messages.json",
                key="ingest_json",
            )
            batch_size = st.number_input(
                "Batch size", min_value=100, max_value=5000,
                value=1000, step=100, key="batch_size",
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
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                        )
                        for line in proc.stdout:
                            lines.append(line.rstrip())
                            _scrollable_log(log_box_ingest, lines[-50:])
                        proc.wait()
                    st.session_state[log_key_ingest] = lines
                    if lines:
                        _scrollable_log(log_box_ingest, lines)
                    if proc.returncode == 0:
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
