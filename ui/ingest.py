"""
ui/ingest.py — Ingest tab: extract Facebook export → JSON, then ingest JSON → ChromaDB.

Both steps run tool scripts from tools/ as subprocesses and stream stdout live.
"""
import subprocess
import sys
from pathlib import Path

import streamlit as st


def render_ingest_tab(collection):
    st.markdown("### ⚙️ Ingest Pipeline")
    st.caption(
        "Run the full ingestion pipeline directly from the UI. "
        "Scripts live in `tools/` and run as subprocesses so the app stays responsive."
    )

    # ── STEP 1: EXTRACT ──────────────────────────────────────────────────────
    with st.expander("📄 Step 1 — Extract Facebook export → JSON", expanded=True):
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
        if st.button("▶️ Run Extract", key="btn_extract"):
            extract_script = Path("tools/extract_facebook.py").resolve()
            if not extract_script.exists():
                st.error(f"Script not found: {extract_script}")
            else:
                log_box = st.empty()
                lines   = []
                with st.spinner("Extracting…"):
                    proc = subprocess.Popen(
                        [sys.executable, str(extract_script),
                         "--input", export_dir, "--output", out_json],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    for line in proc.stdout:
                        lines.append(line.rstrip())
                        log_box.code("\n".join(lines[-40:]), language="")
                    proc.wait()
                if proc.returncode == 0:
                    st.success(f"✅ Extract complete — written to `{out_json}`")
                else:
                    st.error(f"❌ Extract failed (exit {proc.returncode})")

    st.divider()

    # ── STEP 2: INGEST ───────────────────────────────────────────────────────
    with st.expander("🧠 Step 2 — Ingest JSON → ChromaDB", expanded=True):
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
            "⚠️ Reset collection before ingesting (deletes all existing data)",
            value=True, key="reset_collection",
        )

        current_count = collection.count()
        st.caption(f"Current ChromaDB document count: **{current_count:,}**")

        if st.button("▶️ Run Ingest", key="btn_ingest", type="primary"):
            ingest_script = Path("tools/ingest_facebook_messages.py").resolve()
            if not ingest_script.exists():
                st.error(f"Script not found: {ingest_script}")
            else:
                log_box = st.empty()
                lines   = []
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
                        log_box.code("\n".join(lines[-50:]), language="")
                    proc.wait()
                if proc.returncode == 0:
                    new_count = collection.count()
                    st.success(
                        f"✅ Ingest complete! "
                        f"Documents: {current_count:,} → **{new_count:,}** "
                        f"(+{new_count - current_count:,})"
                    )
                    st.balloons()
                else:
                    st.error(f"❌ Ingest failed (exit {proc.returncode})")
