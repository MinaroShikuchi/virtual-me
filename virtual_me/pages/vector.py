"""
virtual_me/pages/vector.py — Vector Store page.

Sections:
  1. ChromaDB connection status + collection stats (per-source document counts)
  2. Run ingestors (extract → ingest pipeline with live log streaming)
"""
import reflex as rx

from config import SOURCES
from virtual_me.state.vector_state import VectorState
from virtual_me.components.layout import layout


# ── Helpers ───────────────────────────────────────────────────────────────────


def _log_box(logs_var) -> rx.Component:
    """Scrollable terminal-style log output."""
    return rx.box(
        rx.foreach(
            logs_var,
            lambda line: rx.text(
                line,
                font_family="monospace",
                font_size="12px",
                white_space="pre-wrap",
            ),
        ),
        max_height="300px",
        overflow_y="auto",
        bg="#1a1a2e",
        color="#00ff41",
        padding="12px",
        border_radius="8px",
        width="100%",
    )


def _source_pill(src_cfg: dict) -> rx.Component:
    """A single source stat pill."""
    chroma_key = src_cfg["chroma_source"]
    label = src_cfg["label"]
    color = src_cfg["color"]
    stat_label = src_cfg["stat_label"]

    return rx.box(
        rx.hstack(
            rx.text(label, weight="bold", size="2", color="#e2e8f0"),
            rx.text(
                VectorState.source_counts[chroma_key].to(str),
                weight="bold",
                size="2",
                color=color,
            ),
            rx.text(stat_label, size="1", color="#cbd5e1"),
            spacing="2",
            align="center",
        ),
        bg="#1e293b",
        border=f"1px solid #334155",
        border_radius="20px",
        padding_x="12px",
        padding_y="4px",
    )


# ── Section A: Collection Overview ────────────────────────────────────────────


def _collection_overview() -> rx.Component:
    """ChromaDB connection status and collection statistics."""
    return rx.vstack(
        # Connection banner
        rx.hstack(
            rx.box(
                width="8px",
                height="8px",
                border_radius="50%",
                bg=rx.cond(VectorState.total_docs > 0, "#4ade80", "#f87171"),
            ),
            rx.text(
                "ChromaDB",
                weight="bold",
                color=rx.cond(VectorState.total_docs > 0, "#4ade80", "#f87171"),
            ),
            rx.text("—", size="2", color="#94a3b8"),
            rx.text(
                VectorState.total_docs.to(str) + " documents",
                size="2",
                color="#e2e8f0",
            ),
            spacing="2",
            align="center",
        ),
        # Episodic memory
        rx.cond(
            VectorState.ep_count > 0,
            rx.hstack(
                rx.box(
                    width="8px",
                    height="8px",
                    border_radius="50%",
                    bg="#4ade80",
                ),
                rx.text("Episodic Memory", weight="bold", color="#4ade80"),
                rx.text("—", size="2", color="#94a3b8"),
                rx.text(
                    VectorState.ep_count.to(str) + " episodes",
                    size="2",
                    color="#e2e8f0",
                ),
                spacing="2",
                align="center",
            ),
            rx.fragment(),
        ),
        rx.divider(color_scheme="gray"),
        # Section heading
        rx.heading("Collection Statistics", size="4"),
        # Source stat pills
        rx.flex(
            *[_source_pill(src) for src in SOURCES],
            rx.box(
                rx.hstack(
                    rx.text("Total", weight="bold", size="2", color="#e2e8f0"),
                    rx.text(
                        VectorState.total_docs.to(str),
                        weight="bold",
                        size="2",
                        color="#6366f1",
                    ),
                    rx.text("documents", size="1", color="#cbd5e1"),
                    spacing="2",
                    align="center",
                ),
                bg="#1e293b",
                border="1px solid #6366f1",
                border_radius="20px",
                padding_x="12px",
                padding_y="4px",
            ),
            flex_wrap="wrap",
            gap="8px",
        ),
        spacing="3",
        width="100%",
    )


# ── Section B: Run Ingestors ─────────────────────────────────────────────────


def _extract_section() -> rx.Component:
    """Step 1 — Extract Facebook export → JSON."""
    return rx.vstack(
        rx.heading("Step 1 — Extract Facebook export → JSON", size="3"),
        rx.text(
            "Point at your Facebook HTML export folder (the one containing messages/inbox/). "
            "This runs tools/extract_facebook.py and writes facebook_messages.json.",
            size="2",
            color="#94a3b8",
        ),
        rx.hstack(
            rx.vstack(
                rx.text("Facebook export folder", size="2", weight="medium"),
                rx.input(
                    value=VectorState.export_dir,
                    on_change=VectorState.set_export_dir,
                    placeholder="./data/facebook",
                    width="100%",
                ),
                width="50%",
            ),
            rx.vstack(
                rx.text("Output JSON file", size="2", weight="medium"),
                rx.input(
                    value=VectorState.out_json,
                    on_change=VectorState.set_out_json,
                    placeholder="./data/facebook/facebook_messages.json",
                    width="100%",
                ),
                width="50%",
            ),
            width="100%",
            spacing="4",
        ),
        # Log output
        rx.cond(
            VectorState.extract_logs.length() > 0,
            _log_box(VectorState.extract_logs),
            rx.fragment(),
        ),
        # Run button
        rx.button(
            rx.cond(
                VectorState.extract_running,
                rx.hstack(
                    rx.spinner(size="1"),
                    rx.text("Extracting…"),
                    spacing="2",
                    align="center",
                ),
                rx.hstack(
                    rx.icon("play", size=16),
                    rx.text("Run Extract"),
                    spacing="2",
                    align="center",
                ),
            ),
            on_click=VectorState.run_extract,
            disabled=VectorState.extract_running,
            variant="outline",
            size="2",
        ),
        spacing="3",
        width="100%",
        padding="16px",
        bg="#111827",
        border="1px solid #1f2937",
        border_radius="12px",
    )


def _ingest_section() -> rx.Component:
    """Step 2 — Ingest JSON → ChromaDB."""
    return rx.vstack(
        rx.heading("Step 2 — Ingest JSON → ChromaDB", size="3"),
        rx.text(
            "Reads the extracted JSON and embeds it into ChromaDB. "
            "Runs tools/ingest_facebook_messages.py.",
            size="2",
            color="#94a3b8",
        ),
        # Parameters row 1
        rx.hstack(
            rx.vstack(
                rx.text("JSON file", size="2", weight="medium"),
                rx.input(
                    value=VectorState.json_file,
                    on_change=VectorState.set_json_file,
                    placeholder="./data/facebook/facebook_messages.json",
                    width="100%",
                ),
                width="50%",
            ),
            rx.vstack(
                rx.text("Batch size", size="2", weight="medium"),
                rx.input(
                    value=VectorState.batch_size.to(str),
                    on_change=VectorState.set_batch_size,
                    type="number",
                    width="100%",
                ),
                width="50%",
            ),
            width="100%",
            spacing="4",
        ),
        # Parameters row 2
        rx.hstack(
            rx.vstack(
                rx.text(
                    "Session gap: " + VectorState.session_gap_h.to(str) + " hours",
                    size="2",
                    weight="medium",
                ),
                rx.slider(
                    min=1,
                    max=48,
                    default_value=[8],
                    on_value_commit=VectorState.set_session_gap,
                    width="100%",
                ),
                width="50%",
            ),
            rx.vstack(
                rx.text("Max msgs per chunk", size="2", weight="medium"),
                rx.input(
                    value=VectorState.max_msgs.to(str),
                    on_change=VectorState.set_max_msgs,
                    type="number",
                    width="100%",
                ),
                width="50%",
            ),
            width="100%",
            spacing="4",
        ),
        # Reset checkbox
        rx.hstack(
            rx.checkbox(
                "Reset collection before ingesting (deletes all existing data)",
                checked=VectorState.reset_collection,
                on_change=VectorState.toggle_reset,
            ),
            width="100%",
        ),
        rx.text(
            "Current ChromaDB document count: "
            + VectorState.total_docs.to(str),
            size="2",
            color="#94a3b8",
        ),
        # Log output
        rx.cond(
            VectorState.ingest_logs.length() > 0,
            _log_box(VectorState.ingest_logs),
            rx.fragment(),
        ),
        # Run button
        rx.button(
            rx.cond(
                VectorState.ingest_running,
                rx.hstack(
                    rx.spinner(size="1"),
                    rx.text("Ingesting…"),
                    spacing="2",
                    align="center",
                ),
                rx.hstack(
                    rx.icon("play", size=16),
                    rx.text("Run Ingest"),
                    spacing="2",
                    align="center",
                ),
            ),
            on_click=VectorState.run_ingest,
            disabled=VectorState.ingest_running,
            size="2",
        ),
        spacing="3",
        width="100%",
        padding="16px",
        bg="#111827",
        border="1px solid #1f2937",
        border_radius="12px",
    )


# ── Page content ──────────────────────────────────────────────────────────────


def vector_content() -> rx.Component:
    """Main content for the Vector Store page."""
    return rx.vstack(
        rx.heading("Vector Store", size="6"),
        rx.text(
            "Semantic memory stored in ChromaDB — extract your data exports "
            "and embed them as searchable documents.",
            size="2",
            color="#94a3b8",
        ),
        _collection_overview(),
        rx.divider(color_scheme="gray"),
        rx.heading("Run Ingestors", size="4"),
        _extract_section(),
        _ingest_section(),
        spacing="4",
        width="100%",
        padding="24px",
    )


def vector_page() -> rx.Component:
    """Vector Store page wrapped in the shared layout."""
    return layout(vector_content())
