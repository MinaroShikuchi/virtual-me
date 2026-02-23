"""
virtual_me/pages/graph.py — Knowledge Graph / Node Extract page.

Sections:
  1. Neo4j connection status + graph statistics (node counts by label)
  2. Interest profile (table view)
  3. Run extractors (platform tabs, live log streaming)
"""
import reflex as rx

from graph.constants import ENTITY_LABELS, LABEL_COLORS
from virtual_me.state.graph_state import (
    GraphState,
    PLATFORMS,
    _get_extractor_labels,
    _find_extractor,
)
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


# ── Section A: Graph Overview ─────────────────────────────────────────────────


def _entity_stat_card(label: str) -> rx.Component:
    """A single entity label stat card."""
    color = LABEL_COLORS.get(label, "#6366f1")
    return rx.box(
        rx.vstack(
            rx.text(label, size="2", weight="bold", color=color),
            rx.text(
                GraphState.graph_stats[label].to(str),
                size="4",
                weight="bold",
                color="#e2e8f0",
            ),
            spacing="1",
            align="center",
        ),
        bg="#111827",
        border=f"1px solid {color}40",
        border_radius="12px",
        padding="12px 16px",
        min_width="100px",
        text_align="center",
    )


def _graph_overview() -> rx.Component:
    """Neo4j connection status and graph statistics."""
    return rx.vstack(
        # Connection banner
        rx.hstack(
            rx.box(
                width="8px",
                height="8px",
                border_radius="50%",
                bg=rx.cond(GraphState.neo4j_alive, "#4ade80", "#f87171"),
            ),
            rx.text(
                "Neo4j",
                weight="bold",
                color=rx.cond(GraphState.neo4j_alive, "#4ade80", "#f87171"),
            ),
            rx.text("—", size="2", color="#94a3b8"),
            rx.cond(
                GraphState.neo4j_alive,
                rx.text("connected", size="2", color="#e2e8f0"),
                rx.text(
                    "unreachable — start with: docker compose up neo4j -d",
                    size="2",
                    color="#f87171",
                ),
            ),
            spacing="2",
            align="center",
        ),
        rx.divider(color_scheme="gray"),
        # Entity stats heading
        rx.heading("Graph Statistics", size="4"),
        # Entity stat cards
        rx.cond(
            GraphState.neo4j_alive,
            rx.flex(
                *[_entity_stat_card(label) for label in ENTITY_LABELS],
                flex_wrap="wrap",
                gap="8px",
            ),
            rx.text(
                "Connect Neo4j to see graph statistics.",
                size="2",
                color="#94a3b8",
            ),
        ),
        spacing="3",
        width="100%",
    )


# ── Interest Profile ─────────────────────────────────────────────────────────


def _interest_row(item: list) -> rx.Component:
    """Render a single interest row: [name, percentage]."""
    return rx.table.row(
        rx.table.cell(rx.text(item[0], size="2")),
        rx.table.cell(
            rx.hstack(
                rx.box(
                    width=item[1].to(str) + "%",
                    height="8px",
                    bg="#6366f1",
                    border_radius="4px",
                    max_width="200px",
                ),
                rx.text(item[1].to(str) + "%", size="1", color="#94a3b8"),
                spacing="2",
                align="center",
            ),
        ),
    )


def _interest_profile() -> rx.Component:
    """Interest profile table (simplified from radar chart)."""
    return rx.cond(
        GraphState.interest_data.length() > 0,
        rx.vstack(
            rx.heading("Interest Profile", size="4"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Interest"),
                        rx.table.column_header_cell("Score"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        GraphState.interest_data.items(),
                        _interest_row,
                    ),
                ),
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        rx.fragment(),
    )


# ── Section B: Run Extractors ─────────────────────────────────────────────────


def _platform_button(platform: dict) -> rx.Component:
    """A platform selection button."""
    label = platform["label"]
    color = platform["color"]
    return rx.button(
        label,
        variant=rx.cond(
            GraphState.selected_platform == label,
            "solid",
            "outline",
        ),
        color_scheme=rx.cond(
            GraphState.selected_platform == label,
            "iris",
            "gray",
        ),
        on_click=GraphState.set_selected_platform(label),
        size="2",
    )


def _extractor_option(ext_label: str) -> rx.Component:
    """A single extractor option in the select dropdown."""
    return rx.select.item(ext_label, value=ext_label)


def _extractor_info_pills() -> rx.Component:
    """Show entities and relationships for the currently selected extractor.
    Since we can't dynamically look up the extractor config in reactive context,
    we render all possible combinations and use rx.cond to show the right one."""
    # Build a flat list of all platform+extractor combos
    fragments = []
    for platform in PLATFORMS:
        for ext in platform["extractors"]:
            ent_pills = rx.flex(
                rx.text("Entities:", size="1", color="#64748b", weight="bold"),
                *[
                    rx.badge(
                        e,
                        color_scheme="iris",
                        variant="outline",
                        size="1",
                    )
                    for e in ext.get("entities", [])
                ],
                flex_wrap="wrap",
                gap="4px",
                align="center",
            )
            rel_pills = rx.flex(
                rx.text("Relationships:", size="1", color="#64748b", weight="bold"),
                *[
                    rx.badge(
                        r,
                        color_scheme="gray",
                        variant="outline",
                        size="1",
                    )
                    for r in ext.get("relationships", [])
                ],
                flex_wrap="wrap",
                gap="4px",
                align="center",
            )
            fragments.append(
                rx.cond(
                    (GraphState.selected_platform == platform["label"])
                    & (GraphState.selected_extractor == ext["label"]),
                    rx.vstack(ent_pills, rel_pills, spacing="2", width="100%"),
                    rx.fragment(),
                )
            )
    return rx.fragment(*fragments)


def _extractor_select_for_platform(platform: dict) -> rx.Component:
    """Render the extractor select dropdown for a specific platform,
    shown only when that platform is selected."""
    ext_labels = [e["label"] for e in platform["extractors"]]
    if len(ext_labels) <= 1:
        return rx.fragment()
    return rx.cond(
        GraphState.selected_platform == platform["label"],
        rx.select(
            ext_labels,
            value=GraphState.selected_extractor,
            on_change=GraphState.set_selected_extractor,
            width="300px",
        ),
        rx.fragment(),
    )


def _extractor_section() -> rx.Component:
    """Run Extractors section with platform tabs and settings."""
    return rx.vstack(
        rx.heading("Run Extractors", size="4"),
        # Global settings row
        rx.hstack(
            rx.vstack(
                rx.text("Your name in the graph", size="2", weight="medium"),
                rx.input(
                    value=GraphState.self_name,
                    on_change=GraphState.set_self_name,
                    placeholder="ME",
                    width="200px",
                ),
                spacing="1",
            ),
            rx.vstack(
                rx.text("Limit chunks (0 = all)", size="2", weight="medium"),
                rx.input(
                    value=GraphState.limit_chunks.to(str),
                    on_change=GraphState.set_limit_chunks,
                    type="number",
                    width="150px",
                ),
                spacing="1",
            ),
            rx.vstack(
                rx.text("Dry run", size="2", weight="medium"),
                rx.switch(
                    checked=GraphState.dry_run,
                    on_change=GraphState.toggle_dry_run,
                ),
                spacing="1",
            ),
            spacing="4",
            align="end",
            width="100%",
        ),
        # Platform selector buttons
        rx.flex(
            *[_platform_button(p) for p in PLATFORMS],
            gap="8px",
            flex_wrap="wrap",
        ),
        # Extractor selector (per platform)
        *[_extractor_select_for_platform(p) for p in PLATFORMS],
        # Entity/relationship info pills
        _extractor_info_pills(),
        # Log output
        rx.cond(
            GraphState.extractor_logs.length() > 0,
            _log_box(GraphState.extractor_logs),
            rx.fragment(),
        ),
        # Run button
        rx.button(
            rx.cond(
                GraphState.extractor_running,
                rx.hstack(
                    rx.spinner(size="1"),
                    rx.text("Running…"),
                    spacing="2",
                    align="center",
                ),
                rx.hstack(
                    rx.icon("play", size=16),
                    rx.text("Run Extractor"),
                    spacing="2",
                    align="center",
                ),
            ),
            on_click=GraphState.run_extractor,
            disabled=GraphState.extractor_running,
            size="2",
        ),
        spacing="4",
        width="100%",
        padding="16px",
        bg="#111827",
        border="1px solid #1f2937",
        border_radius="12px",
    )


# ── Page content ──────────────────────────────────────────────────────────────


def graph_content() -> rx.Component:
    """Main content for the Knowledge Graph page."""
    return rx.vstack(
        rx.heading("Knowledge Graph", size="6"),
        rx.text(
            "Episodic memory stored in Neo4j — extract entities & relationships "
            "from your data sources.",
            size="2",
            color="#94a3b8",
        ),
        _graph_overview(),
        _interest_profile(),
        rx.divider(color_scheme="gray"),
        _extractor_section(),
        spacing="4",
        width="100%",
        padding="24px",
    )


def graph_page() -> rx.Component:
    """Knowledge Graph page wrapped in the shared layout."""
    return layout(graph_content())
