"""
virtual_me/pages/dashboard.py — Dashboard page.
"""
import reflex as rx
from virtual_me.state.dashboard_state import DashboardState
from virtual_me.state.dashboard_state import SourceInfo
from virtual_me.components.layout import layout


def _source_card(src: rx.Var[SourceInfo]) -> rx.Component:
    """Render a single data source card from a reactive SourceInfo var."""
    return rx.box(
        rx.vstack(
            rx.text(src.label, weight="bold", size="3", color="#f1f5f9"),
            rx.text(src.description, size="1", color="#94a3b8"),
            rx.heading(
                "~" + src.approx_total.to(str),
                size="5",
                color=src.color,
            ),
            rx.text(src.stat_label, size="1", color="#64748b"),
            rx.cond(
                src.ingested > 0,
                rx.text(
                    src.ingested.to(str) + " chunks ingested",
                    size="1",
                    color="#4ade80",
                ),
                rx.text("not ingested", size="1", color="#f87171"),
            ),
            rx.cond(
                src.file_count > 0,
                rx.text(
                    src.size_mb.to(str) + " MB in data/",
                    size="1",
                    color="#475569",
                ),
                rx.fragment(),
            ),
            spacing="2",
            padding="20px",
        ),
        border="1px solid #33335580",
        border_left="4px solid " + src.color,
        border_radius="12px",
        bg="#1e203080",
    )


def _metric_card(label: str, value: rx.Var[str]) -> rx.Component:
    """Simple metric display."""
    return rx.box(
        rx.vstack(
            rx.text(label, size="1", color="#94a3b8", weight="medium"),
            rx.heading(value, size="5", color="#f1f5f9"),
            spacing="1",
        ),
        padding="16px",
        border="1px solid #2d3250",
        border_radius="10px",
        bg="#1e2030",
    )


def dashboard_content() -> rx.Component:
    """Dashboard page content."""
    return rx.vstack(
        rx.heading("Dashboard", size="6"),

        # Hero metrics
        rx.grid(
            _metric_card(
                "Total chunks in ChromaDB",
                DashboardState.total_docs.to(str),
            ),
            _metric_card(
                "Active data sources",
                DashboardState.active_sources.to(str) + " / 3",
            ),
            _metric_card("Data folder", "./data"),
            columns="3",
            spacing="4",
            width="100%",
        ),

        rx.divider(),

        # Source cards
        rx.heading("Data Sources", size="4"),
        rx.grid(
            rx.foreach(
                DashboardState.source_scan,
                _source_card,
            ),
            columns="3",
            spacing="4",
            width="100%",
        ),

        spacing="6",
        width="100%",
        on_mount=DashboardState.load_dashboard,
    )


def dashboard_page() -> rx.Component:
    """Dashboard page wrapped in the shared layout."""
    return layout(dashboard_content())
