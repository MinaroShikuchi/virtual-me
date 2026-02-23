"""
virtual_me/pages/rag_explorer.py — RAG Explorer page.
"""
import reflex as rx
from virtual_me.state.rag_explorer_state import RAGExplorerState
from virtual_me.components.layout import layout


def _metric(label: str, value) -> rx.Component:
    """A single metric card."""
    return rx.box(
        rx.vstack(
            rx.text(label, size="1", color="#94a3b8"),
            rx.text(value, size="3", weight="bold", color="#f1f5f9"),
            spacing="1",
        ),
        padding="12px",
        border="1px solid #2d3250",
        border_radius="8px",
        bg="#1e2030",
    )


def _result_card(doc: dict) -> rx.Component:
    """Render a single RAG result card."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(doc["date"], size="1", color="#a5b4fc"),
                rx.text(doc["friend"], size="1", color="#a5b4fc"),
                rx.text(
                    doc["message_count"].to(str) + " msgs",
                    size="1",
                    color="#a5b4fc",
                ),
                spacing="3",
            ),
            rx.el.pre(
                doc["content"],
                style={
                    "white_space": "pre-wrap",
                    "color": "#e2e8f0",
                    "font_size": "0.82rem",
                    "max_height": "200px",
                    "overflow_y": "auto",
                },
            ),
            spacing="2",
            padding="14px",
        ),
        bg="#1e2030",
        border="1px solid #2d3250",
        border_radius="10px",
        margin_bottom="8px",
    )


def _episode_card(ep: dict) -> rx.Component:
    """Render a single episode card."""
    return rx.box(
        rx.hstack(
            rx.text(ep["date"], size="1", color="#a5b4fc"),
            rx.text(ep["content"], size="2", color="#e2e8f0"),
            spacing="3",
        ),
        bg="#1e2030",
        border="1px solid #2d3250",
        border_radius="10px",
        padding="10px 14px",
        margin_bottom="4px",
    )


def rag_explorer_content() -> rx.Component:
    return rx.vstack(
        rx.heading("RAG Explorer", size="6"),
        rx.text(
            "Query ChromaDB directly to inspect retrieved documents — no LLM call.",
            size="2",
            color="#94a3b8",
        ),

        # Query row
        rx.hstack(
            rx.input(
                placeholder="e.g. vacances, boulot, amour…",
                value=RAGExplorerState.query,
                on_change=RAGExplorerState.set_query,
                width="70%",
            ),
            rx.select(
                RAGExplorerState.friend_options,
                value=RAGExplorerState.selected_friend,
                on_change=RAGExplorerState.set_selected_friend,
                width="30%",
            ),
            spacing="3",
            width="100%",
        ),

        # Metadata filters
        rx.accordion.root(
            rx.accordion.item(
                header=rx.text("Metadata Filters", size="2", weight="medium"),
                content=rx.hstack(
                    rx.vstack(
                        rx.text("Source", size="1"),
                        rx.select(
                            [
                                "Any source",
                                "facebook_windowed",
                                "google_locations",
                                "spotify",
                            ],
                            value=RAGExplorerState.source_filter,
                            on_change=RAGExplorerState.set_source_filter,
                        ),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Min messages", size="1"),
                        rx.input(
                            value=RAGExplorerState.min_messages.to(str),
                            on_change=RAGExplorerState.set_min_messages_from_str,
                            type="number",
                            width="100px",
                        ),
                        spacing="1",
                    ),
                    spacing="4",
                ),
                value="filters",
            ),
            type="multiple",
            width="100%",
        ),

        # Search button
        rx.button(
            "Search",
            on_click=RAGExplorerState.search,
            loading=RAGExplorerState.is_searching,
            color_scheme="iris",
            width="100%",
        ),

        # Error
        rx.cond(
            RAGExplorerState.error_message != "",
            rx.callout(
                RAGExplorerState.error_message,
                icon="info",
                color_scheme="amber",
            ),
            rx.fragment(),
        ),

        # Metrics row
        rx.cond(
            RAGExplorerState.result_count > 0,
            rx.hstack(
                _metric("Documents", RAGExplorerState.result_count.to(str)),
                _metric("Strategy", RAGExplorerState.strategy),
                _metric(
                    "Matched friend",
                    rx.cond(
                        RAGExplorerState.matched_friend != "",
                        RAGExplorerState.matched_friend,
                        "—",
                    ),
                ),
                _metric(
                    "Active filters",
                    RAGExplorerState.active_filter_count.to(str),
                ),
                _metric("Mode", RAGExplorerState.mode_label),
                spacing="3",
                width="100%",
                overflow_x="auto",
            ),
            rx.fragment(),
        ),

        # Results — Conversation Chunks
        rx.cond(
            RAGExplorerState.result_count > 0,
            rx.vstack(
                rx.heading(
                    "Conversation Chunks ("
                    + RAGExplorerState.result_count.to(str)
                    + " results)",
                    size="4",
                ),
                rx.foreach(RAGExplorerState.results, _result_card),
                spacing="3",
                width="100%",
            ),
            rx.fragment(),
        ),

        # Episodes
        rx.cond(
            RAGExplorerState.episode_count > 0,
            rx.vstack(
                rx.heading(
                    "Episodic Memories ("
                    + RAGExplorerState.episode_count.to(str)
                    + ")",
                    size="4",
                ),
                rx.foreach(RAGExplorerState.episodes, _episode_card),
                spacing="3",
                width="100%",
            ),
            rx.fragment(),
        ),

        spacing="6",
        width="100%",
        on_mount=RAGExplorerState.load_friends,
    )


def rag_explorer_page() -> rx.Component:
    return layout(rag_explorer_content())
