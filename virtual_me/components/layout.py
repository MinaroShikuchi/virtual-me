"""
virtual_me/components/layout.py — Shared layout: sidebar + navbar + content wrapper.
"""
import reflex as rx
from virtual_me.state.app_state import AppState


def status_badge(label: str, connected: rx.Var[bool], detail: rx.Var[str]) -> rx.Component:
    """A colored status indicator dot + label."""
    return rx.hstack(
        rx.box(
            width="8px",
            height="8px",
            border_radius="50%",
            bg=rx.cond(connected, "#4ade80", "#f87171"),
        ),
        rx.text(
            label,
            size="2",
            weight="medium",
            color=rx.cond(connected, "#4ade80", "#f87171"),
        ),
        rx.text(
            detail,
            size="1",
            color="#94a3b8",
        ),
        spacing="2",
        align="center",
    )


def sidebar_nav_link(text: str, href: str, icon: str) -> rx.Component:
    """A sidebar navigation link."""
    return rx.link(
        rx.hstack(
            rx.icon(icon, size=18),
            rx.text(text, size="2", weight="medium"),
            spacing="3",
            align="center",
            padding_x="12px",
            padding_y="8px",
            border_radius="8px",
            width="100%",
            _hover={"bg": "rgba(99, 102, 241, 0.1)"},
        ),
        href=href,
        underline="none",
        width="100%",
    )


def sidebar() -> rx.Component:
    """Left sidebar with navigation and connection status."""
    return rx.box(
        rx.vstack(
            # Logo / title
            rx.hstack(
                rx.icon("brain", size=24, color="#a5b4fc"),
                rx.heading("Virtual Me", size="4", color="#f1f5f9"),
                spacing="3",
                align="center",
                padding="16px",
            ),
            rx.divider(color_scheme="gray"),

            # Navigation
            rx.vstack(
                sidebar_nav_link("Dashboard", "/", "layout-dashboard"),
                sidebar_nav_link("Chat", "/chat", "message-circle"),
                sidebar_nav_link("Vector", "/vector", "database"),
                sidebar_nav_link("Node Extract", "/graph", "git-branch"),
                sidebar_nav_link("RAG Explorer", "/rag", "search"),
                sidebar_nav_link("Entity Browser", "/browser", "globe"),
                spacing="1",
                padding="8px",
                width="100%",
            ),

            rx.spacer(),

            # Connection status
            rx.vstack(
                rx.text("Connections", size="1", weight="bold", color="#94a3b8"),
                status_badge(
                    "ChromaDB",
                    AppState.chroma_doc_count > 0,
                    rx.cond(
                        AppState.chroma_doc_count > 0,
                        AppState.chroma_doc_count.to(str) + " docs",
                        "disconnected",
                    ),
                ),
                status_badge(
                    "Ollama",
                    AppState.ollama_connected,
                    rx.cond(
                        AppState.ollama_connected,
                        AppState.ollama_model_count.to(str) + " models",
                        "unreachable",
                    ),
                ),
                status_badge(
                    "Neo4j",
                    AppState.neo4j_connected,
                    rx.cond(AppState.neo4j_connected, "connected", "not connected"),
                ),
                spacing="2",
                padding="12px",
                width="100%",
            ),

            rx.divider(color_scheme="gray"),

            # Settings button
            rx.button(
                rx.icon("settings", size=16),
                "Settings",
                variant="ghost",
                width="100%",
                on_click=AppState.open_settings,
            ),

            spacing="1",
            height="100vh",
            width="100%",
        ),
        width="240px",
        min_width="240px",
        bg="linear-gradient(180deg, #1a1d2e 0%, #12141f 100%)",
        border_right="1px solid #2d2f45",
        position="fixed",
        left="0",
        top="0",
        height="100vh",
        overflow_y="auto",
    )


def layout(page_content: rx.Component) -> rx.Component:
    """Wrap a page component with sidebar + settings drawer."""
    from virtual_me.components.settings_drawer import settings_drawer

    return rx.box(
        sidebar(),
        settings_drawer(),
        rx.box(
            page_content,
            margin_left="240px",
            padding="24px",
            min_height="100vh",
            bg="#0f1117",
        ),
        on_mount=AppState.check_connections,
    )
