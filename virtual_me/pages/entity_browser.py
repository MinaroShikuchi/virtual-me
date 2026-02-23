"""
virtual_me/pages/entity_browser.py — Entity Browser page.
"""
import reflex as rx
from virtual_me.state.entity_browser_state import EntityBrowserState
from virtual_me.components.layout import layout
from graph.constants import ENTITY_LABELS


def _neighbour_row(item: dict) -> rx.Component:
    """Render a single neighbour row in the table."""
    return rx.table.row(
        rx.table.cell(rx.text(item["rel"], size="2")),
        rx.table.cell(rx.text(item["label"], size="2")),
        rx.table.cell(rx.text(item["name"], size="2")),
    )


def entity_browser_content() -> rx.Component:
    return rx.vstack(
        rx.heading("Entity Browser", size="6"),
        rx.text(
            "Explore connections for specific entities in the knowledge graph.",
            size="2",
            color="#94a3b8",
        ),

        # Search controls
        rx.hstack(
            rx.vstack(
                rx.text("Entity type", size="2", weight="medium"),
                rx.select(
                    ENTITY_LABELS,
                    value=EntityBrowserState.selected_label,
                    on_change=EntityBrowserState.set_selected_label,
                    width="200px",
                ),
                spacing="1",
            ),
            rx.vstack(
                rx.text("Search name", size="2", weight="medium"),
                rx.input(
                    placeholder="e.g. Paris, Spotify, hiking…",
                    value=EntityBrowserState.search_query,
                    on_change=EntityBrowserState.set_search_query,
                    width="400px",
                ),
                spacing="1",
            ),
            rx.button(
                "Search",
                on_click=EntityBrowserState.search_nodes,
                loading=EntityBrowserState.is_searching,
                color_scheme="iris",
                align_self="end",
            ),
            spacing="4",
            align="end",
        ),

        # Error/info messages
        rx.cond(
            EntityBrowserState.error_message != "",
            rx.callout(
                EntityBrowserState.error_message,
                icon="info",
                color_scheme="amber",
            ),
            rx.fragment(),
        ),

        # Search results — node selector
        rx.cond(
            EntityBrowserState.search_results.length() > 0,
            rx.vstack(
                rx.text("Select node", size="2", weight="medium"),
                rx.select(
                    EntityBrowserState.search_results,
                    value=EntityBrowserState.selected_node,
                    on_change=EntityBrowserState.load_neighbours,
                    width="400px",
                ),
                spacing="1",
            ),
            rx.fragment(),
        ),

        # Neighbours table
        rx.cond(
            EntityBrowserState.neighbours.length() > 0,
            rx.vstack(
                rx.text(
                    EntityBrowserState.neighbours.length().to(str)
                    + " relationships found",
                    size="2",
                    weight="medium",
                ),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Relationship"),
                            rx.table.column_header_cell("Label"),
                            rx.table.column_header_cell("Name"),
                        ),
                    ),
                    rx.table.body(
                        rx.foreach(
                            EntityBrowserState.neighbours,
                            _neighbour_row,
                        ),
                    ),
                    width="100%",
                ),
                spacing="3",
                width="100%",
            ),
            rx.fragment(),
        ),

        spacing="6",
        width="100%",
    )


def entity_browser_page() -> rx.Component:
    return layout(entity_browser_content())
