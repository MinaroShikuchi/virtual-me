"""
virtual_me/state/entity_browser_state.py — Entity Browser state.
"""
import reflex as rx
from virtual_me.state.app_state import AppState


class EntityBrowserState(AppState):
    """State for the Entity Browser page."""

    selected_label: str = "Person"
    search_query: str = ""
    search_results: list[str] = []
    selected_node: str = ""
    neighbours: list[dict] = []
    is_searching: bool = False
    error_message: str = ""

    @rx.event
    def set_selected_label(self, value: str):
        """Set the selected node label."""
        self.selected_label = value

    @rx.event
    def set_search_query(self, value: str):
        """Set the search query."""
        self.search_query = value

    def search_nodes(self):
        """Search Neo4j for nodes matching the query."""
        if not self.search_query:
            self.search_results = []
            self.neighbours = []
            return

        self.is_searching = True
        self.error_message = ""
        try:
            from graph.neo4j_client import get_client
            with get_client(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            ) as client:
                if not client.verify():
                    self.error_message = "Neo4j not connected."
                    self.search_results = []
                    return
                names = client.search_nodes(self.selected_label, self.search_query)
                self.search_results = names if names else []
                if not names:
                    self.error_message = f"No {self.selected_label} nodes matching '{self.search_query}'."
        except Exception as e:
            self.error_message = f"Error: {e}"
            self.search_results = []
        finally:
            self.is_searching = False

    def load_neighbours(self, node_name: str):
        """Load relationships for a selected node."""
        self.selected_node = node_name
        if not node_name:
            self.neighbours = []
            return

        try:
            from graph.neo4j_client import get_client
            with get_client(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            ) as client:
                raw = client.neighbours(self.selected_label, node_name)
                self.neighbours = raw if raw else []
        except Exception as e:
            self.error_message = f"Error loading neighbours: {e}"
            self.neighbours = []
