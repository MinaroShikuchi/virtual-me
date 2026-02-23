"""
virtual_me/virtual_me.py — Reflex app entry point with routing.
"""
import reflex as rx

from virtual_me.pages.dashboard import dashboard_page
from virtual_me.pages.entity_browser import entity_browser_page
from virtual_me.pages.rag_explorer import rag_explorer_page
from virtual_me.pages.chat import chat_page
from virtual_me.pages.vector import vector_page
from virtual_me.pages.graph import graph_page
from virtual_me.state.vector_state import VectorState
from virtual_me.state.graph_state import GraphState


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="iris",
    ),
    style={
        "font_family": "'Inter', sans-serif",
        "bg": "#0f1117",
    },
)

app.add_page(dashboard_page, route="/", title="Dashboard — Virtual Me")
app.add_page(chat_page, route="/chat", title="Chat — Virtual Me")
app.add_page(vector_page, route="/vector", title="Vector Store — Virtual Me", on_load=VectorState.load_vector_info)
app.add_page(graph_page, route="/graph", title="Node Extract — Virtual Me", on_load=GraphState.load_graph_info)
app.add_page(entity_browser_page, route="/browser", title="Entity Browser — Virtual Me")
app.add_page(rag_explorer_page, route="/rag", title="RAG Explorer — Virtual Me")
