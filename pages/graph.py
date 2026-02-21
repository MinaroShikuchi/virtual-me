"""Page: Graph"""
import streamlit as st
from ui.graph import render_graph_tab
from ui.settings import render_settings, init_settings_defaults


def page():
    init_settings_defaults()
    *_, neo4j_uri, neo4j_user, neo4j_password = render_settings()
    render_graph_tab(neo4j_uri, neo4j_user, neo4j_password)
