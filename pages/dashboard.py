"""Page: Dashboard"""
import streamlit as st
from rag.resources import load_chroma
from ui.dashboard import render_dashboard_tab
from ui.settings import render_settings


def page():
    *_, neo4j_uri, neo4j_user, neo4j_password = render_settings()
    collection, _ = load_chroma()
    render_dashboard_tab(collection, neo4j_uri, neo4j_user, neo4j_password)
