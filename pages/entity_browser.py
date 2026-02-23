"""Page: Entity Browser"""
import streamlit as st
from ui.entity_browser import render_entity_browser
from ui.settings import render_settings

def page():
    *_, neo4j_uri, neo4j_user, neo4j_password = render_settings()
    render_entity_browser(neo4j_uri, neo4j_user, neo4j_password)
