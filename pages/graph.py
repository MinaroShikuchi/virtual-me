"""Page: Platform Extract"""
import streamlit as st
from ui.extract import render_extract_page
from ui.settings import render_settings, init_settings_defaults


def page():
    init_settings_defaults()
    *_, neo4j_uri, neo4j_user, neo4j_password = render_settings()
    render_extract_page(neo4j_uri, neo4j_user, neo4j_password)

page()
