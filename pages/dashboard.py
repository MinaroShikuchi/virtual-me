"""Page: Dashboard"""
import streamlit as st
from rag.resources import load_chroma
from ui.dashboard import render_dashboard_tab
from ui.settings import render_settings


def page():
    render_settings()
    collection, _ = load_chroma()
    render_dashboard_tab(collection)
