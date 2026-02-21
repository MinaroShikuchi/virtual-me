"""Page: Vector Store (semantic memory)"""
import streamlit as st
from rag.resources import load_chroma
from ui.ingest import render_vector_tab
from ui.settings import render_settings


def page():
    render_settings()
    collection, episodic = load_chroma()
    render_vector_tab(collection, episodic)
