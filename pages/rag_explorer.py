"""Page: RAG Explorer"""
import streamlit as st
from rag.resources import load_chroma, load_mappings
from ui.rag_explorer import render_rag_tab
from ui.settings import render_settings, init_settings_defaults


def page():
    init_settings_defaults()
    model, intent_model, ollama_host, _, _, _, _, _, n_results, top_k, do_rerank, hybrid, *_ = render_settings()

    collection, episodic = load_chroma()
    id_to_name, name_to_id = load_mappings()

    render_rag_tab(
        collection, episodic, id_to_name, name_to_id,
        n_results, top_k, do_rerank, hybrid,
        model, intent_model, ollama_host
    )
