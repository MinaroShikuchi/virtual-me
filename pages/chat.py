"""Page: Chat"""
import streamlit as st
from rag.resources import load_chroma, load_mappings
from ui.chat import render_chat_tab
from ui.settings import render_settings, init_settings_defaults


def page():
    init_settings_defaults()
    model, intent_model, ollama_host, num_ctx, deliberation_rounds, active_personas, enable_thinking, system_prompt, n_results, top_k, do_rerank, hybrid, \
        *_ = render_settings()

    collection, episodic = load_chroma()
    id_to_name, name_to_id = load_mappings()

    render_chat_tab(
        collection, episodic, id_to_name, name_to_id,
        model, intent_model, ollama_host, num_ctx, deliberation_rounds, active_personas, enable_thinking, system_prompt, n_results, top_k, do_rerank, hybrid,
    )
