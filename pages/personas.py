"""Page: Personas"""
import streamlit as st
from ui.personas import render_personas_page
from ui.settings import render_settings, init_settings_defaults


def page():
    init_settings_defaults()
    render_settings()
    render_personas_page()
