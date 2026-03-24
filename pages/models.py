"""Page: Models"""
from ui.models import render_models_tab
from ui.settings import render_settings

def page():
    # Make sure settings are initialized
    render_settings()
    # Render the actual tab
    render_models_tab()

page()
