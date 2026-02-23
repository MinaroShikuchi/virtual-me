import streamlit as st
import pandas as pd
from graph.neo4j_client import get_client, Neo4jClient
from graph.constants import ENTITY_LABELS

def _try_connect(uri=None, user=None, password=None) -> tuple[Neo4jClient | None, bool]:
    try:
        c = get_client(uri=uri, user=user, password=password)
        return c, c.verify()
    except Exception:
        return None, False

def render_entity_browser(uri=None, user=None, password=None):
    st.markdown("### :material/search: Entity Browser")
    st.caption("Explore connections for specific entities in the knowledge graph.")
    b1, b2 = st.columns([1, 3])
    with b1:
        label = st.selectbox("Entity type", ENTITY_LABELS, key="kg_browse_label_page")
    with b2:
        search = st.text_input("Search name", placeholder="e.g. Paris, Spotify, hiking…",
                                key="kg_browse_search_page")

    if not search:
        st.info("Type a name to search the graph.")
        return

    c, alive = _try_connect(uri=uri, user=user, password=password)
    if not alive or c is None:
        st.warning("Neo4j not connected.")
        return

    try:
        names = c.search_nodes(label, search)
        if not names:
            st.warning(f"No {label} nodes matching '{search}'.")
            return

        selected = st.selectbox("Select node", names, key="kg_browse_node_page")
        if selected:
            neighbours = c.neighbours(label, selected)
            if not neighbours:
                st.info("No relationships found for this node.")
            else:
                st.markdown(
                    f"**{len(neighbours)}** relationships for "
                    f"[{label}] **{selected}**"
                )
                df = pd.DataFrame(neighbours).rename(columns={
                    "rel": "Relationship", "label": "Entity Type", "name": "Name"
                })
                st.dataframe(df, width="stretch", hide_index=True)
    finally:
        c.close()
