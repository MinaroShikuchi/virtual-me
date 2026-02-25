import streamlit as st
import pandas as pd
from graph.neo4j_client import get_client, Neo4jClient
from graph.constants import ENTITY_LABELS, REL_TYPES

def _try_connect(uri=None, user=None, password=None) -> tuple[Neo4jClient | None, bool]:
    try:
        c = get_client(uri=uri, user=user, password=password)
        return c, c.verify()
    except Exception:
        return None, False

def render_entity_browser(uri=None, user=None, password=None):
    st.markdown("### :material/search: Graph Explorer")
    st.caption("Explore connections for specific entities in the knowledge graph.")
    b1, b2 = st.columns([1, 3])
    with b1:
        label = st.selectbox("Entity type", ENTITY_LABELS, key="kg_browse_label_page")
    with b2:
        search = st.text_input("Search name", placeholder="e.g. Paris, Spotify, hiking…",
                                key="kg_browse_search_page")

    c, alive = _try_connect(uri=uri, user=user, password=password)
    if not alive or c is None:
        st.warning("Neo4j not connected.")
        return

    try:
        if not search:
            st.info("Type a name to search the graph.")
        else:
            names = c.search_nodes(label, search)
            if not names:
                st.warning(f"No {label} nodes matching '{search}'.")
            else:
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
                
        st.divider()
        _render_manual_entry(c)

    finally:
        c.close()

def _render_manual_entry(client: Neo4jClient):
    st.markdown("#### :material/edit_note: Manual Entry")
    st.caption("Manually insert a single Node or a Relationship between two Nodes.")
    
    t1, t2 = st.tabs(["Add Node", "Add Relationship"])
    with t1:
        with st.form("add_node_form"):
            col1, col2 = st.columns(2)
            node_label = col1.selectbox("Node Label", ENTITY_LABELS)
            node_name = col2.text_input("Node Name")
            
            submit_node = st.form_submit_button("Insert Node")
            if submit_node:
                if not node_name.strip():
                    st.error("Node Name cannot be empty.")
                else:
                    try:
                        client.merge_entity(node_label, node_name.strip())
                        st.success(f"Successfully inserted {node_label} '{node_name.strip()}'.")
                    except Exception as e:
                        st.error(f"Failed to insert node: {e}")
                        
    with t2:
        with st.form("add_rel_form"):
            col1, col2, col3 = st.columns([2, 1, 2])
            from_label = col1.selectbox("From Node Label", ENTITY_LABELS, key="fl")
            from_name = col1.text_input("From Node Name", key="fn")
            
            rel_type = col2.selectbox("Relationship", REL_TYPES)
            
            to_label = col3.selectbox("To Node Label", ENTITY_LABELS, key="tl")
            to_name = col3.text_input("To Node Name", key="tn")
            
            submit_rel = st.form_submit_button("Insert Relationship")
            if submit_rel:
                if not from_name.strip() or not to_name.strip():
                    st.error("Both node names must be provided.")
                else:
                    try:
                        client.merge_relation(from_label, from_name.strip(), rel_type, to_label, to_name.strip())
                        st.success(f"Successfully inserted relationship: {from_name} -> {rel_type} -> {to_name}.")
                    except Exception as e:
                        st.error(f"Failed to insert relationship: {e}")
