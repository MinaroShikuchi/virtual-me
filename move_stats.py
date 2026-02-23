import re
from pathlib import Path

# 1. Update pages/dashboard.py
p_dash_path = Path('pages/dashboard.py')
p_dash_lines = p_dash_path.read_text().split('\n')
for i, line in enumerate(p_dash_lines):
    if line.strip() == 'render_settings()':
        p_dash_lines[i] = '    *_, neo4j_uri, neo4j_user, neo4j_password = render_settings()'
    if 'render_dashboard_tab(collection)' in line:
        p_dash_lines[i] = line.replace('render_dashboard_tab(collection)', 'render_dashboard_tab(collection, neo4j_uri, neo4j_user, neo4j_password)')
p_dash_path.write_text('\n'.join(p_dash_lines))


# 2. Extract from ui/graph.py
graph_path = Path('ui/graph.py')
graph_content = graph_path.read_text()

# Extract graph stats block
stats_match = re.search(r'    # ── Graph stats ──\n    st\.divider\(\)\n.*?    finally:\n        client\.close\(\)\n\n    st\.divider\(\)\n', graph_content, re.DOTALL)
stats_block = stats_match.group(0)

# Extract _render_interest_chart_from_data function
chart_match = re.search(r'def _render_interest_chart_from_data\(data: dict\):\n.*?    except Exception:\n        pass\n', graph_content, re.DOTALL)
chart_func = chart_match.group(0)

# Remove them from ui/graph.py
graph_content = graph_content.replace(stats_block, '')
graph_content = graph_content.replace(chart_func, '')
graph_path.write_text(graph_content)


# 3. Add to ui/dashboard.py
dash_path = Path('ui/dashboard.py')
dash_content = dash_path.read_text()

imports_to_add = """
from config import NEO4J_URI, SELF_NAME
from graph.neo4j_client import get_client, Neo4jClient
from graph.constants import ENTITY_LABELS, REL_TYPES, LABEL_COLORS, REL_ICONS

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _try_connect(uri=None, user=None, password=None) -> tuple[Neo4jClient | None, bool]:
    try:
        c = get_client(uri=uri, user=user, password=password)
        return c, c.verify()
    except Exception:
        return None, False
"""

# add chart_func to imports_to_add
imports_to_add += "\n\n" + chart_func + "\n"

# Add imports after existing imports
dash_content = dash_content.replace('from config import SOURCES, DATA_DIR\n', 'from config import SOURCES, DATA_DIR\n' + imports_to_add)

# Change signature of render_dashboard_tab
dash_content = dash_content.replace('def render_dashboard_tab(collection):', 'def render_dashboard_tab(collection, neo4j_uri=None, neo4j_user=None, neo4j_password=None):')

# Add the stats block before 'data/ folder status table'
# Find the insertion point
insert_marker = r'    # ── data/ folder status table ──'

# Replace the title inside stats block to be 'Knowledge Graph Statistics' to differentiate
stats_block = stats_block.replace('st.markdown("#### :material/bar_chart: Graph Statistics', 'st.markdown("#### :material/hub: Knowledge Graph Statistics')

stats_injection = f"""    client, alive = _try_connect(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    if alive:
{stats_block.replace('    # ── Graph stats ──', '    # ── Graph stats ──').replace('st.divider()', '', 1)}
    else:
        st.info("Connect to Neo4j to view knowledge graph stats.")

"""

dash_content = dash_content.replace(insert_marker, stats_injection + insert_marker)

dash_path.write_text(dash_content)
