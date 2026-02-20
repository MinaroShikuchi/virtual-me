"""
ui/graph.py — Knowledge Graph tab.

Sections:
  1. Connection status + graph stats (entity/relationship counts)
  2. Run extractors (select sources, dry-run toggle, live log stream)
  3. Entity browser (pick label + name, see neighbours as a table)
"""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, SELF_NAME
from graph.neo4j_client import Neo4jClient, ENTITY_LABELS, REL_TYPES, get_client

# ── Extractor groups ────────────────────────────────────────────────────────
PLATFORMS = [
    {
        "id": "facebook",
        "label": "Facebook",
        "icon": "💬",
        "color": "#1877F2",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/b/b8/2021_Facebook_icon.svg",
        "extractors": [
            {
                "label": "Messages",
                "script": "tools/extractors/facebook_messages.py",
                "args": lambda cfg: ["--json-file", cfg.get("json_file", "facebook_messages.json"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload messages.json", type=["json"], key="fb_msg_up")},
            },
            {
                "label": "Friends / Contacts",
                "script": "tools/extractors/facebook_contacts.py",
                "args": lambda cfg: ["--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload friends.json", type=["json"], key="fb_fr_up")},
            },
        ]
    },
    {
        "id": "linkedin",
        "label": "LinkedIn",
        "icon": "💼",
        "color": "#0A66C2",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png",
        "extractors": [
            {
                "label": "Positions",
                "script": "tools/extractors/linkedin_positions.py",
                "args": lambda cfg: ["--csv-file", cfg.get("csv_file", "Positions.csv"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Positions.csv", type=["csv"], key="li_pos_up")},
            },
            {
                "label": "Connections",
                "script": "tools/extractors/linkedin_connections.py",
                "args": lambda cfg: [
                    "--csv-file",       cfg.get("csv_file", "data/linkedin/Connections.csv"),
                    "--positions-file", str(Path("data/linkedin/Positions.csv")),
                    "--self-name",      cfg.get("self_name", SELF_NAME),
                ],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Connections.csv", type=["csv"], key="li_conn_up")},
            },
            {
                "label": "Education",
                "script": "tools/extractors/linkedin_education.py",
                "args": lambda cfg: [
                    "--csv-file",  cfg.get("csv_file", "data/linkedin/Education.csv"),
                    "--self-name", cfg.get("self_name", SELF_NAME),
                ],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Education.csv", type=["csv"], key="li_edu_up")},
            },
        ]
    },
    {
        "id": "spotify",
        "label": "Spotify",
        "icon": "🎵",
        "color": "#1DB954",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg",
        "extractors": [
            {
                "label": "Listening History",
                "script": "tools/extractors/spotify.py",
                "args": lambda cfg: ["--data-dir", cfg.get("data_dir", "data/spotify"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Spotify History", type=["json"], key="sp_up")},
            }
        ]
    },
    {
        "id": "google",
        "label": "Google",
        "icon": "🗺️",
        "color": "#4285F4",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg",
        "extractors": [
            {
                "label": "Location History",
                "script": "tools/extractors/google_timeline.py",
                "args": lambda cfg: ["--records", cfg.get("records", "Records.json"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Records.json", type=["json"], key="gl_up")},
            }
        ]
    },
    {
        "id": "strava",
        "label": "Strava",
        "icon": "🏃",
        "color": "#FC6100",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/cb/Strava_Logo.svg",
        "extractors": [
            {
                "label": "Activities",
                "script": "tools/extractors/strava.py",
                "args": lambda cfg: ["--data-dir", cfg.get("data_dir", "data/strava"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Strava", type=["json"], key="st_up")},
            }
        ]
    },
]

# ── Entity colour palette ─────────────────────────────────────────────────────
_LABEL_COLORS = {
    "Person":   "#6366f1",
    "Place":    "#22c55e",
    "Song":     "#f59e0b",
    "Artist":   "#ec4899",
    "Company":  "#0ea5e9",
    "Game":     "#8b5cf6",
    "Activity": "#14b8a6",
    "Interest": "#f97316",
}


def _stat_card(label: str, value: int, color: str, selected: bool = False) -> str:
    border = f"2px solid {color}" if selected else f"1px solid {color}40"
    bg     = f"{color}28"         if selected else f"{color}12"
    return (
        f'<div style="background:{bg};border:{border};'
        f'border-left:4px solid {color};border-radius:10px;'
        f'padding:14px 16px;text-align:center;cursor:pointer;">'
        f'<div style="font-size:1.6rem;font-weight:800;color:{color}">{value:,}</div>'
        f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:2px">{label}</div>'
        f'</div>'
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _try_connect(uri=None, user=None, password=None) -> tuple[Neo4jClient | None, bool]:
    """Returns (client, is_connected). Caller must close client."""
    try:
        c = get_client(uri=uri, user=user, password=password)
        return c, c.verify()
    except Exception:
        return None, False


# ── Main render function ──────────────────────────────────────────────────────

def render_graph_tab(neo4j_uri=None, neo4j_user=None, neo4j_password=None):
    st.markdown("### 🕸️ Knowledge Graph")
    st.caption("Extract entities & relationships from your data sources and store them in Neo4j.")

    # ── Connection banner ──
    client, alive = _try_connect(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    if alive:
        st.markdown(
            f'<span class="status-ok">● Neo4j</span> — connected at <code>{neo4j_uri or NEO4J_URI}</code>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-err">○ Neo4j</span> — unreachable at <code>{neo4j_uri or NEO4J_URI}</code>. '
            f'Start it with <code>docker compose up neo4j -d</code>',
            unsafe_allow_html=True,
        )
        if client:
            client.close()
        st.divider()
        _render_extractor_section(alive=False)
        return

    # ── Graph stats ──
    st.divider()
    st.markdown("#### 📊 Graph Statistics  <span style='font-size:0.8rem;color:#888;'>click a type to explore top 10</span>", unsafe_allow_html=True)
    try:
        stats      = client.graph_stats()
        sel_key    = "graph_selected_label"
        chart_data = None

        # ── Entity type buttons ──
        ent_cols = st.columns(len(ENTITY_LABELS))
        for col, label in zip(ent_cols, ENTITY_LABELS):
            color  = _LABEL_COLORS.get(label, "#6366f1")
            count  = stats.get(label, 0)
            is_sel = st.session_state.get(sel_key) == label
            # Colored stat card (visual)
            col.markdown(_stat_card(label, count, color, selected=is_sel),
                         unsafe_allow_html=True)
            # Invisible click button overlaid via CSS
            col.markdown(
                f"<style>div[data-testid='column']:has(button[data-testid='baseButton-secondary'][aria-label='{label}']){{margin-top:-78px}}</style>",
                unsafe_allow_html=True,
            )
            if col.button(" ", key=f"stat_btn_{label}", help=f"Explore top {label} nodes",
                          use_container_width=True, args=None):
                st.session_state[sel_key] = None if is_sel else label
                st.rerun()

        # ── Relationship counts ──
        st.markdown("")
        rel_cols = st.columns(len(REL_TYPES))
        for col, rel in zip(rel_cols, REL_TYPES):
            col.metric(f"→{rel}", f"{stats.get(f'→{rel}', 0):,}")

        # ── Top-10 radar chart ──
        selected = st.session_state.get(sel_key)
        if selected:
            try:
                import plotly.graph_objects as go
                color    = _LABEL_COLORS.get(selected, "#6366f1")
                top_rows = client.top_nodes_by_degree(selected, limit=10)
                if top_rows:
                    names   = [r["name"]   for r in top_rows]
                    degrees = [r["degree"] for r in top_rows]

                    # Radar requires ≥3 axes; pad short lists
                    while len(names) < 3:
                        names.append(""); degrees.append(0)

                    nm_c = names   + [names[0]]
                    dg_c = degrees + [degrees[0]]

                    fig = go.Figure(go.Scatterpolar(
                        r             = dg_c,
                        theta         = nm_c,
                        fill          = "toself",
                        fillcolor     = _hex_to_rgba(color, 0.18),
                        line          = dict(color=color, width=2),
                        marker        = dict(size=6, color=color),
                        hovertemplate = "<b>%{theta}</b><br>Connections: %{r}<extra></extra>",
                    ))
                    fig.update_layout(
                        polar=dict(
                            bgcolor     = "rgba(0,0,0,0)",
                            radialaxis  = dict(
                                visible   = True,
                                tickfont  = dict(size=9, color="#aaa"),
                                gridcolor = "#333",
                                linecolor = "#444",
                            ),
                            angularaxis = dict(
                                tickfont  = dict(size=11, color="#ddd"),
                                gridcolor = "#333",
                                linecolor = "#444",
                            ),
                        ),
                        paper_bgcolor = "rgba(0,0,0,0)",
                        showlegend    = False,
                        margin        = dict(l=70, r=70, t=30, b=30),
                        height        = 360,
                    )
                    st.markdown(f"##### Top {len(top_rows)} **{selected}** nodes by connections")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No {selected} nodes in the graph yet.")
            except ImportError:
                st.info("Install plotly to see charts: `pip install plotly`")
            except Exception as ex:
                st.warning(f"Could not load top nodes: {ex}")

    except Exception as e:
        st.warning(f"Could not load stats: {e}")
    finally:
        client.close()

    st.divider()

    # ── Extractor section ──
    _render_extractor_section(alive=True, uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

    st.divider()

    # ── Entity browser ──
    _render_browser(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)



# ── Interest spider chart ─────────────────────────────────────────────────────
def _render_interest_chart(chart_key: str):
    """Renders a Plotly radar chart if interest data is stored in session_state."""
    data = st.session_state.get(chart_key)
    if not data:
        return

    try:
        import plotly.graph_objects as go

        categories = list(data.keys())
        values     = list(data.values())

        # Close the radar polygon
        cats_closed = categories + [categories[0]]
        vals_closed = values    + [values[0]]

        PLATFORM_COLOR = "#1877F2"  # Facebook blue; override per-platform if needed

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r            = vals_closed,
            theta        = cats_closed,
            fill         = "toself",
            fillcolor    = _hex_to_rgba(PLATFORM_COLOR, 0.18),
            line         = dict(color=PLATFORM_COLOR, width=2),
            marker       = dict(size=6, color=PLATFORM_COLOR),
            name         = "Interests",
            hovertemplate = "<b>%{theta}</b><br>Score: %{r:.1f}%<extra></extra>",
        ))

        fig.update_layout(
            polar=dict(
                bgcolor    = "rgba(0,0,0,0)",
                radialaxis = dict(
                    visible   = True,
                    range     = [0, max(values) * 1.15],
                    tickfont  = dict(size=10, color="#aaa"),
                    gridcolor = "#333",
                    linecolor = "#444",
                ),
                angularaxis = dict(
                    tickfont  = dict(size=12, color="#ddd"),
                    gridcolor = "#333",
                    linecolor = "#444",
                ),
            ),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            showlegend    = False,
            margin        = dict(l=60, r=60, t=40, b=40),
            height        = 400,
        )

        top_interest = categories[0] if categories else "—"
        st.markdown(f"#### 🕷️ Your Interest Profile  ·  Top: **{top_interest.capitalize()}**")
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.info("Install plotly (`pip install plotly`) to see the interest chart.")
    except Exception as e:
        st.warning(f"Could not render interest chart: {e}")


def _render_extractor_section(alive: bool, uri=None, user=None, password=None):
    st.markdown("#### ⚙️ Run Extractors")
    if not alive:
        st.info("Connect Neo4j first to run extractors.")

    # Global settings
    gcol1, gcol2, gcol3 = st.columns([2, 1, 1])
    with gcol1:
        self_name = st.text_input(
            "Your name in the graph",
            value=SELF_NAME or "Me",
            key="kg_self_name",
            help="This anchors 'you' as a Person node in the graph",
        )
    with gcol2:
        dry_run = st.toggle("Dry run (no writes)", value=True, key="kg_dry_run",
                            help="Print extracted triples without writing to Neo4j")
    with gcol3:
        limit = st.number_input("Limit chunks (0 = all)", min_value=0, value=0,
                                step=500, key="kg_limit")

    st.markdown("")

    # Inject CSS to render real brand icons in the tab buttons
    tab_css = "".join([
        f"""
        [data-baseweb="tab-list"] button:nth-child({i+1}) {{
            background-image: url('{p['logo_url']}');
            background-repeat: no-repeat;
            background-size: 18px 18px;
            background-position: 10px center;
            padding-left: 36px !important;
            font-weight: 600;
            color: #bbb;
        }}
        [data-baseweb="tab-list"] button:nth-child({i+1})[aria-selected="true"] {{
            color: {p['color']} !important;
        }}
        [data-baseweb="tab-list"] button:nth-child({i+1}):hover {{
            color: {p['color']} !important;
        }}
        """
        for i, p in enumerate(PLATFORMS)
    ])
    st.markdown(f"<style>{tab_css}</style>", unsafe_allow_html=True)

    tabs = st.tabs([p['label'] for p in PLATFORMS])
    for platform, tab in zip(PLATFORMS, tabs):
        with tab:
            # ── Data type selector ──────────────────────────────────────────
            ext_labels = [e["label"] for e in platform["extractors"]]
            chosen_label = (
                st.selectbox("Data type", ext_labels,
                             key=f"sel_{platform['id']}",
                             label_visibility="collapsed")
                if len(ext_labels) > 1 else ext_labels[0]
            )
            ext = next(e for e in platform["extractors"] if e["label"] == chosen_label)
            log_key = f"log_{platform['id']}_{ext['label']}"

            # ── File upload ───────────────────────────────────────────────
            cfg = ext["extra_fields"]()
            up_file = cfg.pop("uploaded_file", None)
            cfg["self_name"] = self_name

            target_path = None
            if up_file:
                target_dir = Path("data") / platform["id"]
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / up_file.name
                with open(target_path, "wb") as wf:
                    wf.write(up_file.getbuffer())
                if platform["id"] == "facebook" and "Messages" in ext["label"]:
                    cfg["json_file"] = str(target_path)
                elif platform["id"] == "google":
                    cfg["records"] = str(target_path)
                elif platform["id"] == "linkedin":
                    cfg["csv_file"] = str(target_path)
                elif platform["id"] in ["spotify", "strava"]:
                    cfg["data_dir"] = str(target_dir)

            # ── Persistent log display ─────────────────────────────────────
            log_box = st.empty()
            if st.session_state.get(log_key):
                log_box.code("\n".join(st.session_state[log_key]), language="")

            # ── Run button ──────────────────────────────────────────────────
            run_col, _ = st.columns([1, 3])
            with run_col:
                run_clicked = st.button(
                    f"▶️ Run {ext['label']}",
                    key=f"run_{platform['id']}_{ext['label']}",
                    disabled=not (alive or dry_run),
                    use_container_width=True,
                )

                        
            if run_clicked:
                script = Path(ext["script"]).resolve()
                if not script.exists():
                    st.error(f"Script not found: {script}")
                else:
                    cmd = [sys.executable, str(script)] + ext["args"](cfg)
                    if dry_run: cmd.append("--dry-run")
                    if limit:   cmd += ["--limit", str(limit)]
                    if uri:      cmd += ["--neo4j-uri", uri]
                    if user:     cmd += ["--neo4j-user", user]
                    if password: cmd += ["--neo4j-pass", password]

                    pbar  = st.progress(0, text="Starting extraction...")
                    lines: list[str] = []

                    with st.spinner(f"Running {platform['label']} ▸ {ext['label']}…"):
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                        )
                        for line in proc.stdout:
                            ls = line.rstrip()
                            if ls.startswith("PROGRESS:"):
                                try:
                                    val = int(ls.split(":")[1].split("%")[0].strip()) / 100.0
                                    pbar.progress(min(val, 1.0), text=ls)
                                except Exception: pass
                                continue
                            if ls.startswith("INTERESTS_CHART:"):
                                try:
                                    chart_key = f"chart_{platform['id']}_{ext['label']}"
                                    st.session_state[chart_key] = json.loads(ls[len("INTERESTS_CHART:"):].strip())
                                except Exception: pass
                                continue
                            if any(t in ls for t in
                                   ["[ENT]", "[REL]", "📊", "✅", "❌",
                                    "🕵", "🏠", "📦", "💼", "📂"]):
                                lines.append(ls)
                                log_box.code("\n".join(lines[-100:]), language="")
                        proc.wait()

                    pbar.empty()
                    st.session_state[log_key] = lines
                    if lines:
                        log_box.code("\n".join(lines), language="")

                    # ── Interest spider chart (Facebook Messages only) ─────
                    chart_key = f"chart_{platform['id']}_{ext['label']}"
                    _render_interest_chart(chart_key)

                    if proc.returncode == 0:
                        st.success(f"✅ {ext['label']} finished successfully.")
                        if st.button("🔄 Refresh graph stats",
                                     key=f"refresh_{platform['id']}_{ext['label']}"):
                            st.rerun()
                    else:
                        st.error(
                            f"❌ Extraction failed (exit code {proc.returncode}). "
                            "Check the log above — make sure the uploaded file "
                            "matches the expected format."
                        )


def _render_browser(uri=None, user=None, password=None):
    st.markdown("#### 🔎 Entity Browser")
    b1, b2 = st.columns([1, 3])
    with b1:
        label = st.selectbox("Entity type", ENTITY_LABELS, key="kg_browse_label")
    with b2:
        search = st.text_input("Search name", placeholder="e.g. Paris, Spotify, hiking…",
                                key="kg_browse_search")

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

        selected = st.selectbox("Select node", names, key="kg_browse_node")
        if selected:
            neighbours = c.neighbours(label, selected)
            if not neighbours:
                st.info("No relationships found for this node.")
            else:
                st.markdown(
                    f"**{len(neighbours)}** relationships for "
                    f"[{label}] **{selected}**"
                )
                import pandas as pd
                df = pd.DataFrame(neighbours).rename(columns={
                    "rel": "Relationship", "label": "Entity Type", "name": "Name"
                })
                st.dataframe(df, width="stretch", hide_index=True)
    finally:
        c.close()
