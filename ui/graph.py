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
from graph.constants import ENTITY_LABELS, REL_TYPES, LABEL_COLORS, REL_ICONS
from graph.neo4j_client import Neo4jClient, get_client

# ── Extractor groups ────────────────────────────────────────────────────────
PLATFORMS = [
    {
        "id": "facebook",
        "label": "Facebook",
        "icon": "chat",
        "color": "#1877F2",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/b/b8/2021_Facebook_icon.svg",
        "extractors": [
            {
                "label": "Messages",
                "script": "tools/extractors/facebook_messages.py",
                "args": lambda cfg: ["--json-file", cfg.get("json_file", "facebook_messages.json"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload messages.json", type=["json"], key="fb_msg_up")},
                "entities": ["Person", "Place", "Company", "Interest"],
                "relationships": ["MET", "VISITED", "LIVES_IN", "WORKS_AT", "INTERESTED_IN",
                                  "PARTNER_OF", "FAMILY_OF", "COLLEAGUE_OF", "FRIEND_OF"],
            },
            {
                "label": "Friends / Contacts (HTML)",
                "script": "tools/extract_facebook_friends.py",
                "args": lambda cfg: [cfg.get("html_file", "your_friends.html")],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload your_friends.html", type=["html"], key="fb_fr_up")},
                "entities": ["Person"],
                "relationships": ["FRIENDS_WITH"],
            },
            {
                "label": "Removed Friends (HTML)",
                "script": "tools/extract_facebook_friends.py",
                "args": lambda cfg: [cfg.get("html_file", "removed_friends.html"), "--removed"],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload removed_friends.html", type=["html"], key="fb_removed_fr_up")},
                "entities": ["Person"],
                "relationships": ["WAS_FRIENDS_WITH"],
            },
        ]
    },
    {
        "id": "linkedin",
        "label": "LinkedIn",
        "icon": "work",
        "color": "#0A66C2",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png",
        "extractors": [
            {
                "label": "Positions",
                "script": "tools/extractors/linkedin_positions.py",
                "args": lambda cfg: ["--csv-file", cfg.get("csv_file", "Positions.csv"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Positions.csv", type=["csv"], key="li_pos_up")},
                "entities": ["Person", "Company", "Place"],
                "relationships": ["WORKS_AT", "LIVES_IN"],
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
                "entities": ["Person", "Company"],
                "relationships": ["KNOWS", "WORKS_AT", "COLLEAGUE_OF"],
            },
            {
                "label": "Education",
                "script": "tools/extractors/linkedin_education.py",
                "args": lambda cfg: [
                    "--csv-file",  cfg.get("csv_file", "data/linkedin/Education.csv"),
                    "--self-name", cfg.get("self_name", SELF_NAME),
                ],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Education.csv", type=["csv"], key="li_edu_up")},
                "entities": ["Person", "School"],
                "relationships": ["STUDIED_AT"],
            },
        ]
    },
    {
        "id": "spotify",
        "label": "Spotify",
        "icon": "music_note",
        "color": "#1DB954",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg",
        "extractors": [
            {
                "label": "Listening History",
                "script": "tools/extractors/spotify.py",
                "args": lambda cfg: ["--data-dir", cfg.get("data_dir", "data/spotify"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_files": st.file_uploader("Upload Streaming History JSON files", type=["json"], key="sp_up", accept_multiple_files=True)},
                "entities": ["Person", "Artist", "Song", "Activity", "Device"],
                "relationships": ["LISTENED_TO", "INTERESTED_IN", "USED_DEVICE"],
            }
        ]
    },
    {
        "id": "google",
        "label": "Google",
        "icon": "map",
        "color": "#4285F4",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg",
        "extractors": [
            {
                "label": "Location History / Timeline",
                "script": "tools/extractors/google_timeline.py",
                "args": lambda cfg: ["--records", cfg.get("records", "data/google/Timeline.json"), "--self-name", cfg.get("self_name", SELF_NAME)],
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload Timeline.json or Records.json", type=["json"], key="gl_up")},
                "entities": ["Person", "Place"],
                "relationships": ["VISITED", "LIVES_IN"],
            },
            {
                "label": "Calendar Events (.ics)",
                "script": "tools/extractors/gcal.py",
                "args": lambda cfg: (
                    ["--data-dir", cfg["data_dir"], "--self-name", cfg.get("self_name", SELF_NAME)]
                    if "data_dir" in cfg else 
                    ["--data-dir", "data/google", "--self-name", cfg.get("self_name", SELF_NAME)]
                ),
                "extra_fields": lambda: {"uploaded_files": st.file_uploader("Upload Calendar .ics files", type=["ics"], key="gcal_up", accept_multiple_files=True)},
                "entities": ["Person", "Event", "Place"],
                "relationships": ["ATTENDED", "LOCATED_AT"],
            }
        ]
    },
    {
        "id": "strava",
        "label": "Strava",
        "icon": "directions_run",
        "color": "#FC6100",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/cb/Strava_Logo.svg",
        "extractors": [
            {
                "label": "Activities",
                "script": "tools/extractors/strava.py",
                "args": lambda cfg: (
                    ["--csv-file", cfg["csv_file"], "--self-name", cfg.get("self_name", SELF_NAME)]
                    if "csv_file" in cfg else
                    ["--data-dir", cfg.get("data_dir", "data/strava"), "--self-name", cfg.get("self_name", SELF_NAME)]
                ),
                "extra_fields": lambda: {"uploaded_file": st.file_uploader("Upload activities.csv", type=["csv", "json"], key="st_up")},
                "entities": ["Person", "Activity"],
                "relationships": ["INTERESTED_IN"],
            }
        ]
    },
]

# LABEL_COLORS, REL_ICONS imported from graph.constants above



def _scrollable_log(container, lines: list[str], max_height: int = 300):
    """Render log lines inside a scrollable container with reversed layout so newest is at the top."""
    escaped = "\n".join(lines[::-1]).replace("&", "&").replace("<", "<").replace(">", ">")
    container.markdown(
        f'<div style="max-height:{max_height}px;overflow-y:auto;'
        f'background:#0e1117;border:1px solid #333;border-radius:6px;'
        f'padding:10px;font-family:monospace;font-size:13px;'
        f'white-space:pre-wrap;color:#ccc;">{escaped}</div>',
        unsafe_allow_html=True,
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
    st.markdown("### :material/hub: Knowledge Graph")
    st.caption("Episodic memory stored in Neo4j — extract entities & relationships from your data sources.")

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


    # ── Extractor section ──
    _render_extractor_section(alive=True, uri=neo4j_uri, user=neo4j_user, password=neo4j_password)





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
        st.markdown(f"#### :material/radar: Your Interest Profile  ·  Top: **{top_interest.capitalize()}**")
        st.plotly_chart(fig)

    except ImportError:
        st.info("Install plotly (`pip install plotly`) to see the interest chart.")
    except Exception as e:
        st.warning(f"Could not render interest chart: {e}")




def _render_extractor_section(alive: bool, uri=None, user=None, password=None):
    st.markdown("#### :material/manufacturing: Run Extractors")
    if not alive:
        st.info("Connect Neo4j first to run extractors.")

    # Global settings
    gcol1, gcol2, gcol3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    with gcol1:
        self_name = st.text_input(
            "Your name in the graph",
            value=SELF_NAME or "Me",
            key="kg_self_name",
            help="This anchors 'you' as a Person node in the graph",
        )
    with gcol2:
        limit = st.number_input("Limit chunks (0 = all)", min_value=0, value=0,
                                step=500, key="kg_limit")
    with gcol3:
        dry_run = st.toggle("Dry run (no writes)", value=True, key="kg_dry_run",
                            help="Print extracted triples without writing to Neo4j")

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

            # ── Affected entities & relationships ─────────────────────────
            _ent_list = ext.get("entities", [])
            _rel_list = ext.get("relationships", [])
            if _ent_list or _rel_list:
                _ent_pills = "".join(
                    f'<span style="display:inline-flex;align-items:center;gap:3px;'
                    f'background:{LABEL_COLORS.get(e, "#6366f1")}20;'
                    f'color:{LABEL_COLORS.get(e, "#6366f1")};'
                    f'border:1px solid {LABEL_COLORS.get(e, "#6366f1")}40;'
                    f'border-radius:12px;padding:2px 10px;font-size:0.75rem;'
                    f'font-weight:600;">'
                    f'<span class="material-symbols-outlined" style="font-size:14px">category</span>'
                    f'{e}</span>'
                    for e in _ent_list
                )
                _rel_pills = "".join(
                    f'<span style="display:inline-flex;align-items:center;gap:3px;'
                    f'background:#33415520;color:#94a3b8;'
                    f'border:1px solid #33415540;'
                    f'border-radius:12px;padding:2px 10px;font-size:0.75rem;'
                    f'font-weight:500;">'
                    f'<span class="material-symbols-outlined" style="font-size:14px">'
                    f'{REL_ICONS.get(r, "link")}</span>'
                    f'{r}</span>'
                    for r in _rel_list
                )
                st.markdown(
                    f'<div style="display:flex;flex-wrap:wrap;gap:6px;'
                    f'align-items:center;margin:4px 0 10px 0;">'
                    f'<span style="font-size:0.7rem;color:#64748b;font-weight:600;'
                    f'margin-right:2px;">Entities</span>{_ent_pills}'
                    f'<span style="font-size:0.7rem;color:#64748b;font-weight:600;'
                    f'margin-left:8px;margin-right:2px;">Relationships</span>{_rel_pills}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── File upload ───────────────────────────────────────────────
            cfg = ext["extra_fields"]()
            up_file  = cfg.pop("uploaded_file", None)
            up_files = cfg.pop("uploaded_files", None)
            cfg["self_name"] = self_name

            target_path = None

            # Handle multi-file uploads (e.g. Spotify streaming history)
            if up_files:
                target_dir = Path("data") / platform["id"]
                target_dir.mkdir(parents=True, exist_ok=True)
                for uf in up_files:
                    fpath = target_dir / uf.name
                    with open(fpath, "wb") as wf:
                        wf.write(uf.getbuffer())
                cfg["data_dir"] = str(target_dir)
                st.caption(f"📁 {len(up_files)} file(s) saved to `{target_dir}`")

            # Handle single-file uploads
            elif up_file:
                target_dir = Path("data") / platform["id"]
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / up_file.name
                with open(target_path, "wb") as wf:
                    wf.write(up_file.getbuffer())
                if platform["id"] == "facebook" and "Messages" in ext["label"]:
                    cfg["json_file"] = str(target_path)
                elif platform["id"] == "facebook" and "Friends" in ext["label"] and up_file.name.endswith(".html"):
                    cfg["html_file"] = str(target_path)
                elif platform["id"] == "google":
                    cfg["records"] = str(target_path)
                elif platform["id"] == "linkedin":
                    cfg["csv_file"] = str(target_path)
                elif platform["id"] == "spotify":
                    cfg["data_dir"] = str(target_dir)
                elif platform["id"] == "strava":
                    if up_file.name.endswith(".csv"):
                        cfg["csv_file"] = str(target_path)
                    else:
                        cfg["data_dir"] = str(target_dir)

            # ── Persistent log display ─────────────────────────────────────
            log_box = st.empty()
            if st.session_state.get(log_key):
                _scrollable_log(log_box, st.session_state[log_key])

            # ── Run button ──────────────────────────────────────────────────
            run_col, _ = st.columns([1, 3])
            with run_col:
                run_clicked = st.button(
                    f"Run {ext['label']}",
                    key=f"run_{platform['id']}_{ext['label']}",
                    disabled=not (alive or dry_run),
                    width="stretch",
                    icon=":material/play_arrow:",
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
                                _scrollable_log(log_box, lines[-100:])
                        proc.wait()

                    pbar.empty()
                    st.session_state[log_key] = lines
                    if lines:
                        _scrollable_log(log_box, lines)

                    # ── Interest spider chart (Facebook Messages only) ─────
                    chart_key = f"chart_{platform['id']}_{ext['label']}"
                    _render_interest_chart(chart_key)

                    if proc.returncode == 0:
                        st.success(f"{ext['label']} finished successfully.")
                        if st.button("Refresh graph stats", icon=":material/refresh:",
                                     key=f"refresh_{platform['id']}_{ext['label']}"):
                            st.rerun()
                    else:
                        st.error(
                            f"Extraction failed (exit code {proc.returncode}). "
                            "Check the log above — make sure the uploaded file "
                            "matches the expected format."
                        )



