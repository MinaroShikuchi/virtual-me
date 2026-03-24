"""
ui/extract.py — Platform Extract UI.

Upload platform data files and run extractors to populate the knowledge graph.
Supports: Steam, Strava, Spotify, LinkedIn, Google (Timeline + Calendar), Facebook.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, SELF_NAME
from graph.constants import LABEL_COLORS, REL_ICONS
from ui.components.log_viewer import scrollable_log


# ── Platform definitions ──────────────────────────────────────────────────────
# Each extractor entry:
#   label       — display name for the data type selector
#   script      — path to the extractor script (relative to project root)
#   file_label  — label shown on the file uploader
#   file_types  — accepted file extensions
#   multi       — True if multiple files can be uploaded
#   entities    — KG node labels this extractor creates
#   relationships — KG relationship types this extractor creates

PLATFORMS = [
    {
        "id": "steam",
        "label": "Steam",
        "color": "#1b2838",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/8/83/Steam_icon_logo.svg",
        "extractors": [
            {
                "label": "Play Sessions",
                "script": "tools/extractors/steam.py",
                "file_label": "Upload Steam play-session CSV  (columns: appid, start_at, end_at)",
                "file_types": ["csv"],
                "multi": False,
                "entities": ["Person", "Game", "Activity"],
                "relationships": ["PLAYED", "INTERESTED_IN"],
            }
        ],
    },
    {
        "id": "strava",
        "label": "Strava",
        "color": "#FC6100",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/cb/Strava_Logo.svg",
        "extractors": [
            {
                "label": "Activities",
                "script": "tools/extractors/strava.py",
                "file_label": "Upload activities.csv  (Note: GPS/location data requires the local folder path below)",
                "file_types": ["csv", "json"],
                "multi": False,
                "extra_dir": True,          # show local-path text input
                "entities": ["Person", "Activity", "Place", "City", "Country"],
                "relationships": ["PERFORMED", "INTERESTED_IN", "LOCATED_AT", "IN_CITY", "IN_COUNTRY"],
            }
        ],
    },
    {
        "id": "spotify",
        "label": "Spotify",
        "color": "#1DB954",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg",
        "extractors": [
            {
                "label": "Listening History",
                "script": "tools/extractors/spotify.py",
                "file_label": "Upload Streaming History JSON files  (Streaming_History_Audio_*.json)",
                "file_types": ["json"],
                "multi": True,
                "entities": ["Person", "Artist", "Song", "Activity", "Device"],
                "relationships": ["LISTENED_TO", "INTERESTED_IN", "USED_DEVICE"],
            }
        ],
    },
    {
        "id": "linkedin",
        "label": "LinkedIn",
        "color": "#0A66C2",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png",
        "extractors": [
            {
                "label": "Positions",
                "script": "tools/extractors/linkedin_positions.py",
                "file_label": "Upload Positions.csv",
                "file_types": ["csv"],
                "multi": False,
                "entities": ["Person", "Company", "Place", "City", "Country"],
                "relationships": ["WORKS_AT", "LIVES_IN", "IN_CITY", "IN_COUNTRY"],
            },
            {
                "label": "Connections",
                "script": "tools/extractors/linkedin_connections.py",
                "file_label": "Upload Connections.csv",
                "file_types": ["csv"],
                "multi": False,
                "entities": ["Person", "Company"],
                "relationships": ["KNOWS", "WORKS_AT", "COLLEAGUE_OF"],
            },
            {
                "label": "Education",
                "script": "tools/extractors/linkedin_education.py",
                "file_label": "Upload Education.csv",
                "file_types": ["csv"],
                "multi": False,
                "entities": ["Person", "School"],
                "relationships": ["STUDIED_AT"],
            },
        ],
    },
    {
        "id": "google",
        "label": "Google",
        "color": "#4285F4",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg",
        "extractors": [
            {
                "label": "Timeline / Location History",
                "script": "tools/extractors/google_timeline.py",
                "file_label": "Upload Timeline.json or Records.json",
                "file_types": ["json"],
                "multi": False,
                "entities": ["Person", "Place", "City", "Country", "Visit", "Trip"],
                "relationships": [
                    "VISITED", "LIVES_IN", "TOOK_TRIP",
                    "LOCATED_AT", "IN_CITY", "IN_COUNTRY",
                ],
            },
            {
                "label": "Calendar Events",
                "script": "tools/extractors/gcal.py",
                "file_label": "Upload Calendar .ics files",
                "file_types": ["ics"],
                "multi": True,
                "entities": ["Person", "Event", "Place", "City", "Country"],
                "relationships": ["ATTENDED", "LOCATED_AT", "IN_CITY", "IN_COUNTRY"],
            },
        ],
    },
    {
        "id": "facebook",
        "label": "Facebook",
        "color": "#1877F2",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/b/b8/2021_Facebook_icon.svg",
        "extractors": [
            {
                "label": "Messages",
                "script": "tools/extractors/facebook_messages.py",
                "file_label": "Upload facebook_messages.json  (extracted from HTML export via the Data page)",
                "file_types": ["json"],
                "multi": False,
                "entities": ["Person", "Place", "City", "Country", "Company", "Interest"],
                "relationships": [
                    "MET", "VISITED", "LIVES_IN", "WORKS_AT", "INTERESTED_IN",
                    "PARTNER_OF", "FAMILY_OF", "COLLEAGUE_OF", "FRIEND_OF",
                    "IN_CITY", "IN_COUNTRY",
                ],
            },
        ],
    },
]


# ── Helper: colored entity/relationship pills ─────────────────────────────────

def _pills_html(entities: list[str], relationships: list[str]) -> str:
    ent_pills = "".join(
        f'<span style="display:inline-flex;align-items:center;'
        f'background:{LABEL_COLORS.get(e, "#6366f1")}20;'
        f'color:{LABEL_COLORS.get(e, "#6366f1")};'
        f'border:1px solid {LABEL_COLORS.get(e, "#6366f1")}40;'
        f'border-radius:12px;padding:2px 10px;font-size:0.75rem;font-weight:600;'
        f'margin:1px;">{e}</span>'
        for e in entities
    )
    rel_pills = "".join(
        f'<span style="display:inline-flex;align-items:center;'
        f'background:#33415520;color:#94a3b8;border:1px solid #33415540;'
        f'border-radius:12px;padding:2px 10px;font-size:0.75rem;font-weight:500;'
        f'margin:1px;">{r}</span>'
        for r in relationships
    )
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:4px 0 12px 0;">'
        f'<span style="font-size:0.7rem;color:#64748b;font-weight:600;margin-right:2px;">Entities</span>'
        f'{ent_pills}'
        f'<span style="font-size:0.7rem;color:#64748b;font-weight:600;margin-left:8px;margin-right:2px;">'
        f'Relationships</span>{rel_pills}'
        f'</div>'
    )


# ── Arg builder ───────────────────────────────────────────────────────────────

def _build_args(platform_id: str, ext_label: str, cfg: dict) -> list[str]:
    """Build the CLI argument list for an extractor based on platform + config."""
    self_name = cfg.get("self_name", SELF_NAME)
    args: list[str] = ["--self-name", self_name]

    if platform_id == "steam":
        if "csv_file" in cfg:
            args += ["--csv-file", cfg["csv_file"]]
        else:
            args += ["--data-dir", cfg.get("data_dir", "data/steam")]

    elif platform_id == "strava":
        args += ["--data-dir", cfg.get("data_dir", "data/strava")]
        if "csv_file" in cfg:
            args += ["--csv-file", cfg["csv_file"]]

    elif platform_id == "spotify":
        args += ["--data-dir", cfg.get("data_dir", "data/spotify")]

    elif platform_id == "linkedin":
        if "csv_file" in cfg:
            args += ["--csv-file", cfg["csv_file"]]
        if ext_label == "Connections":
            # Connections extractor can optionally use a positions file for enrichment
            pos_file = Path("data/linkedin/Positions.csv")
            if pos_file.exists():
                args += ["--positions-file", str(pos_file)]

    elif platform_id == "google":
        if ext_label == "Timeline / Location History":
            args += ["--records", cfg.get("records", "data/google/Timeline.json")]
        elif ext_label == "Calendar Events":
            args += ["--data-dir", cfg.get("data_dir", "data/google")]

    elif platform_id == "facebook":
        args += ["--json-file", cfg.get("json_file", "facebook_messages.json")]

    return args


# ── Extractor runner ──────────────────────────────────────────────────────────

def _run_extractor(
    platform_id: str,
    ext: dict,
    cfg: dict,
    dry_run: bool,
    uri: str,
    user: str,
    password: str,
) -> None:
    """Build and run the extractor subprocess, streaming logs into the UI."""
    script = Path(ext["script"]).resolve()
    if not script.exists():
        st.error(f"Script not found: {script}")
        return

    args = _build_args(platform_id, ext["label"], cfg)
    cmd = [sys.executable, str(script)] + args
    if dry_run:
        cmd.append("--dry-run")
    if uri:
        cmd += ["--neo4j-uri",  uri]
    if user:
        cmd += ["--neo4j-user", user]
    if password:
        cmd += ["--neo4j-pass", password]

    log_key   = f"log_{platform_id}_{ext['label']}"
    chart_key = f"chart_{platform_id}_{ext['label']}"

    pbar    = st.progress(0, text="Starting…")
    log_box = st.empty()
    lines: list[str] = []

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with st.spinner(f"Running {platform_id} › {ext['label']}…"):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            ls = line.rstrip()

            if ls.startswith("PROGRESS:"):
                try:
                    val = int(ls.split(":")[1].split("%")[0].strip()) / 100.0
                    pbar.progress(min(val, 1.0), text=ls)
                except Exception:
                    pass
                continue

            if ls.startswith("INTERESTS_CHART:"):
                try:
                    st.session_state[chart_key] = json.loads(ls[len("INTERESTS_CHART:"):].strip())
                except Exception:
                    pass
                continue

            lines.append(ls)
            scrollable_log(log_box, lines[-2000:])

        proc.wait()

    pbar.empty()
    st.session_state[log_key] = lines
    scrollable_log(log_box, lines, follow=False)

    # Facebook interest radar chart
    if chart_key in st.session_state:
        _render_interest_chart(chart_key)

    if proc.returncode == 0:
        st.success(f"✅ {ext['label']} finished.")
        if not dry_run:
            if st.button(":material/refresh: Refresh graph stats",
                         key=f"refresh_{platform_id}_{ext['label']}"):
                st.rerun()
    else:
        st.error(
            f"❌ Extraction failed (exit {proc.returncode}). "
            "Check the log above — make sure the file matches the expected format."
        )


# ── Interest radar chart (Facebook Messages) ──────────────────────────────────

def _render_interest_chart(chart_key: str) -> None:
    data = st.session_state.get(chart_key)
    if not data:
        return
    try:
        import plotly.graph_objects as go

        categories = list(data.keys())
        values     = list(data.values())
        cats_c = categories + [categories[0]]
        vals_c = values    + [values[0]]
        COLOR  = "#1877F2"

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats_c, fill="toself",
            fillcolor=f"rgba(24,119,242,0.18)",
            line=dict(color=COLOR, width=2),
            marker=dict(size=6, color=COLOR),
            name="Interests",
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, max(values) * 1.15],
                    tickfont=dict(size=10, color="#aaa"),
                    gridcolor="#333", linecolor="#444",
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="#ddd"),
                    gridcolor="#333", linecolor="#444",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=60, r=60, t=40, b=40),
            height=400,
        )
        top = categories[0] if categories else "—"
        st.markdown(f"#### :material/radar: Interest Profile  ·  Top: **{top.capitalize()}**")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly to see the interest chart.")
    except Exception as e:
        st.warning(f"Could not render interest chart: {e}")


# ── Tab CSS ───────────────────────────────────────────────────────────────────

def _inject_tab_css() -> None:
    css = "".join(
        f"""
        [data-baseweb="tab-list"] button:nth-child({i + 1}) {{
            background-image: url('{p["logo_url"]}');
            background-repeat: no-repeat;
            background-size: 18px 18px;
            background-position: 10px center;
            padding-left: 36px !important;
            font-weight: 600;
            color: #bbb;
        }}
        [data-baseweb="tab-list"] button:nth-child({i + 1})[aria-selected="true"] {{
            color: {p["color"]} !important;
        }}
        [data-baseweb="tab-list"] button:nth-child({i + 1}):hover {{
            color: {p["color"]} !important;
        }}
        """
        for i, p in enumerate(PLATFORMS)
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ── Main render function ──────────────────────────────────────────────────────

def render_extract_page(neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
    st.markdown("### :material/manufacturing: Platform Extract")
    st.caption(
        "Upload data exports from each platform to extract entities & relationships "
        "into your knowledge graph."
    )

    _inject_tab_css()

    tabs = st.tabs([p["label"] for p in PLATFORMS])

    for platform, tab in zip(PLATFORMS, tabs):
        with tab:
            extractors  = platform["extractors"]
            ext_labels  = [e["label"] for e in extractors]

            chosen_label = (
                st.selectbox(
                    "Data type", ext_labels,
                    key=f"sel_{platform['id']}",
                    label_visibility="collapsed",
                )
                if len(ext_labels) > 1
                else ext_labels[0]
            )
            ext     = next(e for e in extractors if e["label"] == chosen_label)
            log_key = f"log_{platform['id']}_{ext['label']}"

            # Entity & relationship pills
            st.markdown(
                _pills_html(ext.get("entities", []), ext.get("relationships", [])),
                unsafe_allow_html=True,
            )

            # ── File upload ───────────────────────────────────────────────────
            cfg: dict = {"self_name": SELF_NAME}

            if ext.get("multi"):
                up_files = st.file_uploader(
                    ext["file_label"],
                    type=ext["file_types"],
                    key=f"up_{platform['id']}_{ext['label']}",
                    accept_multiple_files=True,
                )
                if up_files:
                    target_dir = Path("data") / platform["id"]
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for uf in up_files:
                        (target_dir / uf.name).write_bytes(uf.getbuffer())
                    cfg["data_dir"] = str(target_dir)
                    st.caption(f"📁 {len(up_files)} file(s) saved to `{target_dir}`")

            else:
                # Optional local-folder path (Strava GPS data)
                if ext.get("extra_dir"):
                    local_dir = st.text_input(
                        "Local path to export folder (required for GPS/location data)",
                        value="./data/strava",
                        key=f"dir_{platform['id']}_{ext['label']}",
                    )
                    cfg["data_dir"] = local_dir

                up_file = st.file_uploader(
                    ext["file_label"],
                    type=ext["file_types"],
                    key=f"up_{platform['id']}_{ext['label']}",
                )
                if up_file:
                    target_dir = Path("data") / platform["id"]
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / up_file.name
                    target_path.write_bytes(up_file.getbuffer())

                    # Map uploaded file to the right CLI arg
                    if platform["id"] == "steam":
                        cfg["csv_file"] = str(target_path)
                    elif platform["id"] == "strava" and up_file.name.endswith(".csv"):
                        cfg["csv_file"] = str(target_path)
                    elif platform["id"] == "google":
                        cfg["records"] = str(target_path)
                    elif platform["id"] == "linkedin":
                        cfg["csv_file"] = str(target_path)
                    elif platform["id"] == "facebook":
                        cfg["json_file"] = str(target_path)

                    st.caption(f"📄 Saved to `{target_path}`")

            # ── Persistent log (from previous run) ───────────────────────────
            log_box = st.empty()
            if st.session_state.get(log_key):
                scrollable_log(log_box, st.session_state[log_key], follow=False)

            # ── Controls ──────────────────────────────────────────────────────
            run_col, dry_col = st.columns([1, 1], vertical_alignment="center")
            with dry_col:
                dry_run = st.toggle(
                    "Dry run (no Neo4j write)",
                    value=True,
                    key=f"dry_{platform['id']}_{ext['label']}",
                    help="Print extracted triples without writing to Neo4j",
                )
            with run_col:
                run_clicked = st.button(
                    f":material/play_arrow: Run {ext['label']}",
                    key=f"run_{platform['id']}_{ext['label']}",
                    width="stretch",
                )

            if run_clicked:
                _run_extractor(
                    platform["id"],
                    ext,
                    cfg,
                    dry_run,
                    neo4j_uri  or NEO4J_URI,
                    neo4j_user or NEO4J_USER,
                    neo4j_password or NEO4J_PASSWORD,
                )
