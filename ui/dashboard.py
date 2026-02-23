"""
ui/dashboard.py — Dashboard tab: data source overview, ingestion status,
                  memory timeline, and data/ folder status table.
"""
import streamlit as st
import pandas as pd

from config import SOURCES, DATA_DIR

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


def _render_interest_chart_from_data(data: dict):
    """Renders a Plotly radar chart from interest profile data {name: percentage}."""
    if not data:
        return

    try:
        import plotly.graph_objects as go

        # Sort by percentage descending
        sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        categories = [k for k, _ in sorted_items]
        values     = [v for _, v in sorted_items]

        # Close the radar polygon
        cats_closed = categories + [categories[0]]
        vals_closed = values    + [values[0]]

        COLOR = "#6366f1"

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r            = vals_closed,
            theta        = cats_closed,
            fill         = "toself",
            fillcolor    = _hex_to_rgba(COLOR, 0.18),
            line         = dict(color=COLOR, width=2),
            marker       = dict(size=6, color=COLOR),
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
        st.divider()
        st.markdown(f"#### :material/radar: Interest Profile  ·  Top: **{top_interest.capitalize()}**")
        st.plotly_chart(fig)

    except ImportError:
        pass
    except Exception:
        pass



def scan_data_sources() -> dict:
    """
    Scans DATA_DIR subfolders for known file patterns.
    Returns {source_id: {"files": [Path, ...], "size_mb": float}}
    """
    result = {}
    for src in SOURCES:
        folder = DATA_DIR / src["data_folder"]
        files  = []
        if folder.exists():
            for pat in src["file_patterns"]:
                files.extend(f for f in folder.glob(pat)
                             if f.is_file() and f.suffix != ".md")
        total_bytes = sum(f.stat().st_size for f in files)
        result[src["id"]] = {"files": files, "size_mb": total_bytes / 1_048_576}
    return result


def source_chroma_counts(collection) -> dict:
    """Returns {chroma_source_value: doc_count} for each known source."""
    counts = {}
    for src in SOURCES:
        cs = src["chroma_source"]
        try:
            result   = collection.get(where={"source": {"$eq": cs}}, include=[])
            counts[cs] = len(result["ids"])
        except Exception:
            counts[cs] = 0
    return counts


def render_dashboard_tab(collection, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
    st.markdown("### :material/dashboard: Dashboard")

    data_scan     = scan_data_sources()
    chroma_counts = source_chroma_counts(collection)
    total_docs    = collection.count()
    active_srcs   = sum(1 for src in SOURCES
                        if chroma_counts.get(src["chroma_source"], 0) > 0)

    # ── Hero metrics ──
    h1, h2, h3 = st.columns(3)
    h1.metric("Total chunks in ChromaDB", f"{total_docs:,}")
    h2.metric("Active data sources",       f"{active_srcs} / {len(SOURCES)}")
    h3.metric("Data folder",               str(DATA_DIR.resolve()))

    st.divider()

    # ── Source cards ──
    st.markdown("#### Data Sources")
    cols = st.columns(len(SOURCES))

    for col, src in zip(cols, SOURCES):
        cs       = src["chroma_source"]
        ingested = chroma_counts.get(cs, 0)
        scan     = data_scan[src["id"]]
        files    = scan["files"]
        size_mb  = scan["size_mb"]

        if ingested > 0:
            status_icon, status_label, status_color = "done", f"{ingested:,} chunks", "#22c55e"
        elif files:
            status_icon, status_label, status_color = "folder_open", f"{len(files)} file(s) ready", "#f59e0b"
        else:
            status_icon, status_label, status_color = "○", "No data found", "#64748b"

        with col:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {src['color']}18, {src['color']}08);
                    border: 1px solid {src['color']}40;
                    border-left: 4px solid {src['color']};
                    border-radius: 12px;
                    padding: 20px 16px;
                    margin-bottom: 12px;
                ">
                    <div style="margin-bottom:6px"><span class="material-symbols-outlined" style="font-size:2rem;color:{src['color']}">{src['icon']}</span></div>
                    <div style="font-weight:700;font-size:1rem;color:#f1f5f9;margin-bottom:4px">{src['label']}</div>
                    <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:12px">{src['description']}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:{src['color']};letter-spacing:-0.5px">
                        ~{src['approx_total']:,}
                    </div>
                    <div style="font-size:0.72rem;color:#64748b;margin-bottom:10px">{src['stat_label']}</div>
                    <div style="
                        display:inline-block;
                        background:{status_color}22;
                        color:{status_color};
                        border:1px solid {status_color}55;
                        border-radius:20px;
                        padding:2px 10px;
                        font-size:0.72rem;font-weight:600
                    "><span class="material-symbols-outlined" style="font-size:14px;vertical-align:middle">{status_icon}</span> {status_label}</div>
                    {f'<div style="font-size:0.68rem;color:#475569;margin-top:6px">{size_mb:.1f} MB in data/</div>' if files else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Memory timeline: chunks per year ──
    st.markdown("#### :material/timeline: Memory timeline (chunks by year)")
    try:
        raw = collection.get(include=["metadatas"])
        year_counts: dict[str, int] = {}
        for m in (raw["metadatas"] or []):
            date_str = m.get("date", "")
            if date_str and len(date_str) >= 4:
                y = date_str[:4]
                year_counts[y] = year_counts.get(y, 0) + 1

        if year_counts:
            years  = sorted(year_counts)
            counts = [year_counts[y] for y in years]
            max_c  = max(counts) or 1
            bars   = ""
            for y, c in zip(years, counts):
                pct = c / max_c * 100
                bars += (
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                    f'<div style="width:38px;font-size:0.72rem;color:#94a3b8;text-align:right">{y}</div>'
                    f'<div style="flex:1;background:#1e293b;border-radius:4px;overflow:hidden">'
                    f'<div style="width:{pct:.1f}%;background:linear-gradient(90deg,#6366f1,#a78bfa);'
                    f'height:18px;border-radius:4px"></div></div>'
                    f'<div style="width:50px;font-size:0.72rem;color:#cbd5e1">{c:,}</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="padding:16px 8px;background:#0f172a;border-radius:10px">{bars}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No dated documents found in ChromaDB yet.")
    except Exception as e:
        st.warning(f"Could not build timeline: {e}")

    st.divider()

    client, alive = _try_connect(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    if alive:
        # ── Graph stats ──
    
        st.markdown("#### :material/hub: Knowledge Graph Statistics (Semantic Memory) <span style='font-size:0.8rem;color:#888;'>click a type to explore top 10</span>", unsafe_allow_html=True)
        try:
            stats      = client.graph_stats()
            sel_key    = "graph_selected_label"
            chart_data = None

            # ── Entity type colored buttons ──
            ent_cols = st.columns(len(ENTITY_LABELS))
            for col, label in zip(ent_cols, ENTITY_LABELS):
                count  = stats.get(label, 0)
                is_sel = st.session_state.get(sel_key) == label
                with col:
                    if st.button(
                        f"{label}  {count:,}",
                        key=f"stat_btn_{label}",
                        help=f"Explore top {label} nodes",
                        width="stretch",
                        type="primary" if is_sel else "secondary",
                    ):
                        st.session_state[sel_key] = None if is_sel else label
                        st.rerun()

            # ── Relationship counts (compact pill strip with Material icons) ──
            pills_html = "".join(
                f'<span style="display:inline-flex;align-items:center;gap:4px;'
                f'background:#1e293b;border:1px solid #334155;'
                f'border-radius:20px;padding:4px 12px;margin:3px 4px;font-size:0.78rem;'
                f'color:#cbd5e1;white-space:nowrap;">'
                f'<span class="material-symbols-outlined" style="font-size:16px;color:#94a3b8">'
                f'{REL_ICONS.get(rel, "arrow_forward")}</span>'
                f' <b style="color:#e2e8f0">{rel.replace("_"," ")}</b>'
                f' <span style="color:#6366f1;font-weight:700">{stats.get(f"→{rel}", 0):,}</span>'
                f'</span>'
                for rel in REL_TYPES
            )
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:8px;">'
                f'{pills_html}</div>',
                unsafe_allow_html=True,
            )

            # ── Top-10 radar chart ──
            selected = st.session_state.get(sel_key)
            if selected:
                try:
                    import plotly.graph_objects as go
                    color    = LABEL_COLORS.get(selected, "#6366f1")
                    # Exclude the self-identity node (e.g. "ME") from the chart
                    self_name = st.session_state.get("kg_self_name", SELF_NAME)
                    top_rows = client.top_nodes_by_degree(
                        selected, limit=10, exclude_names=[self_name, "ME"],
                    )
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
                            hovertemplate = "<b>%{theta}</b><br>"
                                            + ({"Activity": "Activities: %{r}",
                                                "Artist":   "Songs listened: %{r}",
                                                "Song":     "Listens: %{r}",
                                               }.get(selected, "Connections: %{r}"))
                                            + "<extra></extra>",
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
                        metric = {"Activity": "activities", "Artist": "songs listened",
                                  "Song": "listens"}.get(selected, "connections")
                        st.markdown(f"##### Top {len(top_rows)} **{selected}** nodes by {metric}")
                        st.plotly_chart(fig)
                    else:
                        st.info(f"No {selected} nodes in the graph yet.")
                except ImportError:
                    st.info("Install plotly to see charts: `pip install plotly`")
                except Exception as ex:
                    st.warning(f"Could not load top nodes: {ex}")

            # ── Interest profile spider chart (from Neo4j) ──
            try:
                self_name = st.session_state.get("kg_self_name", SELF_NAME)
                interest_data = client.interest_profile(self_name=self_name)
                if interest_data:
                    _render_interest_chart_from_data(interest_data)
            except Exception as ex:
                pass  # silently skip if no interest data

        except Exception as e:
            st.warning(f"Could not load stats: {e}")
        finally:
            client.close()

        st.divider()

    else:
        st.info("Connect to Neo4j to view knowledge graph stats.")

    # ── data/ folder status table ──
    st.markdown("#### :material/folder: `data/` folder status")
    rows = []
    for src in SOURCES:
        scan  = data_scan[src["id"]]
        flist = ", ".join(f.name for f in scan["files"][:3])
        if len(scan["files"]) > 3:
            flist += f" (+{len(scan['files'])-3} more)"
        rows.append({
            "Source":    f"{src['icon']} {src['label']}",
            "Folder":    f"data/{src['data_folder']}/",
            "Files":     flist or "—",
            "Size (MB)": f"{scan['size_mb']:.1f}" if scan["files"] else "—",
            "Status":    "Ingested" if chroma_counts.get(src["chroma_source"], 0) > 0
                         else ("Ready" if scan["files"] else "Not found"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
