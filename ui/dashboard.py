"""
ui/dashboard.py — Dashboard tab: data source overview, ingestion status,
                  memory timeline, and data/ folder status table.
"""
import streamlit as st
import pandas as pd

from config import SOURCES, DATA_DIR


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


def render_dashboard_tab(collection):
    st.markdown("### 📊 Dashboard")

    data_scan     = scan_data_sources()
    chroma_counts = source_chroma_counts(collection)
    total_docs    = collection.count()
    active_srcs   = sum(1 for src in SOURCES
                        if chroma_counts.get(src["chroma_source"], 0) > 0)

    # ── Hero metrics ──
    h1, h2, h3 = st.columns(3)
    h1.metric("🧠 Total chunks in ChromaDB", f"{total_docs:,}")
    h2.metric("📦 Active data sources",       f"{active_srcs} / {len(SOURCES)}")
    h3.metric("📁 Data folder",               str(DATA_DIR.resolve()))

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
            status_icon, status_label, status_color = "✅", f"{ingested:,} chunks", "#22c55e"
        elif files:
            status_icon, status_label, status_color = "📂", f"{len(files)} file(s) ready", "#f59e0b"
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
                    <div style="font-size:2rem;margin-bottom:6px">{src['icon']}</div>
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
                    ">{status_icon} {status_label}</div>
                    {f'<div style="font-size:0.68rem;color:#475569;margin-top:6px">{size_mb:.1f} MB in data/</div>' if files else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Memory timeline: chunks per year ──
    st.markdown("#### 📅 Memory timeline (chunks by year)")
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

    # ── data/ folder status table ──
    st.markdown("#### 📂 `data/` folder status")
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
            "Status":    "✅ Ingested" if chroma_counts.get(src["chroma_source"], 0) > 0
                         else ("📂 Ready" if scan["files"] else "○ Not found"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
