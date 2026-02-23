"""
virtual_me/state/dashboard_state.py — Dashboard page state.
"""
import reflex as rx
from pydantic import BaseModel
from virtual_me.state.app_state import AppState


class SourceInfo(BaseModel):
    """Model for a data source card — used in rx.foreach."""

    id: str = ""
    label: str = ""
    icon: str = "database"
    color: str = "#6366f1"
    description: str = ""
    stat_label: str = ""
    approx_total: int = 0
    file_count: int = 0
    size_mb: float = 0.0
    chroma_source: str = ""
    ingested: int = 0


class DashboardState(AppState):
    """State for the Dashboard page."""

    # Source data
    source_scan: list[SourceInfo] = []
    total_docs: int = 0
    active_sources: int = 0
    year_counts: dict[str, int] = {}
    graph_stats: dict[str, int] = {}
    interest_data: dict[str, float] = {}

    def load_dashboard(self):
        """Load all dashboard data."""
        from config import SOURCES, DATA_DIR
        from pathlib import Path

        # Scan data sources
        scan_results: list[SourceInfo] = []
        for src in SOURCES:
            folder = DATA_DIR / src["data_folder"]
            files = []
            if folder.exists():
                for pat in src["file_patterns"]:
                    files.extend(
                        f for f in folder.glob(pat)
                        if f.is_file() and f.suffix != ".md"
                    )
            total_bytes = sum(f.stat().st_size for f in files)
            scan_results.append(
                SourceInfo(
                    id=src["id"],
                    label=src["label"],
                    icon=src.get("icon", "database"),
                    color=src.get("color", "#6366f1"),
                    description=src.get("description", ""),
                    stat_label=src.get("stat_label", ""),
                    approx_total=src.get("approx_total", 0),
                    file_count=len(files),
                    size_mb=round(total_bytes / 1_048_576, 1),
                    chroma_source=src.get("chroma_source", ""),
                    ingested=0,
                )
            )

        # ChromaDB counts per source
        try:
            from services.embedding_service import get_embedding_func
            from services.chroma_service import get_collection

            ef = get_embedding_func(self.embedding_model)
            collection = get_collection(ef)
            self.total_docs = collection.count()

            active = 0
            for info in scan_results:
                try:
                    result = collection.get(
                        where={"source": {"$eq": info.chroma_source}},
                        include=[],
                    )
                    count = len(result["ids"])
                    info.ingested = count
                    if count > 0:
                        active += 1
                except Exception:
                    info.ingested = 0
            self.active_sources = active

            # Year counts for timeline
            try:
                raw = collection.get(include=["metadatas"])
                yc: dict[str, int] = {}
                for m in raw.get("metadatas", []) or []:
                    date_str = m.get("date", "")
                    if date_str and len(date_str) >= 4:
                        year = date_str[:4]
                        yc[year] = yc.get(year, 0) + 1
                self.year_counts = dict(sorted(yc.items()))
            except Exception:
                self.year_counts = {}

        except Exception:
            self.total_docs = 0
            self.active_sources = 0
            self.year_counts = {}

        self.source_scan = scan_results

        # Graph stats
        try:
            from graph.neo4j_client import get_client

            with get_client(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            ) as client:
                self.graph_stats = client.graph_stats()
                try:
                    self.interest_data = client.interest_profile()
                except Exception:
                    self.interest_data = {}
        except Exception:
            self.graph_stats = {}
            self.interest_data = {}
