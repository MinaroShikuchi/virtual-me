"""
virtual_me/state/graph_state.py — Knowledge Graph / Node Extract page state.

Manages Neo4j connection status, graph statistics, interest profile,
and subprocess streaming for platform extractors.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import reflex as rx

from config import SELF_NAME, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from virtual_me.state.app_state import AppState


# ── Extractor configuration (non-reactive, used to build subprocess commands) ─

PLATFORMS = [
    {
        "id": "facebook",
        "label": "Facebook",
        "color": "#1877F2",
        "extractors": [
            {
                "label": "Messages",
                "script": "tools/extractors/facebook_messages.py",
                "default_args": ["--json-file", "facebook_messages.json"],
                "entities": ["Person", "Place", "Company", "Interest"],
                "relationships": [
                    "MET", "VISITED", "LIVES_IN", "WORKS_AT",
                    "INTERESTED_IN", "PARTNER_OF", "FAMILY_OF",
                    "COLLEAGUE_OF", "FRIEND_OF",
                ],
            },
            {
                "label": "Friends / Contacts (HTML)",
                "script": "tools/extract_facebook_friends.py",
                "default_args": ["your_friends.html"],
                "entities": ["Person"],
                "relationships": ["FRIENDS_WITH"],
            },
            {
                "label": "Removed Friends (HTML)",
                "script": "tools/extract_facebook_friends.py",
                "default_args": ["removed_friends.html", "--removed"],
                "entities": ["Person"],
                "relationships": ["WAS_FRIENDS_WITH"],
            },
        ],
    },
    {
        "id": "linkedin",
        "label": "LinkedIn",
        "color": "#0A66C2",
        "extractors": [
            {
                "label": "Positions",
                "script": "tools/extractors/linkedin_positions.py",
                "default_args": ["--csv-file", "Positions.csv"],
                "entities": ["Person", "Company", "Place"],
                "relationships": ["WORKS_AT", "LIVES_IN"],
            },
            {
                "label": "Connections",
                "script": "tools/extractors/linkedin_connections.py",
                "default_args": [
                    "--csv-file", "data/linkedin/Connections.csv",
                    "--positions-file", "data/linkedin/Positions.csv",
                ],
                "entities": ["Person", "Company"],
                "relationships": ["KNOWS", "WORKS_AT", "COLLEAGUE_OF"],
            },
            {
                "label": "Education",
                "script": "tools/extractors/linkedin_education.py",
                "default_args": ["--csv-file", "data/linkedin/Education.csv"],
                "entities": ["Person", "School"],
                "relationships": ["STUDIED_AT"],
            },
        ],
    },
    {
        "id": "spotify",
        "label": "Spotify",
        "color": "#1DB954",
        "extractors": [
            {
                "label": "Listening History",
                "script": "tools/extractors/spotify.py",
                "default_args": ["--data-dir", "data/spotify"],
                "entities": ["Person", "Artist", "Song", "Activity", "Device"],
                "relationships": ["LISTENED_TO", "INTERESTED_IN", "USED_DEVICE"],
            },
        ],
    },
    {
        "id": "google",
        "label": "Google",
        "color": "#4285F4",
        "extractors": [
            {
                "label": "Location History",
                "script": "tools/extractors/google_timeline.py",
                "default_args": ["--records", "data/google/Records.json"],
                "entities": ["Person", "Place"],
                "relationships": ["VISITED", "LIVES_IN"],
            },
        ],
    },
    {
        "id": "strava",
        "label": "Strava",
        "color": "#FC6100",
        "extractors": [
            {
                "label": "Activities",
                "script": "tools/extractors/strava.py",
                "default_args": ["--data-dir", "data/strava"],
                "entities": ["Person", "Activity"],
                "relationships": ["INTERESTED_IN"],
            },
        ],
    },
]


def _get_platform_labels() -> list[str]:
    """Return list of platform labels."""
    return [p["label"] for p in PLATFORMS]


def _get_extractor_labels(platform_label: str) -> list[str]:
    """Return list of extractor labels for a given platform."""
    for p in PLATFORMS:
        if p["label"] == platform_label:
            return [e["label"] for e in p["extractors"]]
    return []


def _find_extractor(platform_label: str, extractor_label: str) -> dict | None:
    """Find extractor config by platform and extractor label."""
    for p in PLATFORMS:
        if p["label"] == platform_label:
            for e in p["extractors"]:
                if e["label"] == extractor_label:
                    return {**e, "platform_id": p["id"]}
    return None


class GraphState(AppState):
    """State for the Knowledge Graph / Node Extract page."""

    # ── Graph overview ──
    graph_stats: dict[str, int] = {}
    neo4j_alive: bool = False
    interest_data: dict[str, float] = {}

    # ── Extractor subprocess ──
    extractor_logs: list[str] = []
    extractor_running: bool = False

    # ── Selection ──
    selected_platform: str = "Facebook"
    selected_extractor: str = "Messages"

    # ── Extractor settings ──
    self_name: str = SELF_NAME or "ME"
    limit_chunks: int = 0
    dry_run: bool = True

    def load_graph_info(self):
        """Connect to Neo4j, load stats and interest profile."""
        try:
            from graph.neo4j_client import get_client

            client = get_client(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            )
            alive = client.verify()
            self.neo4j_alive = alive

            if alive:
                self.graph_stats = client.graph_stats()
                self.interest_data = client.interest_profile(self.self_name)
            else:
                self.graph_stats = {}
                self.interest_data = {}

            client.close()
        except Exception:
            self.neo4j_alive = False
            self.graph_stats = {}
            self.interest_data = {}

    def set_selected_platform(self, val: str):
        """Set the selected platform and reset extractor to first available."""
        self.selected_platform = val
        extractors = _get_extractor_labels(val)
        if extractors:
            self.selected_extractor = extractors[0]
        else:
            self.selected_extractor = ""

    def set_selected_extractor(self, val: str):
        """Set the selected extractor."""
        self.selected_extractor = val

    def set_self_name(self, val: str):
        """Set the self name for graph extraction."""
        self.self_name = val

    def set_limit_chunks(self, val: str):
        """Set the limit chunks value."""
        try:
            self.limit_chunks = int(val)
        except ValueError:
            pass

    def toggle_dry_run(self, val: bool):
        """Toggle dry run mode."""
        self.dry_run = val

    @rx.event(background=True)
    async def run_extractor(self):
        """Run the selected extractor via subprocess, streaming stdout."""
        async with self:
            platform_label = self.selected_platform
            extractor_label = self.selected_extractor
            self_name = self.self_name
            limit = self.limit_chunks
            dry_run = self.dry_run
            neo4j_uri = self.neo4j_uri
            neo4j_user = self.neo4j_user
            neo4j_password = self.neo4j_password

        ext = _find_extractor(platform_label, extractor_label)
        if not ext:
            async with self:
                self.extractor_logs = [f"Error: extractor not found for {platform_label} / {extractor_label}"]
            return

        script = ext["script"]
        if not Path(script).exists():
            async with self:
                self.extractor_logs = [f"Error: script not found: {script}"]
            return

        async with self:
            self.extractor_running = True
            self.extractor_logs = [
                f"Starting {platform_label} ▸ {extractor_label}..."
            ]

        # Build command
        cmd = [sys.executable, script] + ext["default_args"]

        # Add --self-name if the script supports it
        if "--self-name" not in ext["default_args"] and "--json-file" not in " ".join(ext["default_args"][:1]):
            cmd += ["--self-name", self_name]
        else:
            # For scripts that take --self-name, add it
            cmd += ["--self-name", self_name]

        if dry_run:
            cmd.append("--dry-run")
        if limit > 0:
            cmd += ["--limit", str(limit)]
        if neo4j_uri:
            cmd += ["--neo4j-uri", neo4j_uri]
        if neo4j_user:
            cmd += ["--neo4j-user", neo4j_user]
        if neo4j_password:
            cmd += ["--neo4j-pass", neo4j_password]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async for line in proc.stdout:
            text = line.decode().rstrip()
            if not text:
                continue
            # Skip PROGRESS lines (they're for progress bars)
            if text.startswith("PROGRESS:"):
                continue
            # Capture interest chart data
            if text.startswith("INTERESTS_CHART:"):
                try:
                    chart_data = json.loads(text[len("INTERESTS_CHART:"):].strip())
                    async with self:
                        self.interest_data = chart_data
                except Exception:
                    pass
                continue
            async with self:
                self.extractor_logs = self.extractor_logs + [text]

        await proc.wait()
        async with self:
            self.extractor_running = False
            self.extractor_logs = self.extractor_logs + [
                f"Process exited with code {proc.returncode}"
            ]
