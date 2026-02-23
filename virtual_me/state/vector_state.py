"""
virtual_me/state/vector_state.py — Vector Store page state.

Manages ChromaDB collection stats, conversation stats, and subprocess
streaming for the extract/ingest pipeline.
"""
import asyncio
import json
import os
import sys

import reflex as rx

from virtual_me.state.app_state import AppState


class VectorState(AppState):
    """State for the Vector Store page."""

    # ── Collection overview ──
    total_docs: int = 0
    source_counts: dict[str, int] = {}

    # ── Extract subprocess ──
    extract_logs: list[str] = []
    extract_running: bool = False
    export_dir: str = "./data/facebook"
    out_json: str = "./data/facebook/facebook_messages.json"

    # ── Ingest subprocess ──
    ingest_logs: list[str] = []
    ingest_running: bool = False
    json_file: str = "./data/facebook/facebook_messages.json"
    batch_size: int = 32
    session_gap_h: int = 8
    max_msgs: int = 150
    reset_collection: bool = True

    # ── Conversation stats (loaded from JSON) ──
    conversation_stats: dict[str, int] = {}

    # ── Episodic count ──
    ep_count: int = 0

    def load_vector_info(self):
        """Load collection stats from ChromaDB."""
        try:
            from services.chroma_service import get_collection, get_episodic
            from services.embedding_service import get_embedding_func

            ef = get_embedding_func(self.embedding_model)
            collection = get_collection(ef)
            self.total_docs = collection.count()

            # Per-source counts
            source_counts: dict[str, int] = {}
            try:
                all_meta = collection.get(include=["metadatas"])
                metas = all_meta.get("metadatas") or []
                for m in metas:
                    src = (m or {}).get("source", "unknown")
                    source_counts[src] = source_counts.get(src, 0) + 1
            except Exception:
                pass
            self.source_counts = source_counts

            # Episodic
            try:
                ep = get_episodic(ef)
                self.ep_count = ep.count() if ep else 0
            except Exception:
                self.ep_count = 0

        except Exception:
            self.total_docs = 0
            self.source_counts = {}
            self.ep_count = 0

        # Load conversation stats
        self._load_conversation_stats()

    def _load_conversation_stats(self):
        """Load conversation stats from the JSON output file."""
        json_path = self.json_file
        if not os.path.isfile(json_path):
            self.conversation_stats = {}
            return
        try:
            with open(json_path) as f:
                data = json.load(f)
            counts: dict[str, int] = {}
            for msg in data:
                if msg.get("text"):
                    conv = msg.get("conversation", "Unknown")
                    counts[conv] = counts.get(conv, 0) + 1
            self.conversation_stats = counts
        except Exception:
            self.conversation_stats = {}

    def set_export_dir(self, val: str):
        """Set the Facebook export directory."""
        self.export_dir = val

    def set_out_json(self, val: str):
        """Set the output JSON file path."""
        self.out_json = val

    def set_json_file(self, val: str):
        """Set the JSON file path for ingestion."""
        self.json_file = val

    def set_batch_size(self, val: str):
        """Set the batch size for ingestion."""
        try:
            self.batch_size = int(val)
        except ValueError:
            pass

    def set_session_gap(self, val: list[float]):
        """Set the session gap in hours (from slider on_value_commit)."""
        if val:
            self.session_gap_h = val[0]

    def set_max_msgs(self, val: str):
        """Set max messages per chunk."""
        try:
            self.max_msgs = int(val)
        except ValueError:
            pass

    def toggle_reset(self, val: bool):
        """Toggle the reset collection flag."""
        self.reset_collection = val

    @rx.event(background=True)
    async def run_extract(self):
        """Run tools/extract_facebook.py via subprocess, streaming stdout."""
        async with self:
            self.extract_running = True
            self.extract_logs = ["Starting extraction..."]
            export_dir = self.export_dir
            out_json = self.out_json

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "tools/extract_facebook.py",
            "--input", export_dir,
            "--output", out_json,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async for line in proc.stdout:
            text = line.decode().rstrip()
            if text:
                async with self:
                    self.extract_logs = self.extract_logs + [text]

        await proc.wait()
        async with self:
            self.extract_running = False
            self.extract_logs = self.extract_logs + [
                f"Process exited with code {proc.returncode}"
            ]

    @rx.event(background=True)
    async def run_ingest(self):
        """Run tools/ingest_facebook_messages.py via subprocess, streaming stdout."""
        async with self:
            self.ingest_running = True
            self.ingest_logs = ["Starting ingestion..."]
            json_file = self.json_file
            batch_size = self.batch_size
            session_gap_h = self.session_gap_h
            max_msgs = self.max_msgs
            reset = self.reset_collection

        cmd = [
            sys.executable, "tools/ingest_facebook_messages.py",
            "--json-file", json_file,
            "--session-gap", str(session_gap_h * 3600),
            "--max-msgs", str(max_msgs),
            "--batch-size", str(batch_size),
        ]
        if reset:
            cmd.append("--reset")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async for line in proc.stdout:
            text = line.decode().rstrip()
            if text:
                async with self:
                    self.ingest_logs = self.ingest_logs + [text]

        await proc.wait()
        async with self:
            self.ingest_running = False
            self.ingest_logs = self.ingest_logs + [
                f"Process exited with code {proc.returncode}"
            ]
            # Reload stats after ingest
        # Trigger a reload outside the background context
