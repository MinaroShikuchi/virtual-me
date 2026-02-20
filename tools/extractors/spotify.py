#!/usr/bin/env python3
"""
tools/extractors/spotify.py
-----------------------------
Extract entities and relationships from Spotify streaming history.

Relationships extracted:
  Person → LISTENED_TO → Artist  (top N artists by play count)
  Person → LISTENED_TO → Song    (top N songs by play count)

Usage:
  python3 tools/extractors/spotify.py [options]

Options:
  --data-dir DIR    Path to spotify data folder  [default: data/spotify]
  --self-name NAME  Your name in the graph
  --top-n N         How many top artists/songs to emit  [default: 200]
  --dry-run         Print triples without writing to Neo4j
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def extract(streams: list[dict], self_name: str,
            top_n: int = 200, dry_run: bool = False, client=None) -> dict:
    """
    streams: list of Spotify streaming history records.
    Expected fields: master_metadata_track_name, master_metadata_album_artist_name,
                     ms_played
    """
    artist_plays: Counter = Counter()
    song_plays:   Counter = Counter()

    for i, s in enumerate(streams):
        artist = (s.get("master_metadata_album_artist_name") or "").strip()
        track  = (s.get("master_metadata_track_name")        or "").strip()
        ms     = s.get("ms_played", 0)

        if artist and ms > 30_000:   # only count plays > 30 seconds
            artist_plays[artist] += 1
        if track and artist and ms > 30_000:
            song_plays[f"{track} — {artist}"] += 1
        
        if (i + 1) % 500 == 0 or (i + 1) == len(streams):
            pct = int((i + 1) / len(streams) * 100)
            print(f"PROGRESS: {pct}% | Analyzing {i+1}/{len(streams)}", flush=True)

    triples = []
    print("\n[ENT] Emitting top artists/songs...", flush=True)
    for artist, count in artist_plays.most_common(top_n):
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "LISTENED_TO",
            "to_label":   "Artist", "to_name": artist,
            "props":      {"play_count": count},
        })
        print(f"[REL] {self_name!r} --LISTENED_TO--> Artist: {artist!r} ({count} plays)", flush=True)

    for song, count in song_plays.most_common(top_n):
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "LISTENED_TO",
            "to_label":   "Song",   "to_name": song,
            "props":      {"play_count": count},
        })
        print(f"[REL] {self_name!r} --LISTENED_TO--> Song: {song!r} ({count} plays)", flush=True)

    print(f"\n📊 Spotify: {len(artist_plays)} unique artists, {len(song_plays)} unique songs", flush=True)
    print(f"   → Emitting top-{top_n} each ({len(triples)} triples total)", flush=True)

    if dry_run:
        for t in triples[:20]:
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r} "
                  f"(plays: {t['props'].get('play_count', 0)})")
    elif client is not None:
        client.ensure_constraints()
        client.batch_merge_relations(triples)
        print("✅ Written to Neo4j.", flush=True)

    return {"LISTENED_TO": len(triples)}


def main():
    p = argparse.ArgumentParser(description="Extract KG triples from Spotify history")
    p.add_argument("--data-dir",   default="data/spotify")
    p.add_argument("--self-name",  default=os.environ.get("SELF_NAME", "Me"))
    p.add_argument("--top-n",      type=int, default=200)
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--neo4j-uri",  default=os.environ.get("NEO4J_URI",      "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER",     "neo4j"))
    p.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data folder not found: {data_dir}", flush=True)
        sys.exit(1)

    streams = []
    for f in sorted(data_dir.glob("Streaming_History_Audio_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            streams.extend(json.load(fh))

    print(f"📦 {len(streams):,} Spotify streams loaded.", flush=True)

    if args.dry_run:
        extract(streams, args.self_name, top_n=args.top_n, dry_run=True)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        from graph.neo4j_client import Neo4jClient
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            extract(streams, args.self_name, top_n=args.top_n, client=client)


if __name__ == "__main__":
    main()
