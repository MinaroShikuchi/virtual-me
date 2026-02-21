#!/usr/bin/env python3
"""
tools/extractors/spotify.py
-----------------------------
Extract entities and relationships from Spotify streaming history.

Relationships extracted:
  Person → LISTENED_TO → Artist  (top N artists by play count)
  Person → LISTENED_TO → Song    (top N songs by play count)
  Person → INTERESTED_IN → Activity("Music")  (aggregate stats)
  Person → USED_DEVICE → Device  (per-device play counts)

Usage:
  python3 tools/extractors/spotify.py [options]

Options:
  --data-dir DIR    Path to spotify data folder  [default: data/spotify]
  --self-name NAME  Your name in the graph
  --top-n N         How many top artists/songs to emit (0 = all)  [default: 0]
  --dry-run         Print triples without writing to Neo4j
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Device normalisation
# ---------------------------------------------------------------------------
_DEVICE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Android with model info: "Android OS 6.0.1 API 23 (samsung, SM-A300FU)"
    (re.compile(r"Android\s+OS.*\((\w+),", re.I), None),   # brand extracted dynamically
    # Plain "android" / "Android"
    (re.compile(r"^android$", re.I),                        "Android"),
    # iOS / iPhone / iPad
    (re.compile(r"iOS|iphone|ipad", re.I),                  "iOS"),
    # Web player (must come before macOS — "web_player osx …" is a browser)
    (re.compile(r"web_player|web player", re.I),            "Web Player"),
    # Windows
    (re.compile(r"windows", re.I),                          "Windows"),
    # macOS
    (re.compile(r"os\s*x|macos|macintosh", re.I),           "macOS"),
    # Linux (desktop)
    (re.compile(r"linux", re.I),                            "Linux"),
    # Smart TV / casting
    (re.compile(r"cast|chromecast", re.I),                  "Chromecast"),
    (re.compile(r"tv|smart\s*tv", re.I),                    "Smart TV"),
    # Gaming consoles
    (re.compile(r"playstation|ps[345]", re.I),              "PlayStation"),
    (re.compile(r"xbox", re.I),                             "Xbox"),
]


def _normalize_device(platform: str | None) -> str:
    """Turn a raw Spotify ``platform`` string into a clean device category."""
    if not platform:
        return "Unknown"
    platform = platform.strip()
    if not platform:
        return "Unknown"

    # Android with brand in parentheses — extract brand name
    m = re.search(r"Android\s+OS.*\((\w+),", platform, re.I)
    if m:
        brand = m.group(1).capitalize()
        return f"Android ({brand})"

    for pat, label in _DEVICE_PATTERNS:
        if pat.search(platform):
            return label  # type: ignore[return-value]

    # Partner devices (e.g. "Partner lg_tv lg_tv;tv")
    if platform.lower().startswith("partner"):
        return "Smart TV"

    return platform  # keep raw if nothing matched


def extract(streams: list[dict], self_name: str,
            top_n: int = 0, dry_run: bool = False, client=None) -> dict:
    """
    streams: list of Spotify streaming history records.
    Expected fields: master_metadata_track_name, master_metadata_album_artist_name,
                     ms_played, platform
    """
    artist_plays: Counter = Counter()
    song_plays:   Counter = Counter()
    device_plays: Counter = Counter()
    total_ms: int = 0

    for i, s in enumerate(streams):
        artist = (s.get("master_metadata_album_artist_name") or "").strip()
        track  = (s.get("master_metadata_track_name")        or "").strip()
        ms     = s.get("ms_played", 0)

        if artist and ms > 30_000:   # only count plays > 30 seconds
            artist_plays[artist] += 1
            total_ms += ms
        if track and artist and ms > 30_000:
            song_plays[f"{track} — {artist}"] += 1

        # Device tracking (count all plays, even short ones)
        device = _normalize_device(s.get("platform"))
        if device != "Unknown":
            device_plays[device] += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(streams):
            pct = int((i + 1) / len(streams) * 100)
            print(f"PROGRESS: {pct}% | Analyzing {i+1}/{len(streams)}", flush=True)

    triples = []
    n = top_n if top_n > 0 else None          # None = all
    print(f"\n[ENT] Emitting {'all' if n is None else f'top-{n}'} artists/songs...", flush=True)
    for artist, count in artist_plays.most_common(n):
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "LISTENED_TO",
            "to_label":   "Artist", "to_name": artist,
            "props":      {"play_count": count},
        })
        print(f"[REL] {self_name!r} --LISTENED_TO--> Artist: {artist!r} ({count} plays)", flush=True)

    for song, count in song_plays.most_common(n):
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "LISTENED_TO",
            "to_label":   "Song",   "to_name": song,
            "props":      {"play_count": count},
        })
        print(f"[REL] {self_name!r} --LISTENED_TO--> Song: {song!r} ({count} plays)", flush=True)

    # ── Music Activity node ───────────────────────────────────────────────
    total_plays = sum(artist_plays.values())
    total_hours = round(total_ms / 3_600_000, 1)
    triples.append({
        "from_label": "Person",   "from_name": self_name,
        "rel_type":   "INTERESTED_IN",
        "to_label":   "Activity", "to_name": "Music",
        "props":      {"total_plays": total_plays, "total_hours": total_hours,
                       "source": "spotify"},
    })
    print(f"[REL] {self_name!r} --INTERESTED_IN--> Activity: 'Music' "
          f"({total_plays:,} plays, {total_hours:,}h)", flush=True)

    # ── Device nodes ──────────────────────────────────────────────────────
    for device, count in device_plays.most_common():
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "USED_DEVICE",
            "to_label":   "Device", "to_name": device,
            "props":      {"play_count": count, "source": "spotify"},
        })
        print(f"[REL] {self_name!r} --USED_DEVICE--> Device: {device!r} ({count:,} plays)", flush=True)

    print(f"\n📊 Spotify: {len(artist_plays)} unique artists, {len(song_plays)} unique songs, "
          f"{len(device_plays)} devices", flush=True)
    emitted_label = "all" if n is None else f"top-{n}"
    print(f"   → Emitting {emitted_label} each ({len(triples)} triples total)", flush=True)

    if dry_run:
        for t in triples[:20]:
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r} "
                  f"(plays: {t['props'].get('play_count', 0)})")
    elif client is not None:
        client.ensure_constraints()
        client.batch_merge_relations(triples)
        print("✅ Written to Neo4j.", flush=True)

    return {
        "LISTENED_TO":  len(artist_plays.most_common(n)) + len(song_plays.most_common(n)),
        "INTERESTED_IN": 1,
        "USED_DEVICE":   len(device_plays),
    }


def main():
    p = argparse.ArgumentParser(description="Extract KG triples from Spotify history")
    p.add_argument("--data-dir",   default="data/spotify")
    p.add_argument("--self-name",  default=os.environ.get("SELF_NAME", "Me"))
    p.add_argument("--top-n",      type=int, default=0,
                   help="How many top artists/songs to emit (0 = all)")
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
    # Try specific pattern first, fall back to all JSON files
    json_files = sorted(data_dir.glob("Streaming_History_Audio_*.json"))
    if not json_files:
        json_files = sorted(data_dir.glob("*.json"))
    for f in json_files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            streams.extend(data)
        print(f"   📄 {f.name}: {len(data) if isinstance(data, list) else 1} records", flush=True)

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
