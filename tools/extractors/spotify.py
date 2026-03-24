#!/usr/bin/env python3
"""
tools/extractors/spotify.py
-----------------------------
Extract entities and relationships from Spotify streaming history.

Graph model written:
  (Person)-[:LISTENED]->(Play {ts, ms_played})-[:OF_TRACK]->(Track {title, uri})
                                                             (Track)-[:BY]->(Artist)
                                                             (Track)-[:FROM_ALBUM]->(Album)
  (Person)-[:INTERESTED_IN]->(Activity "Music")  (aggregate stats)
  (Person)-[:USED_DEVICE]->(Device)              (per-device play counts)

Usage:
  python3 tools/extractors/spotify.py [options]

Options:
  --data-dir DIR    Path to spotify data folder  [default: data/spotify]
  --self-name NAME  Your name in the graph
  --dry-run         Print plays without writing to Neo4j
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_BATCH_SIZE = 500


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


# ---------------------------------------------------------------------------
# Batch write
# ---------------------------------------------------------------------------

_UPSERT_PLAYS_CYPHER = """
UNWIND $rows AS row
MERGE (person:Person {name: $self_name})
MERGE (artist:Artist {name: row.artist})
MERGE (album:Album   {name: row.album})
MERGE (track:Track   {uri: row.uri})
  ON CREATE SET track.title = row.title
MERGE (play:Play {ts: row.ts})
  ON CREATE SET play.ms_played = row.ms_played
MERGE (person)-[:LISTENED]->(play)
MERGE (play)-[:OF_TRACK]->(track)
MERGE (track)-[:BY]->(artist)
MERGE (track)-[:FROM_ALBUM]->(album)
"""


def _write_plays(driver, self_name: str, plays: list[dict]) -> None:
    """Batch-write Play event nodes to Neo4j using UNWIND."""
    total = len(plays)
    written = 0
    with driver.session() as s:
        for i in range(0, total, _BATCH_SIZE):
            batch = plays[i : i + _BATCH_SIZE]
            s.run(_UPSERT_PLAYS_CYPHER, rows=batch, self_name=self_name)
            written += len(batch)
            pct = int(written / total * 100)
            print(f"PROGRESS: {pct}% | Writing plays {written}/{total}", flush=True)
    print(f"✅ Written {total:,} Play nodes to Neo4j.", flush=True)


def _ensure_play_constraints(driver) -> None:
    """Add unique constraints for Play and Track nodes (idempotent)."""
    with driver.session() as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Play)  REQUIRE n.ts  IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Track) REQUIRE n.uri IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Album) REQUIRE n.name IS UNIQUE")


# ---------------------------------------------------------------------------
# Main extract
# ---------------------------------------------------------------------------

def extract(streams: list[dict], self_name: str,
            dry_run: bool = False, client=None) -> dict:
    """
    streams: list of Spotify streaming history records.
    Expected fields: master_metadata_track_name, master_metadata_album_artist_name,
                     master_metadata_album_album_name, spotify_track_uri,
                     ms_played, platform, ts
    """
    plays: list[dict] = []
    device_plays: Counter = Counter()
    total_ms: int = 0
    skipped = 0

    for i, s in enumerate(streams):
        artist = (s.get("master_metadata_album_artist_name") or "").strip()
        title  = (s.get("master_metadata_track_name")        or "").strip()
        album  = (s.get("master_metadata_album_album_name")  or "").strip()
        uri    = (s.get("spotify_track_uri")                 or "").strip()
        ms     = s.get("ms_played", 0)
        ts     = (s.get("ts") or "").strip()

        # Device tracking (count all plays, even short ones)
        device = _normalize_device(s.get("platform"))
        if device != "Unknown":
            device_plays[device] += 1

        if not (artist and title and ts) or ms <= 30_000:
            skipped += 1
        else:
            total_ms += ms
            # Fallback URI: use title — artist if no Spotify URI
            if not uri:
                uri = f"{title} — {artist}"
            plays.append({
                "ts":       ts,
                "ms_played": ms,
                "title":    title,
                "uri":      uri,
                "artist":   artist,
                "album":    album or "Unknown Album",
            })

        if (i + 1) % 500 == 0 or (i + 1) == len(streams):
            pct = int((i + 1) / len(streams) * 100)
            print(f"PROGRESS: {pct}% | Parsing {i+1}/{len(streams)}", flush=True)

    # ── Summary stats ──────────────────────────────────────────────────────
    unique_tracks  = len({p["uri"]    for p in plays})
    unique_artists = len({p["artist"] for p in plays})
    total_plays    = len(plays)
    total_hours    = round(total_ms / 3_600_000, 1)

    print(f"\n📊 Spotify: {total_plays:,} valid plays, {unique_tracks:,} unique tracks, "
          f"{unique_artists:,} artists, {total_hours:,}h total", flush=True)
    print(f"   Skipped: {skipped:,} (too short / missing metadata)", flush=True)
    print(f"   Devices: {len(device_plays)}", flush=True)

    if dry_run:
        print("\n[DRY RUN] First 20 plays:", flush=True)
        for p in plays[:20]:
            date = p["ts"][:10]
            mins = round(p["ms_played"] / 60_000, 1)
            print(f"  [Play] {date} | '{p['title']}' by {p['artist']} "
                  f"({mins}min) → uri={p['uri']!r}", flush=True)
        return {"LISTENED": total_plays, "tracks": unique_tracks, "artists": unique_artists}

    if client is not None:
        # Ensure Play/Track/Album constraints
        _ensure_play_constraints(client.driver)
        # Ensure standard entity constraints (Person, Artist, etc.)
        client.ensure_constraints()

        # Write Play event graph
        _write_plays(client.driver, self_name, plays)

        # ── Music Activity aggregate node ──────────────────────────────────
        client.merge_relation(
            from_label="Person",   from_name=self_name,
            rel_type="INTERESTED_IN",
            to_label="Activity",   to_name="Music",
            props={"total_plays": total_plays, "total_hours": total_hours,
                   "source": "spotify"},
        )
        print(f"[REL] {self_name!r} --INTERESTED_IN--> Activity: 'Music' "
              f"({total_plays:,} plays, {total_hours:,}h)", flush=True)

        # ── Device nodes ───────────────────────────────────────────────────
        for device, count in device_plays.most_common():
            client.merge_relation(
                from_label="Person", from_name=self_name,
                rel_type="USED_DEVICE",
                to_label="Device",   to_name=device,
                props={"play_count": count, "source": "spotify"},
            )
            print(f"[REL] {self_name!r} --USED_DEVICE--> Device: {device!r} "
                  f"({count:,} plays)", flush=True)

    return {"LISTENED": total_plays, "INTERESTED_IN": 1, "USED_DEVICE": len(device_plays)}


def main():
    p = argparse.ArgumentParser(description="Extract KG triples from Spotify history")
    p.add_argument("--data-dir",   default="data/spotify")
    p.add_argument("--self-name",  default=os.environ.get("SELF_NAME", "Me"))
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
        extract(streams, args.self_name, dry_run=True)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        from graph.neo4j_client import Neo4jClient
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            extract(streams, args.self_name, client=client)


if __name__ == "__main__":
    main()
