#!/usr/bin/env python3
"""
tools/extractors/steam.py
---------------------------
Extract entities and relationships from Steam play-session data.

Input format (CSV):
    appid,start_at,end_at
    252950,2026-02-26 20:13:43,2026-02-26 21:35:51

The extractor resolves numeric Steam ``appid`` values to human-readable game
names via the Steam Store API (with a local JSON cache to avoid repeated
lookups).

Relationships extracted:
    Person → PLAYED          → Game   (per-game aggregate: session_count, total_hours)
    Person → INTERESTED_IN   → Activity("Gaming")  (overall aggregate stats)

Usage:
    python3 tools/extractors/steam.py --csv-file data/steam/steam-data.csv [options]

Options:
    --csv-file FILE   Path to the Steam play-session CSV
    --data-dir DIR    Folder containing CSV files  [default: data/steam]
    --self-name NAME  Your name in the graph
    --dry-run         Print triples without writing to Neo4j
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


# ── Steam App-ID → Name resolution ───────────────────────────────────────────

_APP_NAME_CACHE_FILE = Path("data/steam/app_name_cache.json")


def _load_app_name_cache() -> dict[str, str]:
    """Load the local appid→name cache from disk."""
    if _APP_NAME_CACHE_FILE.exists():
        try:
            with open(_APP_NAME_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_app_name_cache(cache: dict[str, str]) -> None:
    """Persist the appid→name cache to disk."""
    _APP_NAME_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_APP_NAME_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# In-memory master app list (fetched once per run if needed)
_MASTER_APP_LIST: dict[str, str] | None = None


def _fetch_master_app_list() -> dict[str, str]:
    """Fetch the full Steam app list (appid→name) in one bulk request."""
    global _MASTER_APP_LIST
    if _MASTER_APP_LIST is not None:
        return _MASTER_APP_LIST

    if requests is None:
        _MASTER_APP_LIST = {}
        return _MASTER_APP_LIST

    print("  Fetching Steam master app list (one-time bulk download)…", flush=True)
    try:
        resp = requests.get(
            "https://api.steampowered.com/ISteamApps/GetAppList/v2/",
            timeout=30,
        )
        resp.raise_for_status()
        apps = resp.json().get("applist", {}).get("apps", [])
        _MASTER_APP_LIST = {str(a["appid"]): a["name"] for a in apps if a.get("name")}
        print(f"  ✅ Loaded {len(_MASTER_APP_LIST):,} app names from Steam.", flush=True)
    except Exception as e:
        print(f"  ⚠️ Could not fetch master app list: {e}", flush=True)
        _MASTER_APP_LIST = {}

    return _MASTER_APP_LIST


def _resolve_app_name_via_store(appid: str) -> str | None:
    """Fallback: resolve a single appid via the Steam Store API (slow, rate-limited)."""
    if requests is None:
        return None
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        app_data = data.get(str(appid), {})
        if app_data.get("success"):
            return app_data["data"].get("name")
    except Exception:
        pass
    return None


def _resolve_all_app_names(appids: set[str]) -> dict[str, str]:
    """Resolve all unique app IDs using cache → bulk list → individual Store API."""
    cache = _load_app_name_cache()
    to_resolve = [aid for aid in appids if aid not in cache]

    if not to_resolve:
        return cache

    # Step 1: Try the bulk master list first (fast, single request)
    master = _fetch_master_app_list()
    still_missing = []
    for aid in to_resolve:
        if aid in master and master[aid]:
            cache[aid] = master[aid]
        else:
            still_missing.append(aid)

    resolved_bulk = len(to_resolve) - len(still_missing)
    if resolved_bulk:
        print(f"  Resolved {resolved_bulk}/{len(to_resolve)} via bulk list.", flush=True)

    # Step 2: Fall back to individual Store API for any remaining
    if still_missing:
        print(f"  Resolving {len(still_missing)} remaining app IDs via Store API…", flush=True)
        for i, aid in enumerate(still_missing):
            name = _resolve_app_name_via_store(aid)
            cache[aid] = name if name else f"Steam App {aid}"
            if (i + 1) % 10 == 0:
                print(f"    Resolved {i + 1}/{len(still_missing)}…", flush=True)
                time.sleep(0.5)
            else:
                time.sleep(0.2)

    _save_app_name_cache(cache)
    print(f"  ✅ All {len(to_resolve)} app IDs resolved. Cache saved.", flush=True)

    return cache


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(csv_path: str | Path, limit: int = 0) -> list[dict]:
    """
    Load Steam play-session CSV and return a list of session dicts.

    Expected columns: appid, start_at, end_at
    """
    sessions: list[dict] = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}", flush=True)
        return sessions

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            appid = (row.get("appid") or "").strip()
            start_at = (row.get("start_at") or "").strip()
            end_at = (row.get("end_at") or "").strip()

            if not appid or not start_at or not end_at:
                continue

            # Parse timestamps
            try:
                start_dt = datetime.strptime(start_at, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_at, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

            duration_sec = max(0, (end_dt - start_dt).total_seconds())

            sessions.append({
                "appid": appid,
                "start_at": start_dt.isoformat(),
                "end_at": end_dt.isoformat(),
                "duration_sec": duration_sec,
            })

            if limit and len(sessions) >= limit:
                break

    print(f"📦 {len(sessions):,} Steam play sessions loaded from CSV.", flush=True)
    return sessions


# ── Extraction ────────────────────────────────────────────────────────────────

def extract(sessions: list[dict], self_name: str,
            dry_run: bool = False, client=None) -> dict:
    """
    Extract KG triples from a list of Steam play-session dicts.

    Creates:
      - Person → PLAYED → Game  (with session_count, total_hours)
      - Person → INTERESTED_IN → Activity("Gaming")
    """
    # Collect unique app IDs and resolve names
    unique_appids = {s["appid"] for s in sessions}
    print(f"  Unique games: {len(unique_appids)}", flush=True)

    app_names = _resolve_all_app_names(unique_appids)

    # Aggregate per-game stats
    game_sessions: Counter = Counter()
    game_duration: defaultdict[str, float] = defaultdict(float)
    total_duration_sec: float = 0

    for i, s in enumerate(sessions):
        appid = s["appid"]
        game_name = app_names.get(appid, f"Steam App {appid}")
        dur = s["duration_sec"]

        game_sessions[game_name] += 1
        game_duration[game_name] += dur
        total_duration_sec += dur

        if (i + 1) % 500 == 0 or (i + 1) == len(sessions):
            pct = int((i + 1) / len(sessions) * 100)
            print(f"PROGRESS: {pct}% | Analyzing {i + 1}/{len(sessions)}", flush=True)

    # Build triples
    triples = []

    for game_name, count in game_sessions.most_common():
        hours = round(game_duration[game_name] / 3600, 1)
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type": "PLAYED",
            "to_label": "Game", "to_name": game_name,
            "props": {"session_count": count, "total_hours": hours, "source": "steam"},
        })
        print(f"[REL] {self_name!r} --PLAYED--> Game: {game_name!r} "
              f"({count} sessions, {hours}h)", flush=True)

    # Gaming Activity aggregate node
    total_sessions = sum(game_sessions.values())
    total_hours = round(total_duration_sec / 3600, 1)
    triples.append({
        "from_label": "Person", "from_name": self_name,
        "rel_type": "INTERESTED_IN",
        "to_label": "Activity", "to_name": "Gaming",
        "props": {
            "total_sessions": total_sessions,
            "total_hours": total_hours,
            "source": "steam",
        },
    })
    print(f"[REL] {self_name!r} --INTERESTED_IN--> Activity: 'Gaming' "
          f"({total_sessions:,} sessions, {total_hours:,}h)", flush=True)

    print(f"\n📊 Steam: {len(unique_appids)} unique games, "
          f"{total_sessions:,} sessions, {total_hours:,}h total playtime", flush=True)
    print(f"   → Emitting {len(triples)} triples", flush=True)

    if dry_run:
        for t in triples[:20]:
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r} "
                  f"  {t['props']}")
    elif client is not None:
        client.ensure_constraints()
        client.batch_merge_relations(triples)
        print("✅ Written to Neo4j.", flush=True)

    return {
        "PLAYED": len(game_sessions),
        "INTERESTED_IN": 1,
    }


def main():
    p = argparse.ArgumentParser(description="Extract KG triples from Steam play-session data")
    p.add_argument("--csv-file", default=None,
                   help="Path to Steam play-session CSV file")
    p.add_argument("--data-dir", default="data/steam",
                   help="Folder containing Steam CSV files")
    p.add_argument("--self-name", default=os.environ.get("SELF_NAME", "Me"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    args = p.parse_args()

    # Load sessions
    sessions: list[dict] = []
    if args.csv_file:
        sessions = load_csv(args.csv_file, limit=args.limit)
    else:
        # Auto-detect CSV files in data_dir
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"❌ Data folder not found: {data_dir}", flush=True)
            sys.exit(1)
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            print(f"❌ No CSV files found in {data_dir}", flush=True)
            sys.exit(1)
        for cf in csv_files:
            sessions.extend(load_csv(cf, limit=args.limit))

    if not sessions:
        print("❌ No play sessions found.", flush=True)
        sys.exit(1)

    if args.dry_run:
        extract(sessions, args.self_name, dry_run=True)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        from graph.neo4j_client import Neo4jClient
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            extract(sessions, args.self_name, client=client)


if __name__ == "__main__":
    main()
