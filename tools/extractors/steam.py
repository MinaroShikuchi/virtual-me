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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_BATCH_SIZE = 500

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


# ── Batch write ───────────────────────────────────────────────────────────────

_UPSERT_SESSIONS_CYPHER = """
UNWIND $rows AS row
MERGE (person:Person {name: $self_name})
MERGE (game:Game {name: row.game})
MERGE (session:Session {start_at: row.start_at})
  ON CREATE SET session.end_at       = row.end_at,
                session.duration_sec = row.duration_sec,
                session.date         = row.date
MERGE (person)-[:PLAYED]->(session)
MERGE (session)-[:OF_GAME]->(game)
"""


def _write_sessions(driver, self_name: str, rows: list[dict]) -> None:
    total, written = len(rows), 0
    with driver.session() as s:
        for i in range(0, total, _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            s.run(_UPSERT_SESSIONS_CYPHER, rows=batch, self_name=self_name)
            written += len(batch)
            pct = int(written / total * 100)
            print(f"PROGRESS: {pct}% | Writing sessions {written}/{total}", flush=True)
    print(f"✅ Written {total:,} Session nodes to Neo4j.", flush=True)


def _ensure_session_constraints(driver) -> None:
    with driver.session() as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Session) REQUIRE n.start_at IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Game)    REQUIRE n.name     IS UNIQUE")


# ── Extraction ────────────────────────────────────────────────────────────────

def extract(sessions: list[dict], self_name: str,
            dry_run: bool = False, client=None) -> dict:
    """
    Extract KG triples from a list of Steam play-session dicts.

    Creates:
      - Person → PLAYED → Session {start_at, end_at, duration_sec, date} → OF_GAME → Game
      - Person → INTERESTED_IN → Activity("Gaming")
    """
    unique_appids = {s["appid"] for s in sessions}
    print(f"  Unique games: {len(unique_appids)}", flush=True)

    app_names = _resolve_all_app_names(unique_appids)

    # Build per-session rows + track per-game aggregates for summary log
    rows: list[dict] = []
    game_duration: defaultdict[str, float] = defaultdict(float)
    game_count:    defaultdict[str, int]   = defaultdict(int)
    game_first:    defaultdict[str, str]   = defaultdict(lambda: "9999")
    game_last:     defaultdict[str, str]   = defaultdict(lambda: "0")
    total_duration_sec: float = 0

    for i, s in enumerate(sessions):
        appid     = s["appid"]
        game_name = app_names.get(appid, f"Steam App {appid}")
        dur       = s["duration_sec"]
        start     = s.get("start_at", "")
        end       = s.get("end_at", "")
        date      = start[:10] if start else ""

        rows.append({"game": game_name, "start_at": start,
                     "end_at": end, "duration_sec": dur, "date": date})

        game_duration[game_name] += dur
        game_count[game_name]    += 1
        total_duration_sec       += dur
        if start and start < game_first[game_name]:
            game_first[game_name] = start
        if end   and end   > game_last[game_name]:
            game_last[game_name]  = end

        if (i + 1) % 500 == 0 or (i + 1) == len(sessions):
            pct = int((i + 1) / len(sessions) * 100)
            print(f"PROGRESS: {pct}% | Parsing {i + 1}/{len(sessions)}", flush=True)

    # Per-game summary
    total_sessions = len(rows)
    total_hours    = round(total_duration_sec / 3600, 1)
    for game_name in sorted(game_duration, key=lambda g: game_duration[g], reverse=True):
        hours = round(game_duration[game_name] / 3600, 1)
        first = game_first[game_name][:10]
        last  = game_last[game_name][:10]
        print(f"[REL] {self_name!r} --PLAYED--> Game: {game_name!r} "
              f"({game_count[game_name]} sessions, {hours}h, {first} → {last})", flush=True)

    print(f"\n📊 Steam: {len(unique_appids)} unique games, "
          f"{total_sessions:,} sessions, {total_hours:,}h total playtime", flush=True)

    if dry_run:
        print("\n[DRY RUN] First 20 sessions:", flush=True)
        for r in rows[:20]:
            mins = round(r["duration_sec"] / 60, 1)
            print(f"  [Session] {r['date']} | {r['game']!r} ({mins}min)", flush=True)
        return {"PLAYED": total_sessions, "games": len(unique_appids)}

    if client is not None:
        _ensure_session_constraints(client.driver)
        client.ensure_constraints()

        _write_sessions(client.driver, self_name, rows)

        # Gaming aggregate
        all_starts = [v for v in game_first.values() if v != "9999"]
        all_ends   = [v for v in game_last.values()  if v != "0"]
        gaming_props: dict = {"total_sessions": total_sessions,
                               "total_hours": total_hours, "source": "steam"}
        if all_starts:
            gaming_props["first_session"] = min(all_starts)[:10]
        if all_ends:
            gaming_props["last_session"]  = max(all_ends)[:10]
        client.merge_relation(
            from_label="Person", from_name=self_name,
            rel_type="INTERESTED_IN",
            to_label="Activity",   to_name="Gaming",
            props=gaming_props,
        )
        print(f"[REL] {self_name!r} --INTERESTED_IN--> Activity: 'Gaming' "
              f"({total_sessions:,} sessions, {total_hours:,}h)", flush=True)

    return {"PLAYED": total_sessions, "INTERESTED_IN": 1}


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
