#!/usr/bin/env python3
"""
tools/extractors/strava.py
---------------------------
Extract entities and relationships from Strava activity data.

Supports two input formats:
  1. CSV  — standard Strava bulk export (activities.csv)
  2. JSON — list of activity dicts (legacy)

Relationships extracted:
  Person → INTERESTED_IN → Activity  (activity type, e.g. "cycling", "running")

Each Activity node stores aggregate stats:
  total_distance_km, total_elapsed_hours, activity_count

Usage:
  python3 tools/extractors/strava.py --csv-file data/strava/activities.csv [options]
  python3 tools/extractors/strava.py --data-dir data/strava [options]  (JSON mode)

Options:
  --csv-file FILE   Path to Strava activities.csv export
  --data-dir DIR    Path to strava JSON data folder  [default: data/strava]
  --self-name NAME  Your name in the graph
  --dry-run         Print triples without writing to Neo4j
  --limit N         Process only the first N activities
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── Activity type normalisation ───────────────────────────────────────────────
ACTIVITY_NAME_MAP = {
    "Run":             "running",
    "Ride":            "cycling",
    "Swim":            "swimming",
    "Hike":            "hiking",
    "Walk":            "walking",
    "Workout":         "fitness",
    "WeightTraining":  "weight training",
    "VirtualRide":     "virtual cycling",
    "VirtualRun":      "virtual running",
    "Yoga":            "yoga",
    "Rowing":          "rowing",
    "Kayaking":        "kayaking",
    "Skiing":          "skiing",
    "Snowboard":       "snowboarding",
    "IceSkate":        "ice skating",
    "RockClimbing":    "rock climbing",
    "Surfing":         "surfing",
    "Elliptical":      "elliptical",
    "StairStepper":    "stair stepper",
    "Crossfit":        "crossfit",
    "Soccer":          "soccer",
    "Tennis":          "tennis",
    "Golf":            "golf",
    "Badminton":       "badminton",
    "Squash":          "squash",
    "Skateboard":      "skateboarding",
    "InlineSkate":     "inline skating",
    "Windsurf":        "windsurfing",
    "Kitesurf":        "kitesurfing",
    "Canoeing":        "canoeing",
    "StandUpPaddling": "stand up paddling",
    "Handcycle":       "handcycling",
    "Wheelchair":      "wheelchair",
    "EBikeRide":       "e-bike riding",
    "Velomobile":      "velomobile",
    "MountainBikeRide":"mountain biking",
    "GravelRide":      "gravel cycling",
    "TrailRun":        "trail running",
    "NordicSki":       "nordic skiing",
    "AlpineSki":       "alpine skiing",
    "BackcountrySki":  "backcountry skiing",
    "Snowshoe":        "snowshoeing",
}


def _normalise_type(raw: str) -> str:
    """Map a raw Strava activity type to a human-readable interest name."""
    return ACTIVITY_NAME_MAP.get(raw, raw.lower().replace("_", " "))


def _parse_float(val: str, default: float = 0.0) -> float:
    """Safely parse a float from a CSV cell."""
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _parse_int(val: str, default: int = 0) -> int:
    """Safely parse an int from a CSV cell."""
    try:
        return int(float(val)) if val else default
    except (ValueError, TypeError):
        return default


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(csv_path: str | Path, limit: int = 0) -> list[dict]:
    """
    Load Strava activities.csv and return a list of normalised activity dicts.
    Only keeps the fields we care about: date, type, elapsed_time, distance.
    """
    activities = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}", flush=True)
        return activities

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            activities.append({
                "type":         row.get("Activity Type", "Unknown"),
                "start_date":   row.get("Activity Date", ""),
                "elapsed_time": _parse_int(row.get("Elapsed Time", "0")),
                "distance":     _parse_float(row.get("Distance", "0")),
            })
            if limit and len(activities) >= limit:
                break

    print(f"📦 {len(activities):,} Strava activities loaded from CSV.", flush=True)
    return activities


# ── JSON loading (legacy) ─────────────────────────────────────────────────────

def load_json(data_dir: str | Path, limit: int = 0) -> list[dict]:
    """Load Strava activities from JSON files in a directory."""
    activities = []
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"❌ Data folder not found: {data_dir}", flush=True)
        return activities

    for f in sorted(data_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            activities.extend(data)
        elif isinstance(data, dict):
            activities.append(data)

    if limit:
        activities = activities[:limit]

    print(f"📦 {len(activities):,} Strava activities loaded from JSON.", flush=True)
    return activities


# ── Extraction ────────────────────────────────────────────────────────────────

def extract(activities: list[dict], self_name: str,
            dry_run: bool = False, client=None) -> dict:
    """
    Extract triples from a list of Strava activity dicts.

    Creates one Activity node per activity type with aggregate stats:
      - total_distance_km
      - total_elapsed_hours
      - activity_count
    """
    triples = []
    counters: Counter = Counter()

    # Aggregate stats per activity type
    type_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "total_distance": 0.0, "total_elapsed": 0,
    })

    for i, act in enumerate(activities):
        act_type = act.get("type", "Unknown")
        interest = _normalise_type(act_type)
        distance = _parse_float(str(act.get("distance", 0)))
        elapsed  = _parse_int(str(act.get("elapsed_time", 0)))

        type_stats[interest]["count"] += 1
        type_stats[interest]["total_distance"] += distance
        type_stats[interest]["total_elapsed"] += elapsed

        if (i + 1) % 200 == 0 or (i + 1) == len(activities):
            pct = int((i + 1) / len(activities) * 100)
            print(f"PROGRESS: {pct}% | {i+1}/{len(activities)}", flush=True)

    # Build triples from aggregated stats
    for interest, stats in sorted(type_stats.items()):
        distance_km = round(stats["total_distance"] / 1000, 1)
        elapsed_hrs = round(stats["total_elapsed"] / 3600, 1)

        triples.append({
            "from_label": "Person",   "from_name": self_name,
            "rel_type":   "INTERESTED_IN",
            "to_label":   "Activity", "to_name": interest,
            "props": {
                "activity_count":      stats["count"],
                "total_distance_km":   distance_km,
                "total_elapsed_hours": elapsed_hrs,
                "source":              "strava",
            },
        })
        counters["INTERESTED_IN"] += 1
        print(
            f"[ACT] {interest!r}: {stats['count']} activities, "
            f"{distance_km} km, {elapsed_hrs} hrs",
            flush=True,
        )

    print(f"\n📊 Strava: {len(triples)} triples from {len(activities)} activities", flush=True)
    for rel, cnt in sorted(counters.items()):
        print(f"   {rel:20s} {cnt:>6,}", flush=True)

    if dry_run:
        for t in triples[:20]:
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r} "
                  f"  {t['props']}")
    elif client is not None:
        client.ensure_constraints()
        client.batch_merge_relations(triples)
        print("✅ Written to Neo4j.", flush=True)

    return dict(counters)


def main():
    p = argparse.ArgumentParser(description="Extract KG triples from Strava data")
    p.add_argument("--csv-file",   default=None,
                   help="Path to Strava activities.csv export")
    p.add_argument("--data-dir",   default="data/strava",
                   help="Path to strava JSON data folder (legacy)")
    p.add_argument("--self-name",  default=os.environ.get("SELF_NAME", "Me"))
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--limit",      type=int, default=0)
    p.add_argument("--neo4j-uri",  default=os.environ.get("NEO4J_URI",      "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER",     "neo4j"))
    p.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    args = p.parse_args()

    # Load activities from CSV or JSON
    if args.csv_file:
        activities = load_csv(args.csv_file, limit=args.limit)
    else:
        activities = load_json(args.data_dir, limit=args.limit)

    if not activities:
        print("❌ No activities found.", flush=True)
        sys.exit(1)

    if args.dry_run:
        extract(activities, args.self_name, dry_run=True)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        from graph.neo4j_client import Neo4jClient
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            extract(activities, args.self_name, client=client)


if __name__ == "__main__":
    main()
