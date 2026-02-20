#!/usr/bin/env python3
"""
tools/extractors/strava.py
---------------------------
Extract entities and relationships from Strava activity data.

Relationships extracted:
  Person → INTERESTED_IN → Activity  (activity type)
  Person → TRAVELLED_TO  → Place     (start/end location from activity metadata)

Usage:
  python3 tools/extractors/strava.py [options]

Options:
  --data-dir DIR    Path to strava data folder  [default: data/strava]
  --self-name NAME  Your name in the graph
  --dry-run         Print triples without writing to Neo4j
  --limit N         Process only the first N activities
"""

import argparse
import json
import os
import sys
from pathlib import Path


def extract(activities: list[dict], self_name: str,
            dry_run: bool = False, client=None) -> dict:
    """
    Extract triples from a list of Strava activity dicts.
    Expected fields: type, name, start_latlng, end_latlng,
                     start_date, location_city, location_country
    """
    from collections import Counter, defaultdict
    triples = []
    counters: Counter = Counter()

    # Activity type → INTERESTED_IN
    # e.g. "Run", "Ride", "Swim", "Hike", "Walk"
    ACTIVITY_NAME_MAP = {
        "Run":   "running",   "Ride": "cycling",
        "Swim":  "swimming",  "Hike": "hiking",
        "Walk":  "walking",   "Workout": "fitness",
        "WeightTraining": "weight training",
        "VirtualRide": "cycling",
    }

    seen_activities = set()
    for i, act in enumerate(activities):
        act_type = act.get("type", "Unknown")
        interest = ACTIVITY_NAME_MAP.get(act_type, act_type.lower())
        date     = act.get("start_date", "")[:10]

        if interest not in seen_activities:
            triples.append({
                "from_label": "Person",   "from_name": self_name,
                "rel_type":   "INTERESTED_IN",
                "to_label":   "Activity", "to_name": interest,
                "props":      {},
            })
            seen_activities.add(interest)
            counters["INTERESTED_IN"] += 1
            print(f"[ENT] Detected Activity/Interest: {interest!r}", flush=True)

        # Location fields
        city    = act.get("location_city", "")
        country = act.get("location_country", "")
        place   = city or country
        if place:
            triples.append({
                "from_label": "Person", "from_name": self_name,
                "rel_type":   "TRAVELLED_TO",
                "to_label":   "Place",  "to_name": place,
                "props":      {"date": date},
            })
            counters["TRAVELLED_TO"] += 1
            print(f"[REL] {self_name!r} --TRAVELLED_TO--> {place!r}", flush=True)

        if (i + 1) % 50 == 0 or (i + 1) == len(activities):
            pct = int((i + 1) / len(activities) * 100)
            print(f"PROGRESS: {pct}% | {i+1}/{len(activities)}", flush=True)

    print(f"\n📊 Strava: {len(triples)} triples from {len(activities)} activities", flush=True)
    for rel, cnt in sorted(counters.items()):
        print(f"   {rel:20s} {cnt:>6,}", flush=True)

    if dry_run:
        for t in triples[:20]:
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r}")
    elif client is not None:
        client.ensure_constraints()
        client.batch_merge_relations(triples)
        print("✅ Written to Neo4j.", flush=True)

    return dict(counters)


def main():
    p = argparse.ArgumentParser(description="Extract KG triples from Strava data")
    p.add_argument("--data-dir",   default="data/strava")
    p.add_argument("--self-name",  default=os.environ.get("SELF_NAME", "Me"))
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--limit",      type=int, default=0)
    p.add_argument("--neo4j-uri",  default=os.environ.get("NEO4J_URI",      "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER",     "neo4j"))
    p.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data folder not found: {data_dir}", flush=True)
        sys.exit(1)

    activities = []
    for f in sorted(data_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            activities.extend(data)
        elif isinstance(data, dict):
            activities.append(data)

    if args.limit:
        activities = activities[:args.limit]

    print(f"📦 {len(activities):,} Strava activities loaded.", flush=True)

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
