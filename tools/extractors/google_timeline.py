#!/usr/bin/env python3
"""
tools/extractors/google_timeline.py
--------------------------------------
Extract entities and relationships from Google Location History (Records.json).

Relationships extracted:
  Person → VISITED   → Place   (significant stay points by cluster)
  Person → LIVES_IN  → Place   (most-visited cluster = home)

Algorithm:
  1. Load lat/lng records from Records.json
  2. Cluster nearby points with a simple grid (0.01° ≈ 1km)
  3. Emit VISITED for every cluster with ≥5 visits
  4. Emit LIVES_IN for the most-visited cluster

Usage:
  python3 tools/extractors/google_timeline.py [options]

Options:
  --records FILE    Path to Records.json   [default: data/google/Records.json]
  --self-name NAME  Your name in the graph
  --min-visits N    Minimum visits to emit VISITED  [default: 5]
  --dry-run         Print triples without writing to Neo4j
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def _grid_key(lat: float, lng: float, precision: float = 0.01) -> tuple:
    """Snap lat/lng to a grid cell."""
    return (round(lat / precision) * precision,
            round(lng / precision) * precision)


def extract(records: list[dict], self_name: str, min_visits: int = 5,
            dry_run: bool = False, client=None) -> dict:
    """
    records: list of Google Location History location objects.
    Expected fields: latitudeE7, longitudeE7, timestamp (ISO string)
    """
    grid: Counter = Counter()

    for i, r in enumerate(records):
        lat = r.get("latitudeE7",  0) / 1e7
        lng = r.get("longitudeE7", 0) / 1e7
        if lat == 0 and lng == 0:
            continue
        grid[_grid_key(lat, lng)] += 1
        
        if (i + 1) % 1000 == 0 or (i + 1) == len(records):
            pct = int((i + 1) / len(records) * 100)
            print(f"PROGRESS: {pct}% | Clumping {i+1}/{len(records)}", flush=True)

    triples = []
    counters: Counter = Counter()

    # Home = most frequent cell
    if grid:
        home_key, home_count = grid.most_common(1)[0]
        home_label = f"lat{home_key[0]:.2f}_lng{home_key[1]:.2f}"
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "LIVES_IN",
            "to_label":   "Place",  "to_name": home_label,
            "props":      {"visits": home_count, "lat": home_key[0], "lng": home_key[1]},
        })
        counters["LIVES_IN"] += 1
        print(f"[REL] Inferred LIVES_IN: {home_label!r}", flush=True)

    for cell, count in grid.items():
        if count < min_visits:
            continue
        label = f"lat{cell[0]:.2f}_lng{cell[1]:.2f}"
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "VISITED",
            "to_label":   "Place",  "to_name": label,
            "props":      {"visits": count, "lat": cell[0], "lng": cell[1]},
        })
        counters["VISITED"] += 1
        print(f"[REL] {self_name!r} --VISITED--> {label!r}", flush=True)

    print(f"\n📊 Google Timeline: {len(grid)} grid cells, "
          f"{counters.get('VISITED',0)} VISITED + {counters.get('LIVES_IN',0)} LIVES_IN", flush=True)

    if dry_run:
        for t in triples[:10]:
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r} "
                  f"(visits: {t['props'].get('visits',0)})")
    elif client is not None:
        client.ensure_constraints()
        client.batch_merge_relations(triples)
        print("✅ Written to Neo4j.", flush=True)

    return dict(counters)


def main():
    p = argparse.ArgumentParser(description="Extract KG triples from Google Location History")
    p.add_argument("--records",    default="data/google/Records.json")
    p.add_argument("--self-name",  default=os.environ.get("SELF_NAME", "Me"))
    p.add_argument("--min-visits", type=int, default=5)
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--neo4j-uri",  default=os.environ.get("NEO4J_URI",      "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER",     "neo4j"))
    p.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    args = p.parse_args()

    records_path = Path(args.records)
    if not records_path.exists():
        print(f"❌ File not found: {records_path}", flush=True)
        sys.exit(1)

    print(f"📂 Loading {records_path} …", flush=True)
    with open(records_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    records = data if isinstance(data, list) else data.get("locations", [])
    print(f"📦 {len(records):,} location records loaded.", flush=True)

    if args.dry_run:
        extract(records, args.self_name, min_visits=args.min_visits, dry_run=True)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        from graph.neo4j_client import Neo4jClient
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            extract(records, args.self_name, min_visits=args.min_visits, client=client)


if __name__ == "__main__":
    main()
