#!/usr/bin/env python3
"""
tools/extractors/linkedin_positions.py
--------------------------------------
Extract career history and location from LinkedIn "Positions.csv"
and write them to the Neo4j knowledge graph.

Format:
  Company Name,Title,Description,Location,Started On,Finished On
  Nord Security,DevOps Engineer,"...","Vilnius, Lithuania",Sep 2025,

Relationships:
  Person → WORKS_AT → Company
  Person → LIVES_IN → Place (inferred from position location)
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

def parse_date(date_str: str) -> str:
    """Convert 'Sep 2025' or '2025' to 'YYYY-MM-DD'."""
    if not date_str or date_str.lower() in ("present", "now", ""):
        return ""
    
    # Try common LinkedIn formats
    formats = ["%b %Y", "%Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-01")
        except ValueError:
            continue
    return date_str # Fallback

def extract(csv_path: str, self_name: str = "ME", dry_run: bool = False, client=None):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}", flush=True)
        return

    triples = []
    ME_NODE = "ME"
    
    print(f"📂 Loading LinkedIn positions from {csv_path}...", flush=True)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"📦 Found {len(rows)} positions.", flush=True)

    for i, row in enumerate(rows):
        company = row.get("Company Name", "").strip()
        title = row.get("Title", "").strip()
        desc = row.get("Description", "").strip()
        location = row.get("Location", "").strip()
        started = parse_date(row.get("Started On", ""))
        finished = parse_date(row.get("Finished On", ""))

        if not company:
            continue

        # ── WORKS_AT ────────────────────────────────────────────────────────
        props = {
            "title": title,
            "description": desc[:500], # Keep it reasonable
            "confidence": 1.0, # Structured data
            "source": "linkedin"
        }
        if started: props["since"] = started
        if finished: props["until"] = finished

        triples.append({
            "from_label": "Person", "from_name": ME_NODE,
            "rel_type": "WORKS_AT",
            "to_label": "Company", "to_name": company,
            "props": props
        })
        print(f"[REL] {ME_NODE!r} --WORKS_AT--> {company!r} ({title}) since {started}", flush=True)

        # ── LIVES_IN (Inferred from Office) ─────────────────────────────────
        if location:
            # We assume if you work there, you likely live in that city
            # We'll use the same since/until logic
            loc_props = {
                "confidence": 0.8, # Inferred but high quality
                "source": "linkedin_office"
            }
            if started: loc_props["since"] = started
            if finished: loc_props["until"] = finished
            
            triples.append({
                "from_label": "Person", "from_name": ME_NODE,
                "rel_type": "LIVES_IN",
                "to_label": "Place", "to_name": location,
                "props": loc_props
            })
            print(f"[REL] {ME_NODE!r} --LIVES_IN--> {location!r} (office) since {started}", flush=True)

        # Progress
        pct = int((i + 1) / len(rows) * 100)
        print(f"PROGRESS: {pct}% | {i+1}/{len(rows)}", flush=True)

    if dry_run:
        print("\n🔍 DRY RUN - completion", flush=True)
    elif client:
        client.ensure_constraints()
        print(f"\n✍️ Writing {len(triples)} triples to Neo4j...", flush=True)
        client.batch_merge_relations(triples)
        print("✅ Done.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Extract career history from LinkedIn CSV")
    parser.add_argument("--csv-file", default="Positions.csv")
    parser.add_argument("--self-name", default="ME")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    args = parser.parse_args()

    # Import graph client (needs the project root on sys.path)
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    try:
        from graph.neo4j_client import Neo4jClient
    except ImportError:
        print("❌ Could not import Neo4jClient. Make sure graph/ package exists.")
        sys.exit(1)

    if args.dry_run:
        extract(args.csv_file, args.self_name, dry_run=True)
    else:
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            if not client.verify():
                print(f"❌ Cannot connect to Neo4j at {args.neo4j_uri}")
                sys.exit(1)
            extract(args.csv_file, args.self_name, dry_run=False, client=client)

if __name__ == "__main__":
    main()
