#!/usr/bin/env python3
"""
tools/extractors/linkedin_education.py
----------------------------------------
Parses LinkedIn "Education.csv" export and writes STUDIED_AT relationships
into the Neo4j knowledge graph.

Expected CSV format:
  School Name,Start Date,End Date,Notes,Degree Name,Activities
"""

import argparse
import csv
import os
import sys

# ── validation ─────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"School Name", "Start Date"}


def validate_csv(path: str) -> tuple[bool, str]:
    if not os.path.exists(path):
        return False, f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - cols
        if missing:
            return False, (
                f"Missing required columns: {missing}. "
                f"Expected: School Name, Start Date, End Date, Degree Name, Activities"
            )
        return True, ""
    except Exception as e:
        return False, str(e)


# ── main extractor ─────────────────────────────────────────────────────────────
def extract(csv_path: str, self_name: str = "ME", dry_run: bool = False, client=None):
    ok, err = validate_csv(csv_path)
    if not ok:
        print(f"❌ Invalid CSV: {err}", flush=True)
        sys.exit(1)

    ME_NODE = "ME"
    triples = []

    print(f"📂 Loading LinkedIn education from {csv_path}...", flush=True)

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})

    print(f"📦 Found {len(rows)} education entries.", flush=True)

    for i, row in enumerate(rows):
        school     = row.get("School Name", "").strip()
        start      = row.get("Start Date", "").strip()
        end        = row.get("End Date", "").strip()
        degree     = row.get("Degree Name", "").strip()
        activities = row.get("Activities", "").strip()
        notes      = row.get("Notes", "").strip()

        if not school:
            continue

        # Normalize dates to ISO-ish format (year only for education)
        since = start if start else ""
        until = end   if end   else ""

        props = {"source": "linkedin"}
        if degree:     props["degree"]     = degree
        if activities: props["activities"] = activities
        if notes:      props["notes"]      = notes
        if since:      props["since"]      = since
        if until:      props["until"]      = until

        triples.append({
            "from_label": "Person",  "from_name": ME_NODE,
            "rel_type":   "STUDIED_AT",
            "to_label":   "School",  "to_name":   school,
            "props":      props,
        })

        date_range = f"{since}–{until}" if until else (since or "?")
        deg_info   = f" ({degree})"     if degree else ""
        print(f"[REL] {ME_NODE!r} --STUDIED_AT--> {school!r}{deg_info} [{date_range}]", flush=True)

        pct = int((i + 1) / len(rows) * 100)
        print(f"PROGRESS: {pct}% | {i+1}/{len(rows)}", flush=True)

    print(f"\n📊 Extracted {len(triples)} triples from {len(rows)} entries.", flush=True)

    if dry_run:
        print("\n🔍 DRY RUN — no writes to Neo4j\n", flush=True)
        return

    if client is None:
        print("❌ No Neo4j client provided.", flush=True)
        sys.exit(1)

    client.batch_merge_relations(triples)
    print("✅ Education written to Neo4j.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract LinkedIn education history")
    parser.add_argument("--csv-file",   default="Education.csv")
    parser.add_argument("--self-name",  default="ME")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--neo4j-uri",  default=os.environ.get("NEO4J_URI",      "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER",     "neo4j"))
    parser.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--limit",      type=int, default=0)
    args = parser.parse_args()

    client = None
    if not args.dry_run:
        sys.path.insert(0, str((Path(__file__).parent.parent.parent).resolve()))
        from graph.neo4j_client import get_client
        client = get_client(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)

    extract(args.csv_file, self_name=args.self_name, dry_run=args.dry_run, client=client)

    if client:
        client.close()


if __name__ == "__main__":
    from pathlib import Path
    sys.path.insert(0, str((Path(__file__).parent.parent.parent).resolve()))
    main()
