#!/usr/bin/env python3
"""
tools/extractors/linkedin_connections.py
-----------------------------------------
Parses LinkedIn "Connections.csv" export and writes KNOWS relationships
(with company/position context) into the Neo4j knowledge graph.

Expected LinkedIn CSV format (after the Notes header block):
  First Name,Last Name,URL,Email Address,Company,Position,Connected On
"""

import argparse
import csv
import os
import sys
from datetime import datetime

# ── date parser ────────────────────────────────────────────────────────────────
_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def _parse_date(raw: str) -> str:
    """Convert '03 Jan 2026' → '2026-01-03'"""
    raw = raw.strip()
    if not raw:
        return ""
    try:
        parts = raw.split()
        if len(parts) == 3:
            day, mon, year = parts
            return f"{year}-{_MONTHS.get(mon[:3].lower(), '00')}-{day.zfill(2)}"
    except Exception:
        pass
    return raw


# ── validation ─────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"First Name", "Last Name", "Connected On"}

def validate_csv(path: str) -> tuple[bool, str]:
    """Return (ok, error_message)."""
    if not os.path.exists(path):
        return False, f"File not found: {path}"
    try:
        # utf-8-sig strips the BOM that LinkedIn sometimes adds
        with open(path, "r", encoding="utf-8-sig") as f:
            header_line = None
            for i, line in enumerate(f):
                stripped = line.strip().strip('"')
                if stripped.startswith("First Name"):
                    header_line = line
                    break
                if i > 30:
                    return False, (
                        "Could not find CSV header (First Name, Last Name, …) "
                        "in the first 30 lines. Is this a LinkedIn Connections.csv?"
                    )
            if not header_line:
                return False, "File appears empty or has no valid CSV header."
        cols = {c.strip().strip('"') for c in header_line.split(",")}
        missing = REQUIRED_COLS - cols
        if missing:
            return False, f"Missing required columns: {missing}. Got: {cols}"
        return True, ""
    except Exception as e:
        return False, str(e)

# ── load ME's known companies from Positions.csv ───────────────────────────────
def _load_my_companies(positions_path: str) -> dict[str, dict]:
    """
    Returns a dict of {company_name_lower: {name, since, until}}
    by parsing the Positions.csv produced by linkedin_positions.py.
    """
    if not positions_path or not os.path.exists(positions_path):
        return {}
    companies = {}
    try:
        with open(positions_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Company Name") or row.get("Company") or "").strip()
                if name:
                    companies[name.lower()] = {
                        "name":  name,
                        "since": row.get("Started On", "").strip(),
                        "until": row.get("Finished On", "").strip(),
                    }
    except Exception as e:
        print(f"⚠️ Could not load positions file: {e}", flush=True)
    return companies


def extract(csv_path: str, self_name: str = "ME",
            positions_path: str = "",
            dry_run: bool = False, client=None):
    ok, err = validate_csv(csv_path)
    if not ok:
        print(f"❌ Invalid CSV: {err}", flush=True)
        sys.exit(1)

    ME_NODE = "ME"
    triples = []

    # Load ME's companies for COLLEAGUE_OF detection
    my_companies = _load_my_companies(positions_path)
    if my_companies:
        print(f"🏢 ME's known companies: {', '.join(c['name'] for c in my_companies.values())}", flush=True)
    else:
        print("ℹ️  No Positions.csv provided — skipping COLLEAGUE_OF detection.", flush=True)

    print(f"📂 Loading LinkedIn connections from {csv_path}...", flush=True)

    rows = []
    # utf-8-sig strips BOM automatically
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        header_line = None
        for line in f:
            stripped = line.strip().strip('"')
            if stripped.startswith("First Name"):
                header_line = line
                break
        if not header_line:
            print("❌ Could not find CSV header row.", flush=True)
            sys.exit(1)

        fieldnames = [c.strip().strip('"') for c in header_line.split(",")]
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})

    print(f"📦 Found {len(rows)} connections.", flush=True)

    for i, row in enumerate(rows):
        first     = row.get("First Name", "")
        last      = row.get("Last Name", "")
        company   = row.get("Company", "")
        position  = row.get("Position", "")
        connected = _parse_date(row.get("Connected On", ""))
        url       = row.get("URL", "")

        full_name = f"{first} {last}".strip()
        if not full_name:
            continue

        # KNOWS relationship (ME → Person)
        knows_props = {"since": connected, "source": "linkedin"}
        if url:
            knows_props["url"] = url
        triples.append({
            "from_label": "Person", "from_name": ME_NODE,
            "rel_type": "KNOWS",
            "to_label": "Person", "to_name": full_name,
            "props": knows_props,
        })
        print(f"[REL] {ME_NODE!r} --KNOWS--> {full_name!r} (since {connected})", flush=True)

        # WORKS_AT relationship (Person → Company) if known
        if company:
            works_props = {"title": position, "source": "linkedin_connection"}
            if connected:
                works_props["since"] = connected
            triples.append({
                "from_label": "Person", "from_name": full_name,
                "rel_type": "WORKS_AT",
                "to_label": "Company", "to_name": company,
                "props": works_props,
            })
            print(f"[ENT] {full_name!r} works at {company!r} as {position!r}", flush=True)

            # COLLEAGUE_OF — shared company with ME?
            shared = my_companies.get(company.lower())
            if shared:
                colleague_props = {
                    "company": company,
                    "source": "linkedin",
                    "since": connected,
                }
                triples.append({
                    "from_label": "Person", "from_name": ME_NODE,
                    "rel_type": "COLLEAGUE_OF",
                    "to_label": "Person", "to_name": full_name,
                    "props": colleague_props,
                })
                print(f"[REL] {ME_NODE!r} --COLLEAGUE_OF--> {full_name!r} (via {company!r}) 🏗️", flush=True)

        pct = int((i + 1) / len(rows) * 100)
        print(f"PROGRESS: {pct}% | {i+1}/{len(rows)}", flush=True)

    print(f"\n📊 Extracted {len(triples)} triples from {len(rows)} connections.", flush=True)

    if dry_run:
        print("\n🔍 DRY RUN — no writes to Neo4j\n", flush=True)
        return

    if client is None:
        print("❌ No Neo4j client provided.", flush=True)
        sys.exit(1)

    client.batch_merge_relations(triples)
    print("✅ All connections written to Neo4j.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract LinkedIn connections")
    parser.add_argument("--csv-file",       default="Connections.csv")
    parser.add_argument("--positions-file", default="",
                        help="Optional path to Positions.csv to detect COLLEAGUE_OF")
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

    extract(args.csv_file,
            self_name=args.self_name,
            positions_path=args.positions_file,
            dry_run=args.dry_run,
            client=client)

    if client:
        client.close()


if __name__ == "__main__":
    from pathlib import Path
    sys.path.insert(0, str((Path(__file__).parent.parent.parent).resolve()))
    main()
