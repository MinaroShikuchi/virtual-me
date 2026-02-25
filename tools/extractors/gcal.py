#!/usr/bin/env python3
"""
tools/extractors/calendar.py
----------------------------
Extract entities and relationships from Google Calendar .ics files.

Properly expands recurring events (RRULE) and handles timezone 
conversions using `icalendar` and `recurring_ical_events`.

Relationships extracted:
  Person -> ATTENDED -> Event
  Event -> LOCATED_AT -> Place (if location is provided)

Usage:
  python3 tools/extractors/calendar.py --data-dir data/google/Calendar [options]
"""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

try:
    import icalendar
    import recurring_ical_events
except Exception as e:
    print(f"❌ Missing dependencies: {e}. Please run: pip install icalendar recurring_ical_events", flush=True)
    sys.exit(1)


def parse_ics_files(data_dir: Path, start_year: int = 2000, end_year: int = 2030) -> list[dict]:
    """Parse all .ics files in the given directory and expand recurrences."""
    events = []
    
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"❌ Calendar directory not found: {data_dir}", flush=True)
        return events
        
    ics_files = list(data_dir.glob("*.ics"))
    if not ics_files:
        print(f"⚠️ No .ics files found in {data_dir}", flush=True)
        return events
        
    print(f"📂 Found {len(ics_files)} .ics files. Expanding events from {start_year} to {end_year}...", flush=True)
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    for ics_file in ics_files:
        cal_name = ics_file.stem
        try:
            with open(ics_file, "r", encoding="utf-8") as f:
                cal = icalendar.Calendar.from_ical(f.read())
                
            # Expand recurring events
            events_in_cal = recurring_ical_events.of(cal).between(start_date, end_date)
            
            for ev in events_in_cal:
                uid = str(ev.get("UID", ""))
                summary = str(ev.get("SUMMARY", "Untitled Event"))
                desc = str(ev.get("DESCRIPTION", ""))
                location = str(ev.get("LOCATION", ""))
                
                # Handle dates (could be date or datetime)
                dtstart = ev.get("DTSTART")
                dtend = ev.get("DTEND")
                
                if not dtstart:
                    continue
                    
                start_dt = dtstart.dt
                # If it's a date object without time, convert to datetime at midnight
                if not isinstance(start_dt, datetime):
                    start_dt = datetime.combine(start_dt, datetime.min.time())
                    
                if dtend:
                    end_dt = dtend.dt
                    if not isinstance(end_dt, datetime):
                        end_dt = datetime.combine(end_dt, datetime.min.time())
                else:
                    # Default duration if end not provided
                    end_dt = start_dt + timedelta(hours=1)
                
                # Make sure timezone info is removed or converted to UTC string for Neo4j
                # Actually, Neo4j handles ISO strings with offsets perfectly. 
                # Just need to make sure we serialize correctly.
                # If tzinfo is missing, assume UTC or local. Let's just use isoformat.
                
                start_iso = start_dt.isoformat()
                end_iso = end_dt.isoformat()
                
                # Generate unique ID for this instance
                # uid might be the same for recurrences, so we append the start time to make it unique
                instance_id = f"evt_{uid}_{start_iso.replace(':', '').replace('-', '')}"
                
                events.append({
                    "id": instance_id,
                    "calendar": cal_name,
                    "summary": summary,
                    "description": desc,
                    "location": location,
                    "start": start_iso,
                    "end": end_iso
                })
        except Exception as e:
            print(f"⚠️ Error parsing {ics_file.name}: {e}", flush=True)
            
    print(f"🗓️ Extracted {len(events)} distinct event instances.", flush=True)
    return events


def extract(events: list[dict], self_name: str, dry_run: bool = False, client=None) -> dict:
    triples = []
    counters = Counter()
    
    for i, ev in enumerate(events):
        event_id = ev["id"]
        
        # 1. Person -> ATTENDED -> Event
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type": "ATTENDED",
            "to_label": "Event", "to_name": event_id,
            "props": ev
        })
        counters["ATTENDED"] += 1
        print(f"[REL] {self_name!r} --ATTENDED--> {ev.get('summary', 'Unknown Event')!r} ({ev.get('start', '')})", flush=True)
        
        # 2. Event -> LOCATED_AT -> Place (if location exists)
        location = ev.get("location", "").strip()
        if location:
            triples.append({
                "from_label": "Event", "from_name": event_id,
                "rel_type": "LOCATED_AT",
                "to_label": "Place", "to_name": location,
                "props": {
                    "name": location,
                    "address": location
                }
            })
            counters["LOCATED_AT"] += 1
            print(f"      L--LOCATED_AT--> {location!r}", flush=True)
            
        if (i + 1) % 1000 == 0 or (i + 1) == len(events):
            pct = int((i + 1) / len(events) * 100)
            print(f"PROGRESS: {pct}% | {i+1}/{len(events)}", flush=True)
            
    print(f"\n📊 Calendar: {counters.get('ATTENDED', 0)} events created.", flush=True)
    
    if dry_run:
        for t in triples[:10]:
            loc_str = f" -> {t['to_name']!r}" if t["rel_type"] == "LOCATED_AT" else ""
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r}")
            if t["rel_type"] == "ATTENDED":
                print(f"    Props: {t['props']['summary']} ({t['props']['start']})")
    elif client is not None:
        client.ensure_constraints()
        print("✍️ Executing Cypher ingestion for Calendar schema...", flush=True)
        with client.driver.session() as s:
            for t in triples:
                if t["rel_type"] == "ATTENDED":
                    p = t["props"]
                    s.run(
                        "MERGE (a:Person {name: $fn}) "
                        "MERGE (e:Event {id: $eid}) "
                        "ON CREATE SET e.name = $eid, e.summary = $summary, e.description = $desc, "
                        "e.start = datetime($st), e.end = datetime($et), e.calendar = $cal, e.source = 'google_calendar' "
                        "ON MATCH SET e.name = $eid, e.summary = $summary, e.description = $desc, "
                        "e.start = datetime($st), e.end = datetime($et), e.calendar = $cal, e.source = 'google_calendar' "
                        "MERGE (a)-[:ATTENDED]->(e)",
                        fn=t["from_name"], eid=p["id"], summary=p["summary"], desc=p["description"],
                        st=p["start"], et=p["end"], cal=p["calendar"]
                    )
                elif t["rel_type"] == "LOCATED_AT":
                    p = t["props"]
                    s.run(
                        "MERGE (e:Event {id: $eid}) "
                        "MERGE (pl:Place {name: $pname}) "
                        "ON CREATE SET pl.address = $pname "
                        "MERGE (e)-[:LOCATED_AT]->(pl)",
                        eid=t["from_name"], pname=p["name"]
                    )
        print("✅ Written custom Calendar schema to Neo4j.", flush=True)
        
    return dict(counters)

def main():
    p = argparse.ArgumentParser(description="Extract KG triples from Google Calendar ICS files")
    p.add_argument("--data-dir",   default="data/google/Calendar", help="Path to folder containing .ics files")
    p.add_argument("--self-name",  default=os.environ.get("SELF_NAME", "Me"))
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--limit",      type=int, default=0)
    p.add_argument("--neo4j-uri",  default=os.environ.get("NEO4J_URI",      "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER",     "neo4j"))
    p.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "password"))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    # The user has files in Documents/Takeout/Calendar
    # the arg default is data/google/Calendar. We'll override if the default doesn't exist but Takeout does.
    if not data_dir.exists():
        alt_path = Path("/home/minaro/Documents/Takeout/Calendar")
        if alt_path.exists():
            data_dir = alt_path

    events = parse_ics_files(data_dir)
    if not events:
        sys.exit(1)
        
    if args.limit:
        events = events[:args.limit]

    if args.dry_run:
        extract(events, args.self_name, dry_run=True)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        from graph.neo4j_client import Neo4jClient
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            extract(events, args.self_name, client=client)

if __name__ == "__main__":
    main()
