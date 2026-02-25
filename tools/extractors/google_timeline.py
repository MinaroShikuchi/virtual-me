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


from geo_utils import Geocacher, parse_address_hierarchy




def extract(data: dict | list, self_name: str, min_visits: int = 5,
            dry_run: bool = False, client=None) -> dict:
    """
    data: parsed JSON from Records.json or Timeline.json.
    Supports both legacy {locations: [{latitudeE7...}]} 
    and modern {semanticSegments: [{visit...}]} formats.
    """
    import datetime

    # 1. Parse into a list of "visits": {lat, lng, start_time, end_time}
    #    and "trips": {start_lat, start_lng, end_lat, end_lng, start_time, end_time, distance, mode}
    visits = []
    trips = []
    
    # Modern format (semanticSegments)
    segments = data.get("semanticSegments", []) if isinstance(data, dict) else []
    if segments:
        print(f"📦 Found {len(segments):,} modern semantic segments.", flush=True)
        for i, seg in enumerate(segments):
            visit_data = seg.get("visit", {})
            if visit_data:
                top_cand = visit_data.get("topCandidate", {})
                loc = top_cand.get("placeLocation", {})
                lat_lng_str = loc.get("latLng", "")
                if lat_lng_str and "°" in lat_lng_str:
                    try:
                        lat_str, lng_str = lat_lng_str.replace("°", "").split(", ")
                        lat, lng = float(lat_str), float(lng_str)
                        st = seg.get("startTime", "")
                        et = seg.get("endTime", "")
                        # Store raw float coords; we will snap to grid later for grouping/address lookups
                        visits.append({"lat": lat, "lng": lng, "start_time": st, "end_time": et})
                    except ValueError:
                        pass
            
            activity_data = seg.get("activity", {})
            if activity_data:
                start_loc = activity_data.get("start", {}).get("latLng", "")
                end_loc = activity_data.get("end", {}).get("latLng", "")
                if start_loc and end_loc and "°" in start_loc and "°" in end_loc:
                    try:
                        slat_str, slng_str = start_loc.replace("°", "").split(", ")
                        elat_str, elng_str = end_loc.replace("°", "").split(", ")
                        slat, slng = float(slat_str), float(slng_str)
                        elat, elng = float(elat_str), float(elng_str)
                        st = seg.get("startTime", "")
                        et = seg.get("endTime", "")
                        dist = activity_data.get("distanceMeters", 0)
                        mode = activity_data.get("topCandidate", {}).get("type", "UNKNOWN")
                        trips.append({
                            "start_lat": slat, "start_lng": slng,
                            "end_lat": elat, "end_lng": elng,
                            "start_time": st, "end_time": et,
                            "distance": dist, "mode": mode
                        })
                    except ValueError:
                        pass
    
    # Legacy format (locations array) -> requires heuristic grouping
    records = data if isinstance(data, list) else data.get("locations", [])
    if records:
        print(f"📦 Found {len(records):,} legacy location records. Applying heuristic grouping...", flush=True)
        # Assuming records are somewhat sorted, but let's be safe
        # Records.json typically has a "timestamp" field
        parsed_records = []
        for r in records:
            lat = r.get("latitudeE7", 0) / 1e7
            lng = r.get("longitudeE7", 0) / 1e7
            ts = r.get("timestamp", "")
            if lat != 0 and lng != 0 and ts:
                parsed_records.append((ts, lat, lng))
        
        parsed_records.sort(key=lambda x: x[0])
        
        current_visit = None
        for ts, lat, lng in parsed_records:
            try:
                # Basic parsing, might need to handle different ISO formats
                dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
                
            grid_loc = _grid_key(lat, lng)
            
            if current_visit is None:
                current_visit = {"grid_loc": grid_loc, "lat": lat, "lng": lng, "start": dt, "end": dt}
            else:
                # If within same grid cell and within 3 hours, extend visit
                time_diff = (dt - current_visit["end"]).total_seconds()
                if current_visit["grid_loc"] == grid_loc and time_diff <= 3 * 3600:
                    current_visit["end"] = dt
                else:
                    # New visit
                    # Only keep if duration is > 5 mins to filter noise
                    duration = (current_visit["end"] - current_visit["start"]).total_seconds()
                    if duration >= 300:
                        visits.append({
                            "lat": current_visit["lat"], 
                            "lng": current_visit["lng"],
                            "start_time": current_visit["start"].isoformat(),
                            "end_time": current_visit["end"].isoformat()
                        })
                    current_visit = {"grid_loc": grid_loc, "lat": lat, "lng": lng, "start": dt, "end": dt}

        # Handle last
        if current_visit:
            duration = (current_visit["end"] - current_visit["start"]).total_seconds()
            if duration >= 300:
                visits.append({
                    "lat": current_visit["lat"], 
                    "lng": current_visit["lng"],
                    "start_time": current_visit["start"].isoformat(),
                    "end_time": current_visit["end"].isoformat()
                })

    if not visits and not trips:
        print("⚠️ No valid temporal visits or trips found.", flush=True)
        return {}

    print(f"🕰️ Extracted {len(visits)} distinct temporal visits.", flush=True)
    print(f"🛣️ Extracted {len(trips)} distinct trips.", flush=True)

    # 2. Count grid occurrences to find "home" and filter noise
    grid_counts = Counter()
    for v in visits:
        grid_key = _grid_key(v["lat"], v["lng"])
        grid_counts[grid_key] += 1

    home_key = grid_counts.most_common(1)[0][0] if grid_counts else None

    # 3. Resolve Addresses and Generate Triples
    triples = []
    counters = Counter()
    geocoder = Geocacher(Path("data/google/geocache.json"))

    # LIVES_IN logic (Home)
    if home_key:
        home_addr = geocoder.get_address(home_key[0], home_key[1])
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "LIVES_IN",
            "to_label":   "Place",  "to_name": home_addr,
            "props":      {"lat": home_key[0], "lng": home_key[1]},
        })
        counters["LIVES_IN"] += 1
        
        # Hierarchical Locations for Home
        hierarchy = parse_address_hierarchy(home_addr)
        if hierarchy.get("city"):
            triples.append({
                "from_label": "Place", "from_name": home_addr,
                "rel_type":   "IN_CITY",
                "to_label":   "City",  "to_name": hierarchy["city"],
                "props":      {}
            })
            counters["IN_CITY"] += 1
        if hierarchy.get("country"):
            city_name = hierarchy.get("city", "Unknown City")
            triples.append({
                "from_label": "City", "from_name": city_name,
                "rel_type":   "IN_COUNTRY",
                "to_label":   "Country",  "to_name": hierarchy["country"],
                "props":      {}
            })
            counters["IN_COUNTRY"] += 1

        print(f"[REL] Inferred Home: {home_addr!r}", flush=True)

    for i, v in enumerate(visits):
        grid_key = _grid_key(v["lat"], v["lng"])
        
        # Filter sparse locations (min_visits)
        if grid_counts[grid_key] < min_visits:
            continue
            
        addr = geocoder.get_address(grid_key[0], grid_key[1])
        
        # Create Visit ID (e.g. visit_2025_01_10_paris_001)
        short_name = addr.split(",")[0] if addr else "Location"
        date_str_clean = v["start_time"][:10].replace("-", "_") if v["start_time"] else "UnknownDate"
        short_name_clean = "".join(c for c in short_name if c.isalnum() or c.isspace()).strip().replace(" ", "_").lower()
        if not short_name_clean:
            short_name_clean = "loc"
        visit_id = f"visit_{date_str_clean}_{short_name_clean}_{i:03d}"
        
        # 1. Person -> ATTENDED -> Visit
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "ATTENDED",
            "to_label":   "Visit",  "to_name": visit_id,
            "props":      {
                "id": visit_id,
                "start": v["start_time"],
                "end": v["end_time"]
            }
        })
        counters["ATTENDED"] += 1
        
        # 2. Visit -> LOCATED_AT -> Place
        triples.append({
            "from_label": "Visit", "from_name": visit_id,
            "rel_type":   "LOCATED_AT",
            "to_label":   "Place",  "to_name": addr,
            "props":      {"address": addr, "lat": v["lat"], "lon": v["lng"]}
        })
        counters["LOCATED_AT"] += 1
        
        # 3. Hierarchical Locations
        hierarchy = parse_address_hierarchy(addr)
        if hierarchy.get("city"):
            triples.append({
                "from_label": "Place", "from_name": addr,
                "rel_type":   "IN_CITY",
                "to_label":   "City",  "to_name": hierarchy["city"],
                "props":      {}
            })
            counters["IN_CITY"] += 1
        if hierarchy.get("country"):
            city_name = hierarchy.get("city", "Unknown City")
            triples.append({
                "from_label": "City", "from_name": city_name,
                "rel_type":   "IN_COUNTRY",
                "to_label":   "Country",  "to_name": hierarchy["country"],
                "props":      {}
            })
            counters["IN_COUNTRY"] += 1

        print(f"[REL] {self_name!r} --ATTENDED--> {visit_id!r} --LOCATED_AT--> {addr!r}", flush=True)

    print(f"\n📊 Google Timeline: {counters.get('ATTENDED',0)} Visits created linking to Places.", flush=True)

    # Process Trips
    for i, t in enumerate(trips):
        start_grid = _grid_key(t["start_lat"], t["start_lng"])
        end_grid = _grid_key(t["end_lat"], t["end_lng"])
        
        start_addr = geocoder.get_address(start_grid[0], start_grid[1])
        end_addr = geocoder.get_address(end_grid[0], end_grid[1])
        
        date_str_clean = t["start_time"][:10].replace("-", "_") if t["start_time"] else "UnknownDate"
        mode_clean = t["mode"].replace("IN_", "").lower()
        trip_id = f"trip_{date_str_clean}_{mode_clean}_{i:03d}"
        
        # 1. Person -> TOOK_TRIP -> Trip
        triples.append({
            "from_label": "Person", "from_name": self_name,
            "rel_type":   "TOOK_TRIP",
            "to_label":   "Trip",  "to_name": trip_id,
            "props":      {
                "id": trip_id,
                "start": t["start_time"],
                "end": t["end_time"],
                "mode": t["mode"],
                "distance": t["distance"]
            }
        })
        counters["TOOK_TRIP"] += 1
        
        # 2. Trip -> STARTED_AT -> Place
        triples.append({
            "from_label": "Trip", "from_name": trip_id,
            "rel_type":   "STARTED_AT",
            "to_label":   "Place",  "to_name": start_addr,
            "props":      {"address": start_addr, "lat": t["start_lat"], "lon": t["start_lng"]}
        })
        counters["STARTED_AT"] += 1
        
        # 3. Trip -> ENDED_AT -> Place
        triples.append({
            "from_label": "Trip", "from_name": trip_id,
            "rel_type":   "ENDED_AT",
            "to_label":   "Place",  "to_name": end_addr,
            "props":      {"address": end_addr, "lat": t["end_lat"], "lon": t["end_lng"]}
        })
        counters["ENDED_AT"] += 1
        
        # 4. Hierarchical Locations for Start and End Places
        for loc_addr in (start_addr, end_addr):
            hierarchy = parse_address_hierarchy(loc_addr)
            if hierarchy.get("city"):
                triples.append({
                    "from_label": "Place", "from_name": loc_addr,
                    "rel_type":   "IN_CITY",
                    "to_label":   "City",  "to_name": hierarchy["city"],
                    "props":      {}
                })
                counters["IN_CITY"] += 1
            if hierarchy.get("country"):
                city_name = hierarchy.get("city", "Unknown City")
                triples.append({
                    "from_label": "City", "from_name": city_name,
                    "rel_type":   "IN_COUNTRY",
                    "to_label":   "Country",  "to_name": hierarchy["country"],
                    "props":      {}
                })
                counters["IN_COUNTRY"] += 1

        print(f"[REL] {self_name!r} --TOOK_TRIP--> {trip_id!r} [{t['mode']} {t['distance']:.0f}m]", flush=True)

    print(f"\n📊 Google Timeline: {counters.get('TOOK_TRIP',0)} Trips created linking to start/end Places.", flush=True)

    if dry_run:
        for t in triples[:10]:
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r}")
    elif client is not None:
        client.ensure_constraints()
        print("✍️ Executing custom Cypher ingestion for timeline schema...", flush=True)
        with client.driver.session() as s:
            for t in triples:
                if t["rel_type"] == "LIVES_IN":
                    s.run(
                        "MERGE (a:Person {name: $fn}) "
                        "MERGE (b:Place {name: $tn}) "
                        "MERGE (a)-[r:LIVES_IN]->(b) "
                        "ON CREATE SET r.lat = $lat, r.lng = $lng "
                        "ON MATCH SET r.lat = $lat, r.lng = $lng",
                        fn=t["from_name"], tn=t["to_name"],
                        lat=t["props"].get("lat"), lng=t["props"].get("lng")
                    )
                elif t["rel_type"] == "ATTENDED":
                    p = t["props"]
                    # Create Visit node with id and exact datetimes
                    s.run(
                        "MERGE (a:Person {name: $fn}) "
                        "MERGE (v:Visit {id: $vid}) "
                        "ON CREATE SET v.name = $vid, v.start = datetime($st), v.end = datetime($et) "
                        "MERGE (a)-[:ATTENDED]->(v)",
                        fn=t["from_name"], vid=p["id"],
                        st=p["start"], et=p["end"]
                    )
                elif t["rel_type"] == "LOCATED_AT":
                    p = t["props"]
                    # Link Visit to Place with specific props
                    s.run(
                        "MERGE (v:Visit {id: $vid}) "
                        "MERGE (p:Place {name: $addr}) "
                        "ON CREATE SET p.address = $addr, p.lat = $lat, p.lon = $lon "
                        "ON MATCH SET p.address = $addr, p.lat = $lat, p.lon = $lon "
                        "MERGE (v)-[:LOCATED_AT]->(p)",
                        vid=t["from_name"], addr=t["to_name"],
                        lat=p["lat"], lon=p["lon"]
                    )
                elif t["rel_type"] == "TOOK_TRIP":
                    p = t["props"]
                    s.run(
                        "MERGE (a:Person {name: $fn}) "
                        "MERGE (tr:Trip {id: $tid}) "
                        "ON CREATE SET tr.name = $tid, tr.start = datetime($st), tr.end = datetime($et), tr.mode = $mode, tr.distance = $dist "
                        "MERGE (a)-[:TOOK_TRIP]->(tr)",
                        fn=t["from_name"], tid=p["id"],
                        st=p["start"], et=p["end"],
                        mode=p["mode"], dist=p["distance"]
                    )
                elif t["rel_type"] in ("STARTED_AT", "ENDED_AT"):
                    p = t["props"]
                    rel = t["rel_type"]
                    s.run(
                        "MERGE (tr:Trip {id: $tid}) "
                        "MERGE (pl:Place {name: $addr}) "
                        "ON CREATE SET pl.address = $addr, pl.lat = $lat, pl.lon = $lon "
                        "ON MATCH SET pl.address = $addr, pl.lat = $lat, pl.lon = $lon "
                        f"MERGE (tr)-[:{rel}]->(pl)",
                        tid=t["from_name"], addr=t["to_name"],
                        lat=p.get("lat"), lon=p.get("lon")
                    )
                elif t["rel_type"] == "IN_CITY":
                    s.run(
                        "MERGE (p:Place {name: $pn}) "
                        "MERGE (c:City {name: $cn}) "
                        "MERGE (p)-[:IN_CITY]->(c)",
                        pn=t["from_name"], cn=t["to_name"]
                    )
                elif t["rel_type"] == "IN_COUNTRY":
                    s.run(
                        "MERGE (c:City {name: $cn}) "
                        "MERGE (co:Country {name: $con}) "
                        "MERGE (c)-[:IN_COUNTRY]->(co)",
                        cn=t["from_name"], con=t["to_name"]
                    )
        print("✅ Written custom Timeline schema to Neo4j.", flush=True)

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

    if args.dry_run:
        extract(data, args.self_name, min_visits=args.min_visits, dry_run=True)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        from graph.neo4j_client import Neo4jClient
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            extract(data, args.self_name, min_visits=args.min_visits, client=client)


if __name__ == "__main__":
    main()
