
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
import gzip
import re
import datetime
from collections import Counter, defaultdict
from pathlib import Path

from geo_utils import Geocacher, parse_address_hierarchy


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


def _parse_date(date_str: str) -> str:
    """Safely parse various date strings into ISO-8601 format."""
    if not date_str:
        return "1970-01-01T00:00:00"
    
    # Try ISO
    try:
        from datetime import datetime
        if "Z" in date_str:
            date_str = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(date_str).isoformat()
    except ValueError:
        pass
    
    # Try Strava CSV format: Jun 3, 2018, 7:00:34 PM
    try:
        from datetime import datetime
        return datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p").isoformat()
    except ValueError:
        pass
        
    return "1970-01-01T00:00:00"


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(csv_path: str | Path, limit: int = 0) -> list[dict]:
    """
    Load Strava activities.csv and return a list of activity dicts.
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
                "id":           row.get("Activity ID", ""),
                "name":         row.get("Activity Name", ""),
                "type":         row.get("Activity Type", "Unknown"),
                "start_date":   row.get("Activity Date", ""),
                "elapsed_time": _parse_int(row.get("Elapsed Time", "0")),
                "distance":     _parse_float(row.get("Distance", "0")),
                "avg_hr":       _parse_float(row.get("Average Heart Rate", "0")),
                "calories":     _parse_float(row.get("Calories", "0")),
                "start_lat":    _parse_float(row.get("Start Latitude", "0")),
                "start_lng":    _parse_float(row.get("Start Longitude", "0")),
                "filename":     row.get("Filename", "")
            })
            if limit and len(activities) >= limit:
                break

    print(f"📦 {len(activities):,} Strava activities loaded from CSV.", flush=True)
    return activities


# ── Raw File Parsing ──────────────────────────────────────────────────────────

def _extract_start_coords(file_path: Path) -> tuple[float, float]:
    """
    Parse the first coordinate point from GPX or TCX files.
    Supports .gz compressed files.
    """
    if not file_path.exists():
        return 0.0, 0.0

    content = b""
    try:
        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                content = f.read()
        else:
            with open(file_path, "rb") as f:
                content = f.read()
    except Exception as e:
        print(f"      ⚠️ Failed to read {file_path.name}: {e}", flush=True)
        return 0.0, 0.0

    if not content:
        return 0.0, 0.0

    text = content.decode("utf-8", errors="ignore")

    # Try GPS parsing with XML first (more robust)
    try:
        if "<gpx" in text.lower():
            # Basic XML parsing for GPX
            root = ET.fromstring(text)
            # Find first trkpt (ignore namespaces for simplicity if possible, or handle them)
            # Standard approach: find {namespace}trkpt
            ns = ""
            if "}" in root.tag:
                ns = root.tag.split("}")[0] + "}"
            
            trk = root.find(f".//{ns}trkpt")
            if trk is not None:
                return float(trk.attrib["lat"]), float(trk.attrib["lon"])
        
        elif "<track" in text.lower():
            # Basic parsing for TCX
            root = ET.fromstring(text)
            ns = ""
            if "}" in root.tag:
                ns = root.tag.split("}")[0] + "}"
            
            lat_node = root.find(f".//{ns}LatitudeDegrees")
            lng_node = root.find(f".//{ns}LongitudeDegrees")
            if lat_node is not None and lng_node is not None:
                return float(lat_node.text), float(lng_node.text)
    except Exception:
        pass # Fallback to regex if XML fails

    # Regex Fallback
    m_gpx = re.search(r'<trkpt\s+lat=["\'](-?\d+\.\d+)["\']\s+lon=["\'](-?\d+\.\d+)["\']', text)
    if m_gpx:
        return float(m_gpx.group(1)), float(m_gpx.group(2))

    m_lat = re.search(r'<LatitudeDegrees>([^<]+)</LatitudeDegrees>', text)
    m_lng = re.search(r'<LongitudeDegrees>([^<]+)</LongitudeDegrees>', text)
    if m_lat and m_lng:
        try:
            return float(m_lat.group(1)), float(m_lng.group(1))
        except ValueError: pass

    return 0.0, 0.0


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
            dry_run: bool = False, client=None, source_path: Path = None) -> dict:
    """
    Extract triples from a list of Strava activity dicts.
    """
    triples = []
    counters: Counter = Counter()

    geocoder = Geocacher(Path("data/geocache.json"))

    for i, act in enumerate(activities):
        act_id = str(act.get("id", act.get("Activity ID", "")))
        if not act_id:
            act_id = f"unknown_{i}"
        
        full_id = f"strava_{act_id}"
        
        act_name = act.get("name", act.get("Activity Name", "Strava Activity"))
        act_type = act.get("type", "Unknown")
        
        start_date = act.get("start_date", act.get("start_date_local", ""))
        iso_date = _parse_date(start_date)
        
        elapsed = _parse_int(str(act.get("elapsed_time", "0")))
        dist = _parse_float(str(act.get("distance", "0")))
        avg_hr = _parse_float(str(act.get("avg_hr", act.get("average_heartrate", "0"))))
        cals = _parse_float(str(act.get("calories", act.get("Calories", "0"))))

        # Coordinate resolution: Prefer GPX/TCX files if available
        lat, lng = None, None
        
        if act.get("filename") and source_path:
            raw_path = source_path / act["filename"]
            if raw_path.exists():
                f_lat, f_lng = _extract_start_coords(raw_path)
                if abs(f_lat) > 0.001 or abs(f_lng) > 0.001:
                    lat, lng = f_lat, f_lng
                    act["start_lat"] = lat
                    act["start_lng"] = lng
                else:
                    if act_id == "14891243176":
                        print(f"      ❗ Debug: GPX file found but returned 0.0 coords: {raw_path}", flush=True)
            else:
                if act_id == "14891243176":
                    print(f"      ❌ Debug: GPX file NOT FOUND at: {raw_path}", flush=True)

        # Fallback to CSV/JSON fields if GPX failed or wasn't provided
        if lat is None:
            lat = _parse_float(str(act.get("start_lat", "0")))
            lng = _parse_float(str(act.get("start_lng", "0")))
            # Some Strava exports have start_latlng as a list
            if lat == 0 and lng == 0 and act.get("start_latlng"):
                sl = act.get("start_latlng")
                if isinstance(sl, list) and len(sl) == 2:
                    lat, lng = sl[0], sl[1]
            
        # Final safety check for downstream geocoding
        if lat is None: lat = 0.0
        if lng is None: lng = 0.0

        # 1. Person -> PERFORMED -> Activity
        triples.append({
            "from_label": "Person",   "from_name": self_name,
            "rel_type":   "PERFORMED",
            "to_label":   "Activity", "to_name": full_id,
            "props": {
                "id": full_id,
                "type": act_type,
                "start": iso_date,
                "duration_sec": elapsed,
                "distance_m": dist,
                "avg_hr": avg_hr,
                "calories": cals,
                "source": "strava",
            },
        })
        counters["PERFORMED"] += 1
        
        # 2. Activity -> LOCATED_AT -> Place
        route_name = act_name if act_name else f"{act_type} Route"
        
        # If we have coordinates, try to get a real address
        place_name = route_name
        has_geo = False
        if abs(lat) > 0.1 and abs(lng) > 0.1:
            addr = geocoder.get_address(lat, lng)
            if act_id == "14891243176":
                print(f"      📍 Debug: geocoder returned: {addr!r}", flush=True)
            
            if addr and not addr.startswith("lat"):
                place_name = addr
                has_geo = True
            elif addr: # fallback lat_lng name
                place_name = addr
            else:
                # Absolute fallback if geocoder returned empty (shouldn't happen with new geo_utils)
                place_name = f"lat{lat:.6f}_lng{lng:.6f}"

        triples.append({
            "from_label": "Activity", "from_name": full_id,
            "rel_type":   "LOCATED_AT",
            "to_label":   "Place",  "to_name": place_name,
            "props": {
                "name": place_name,
                "lat": lat if lat != 0 else None,
                "lng": lng if lng != 0 else None
            }
        })
        counters["LOCATED_AT"] += 1

        print(f"[REL] {self_name!r} --PERFORMED--> {full_id!r} [{act_type} {dist:.0f}m]", flush=True)
        print(f"[REL] {full_id!r} --LOCATED_AT--> {place_name!r}", flush=True)

        # 3. Hierarchical Locations (only if geocoded successfully)
        if has_geo:
            hierarchy = parse_address_hierarchy(place_name)
            if hierarchy.get("city"):
                city_name = hierarchy["city"]
                triples.append({
                    "from_label": "Place", "from_name": place_name,
                    "rel_type":   "IN_CITY",
                    "to_label":   "City",  "to_name": city_name,
                    "props":      {}
                })
                counters["IN_CITY"] += 1
                print(f"[REL] {place_name!r} --IN_CITY--> {city_name!r}", flush=True)

            if hierarchy.get("country"):
                city_name = hierarchy.get("city", "Unknown City")
                country_name = hierarchy["country"]
                triples.append({
                    "from_label": "City", "from_name": city_name,
                    "rel_type":   "IN_COUNTRY",
                    "to_label":   "Country",  "to_name": country_name,
                    "props":      {}
                })
                counters["IN_COUNTRY"] += 1
                print(f"[REL] {city_name!r} --IN_COUNTRY--> {country_name!r}", flush=True)

        if (i + 1) % 500 == 0 or (i + 1) == len(activities):
            pct = int((i + 1) / len(activities) * 100)
            print(f"PROGRESS: {pct}% | {i+1}/{len(activities)}", flush=True)

    print(f"\n📊 Strava: {counters.get('PERFORMED', 0)} exact activities created.", flush=True)

    if dry_run:
        for t in triples[:10]:
            print(f"  [{t['from_label']}] {t['from_name']!r} "
                  f"--{t['rel_type']}--> [{t['to_label']}] {t['to_name']!r} "
                  f"  {t['props']}")
    elif client is not None:
        client.ensure_constraints()
        print("✍️ Executing custom Cypher ingestion for Strava schema...", flush=True)
        with client.driver.session() as s:
            for t in triples:
                if t["rel_type"] == "PERFORMED":
                    p = t["props"]
                    s.run(
                        "MERGE (a:Person {name: $fn}) "
                        "MERGE (act:Activity {id: $act_id}) "
                        "ON CREATE SET act.name = $act_id, act.type = $type, act.start = datetime($st), "
                        "act.duration_sec = $dur, act.distance_m = $dist, act.avg_hr = $hr, act.calories = $cal, act.source = $src "
                        "ON MATCH SET act.name = $act_id, act.type = $type, act.start = datetime($st), "
                        "act.duration_sec = $dur, act.distance_m = $dist, act.avg_hr = $hr, act.calories = $cal, act.source = $src "
                        "MERGE (a)-[:PERFORMED]->(act)",
                        fn=t["from_name"], act_id=p["id"],
                        type=p["type"], st=p["start"], dur=p["duration_sec"],
                        dist=p["distance_m"], hr=p["avg_hr"], cal=p["calories"], src=p["source"]
                    )
                elif t["rel_type"] == "LOCATED_AT":
                    p = t["props"]
                    s.run(
                        "MERGE (act:Activity {id: $act_id}) "
                        "MERGE (pl:Place {name: $pname}) "
                        "ON CREATE SET pl.lat = $lat, pl.lng = $lng "
                        "ON MATCH SET pl.lat = $lat, pl.lng = $lng "
                        "MERGE (act)-[:LOCATED_AT]->(pl)",
                        act_id=t["from_name"], pname=p["name"],
                        lat=p.get("lat"), lng=p.get("lng")
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
        print("✅ Written custom Strava schema to Neo4j.", flush=True)

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
    activities = []
    # source_path is the base directory for GPX/TCX file resolution.
    # We prioritize the provided data_dir, but if it's default we check if it actually exists.
    source_path = Path(args.data_dir)
    
    if args.csv_file:
        activities = load_csv(args.csv_file, limit=args.limit)
    else:
        # Smart detection: if data_dir contains activities.csv, use CSV mode
        csv_check = source_path / "activities.csv"
        if csv_check.exists():
            print(f"✨ Found activities.csv in {source_path}. Using CSV mode.", flush=True)
            activities = load_csv(csv_check, limit=args.limit)
        else:
            activities = load_json(source_path, limit=args.limit)

    if not activities:
        print("❌ No activities found.", flush=True)
        sys.exit(1)

    if args.dry_run:
        extract(activities, args.self_name, dry_run=True, 
                source_path=source_path)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        from graph.neo4j_client import Neo4jClient
        with Neo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_pass) as client:
            extract(activities, args.self_name, client=client,
                    source_path=source_path)


if __name__ == "__main__":
    main()
