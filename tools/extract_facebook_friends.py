#!/usr/bin/env python3
"""
Create Person nodes and FRIENDS_WITH relationships in Neo4j from a Facebook HTML export.

Usage:
    python tools/extract_facebook_friends.py <path_to_html> [--removed]

Expects the html file exported by Facebook containing the user's friend list.
If `--removed` is set, ingests WAS_FRIENDS_WITH instead of FRIENDS_WITH.
"""
import sys
import os
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import argparse

# Add parent directory to path to import config and graph utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import SELF_NAME
    from graph.neo4j_client import get_client
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

def parse_date(date_str: str) -> str:
    """
    Parses connection dates from the Facebook HTML format: "Jul 29, 2025 5:42:32 pm"
    Returns ISO 8601 date string.
    """
    try:
        # e.g., "Jul 29, 2025 5:42:32 pm"
        dt = datetime.strptime(date_str, "%b %d, %Y %I:%M:%S %p")
        return dt.isoformat()
    except ValueError:
        # Fallback if the format is weird
        return date_str

def extract_friends(html_path: str, is_removed: bool = False, dry_run: bool = False, limit: int = 0,
                    neo4j_uri: str = None, neo4j_user: str = None, neo4j_pass: str = None):
    print(f"📂 Opening: {html_path}")
    if not os.path.exists(html_path):
        print("Error: File not found.")
        sys.exit(1)

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # The HTML has many <section class="_a6-g"> that hold each friend
    sections = soup.find_all("section", class_="_a6-g")
    
    if not sections:
        print("❌ No friends found. Is this the right file? Ensure it is the Facebook friends HTML export.")
        return

    print(f"🕵 Found {len(sections)} connections.")
    
    friend_data = []
    
    for section in sections:
        # Extract the name from the h2 element
        name_elem = section.find("h2")
        name = name_elem.get_text(strip=True) if name_elem else "Unknown"
        
        # Avoid empty names and 'Unknown'
        if not name or name == "Unknown":
            continue
            
        # Extract the connection date from the <div class="_a72d"> element inside the footer
        date_elem = section.find("div", class_="_a72d")
        date_str = date_elem.get_text(strip=True) if date_elem else ""
        iso_date = parse_date(date_str) if date_str else ""
        
        friend_data.append({
            "name": name,
            "date": iso_date
        })

    print(f"✅ Extracted {len(friend_data)} valid names.")
    
    if limit > 0:
        friend_data = friend_data[:limit]
        print(f"📊 Limited extracted records to {limit}.")

    if not friend_data:
        return

    if dry_run:
        print("\n[DRY RUN] Would ingest the following relationships:")
        rel_type = "WAS_FRIENDS_WITH" if is_removed else "FRIENDS_WITH"
        date_prop = "removed_on" if is_removed else "since"
        for idx, friend in enumerate(friend_data[:10]):  # Print up to 10 for dry-run
            date_str = f" {date_prop}: {friend['date']}" if friend['date'] else ""
            print(f"  [REL] (Me)-[{rel_type}{date_str}]->({friend['name']})")
        if len(friend_data) > 10:
            print(f"  ... and {len(friend_data) - 10} more.")
        return

    # Ingest into Neo4j
    with get_client(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass) as client:
        print("📡 Checking connection to Neo4j...")
        if not client.verify():
            print("❌ Failed to connect to Neo4j. Is it running?")
            sys.exit(1)
            
        client.ensure_constraints()
        
        print("\n✅ Ingesting to Neo4j...")
        # We ensure the self node exists
        client.merge_entity("Person", SELF_NAME, {"is_self": True})
        
        # Batching relations makes this much faster
        rows = []
        rel_type = "WAS_FRIENDS_WITH" if is_removed else "FRIENDS_WITH"
        date_prop = "removed_on" if is_removed else "since"
        
        for friend in friend_data:
            rel_props = {"source": "facebook_friends_export"}
            if friend["date"]:
                rel_props[date_prop] = friend["date"]
                
            rows.append({
                "from_label": "Person",
                "from_name": SELF_NAME,
                "rel_type": rel_type,
                "to_label": "Person",
                "to_name": friend["name"],
                "props": rel_props
            })
            
        # Write batch to DB
        # chunk size of 1000 to be safe
        chunk_size = 1000
        total_chunks = (len(rows) + chunk_size - 1) // chunk_size
        
        with tqdm(total=len(rows), desc=f"MERGE {rel_type}") as pbar:
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i:i + chunk_size]
                client.batch_merge_relations(chunk)
                pbar.update(len(chunk))
                
        print(f"\n✅ Done! Successfully merged {len(friend_data)} {rel_type} relationships.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract friends from Facebook HTML export and ingest to Neo4j.")
    parser.add_argument("html_path", help="Path to the HTML file (e.g., your_friends.html)")
    parser.add_argument("--removed", action="store_true", help="Set this if importing removed friends to create WAS_FRIENDS_WITH relationships.")
    
    # Standard extract args passed by the UI Graph module
    parser.add_argument("--dry-run", action="store_true", help="Print extracted data without writing to Neo4j")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of edges to parse (0 = all)")
    parser.add_argument("--neo4j-uri", help="Neo4j URI")
    parser.add_argument("--neo4j-user", help="Neo4j username")
    parser.add_argument("--neo4j-pass", help="Neo4j password")
    
    args = parser.parse_args()
    
    extract_friends(
        args.html_path, 
        is_removed=args.removed,
        dry_run=args.dry_run,
        limit=args.limit,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pass=args.neo4j_pass
    )
