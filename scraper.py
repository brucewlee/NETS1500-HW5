#!/usr/bin/env python3
"""
scrape_submarine_cables_streaming.py
------------------------------------
An improved version of our submarine cable scraper that saves data incrementally!

This script fetches information about global submarine cables and their landing points,
but unlike our previous version, this one writes data to disk as it processes each cable.

Main benefits:
- CSV file grows row-by-row as we process
- NDJSON file grows cable-by-cable
- If you need to stop the script midway (Ctrl+C), you won't lose progress
- Much safer for long-running scrapes where something might go wrong

Think of it as checkpointing our work as we go along!
"""

import json, csv, time, re, os
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm

# Basic configuration - where to find the data and where to save it
BASE = "https://www.submarinecablemap.com/api/v3"  # API endpoint
UA   = "network-sim/1.1 (contact: you@example.com)"  # Be nice and identify ourselves
OUT  = Path("output_cable_map")  # Where we'll save our findings
OUT.mkdir(exist_ok=True)  # Make sure this folder exists

#############################################################################
# Helper functions - the building blocks we'll use throughout
#############################################################################
def fetch_json(url: str, tries: int = 3, timeout: int = 30) -> dict:
    """
    Get JSON data from a URL with retries if something goes wrong.
    
    We'll try up to 3 times with exponential backoff - wait longer
    between each retry to be gentle on the server if it's struggling.
    """
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
            r.raise_for_status()  # Will trigger an exception if 4xx/5xx error
            return r.json()
        except Exception as exc:
            # Last attempt? Then give up and let the exception bubble up
            if k == tries - 1:
                raise
            # Otherwise, wait a bit (longer after each failure) and retry
            wait = 2 ** k  # 1s, then 2s, then 4s...
            print(f"[warn] {exc} — retrying in {wait}s")
            time.sleep(wait)

def slugify(text: str) -> str:
    """
    Convert text into a URL-friendly slug (lowercase, no spaces or special chars).
    
    This helps us create consistent identifiers for cables when needed.
    """
    text = re.sub(r"[^\w\s-]", "", (text or "").lower())  # Remove weird characters
    return re.sub(r"[-\s]+", "-", text).strip("-_") or "unknown"  # Clean up spaces/hyphens

#############################################################################
# Step 1: First, let's get geographic info for all landing points
#############################################################################
print("Downloading landing-point GeoJSON...")
lp_geo = fetch_json(f"{BASE}/landing-point/landing-point-geo.json")

# Create a lookup dictionary so we can easily find landing point info by ID
landing_by_id: Dict[str, dict] = {}
for feat in lp_geo["features"]:
    props = feat.get("properties", {})
    # Try several possible ID fields (the API isn't totally consistent)
    lp_id = str(
        props.get("id") or feat.get("id") or props.get("fid") or props.get("objectid")
    )
    if not lp_id:
        continue  # Skip if we can't find an ID
    
    # Store the useful info for this landing point
    landing_by_id[lp_id] = {
        "name": props.get("name"),
        "country": props.get("country_name") or props.get("country"),
        "latitude": feat["geometry"]["coordinates"][1],
        "longitude": feat["geometry"]["coordinates"][0],
    }

#############################################################################
# Step 2: Get a list of all submarine cable IDs we need to process
#############################################################################
print("Downloading cable-geo index...")
cable_geo = fetch_json(f"{BASE}/cable/cable-geo.json")
# Extract just the cable IDs we need to look up details for
cable_ids: List[str] = sorted(
    {str(f["properties"].get("id")) for f in cable_geo["features"] if f["properties"].get("id")}
)

#############################################################################
# Step 3: Set up our output files (in append mode so we can add as we go)
#############################################################################
csv_path = OUT / "cables_with_landing_points.csv"
ndjson_path = OUT / "cables_with_landing_points.ndjson"

# Only write the CSV header if the file is new or empty
# This way we can resume a partial run without duplicate headers
write_header = not csv_path.exists() or csv_path.stat().st_size == 0
csv_file = csv_path.open("a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(
        ["cable_key", "cable_name", "rfs_year", "length_km",
         "lp_id", "lp_name", "country", "latitude", "longitude"]
    )
    csv_file.flush()  # Make sure the header gets written to disk immediately

# Open our NDJSON file in append mode too
ndjson_file = ndjson_path.open("a", encoding="utf-8")

#############################################################################
# Step 4: Process each cable and save info as we go
#############################################################################
for cab_id in tqdm(cable_ids, desc="Fetching cables", unit="cable"):
    try:
        # Get detailed info for this specific cable
        cab = fetch_json(f"{BASE}/cable/{cab_id}.json")
    except Exception as e:
        # If something goes wrong with this cable, log it and move on
        # No need to crash the whole script for one problematic cable
        print(f"[error] skipping cable id={cab_id}  ({e})")
        continue

    # Be nice to the server - don't hammer it with requests
    time.sleep(0.1)
    
    # Use the cable's slug as a key, or fallback to its ID
    key = cab.get("slug") or f"id-{cab_id}"

    # Process and enrich each landing point associated with this cable
    lp_objs = []
    for lp in cab.get("landing_points", []):
        lp_id = str(lp.get("id") or lp)
        geo = landing_by_id.get(lp_id, {})  # Look up geographic info we fetched earlier
        
        # Combine specific landing point info with our geographic lookup
        lp_rec = {
            "id"       : lp_id,
            "name"     : lp.get("name")    or geo.get("name"),
            "country"  : lp.get("country") or geo.get("country"),
            "latitude" : geo.get("latitude"),
            "longitude": geo.get("longitude"),
        }
        lp_objs.append(lp_rec)

        # Write one CSV row for each cable-landing point combination
        # This creates a flattened representation that's easy to analyze
        csv_writer.writerow(
            [key, cab.get("name"), cab.get("rfs_year"), cab.get("length"),
             lp_rec["id"], lp_rec["name"], lp_rec["country"],
             lp_rec["latitude"], lp_rec["longitude"]]
        )

    # Make sure our CSV progress is saved to disk
    # This is crucial for our incremental approach
    csv_file.flush()

    # Also save the full cable object (with landing points) to our NDJSON file
    # NDJSON = "Newline-Delimited JSON" - one complete JSON object per line
    ndjson_file.write(json.dumps({
        "cable_key"    : key,
        "name"         : cab.get("name"),
        "rfs_year"     : cab.get("rfs_year"),
        "length_km"    : cab.get("length"),
        "owners"       : cab.get("owners"),
        "landing_points": lp_objs,
    }) + "\n")
    ndjson_file.flush()  # Save this cable to disk right away

#############################################################################
# Step 5: Wrap up - close files and show a preview
#############################################################################
csv_file.close()
ndjson_file.close()

print(f"\n✓ Streaming data collection complete!")
print(f"   CSV output   → {csv_path}")
print(f"   NDJSON output→ {ndjson_path}")

# Show a little preview of what we collected
print("\nSample lines from NDJSON (first 3 cables):")
with ndjson_path.open() as f:
    for _ in range(3):
        print(" ", f.readline().strip())