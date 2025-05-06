#!/usr/bin/env python3
"""
scrape_submarine_cables_streaming.py
------------------------------------
Same logic as before, but *saves as it goes*:

• CSV grows row-by-row.
• NDJSON grows cable-by-cable.
• Safe to ^C part-way—data gathered so far is already persisted.
"""

import json, csv, time, re, os
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm

BASE = "https://www.submarinecablemap.com/api/v3"
UA   = "network-sim/1.1 (contact: you@example.com)"
OUT  = Path("output_cable_map");  OUT.mkdir(exist_ok=True)

###############################################################################
# Helpers
###############################################################################
def fetch_json(url: str, tries: int = 3, timeout: int = 30) -> dict:
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if k == tries - 1:
                raise
            wait = 2 ** k
            print(f"[warn] {exc} — retrying in {wait}s")
            time.sleep(wait)

def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", (text or "").lower())
    return re.sub(r"[-\s]+", "-", text).strip("-_") or "unknown"

###############################################################################
# 1. Landing-point lookup  (lat/lon by id)
###############################################################################
print("Downloading landing-point GeoJSON …")
lp_geo = fetch_json(f"{BASE}/landing-point/landing-point-geo.json")

landing_by_id: Dict[str, dict] = {}
for feat in lp_geo["features"]:
    props = feat.get("properties", {})
    lp_id = str(
        props.get("id") or feat.get("id") or props.get("fid") or props.get("objectid")
    )
    if not lp_id:
        continue
    landing_by_id[lp_id] = {
        "name": props.get("name"),
        "country": props.get("country_name") or props.get("country"),
        "latitude": feat["geometry"]["coordinates"][1],
        "longitude": feat["geometry"]["coordinates"][0],
    }

###############################################################################
# 2. Cable ID list
###############################################################################
print("Downloading cable-geo index …")
cable_geo = fetch_json(f"{BASE}/cable/cable-geo.json")
cable_ids: List[str] = sorted(
    {str(f["properties"].get("id")) for f in cable_geo["features"] if f["properties"].get("id")}
)

###############################################################################
# 3. Prepare output sinks  (append mode)
###############################################################################
csv_path  = OUT / "cables_with_landing_points.csv"
ndjson_path = OUT / "cables_with_landing_points.ndjson"

# write CSV header once if file absent / empty
write_header = not csv_path.exists() or csv_path.stat().st_size == 0
csv_file = csv_path.open("a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(
        ["cable_key", "cable_name", "rfs_year", "length_km",
         "lp_id", "lp_name", "country", "latitude", "longitude"]
    )
    csv_file.flush()

ndjson_file = ndjson_path.open("a", encoding="utf-8")

###############################################################################
# 4. Stream processing
###############################################################################
for cab_id in tqdm(cable_ids, desc="Fetching cables", unit="cable"):
    try:
        cab = fetch_json(f"{BASE}/cable/{cab_id}.json")
    except Exception as e:
        print(f"[error] skipping cable id={cab_id}  ({e})")
        continue

    time.sleep(0.1)          # gentle on the server

    key = cab.get("slug") or f"id-{cab_id}"

    # enrich landing points
    lp_objs = []
    for lp in cab.get("landing_points", []):
        lp_id = str(lp.get("id") or lp)
        geo   = landing_by_id.get(lp_id, {})
        lp_rec = {
            "id"       : lp_id,
            "name"     : lp.get("name")    or geo.get("name"),
            "country"  : lp.get("country") or geo.get("country"),
            "latitude" : geo.get("latitude"),
            "longitude": geo.get("longitude"),
        }
        lp_objs.append(lp_rec)

        # CSV row per cable–landing-point pair
        csv_writer.writerow(
            [key, cab.get("name"), cab.get("rfs_year"), cab.get("length"),
             lp_rec["id"], lp_rec["name"], lp_rec["country"],
             lp_rec["latitude"], lp_rec["longitude"]]
        )

    csv_file.flush()   # ensure progress is on disk

    # Append this cable’s JSON (compact) to NDJSON
    ndjson_file.write(json.dumps({
        "cable_key"    : key,
        "name"         : cab.get("name"),
        "rfs_year"     : cab.get("rfs_year"),
        "length_km"    : cab.get("length"),
        "owners"       : cab.get("owners"),
        "landing_points": lp_objs,
    }) + "\n")
    ndjson_file.flush()

###############################################################################
# 5. Close files cleanly
###############################################################################
csv_file.close()
ndjson_file.close()

print(f"\n✓ streaming complete")
print(f"   CSV   → {csv_path}")
print(f"   NDJSON→ {ndjson_path}")

# preview first few cables (optional)
print("\nSample lines from NDJSON:")
with ndjson_path.open() as f:
    for _ in range(3):
        print(" ", f.readline().strip())
