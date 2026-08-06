#!/usr/bin/env python3
"""
Description: Fetch GAEZ v4 Agro-Climatic Zone (ACZ) Labels for CyBench Dataset. 
This script fetches agro-climatic zone labels from FAO's GAEZ v4 dataset for all locations in your CyBench dataset. It:

1. Reads location files (lat/lon coordinates for each adm_id)
2. Downloads the appropriate GAEZ v4 suitability rasters per crop
3. Samples each location's suitability class
4. Saves results with full provenance for reproducibility

USAGE:
# Fetch ACZ labels for ALL maize locations (auto-discovers all countries)
python fetch_acz.py --crop maize

# Fetch for specific countries only
python fetch_acz.py --crop maize --countries LS IN US

# Validate/spot-check mode (writes GeoJSON for manual verification)
python fetch_acz.py --crop maize --validate

# Resume from previous run (uses cached results)
python fetch_acz.py --crop maize --resume

ASSUMPTIONS & LIMITATIONS: 
Water source: Fixed to 'Rainfed' globally. Real-world water sources vary
by location; upgrade to MIRCA2000-based selection if needed.

Temporal baseline: 1981-2010 historical climatology. GAEZ does not
provide observed climate beyond 2010; future periods require GCM scenarios.

Spatial resolution: ~9km grid cells. Points near coasts/urban edges
may reflect averaged suitability.

Input level: High (standard high-input assumption).
Scope: All Land in Grid Cell (full-cell suitability).
CO2 fertilization: Enabled (reality-consistent).

CLASS LEGEND -- Source: FAO GAEZ v4 data catalog, "Crop suitability index in
classes, all land in grid cell" (map code RES05-SCI), matching the
DEFAULT_RENDERER used below.
https://data.apps.fao.org/catalog/iso/a6965302-55bf-46e9-8e42-6d046938333b

Code | Label              | SI Range (approximate)
-----|--------------------|------------------------
0    | 0                  | -
1    | Very high          | SI > 85
2    | High               | 70 < SI < 85
3    | Good               | 55 < SI < 70
4    | Medium             | 40 < SI < 55
5    | Moderate           | 25 < SI < 40
6    | Marginal           | 10 < SI < 25
7    | Very marginal      | 0 < SI < 10
8    | Not suitable       | -
9    | Water              | -
-9   | NoData             | Outside coverage / masked


OUTPUT
Creates:acz_labels_{crop}.csv with columns:
- adm_id: Location identifier from CyBench
- latitude, longitude: Coordinates
- crop: Crop name (maize/wheat)
- raw_class_code: GAEZ suitability class (0-9, or -9 for nodata)
- suitability_class: Human-readable label
- water_supply: Water source assumption (default: Rainfed)
- input_level: Input level assumption (High)
- baseline_period: GAEZ temporal period (1981-2010)
- source_file: URI of downloaded raster
- fetch_timestamp: When this label was fetched
- confidence: high/low based on suitability threshold

In validate mode, also creates:acz_labels_{crop}.geojson for QGIS
visualization and spot-checking against GAEZ web portal.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
import rasterio
from tqdm import tqdm

# =============================================================================
# CY-BENCH IMPORTS
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cybench.config
from cybench.config import set_forecast_type

# Set forecast type to avoid "end-of-season" error
set_forecast_type("0-days")

# =============================================================================
# CONFIGURATION
# =============================================================================

# GAEZ v4 Service Endpoints
GAEZ_BASE = "https://gaez-services.fao.org/server/rest/services"
SUITABILITY_SERVICE = f"{GAEZ_BASE}/res05/ImageServer"  # Theme 4: Suitability

# Fixed parameters (documented in ASSUMPTIONS above)
DEFAULT_WATER_SUPPLY = "Rainfed"
DEFAULT_INPUT_LEVEL = "High"
DEFAULT_TIME_PERIOD = "1981-2010"
DEFAULT_RENDERER = "Crop Suitability Index in Classes - All Land in Grid Cell"

# GAEZ v4 Suitability Class Legend
# Source: FAO GAEZ v4 data catalog (map code RES05-SCI, "all land in grid cell"),
# https://data.apps.fao.org/catalog/iso/a6965302-55bf-46e9-8e42-6d046938333b
# This matches DEFAULT_RENDERER above.
SUITABILITY_CLASS_LEGEND = {
    0: "0",
    1: "Very high (SI > 85)",
    2: "High (70 < SI < 85)",
    3: "Good (55 < SI < 70)",
    4: "Medium (40 < SI < 55)",
    5: "Moderate (25 < SI < 40)",
    6: "Marginal (10 < SI < 25)",
    7: "Very marginal (0 < SI < 10)",
    8: "Not suitable",
    9: "Water",
}
SUITABILITY_CLASS_NODATA = -9

# Threshold for "low confidence" flag (marginal/very marginal areas)
LOW_CONFIDENCE_THRESHOLD = 20  # SI <= 20 is flagged as low confidence

# Local cache for downloaded rasters
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "gaez_v4_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


# =============================================================================
# DATA PATHS
# =============================================================================

def get_cybench_root():
    """Find the CyBench root directory (parent of cybench/)."""
    current = Path(__file__).resolve().parent
    while current.name != "cybench" and current.parent != current:
        current = current.parent
    if current.name == "cybench":
        return current.parent
    raise RuntimeError("Could not find CyBench root directory")


CYBENCH_ROOT = get_cybench_root()
DATA_DIR = CYBENCH_ROOT / "cybench" / "data"
OUTPUT_DIR = CYBENCH_ROOT / "cybench" / "data"/ "acz_labels"


# =============================================================================
# LOCATION FILE READING
# =============================================================================

MIN_YEARS_THRESHOLD = 8  # Minimum years of data required for a country


def filter_countries_by_min_years(crop: str, countries: list = None, min_years: int = MIN_YEARS_THRESHOLD) -> list:
    """
    Filter countries to only those with at least min_years of data
    using the years_dict.json configuration.

    Args:
        crop: Crop name (maize/wheat)
        countries: List of country codes, or None for all
        min_years: Minimum years of data required (default: 8)

    Returns:
        List of country codes that meet the minimum years threshold
    """
    import json

    country_codes = get_country_codes(crop, countries)
    valid_countries = []

    # Load years_dict.json
    years_dict_path = CYBENCH_ROOT / 'cybench' / 'setups' / 'configurations' / 'years_dict.json'
    with open(years_dict_path, 'r') as f:
        years_dict = json.load(f)

    crop_years = years_dict.get(crop, {})

    print(f"\n[info] Filtering countries with at least {min_years} years of data...")

    for country in country_codes:
        if country not in crop_years:
            print(f"  [skip] {country}: Not in years_dict.json")
            continue

        num_years = len(crop_years[country])

        if num_years >= min_years:
            valid_countries.append(country)
            print(f"  [ok] {country}: {num_years} years")
        else:
            print(f"  [skip] {country}: {num_years} years (< {min_years})")

    print(f"[info] {len(valid_countries)}/{len(country_codes)} countries meet the {min_years}-year threshold")
    return valid_countries

def get_country_codes(crop: str, countries: list = None):
    """Get list of country codes for a crop, optionally filtered.

    Args:
        crop: Crop name (maize/wheat)
        countries: List of country codes, or ["all"] to process all countries

    Returns:
        List of country codes to process
    """
    # Handle explicit "all" request
    if countries and "all" in [c.lower() for c in countries]:
        # Fall through to auto-discovery below
        pass
    elif countries:
        return countries

    # Auto-discover all available countries for this crop
    crop_dir = DATA_DIR / crop
    if not crop_dir.exists():
        raise FileNotFoundError(f"No data directory found for crop '{crop}': {crop_dir}")

    country_codes = [d.name for d in crop_dir.iterdir() if d.is_dir() and (d / f"location_{crop}_{d.name}.csv").exists()]
    if not country_codes:
        raise FileNotFoundError(f"No location files found for crop '{crop}' in {crop_dir}")

    return sorted(country_codes)


def read_location_file(crop: str, country_code: str) -> list:
    """
    Read a location CSV file and extract (adm_id, lat, lon) tuples.

    Expected format: crop_name,adm_id,latitude,longitude,region_area,...

    Args:
        crop: Crop name
        country_code: Country code

    Returns:
        list of (adm_id, lat, lon) tuples
    """
    location_file = DATA_DIR / crop / country_code / f"location_{crop}_{country_code}.csv"

    if not location_file.exists():
        raise FileNotFoundError(f"Location file not found: {location_file}")

    points = []
    with open(location_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                adm_id = row['adm_id']
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                points.append((adm_id, lat, lon))
            except (ValueError, KeyError) as e:
                print(f"[warning] Skipping row due to error: {e}", file=sys.stderr)

    print(f"[info] Read {len(points)} locations from {location_file.name}")
    return points


def read_all_locations(crop: str, countries: list = None, apply_filter: bool = True) -> list:
    """
    Read all location files for a crop.

    Args:
        crop: Crop name (maize/wheat)
        countries: List of country codes, or ["all"] for all available
        apply_filter: Whether to filter countries by minimum years (default: True)

    Returns:
        list: [(adm_id, lat, lon, crop, country_code), ...]
    """
    # First, get country codes (handles "all" case)
    country_codes = get_country_codes(crop, countries)

    # Then apply the minimum years filter
    if apply_filter:
        country_codes = filter_countries_by_min_years(crop, country_codes)
        if not country_codes:
            raise ValueError(f"No countries meet the minimum {MIN_YEARS_THRESHOLD}-year threshold")

    all_points = []
    for country_code in country_codes:
        points = read_location_file(crop, country_code)
        for adm_id, lat, lon in points:
            all_points.append((adm_id, lat, lon, crop, country_code))

    return all_points


# =============================================================================
# GAEZ v4 API INTERACTION
# =============================================================================

def discover_crop_options(crop: str, service_url: str = SUITABILITY_SERVICE):
    """
    Discover available water_supply/input_level options for a crop.
    Useful for debugging or changing defaults.
    """
    params = {
        "where": f"UPPER(crop) = UPPER('{crop}')",
        "outFields": "crop,water_supply,input_level,units,renderer",
        "returnGeometry": "false",
        "returnDistinctValues": "true",
        "f": "json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(f"{service_url}/query", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(f"GAEZ query error: {data['error']}")
            return [f["attributes"] for f in data.get("features", [])]
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[warning] Retry {attempt + 1}/{MAX_RETRIES} for discover_crop_options: {e}")
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise


def find_crop_raster(
    crop,
    water_supply=DEFAULT_WATER_SUPPLY,
    input_level=DEFAULT_INPUT_LEVEL,
    renderer=DEFAULT_RENDERER,
    time_period=DEFAULT_TIME_PERIOD,
    service_url=SUITABILITY_SERVICE,
    units="Class",  # "Class" for categorical codes, "Index" for continuous SI values
    verbose=True,
):
    """
    Find the GAEZ v4 raster for a crop with specified parameters.

    Args:
        units: "Class" for categorical codes (0-10), "Index" for continuous SI values

    Returns:
        dict: Raster attributes including download_url, year, etc.
    """
    # Build query filter - case-insensitive matching
    where = (
        f"UPPER(crop) = UPPER('{crop}') "
        f"AND UPPER(water_supply) = UPPER('{water_supply}') "
        f"AND UPPER(input_level) = UPPER('{input_level}') "
        f"AND UPPER(units) = UPPER('{units}') "
        f"AND renderer = '{renderer}' "
        f"AND UPPER(year) = UPPER('{time_period}') "
        f"AND UPPER(rcp) = UPPER('Historical')"
    )

    # First, get object IDs matching our criteria
    id_params = {"where": where, "returnIdsOnly": "true", "f": "json"}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(f"{service_url}/query", params=id_params, timeout=30)
            r.raise_for_status()
            id_data = r.json()
            break
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[warning] Retry {attempt + 1}/{MAX_RETRIES} for raster ID query: {e}")
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise

    if "error" in id_data:
        raise RuntimeError(f"GAEZ query error: {id_data['error']}")

    object_ids = id_data.get("objectIds") or []
    if not object_ids:
        raise ValueError(
            f"No matching raster for crop='{crop}', water_supply='{water_supply}', "
            f"input_level='{input_level}', year='{time_period}'.\n"
            f"Try running discover_crop_options('{crop}') to see valid combinations."
        )

    # Handle ambiguous matches (usually CO2 fertilization variants)
    matched_id = object_ids[0]
    if len(object_ids) > 1 and verbose:
        print(
            f"[warning] {len(object_ids)} rasters matched for crop={crop}, "
            f"water_supply={water_supply}, input_level={input_level}. Using first match (ID={matched_id})."
        )

    # Fetch full attributes for the selected raster
    attr_params = {
        "objectIds": str(matched_id),
        "outFields": "*",
        "returnGeometry": "false",
        "f": "json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(f"{service_url}/query", params=attr_params, timeout=30)
            r.raise_for_status()
            features = r.json().get("features", [])
            break
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[warning] Retry {attempt + 1}/{MAX_RETRIES} for raster attributes: {e}")
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise

    if not features:
        raise RuntimeError(f"Could not fetch attributes for objectid={matched_id}")

    attrs = features[0]["attributes"]
    if verbose:
        print(f"[info] Found raster: {attrs.get('Name', 'unknown')} (period: {attrs.get('year', 'unknown')})")

    return attrs


def download_raster(download_url: str, verbose=True) -> str:
    """
    Download a GAEZ raster to local cache and return the local path.
    """
    filename = download_url.rstrip("/").split("/")[-1]
    local_path = os.path.join(_CACHE_DIR, filename)

    if os.path.exists(local_path):
        if verbose:
            print(f"[info] Using cached raster: {filename}")
        return local_path

    if verbose:
        print(f"[info] Downloading raster: {filename}")

    for attempt in range(MAX_RETRIES):
        try:
            urllib.request.urlretrieve(download_url, local_path)
            if verbose:
                print(f"[info] Downloaded to: {local_path}")
            return local_path
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[warning] Retry {attempt + 1}/{MAX_RETRIES} for download: {e}")
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise


def decode_suitability(raw_value):
    """
    Decode a raw GAEZ suitability value to (code, label, confidence).

    Returns:
        tuple: (code, label, confidence) where confidence is "high" or "low"
    """
    if raw_value is None or raw_value == SUITABILITY_CLASS_NODATA:
        return None, "NoData (outside raster coverage / masked)", "unknown"

    label = SUITABILITY_CLASS_LEGEND.get(raw_value, f"unrecognized class code {raw_value}")

    # Determine confidence based on suitability threshold
    # Classes 5-7 (SI <= 25) are considered low confidence
    if raw_value in [5, 6, 7]:
        confidence = "low"
    elif raw_value == 8:  # Not suitable
        confidence = "low"
    elif raw_value in [9, 0]:  # Water, undefined
        confidence = "low"
    else:  # Classes 1-4 (SI > 40)
        confidence = "high"

    return raw_value, label, confidence


# BATCH PROCESSING
def fetch_acz_labels_batch(
    points,  # list of (adm_id, lat, lon, crop, country) tuples
    water_supply=DEFAULT_WATER_SUPPLY,
    input_level=DEFAULT_INPUT_LEVEL,
    time_period=DEFAULT_TIME_PERIOD,
    resume_from=None,
    verbose=True,
) -> tuple:
    """
    Fetch ACZ labels for a batch of points.

    Args:
        points: List of (adm_id, lat, lon, crop, country) tuples
        water_supply: Water source (default: Rainfed)
        input_level: Input level (default: High)
        time_period: Time period (default: 1981-2010)
        resume_from: Path to resume file (JSONL with cached results)
        verbose: Print progress info

    Returns:
        tuple: (results, failed) where:
            - results: List of successful result dicts
            - failed: List of (point, error) tuples
    """
    # Group points by crop for efficient raster usage
    points_by_crop = defaultdict(list)
    for idx, point in enumerate(points):
        adm_id, lat, lon, crop, country = point
        points_by_crop[crop].append((idx, adm_id, lat, lon, country))

    # Load resume cache if provided
    resume_cache = {}
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, 'r') as f:
            for line in f:
                try:
                    cached = json.loads(line.strip())
                    # For backward compatibility, handle both old and new cache formats
                    if 'country' in cached:
                        key = (cached['adm_id'], cached['latitude'], cached['longitude'], cached['crop'], cached['country'])
                    else:
                        key = (cached['adm_id'], cached['latitude'], cached['longitude'], cached['crop'], 'unknown')
                    resume_cache[key] = cached
                except (json.JSONDecodeError, KeyError):
                    continue
        if verbose:
            print(f"[info] Loaded {len(resume_cache)} cached results from {resume_from}")

    results = []
    failed = []

    for crop, crop_points in points_by_crop.items():
        if verbose:
            print(f"\n[info] Processing {len(crop_points)} points for crop: {crop}")

        try:
            # Find and download the class raster
            raster_attrs = find_crop_raster(
                crop, water_supply, input_level,
                time_period=time_period, units="Class", verbose=verbose
            )
            local_path = download_raster(raster_attrs["download_url"], verbose=verbose)

            # Sample all points for this crop
            coords = [(lon, lat) for _, _, lat, lon, _ in crop_points]  # rasterio expects (lon, lat)

            with rasterio.open(local_path) as dataset:
                sampled = list(dataset.sample(coords))

            # Process samples
            for (idx, adm_id, lat, lon, country), raw in zip(crop_points, sampled):
                point_key = (adm_id, lat, lon, crop, country)

                # Check if already in resume cache
                if point_key in resume_cache:
                    results.append(resume_cache[point_key])
                    continue

                raw_value = int(raw[0])
                code, label, confidence = decode_suitability(raw_value)

                result = {
                    "adm_id": adm_id,
                    "latitude": lat,
                    "longitude": lon,
                    "crop": crop,
                    "country": country,
                    "raw_class_code": code,
                    "latitude": lat,
                    "longitude": lon,
                    "crop": crop,
                    "raw_class_code": code,
                    "suitability_class": label,
                    "confidence": confidence,
                    "water_supply": water_supply,
                    "input_level": input_level,
                    "baseline_period": time_period,
                    "source_file": raster_attrs["download_url"],
                    "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
                }
                results.append(result)

        except Exception as e:
            if verbose:
                print(f"[error] Failed to process crop {crop}: {e}", file=sys.stderr)
            for idx, adm_id, lat, lon, country in crop_points:
                failed.append(((adm_id, lat, lon, crop, country), str(e)))

    return results, failed


# OUTPUT WRITING
def write_csv(results, output_path):
    """Write results to CSV format."""
    if not results:
        print("[warning] No results to write", file=sys.stderr)
        return

    fieldnames = [
        "adm_id", "latitude", "longitude", "crop", "country",
        "raw_class_code", "suitability_class", "confidence",
        "water_supply", "input_level", "baseline_period",
        "source_file", "fetch_timestamp"
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[info] Wrote {len(results)} results to {output_path}")


def write_jsonl(results, output_path):
    """Write results to JSONL format (one JSON per line, appendable for resume)."""
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"[info] Wrote {len(results)} results to {output_path}")


def write_geojson(results, output_path):
    """
    Write results to GeoJSON format for QGIS visualization and validation.
    Useful for spot-checking against GAEZ web portal.
    """
    features = []
    for r in results:
        if r['raw_class_code'] is None:
            continue

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r['longitude'], r['latitude']],
            },
            "properties": {
                "adm_id": r['adm_id'],
                "crop": r['crop'],
                "raw_class_code": r['raw_class_code'],
                "suitability_class": r['suitability_class'],
                "confidence": r['confidence'],
                "water_supply": r['water_supply'],
                "baseline_period": r['baseline_period'],
                "fetch_timestamp": r['fetch_timestamp'],
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "name": f"ACZ Labels for Validation",
        "features": features,
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"[info] Wrote {len(features)} features to {output_path} (for QGIS validation)")


def write_failed(failed, output_path):
    """Write failed points to JSONL for debugging."""
    with open(output_path, 'w') as f:
        for (point, error) in failed:
            # Handle both old (4-element) and new (5-element) point formats
            if len(point) == 5:
                adm_id, lat, lon, crop, country = point
            else:
                adm_id, lat, lon, crop = point[:4]
                country = 'unknown'
            f.write(json.dumps({
                "adm_id": adm_id,
                "latitude": lat,
                "longitude": lon,
                "crop": crop,
                "country": country,
                "error": error,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }) + '\n')

    if failed:
        print(f"[info] Wrote {len(failed)} failed points to {output_path}")


# MAIN block
def main():
    parser = argparse.ArgumentParser(
        description="Fetch GAEZ v4 Agro-Climatic Zone labels for CyBench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--crop", required=True, choices=["maize", "wheat"],
                        help="Crop to process")
    parser.add_argument("--countries", nargs="+",
                        help="Country codes to process (default: all available). Pass 'all' to be explicit.")
    parser.add_argument("--water-supply", default=DEFAULT_WATER_SUPPLY,
                        help=f"Water source (default: {DEFAULT_WATER_SUPPLY})")
    parser.add_argument("--input-level", default=DEFAULT_INPUT_LEVEL,
                        help=f"Input level (default: {DEFAULT_INPUT_LEVEL})")
    parser.add_argument("--time-period", default=DEFAULT_TIME_PERIOD,
                        help=f"GAEZ time period (default: {DEFAULT_TIME_PERIOD})")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run (uses existing JSONL cache)")
    parser.add_argument("--validate", action="store_true",
                        help="Validation mode: write GeoJSON for QGIS spot-checking")
    parser.add_argument("--discover", action="store_true",
                        help="Discover available GAEZ options for the crop (debugging)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed progress (default: True)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip the minimum 8-year data filter (include all countries)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover mode: just show available options
    if args.discover:
        print(f"[info] Discovering GAEZ options for crop: {args.crop}")
        options = discover_crop_options(args.crop)
        print("\nAvailable combinations:")
        for opt in options:
            print(f"  - water_supply: {opt['water_supply']}, input_level: {opt['input_level']}, "
                  f"units: {opt['units']}, renderer: {opt.get('renderer', 'N/A')}")
        return

    # Read all locations for the crop
    print(f"\n[info] Reading locations for crop: {args.crop}")
    if args.countries:
        print(f"[info] Countries: {', '.join(args.countries)}")
    else:
        print(f"[info] Processing ALL available countries for this crop")

    points = read_all_locations(
        args.crop,
        args.countries,
        apply_filter=not args.no_filter
    )

    print(f"[info] Total points to process: {len(points)}")

    # Set up resume file
    resume_file = output_dir / f"acz_cache_{args.crop}.jsonl"
    if args.resume:
        print(f"[info] Resume mode: will use cached results from {resume_file}")

    # Fetch labels
    results, failed = fetch_acz_labels_batch(
        points,
        water_supply=args.water_supply,
        input_level=args.input_level,
        time_period=args.time_period,
        resume_from=str(resume_file) if args.resume else None,
        verbose=args.verbose,
    )

    # Write outputs
    csv_path = output_dir / f"acz_labels_{args.crop}.csv"
    jsonl_path = output_dir / f"acz_cache_{args.crop}.jsonl"
    failed_path = output_dir / f"acz_failed_{args.crop}.jsonl"

    write_csv(results, csv_path)
    write_jsonl(results, jsonl_path)
    write_failed(failed, failed_path)

    if args.validate:
        geojson_path = output_dir / f"acz_labels_{args.crop}.geojson"
        write_geojson(results, geojson_path)
        print(f"\n[info] VALIDATION MODE: Import {geojson_path} into QGIS to spot-check against GAEZ web portal")
        print(f"[info] GAEZ Web Portal: https://gaez.fao.org/")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total points processed: {len(points)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\n[warning] Some points failed. See {failed_path} for details.")
        print(f"[info] You can retry with --resume to skip successful points.")

    # Print class distribution
    class_counts = defaultdict(int)
    for r in results:
        class_counts[r['suitability_class']] += 1

    print(f"\nClass Distribution:")
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    print(f"\nOutputs:")
    print(f"  - {csv_path} (CSV for analysis)")
    print(f"  - {jsonl_path} (JSONL cache for resume)")
    if failed:
        print(f"  - {failed_path} (Failed points for debugging)")
    if args.validate:
        print(f"  - {geojson_path} (GeoJSON for QGIS validation)")


if __name__ == "__main__":
    main()