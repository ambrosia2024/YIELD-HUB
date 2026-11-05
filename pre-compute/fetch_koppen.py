#!/usr/bin/env python3
"""
Fetch Beck et al. (2023) Köppen-Geiger climate zone labels for all CyBench locations.

Unlike GAEZ ACZ (crop-specific suitability), Köppen-Geiger zones are climate-only
and crop-agnostic: a single global raster covers all crops.  Run once to produce
koppen_labels.csv; join to results on adm_id for both maize and wheat.

SOURCE
    Beck, H.E. et al. (2023) High-resolution (1 km) Köppen-Geiger maps for
    1901–2099 based on constrained CMIP6 projections. Scientific Data 10, 724.
    https://doi.org/10.1038/s41597-023-02549-6
    Data (figshare v2, Jan 2026 corrected release):
    https://figshare.com/articles/dataset/21789074

RASTER USED
    Period   : 1991–2020  (observation-based climatology; aligns with CyBench
               training data which typically runs through the early 2020s)
    File     : 1991_2020/koppen_geiger_0p00833333.tif  inside koppen_geiger_tif.zip
    Resolution: ~1 km (0.00833333°, 30 arc-second)
    Accuracy : 79–86 % (station-validated; Beck 2023 Table 2)
    Licence  : CC BY 4.0

    Why 1991–2020 over the older Beck 2018 (1986–2010) baseline?
      • Period better matches actual CyBench yield data years.
      • Derived from higher-accuracy observation-based climatologies (CHELSA v2,
        ERA5-Land, TerraClimate) versus the V1 WorldClim/GPCC mix.
      • January 2026 corrected release fixes a small calculation error in V2.
    The 30-class legend is identical to Beck 2018, so no downstream changes are
    needed in the encoding pipeline.

USAGE
    # All crops, all countries with >= 8 years of data
    python fetch_koppen.py

    # Specific crops and/or countries
    python fetch_koppen.py --crop maize wheat --countries NL DE FR

    # Validate output: write GeoJSON for QGIS spot-check against the online portal
    # https://koppen.earth/
    python fetch_koppen.py --validate

    # Resume a partial run (skips adm_ids already in koppen_labels.csv)
    python fetch_koppen.py --resume

    # Include countries with fewer than 8 years of data
    python fetch_koppen.py --no-filter

OUTPUT  (cybench/data/koppen_labels/koppen_labels.csv)
    adm_id, latitude, longitude, crop, country,
    koppen_code, koppen_label, koppen_main_group, koppen_description,
    source_file, fetch_timestamp

NOTES
    - A single CSV covers ALL crops because the zone is location-based.
      The `crop` column is kept so downstream joins are explicit.
    - The zip archive (~130 MB) is downloaded once and cached; re-runs use
      the cached extracted GeoTIFF.
    - For cross-reference with fetch_acz.py output: join both CSVs on adm_id
      to get (Koppen climate zone, GAEZ suitability class) per region.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import zipfile
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
import rasterio

# ──────────────────────────────────────────────────── CY-BENCH SETUP ─────────
# pre-compute/ → YIELD-HUB/ → AgML-CY-Bench/  (CyBench root)
CYBENCH_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CYBENCH_ROOT))

import cybench.config
from cybench.config import set_forecast_type
set_forecast_type("0-days")

# ──────────────────────────────────────────────────── PATHS ──────────────────
DATA_DIR   = CYBENCH_ROOT / "cybench" / "data"
OUTPUT_DIR = CYBENCH_ROOT / "cybench" / "data" / "koppen_labels"

# ──────────────────────────────────────────────────── KOPPEN LEGEND ──────────
# Beck et al. (2018) 30-class scheme.
# Class code → 2–3 letter symbol.
KOPPEN_LEGEND = {
    1:  "Af",   2:  "Am",   3:  "Aw",
    4:  "BWh",  5:  "BWk",
    6:  "BSh",  7:  "BSk",
    8:  "Csa",  9:  "Csb",  10: "Csc",
    11: "Cwa",  12: "Cwb",  13: "Cwc",
    14: "Cfa",  15: "Cfb",  16: "Cfc",
    17: "Dsa",  18: "Dsb",  19: "Dsc",  20: "Dsd",
    21: "Dwa",  22: "Dwb",  23: "Dwc",  24: "Dwd",
    25: "Dfa",  26: "Dfb",  27: "Dfc",  28: "Dfd",
    29: "ET",   30: "EF",
}

KOPPEN_DESCRIPTIONS = {
    "Af":  "Tropical rainforest",
    "Am":  "Tropical monsoon",
    "Aw":  "Tropical savanna (dry winter)",
    "BWh": "Hot desert",
    "BWk": "Cold desert",
    "BSh": "Hot semi-arid (steppe)",
    "BSk": "Cold semi-arid (steppe)",
    "Csa": "Mediterranean — hot dry summer",
    "Csb": "Mediterranean — warm dry summer",
    "Csc": "Mediterranean — cold dry summer",
    "Cwa": "Humid subtropical — dry winter",
    "Cwb": "Subtropical highland — dry winter",
    "Cwc": "Cold subtropical highland — dry winter",
    "Cfa": "Humid subtropical",
    "Cfb": "Oceanic (marine west coast)",
    "Cfc": "Subpolar oceanic",
    "Dsa": "Hot-summer continental — dry summer",
    "Dsb": "Warm-summer continental — dry summer",
    "Dsc": "Cool continental — dry summer",
    "Dsd": "Very cold continental — dry summer",
    "Dwa": "Monsoon humid continental — hot summer",
    "Dwb": "Monsoon humid continental — warm summer",
    "Dwc": "Monsoon subarctic",
    "Dwd": "Monsoon extremely cold subarctic",
    "Dfa": "Humid continental — hot summer",
    "Dfb": "Humid continental — warm summer",
    "Dfc": "Subarctic / boreal",
    "Dfd": "Extremely cold subarctic",
    "ET":  "Tundra",
    "EF":  "Ice cap",
}

# First character of the symbol → main climate group
KOPPEN_MAIN_GROUP = {sym: sym[0] for sym in KOPPEN_DESCRIPTIONS}

# ──────────────────────────────────────────────────── RASTER DOWNLOAD ────────
# Beck et al. (2023) — figshare v2, January 2026 corrected release.
# Article: https://figshare.com/articles/dataset/21789074
# The zip contains all periods and SSP scenarios; we extract 1991-2020 only.
FIGSHARE_ARTICLE_ID  = 21789074
RASTER_PERIOD        = "1991-2020"   # observation-based historical baseline
# Direct URL verified via figshare API 2026-11 (file ID 61012822, ~130 MB).
BECK_2023_ZIP_URL    = "https://ndownloader.figshare.com/files/61012822"
BECK_2023_ZIP_NAME   = "koppen_geiger_tif.zip"
# Path of the 1 km target raster inside the zip:  1991_2020/koppen_geiger_0p00833333.tif
_TARGET_MEMBER       = f"{RASTER_PERIOD.replace('-', '_')}/koppen_geiger_0p00833333.tif"
_CACHE_DIR           = os.path.join(tempfile.gettempdir(), "koppen_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAY = 2   # seconds (doubled on each retry)


def _get_raster_url() -> tuple:
    """
    Return (download_url, filename) for the Beck 2023 Köppen-Geiger zip archive.

    The URL points to the January 2026 corrected figshare v2 release
    (article 21789074, file ID 61012822, ~130 MB).  The zip contains rasters
    for all historical periods and SSP future scenarios; _extract_zip() will
    pull out only the 1991-2020 layer at ~1 km resolution.
    """
    return BECK_2023_ZIP_URL, BECK_2023_ZIP_NAME


def _extract_zip(zip_path: str, verbose: bool = True) -> str:
    """
    Extract the 1991-2020 ~1 km raster from the Beck 2023 zip archive.

    Primary target : 1991_2020/koppen_geiger_0p00833333.tif  (~1 km, 30 arc-sec)
    Fallback       : 1991_2020/koppen_geiger_0p1.tif          (~11 km)
    Last resort    : any .tif in the 1991_2020/ folder

    Returns the path of the extracted file that rasterio can open.
    """
    period_dir = RASTER_PERIOD.replace("-", "_")   # "1991_2020"

    candidates = [
        _TARGET_MEMBER,                                          # 1 km (preferred)
        f"{period_dir}/koppen_geiger_0p1.tif",                  # ~11 km fallback
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        available = set(zf.namelist())

        # Try candidates in order
        for member in candidates:
            if member in available:
                zf.extract(member, _CACHE_DIR)
                extracted = os.path.join(_CACHE_DIR, member)
                if verbose:
                    print(f"[info] Extracted raster: {extracted}")
                return extracted

        # Last resort: any .tif in the period folder
        for member in sorted(available):
            if member.startswith(period_dir + "/") and member.endswith(".tif"):
                zf.extract(member, _CACHE_DIR)
                extracted = os.path.join(_CACHE_DIR, member)
                if verbose:
                    print(f"[info] Extracted raster (last resort): {extracted}")
                return extracted

    raise RuntimeError(
        f"No raster for period '{RASTER_PERIOD}' found in {zip_path}. "
        f"Expected member: {_TARGET_MEMBER}"
    )


def _download_raster(verbose: bool = True) -> str:
    """
    Return a rasterio-openable path to the Beck 2023 1991-2020 Köppen raster.

    On first call: downloads koppen_geiger_tif.zip (~130 MB) and extracts
    1991_2020/koppen_geiger_0p00833333.tif into _CACHE_DIR.
    On subsequent calls: reuses the already-extracted file.

    NOTE: If you have an old Beck 2018 raster in _CACHE_DIR from a prior run,
    delete it manually (rm -r <TMPDIR>/koppen_cache/) so this function
    downloads the correct Beck 2023 file instead of reusing the stale one.
    The period tag '1991_2020' in the subdirectory path makes the new file
    unambiguous — the old Beck 2018 files use flat names like 'Beck_KG_V1_present_0p0083.tif'.
    """
    period_dir = RASTER_PERIOD.replace("-", "_")   # "1991_2020"

    # Cache hit: extracted file already present (walk subdirectories)
    for root, _dirs, files in os.walk(_CACHE_DIR):
        if period_dir in os.path.relpath(root, _CACHE_DIR).replace(os.sep, "/"):
            for fname in sorted(files):
                if fname.endswith((".tif", ".tiff")):
                    fpath = os.path.join(root, fname)
                    if verbose:
                        print(f"[info] Using cached raster: {fpath}")
                    return fpath

    # Zip may already be cached but not yet extracted
    zip_local = os.path.join(_CACHE_DIR, BECK_2023_ZIP_NAME)
    if os.path.exists(zip_local):
        if verbose:
            print(f"[info] Using cached zip: {zip_local} — extracting ...")
        return _extract_zip(zip_local, verbose=verbose)

    # Download zip
    url, filename = _get_raster_url()
    if verbose:
        print(f"[info] Downloading Beck 2023 Köppen zip: {filename} (~130 MB)")
        print(f"[info] Source: {url}")

    for attempt in range(MAX_RETRIES):
        try:
            urllib.request.urlretrieve(url, zip_local)
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[warning] Retry {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts: {e}")

    if verbose:
        print(f"[info] Saved to: {zip_local}")

    return _extract_zip(zip_local, verbose=verbose)


# ──────────────────────────────────────────────────── LOCATION READING ───────
MIN_YEARS_THRESHOLD = 8


def _load_years_dict() -> dict:
    path = CYBENCH_ROOT / "YIELD-HUB" / "setups" / "configurations" / "years_dict.json"
    with open(path) as f:
        return json.load(f)


def _filter_by_min_years(crop: str, countries: list, min_years: int = MIN_YEARS_THRESHOLD,
                          verbose: bool = True) -> list:
    years_dict = _load_years_dict()
    crop_years = years_dict.get(crop, {})
    valid   = [c for c in countries if len(crop_years.get(c, [])) >= min_years]
    skipped = set(countries) - set(valid)
    if skipped and verbose:
        print(f"  [skip] {sorted(skipped)} — fewer than {min_years} years")
    if verbose:
        print(f"  [{crop}] {len(valid)}/{len(countries)} countries pass the "
              f"{min_years}-year filter")
    return valid


def _get_country_codes(crop: str, countries: list) -> list:
    """Return country codes for a crop; auto-discovers all if countries is None/'all'."""
    if countries and "all" not in [c.lower() for c in countries]:
        return countries
    crop_dir = DATA_DIR / crop
    if not crop_dir.exists():
        raise FileNotFoundError(f"No data directory for crop '{crop}': {crop_dir}")
    return sorted(
        d.name for d in crop_dir.iterdir()
        if d.is_dir() and (d / f"location_{crop}_{d.name}.csv").exists()
    )


def _read_locations(crop: str, country: str) -> list:
    """Returns list of (adm_id, lat, lon) tuples."""
    loc_file = DATA_DIR / crop / country / f"location_{crop}_{country}.csv"
    if not loc_file.exists():
        raise FileNotFoundError(f"Location file not found: {loc_file}")
    points = []
    with open(loc_file) as f:
        for row in csv.DictReader(f):
            try:
                points.append((
                    row["adm_id"],
                    float(row["latitude"]),
                    float(row["longitude"]),
                ))
            except (KeyError, ValueError) as e:
                print(f"[warning] Skipping row: {e}", file=sys.stderr)
    return points


def _collect_all_points(crops: list, countries, apply_filter: bool,
                         verbose: bool = True) -> list:
    """
    Collect (adm_id, lat, lon, crop, country) tuples across all requested
    crops and countries.  De-duplicates (adm_id, crop, country) triples.
    """
    seen    = set()
    all_pts = []
    for crop in crops:
        codes = _get_country_codes(crop, countries)
        if apply_filter:
            codes = _filter_by_min_years(crop, codes, verbose=verbose)
        for country in codes:
            try:
                pts = _read_locations(crop, country)
            except FileNotFoundError as e:
                if verbose:
                    print(f"[warning] {e}")
                continue
            for adm_id, lat, lon in pts:
                key = (adm_id, crop, country)
                if key not in seen:
                    seen.add(key)
                    all_pts.append((adm_id, lat, lon, crop, country))
    return all_pts


# ──────────────────────────────────────────────────── RASTER SAMPLING ────────
def _decode_koppen(raw_value) -> tuple:
    """
    Decode a raw Beck 2023 pixel value to
    (code, label, main_group, description).
    The 30-class integer legend is identical to Beck 2018.
    """
    if raw_value is None:
        return None, "NoData", "NoData", "Outside raster coverage"
    code = int(raw_value)
    if code == 0 or code not in KOPPEN_LEGEND:
        return None, "NoData", "NoData", "Outside raster coverage"
    label       = KOPPEN_LEGEND[code]
    main_group  = KOPPEN_MAIN_GROUP.get(label, label[0])
    description = KOPPEN_DESCRIPTIONS.get(label, "Unknown")
    return code, label, main_group, description


def sample_koppen(points: list, raster_path: str, resume_cache: dict,
                  verbose: bool = True) -> tuple:
    """
    Sample the Koppen raster for all points.
    Returns (results, failed).
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    results, failed = [], []

    if verbose:
        print(f"[info] Sampling {len(points)} points from: {raster_path}")

    with rasterio.open(raster_path) as dataset:
        coords  = [(lon, lat) for (_, lat, lon, _, _) in points]   # rasterio: (lon, lat)
        sampled = list(dataset.sample(coords))

    for point, raw in zip(points, sampled):
        adm_id, lat, lon, crop, country = point
        cache_key = (adm_id, crop, country)
        if cache_key in resume_cache:
            results.append(resume_cache[cache_key])
            continue
        try:
            code, label, main_group, description = _decode_koppen(raw[0])
            results.append({
                "adm_id":              adm_id,
                "latitude":            lat,
                "longitude":           lon,
                "crop":                crop,
                "country":             country,
                "koppen_code":         code,
                "koppen_label":        label,
                "koppen_main_group":   main_group,
                "koppen_description":  description,
                "source_file":         raster_path,
                "fetch_timestamp":     timestamp,
            })
        except Exception as e:
            failed.append((point, str(e)))

    return results, failed


# ──────────────────────────────────────────────────── OUTPUT ─────────────────
OUTPUT_FIELDS = [
    "adm_id", "latitude", "longitude", "crop", "country",
    "koppen_code", "koppen_label", "koppen_main_group", "koppen_description",
    "source_file", "fetch_timestamp",
]


def write_csv(results: list, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[info] Wrote {len(results)} rows → {path}")


def write_geojson(results: list, path: Path):
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [r["longitude"], r["latitude"]]},
            "properties": {k: r[k] for k in (
                "adm_id", "crop", "country",
                "koppen_code", "koppen_label", "koppen_main_group",
            )},
        }
        for r in results if r.get("koppen_code") is not None
    ]
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
    print(f"[info] Wrote {len(features)} features → {path}")


# ──────────────────────────────────────────────────── MAIN ───────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch Beck et al. (2023) Köppen-Geiger labels for CyBench locations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--crop", nargs="+", default=["maize", "wheat"],
        choices=["maize", "wheat"],
        help="Crops whose location files to read (default: maize wheat). "
             "Koppen is crop-agnostic but location files are per-crop.",
    )
    parser.add_argument(
        "--countries", nargs="+", default=None,
        help="Country codes (default: all countries with >= 8 years of data). "
             "Pass 'all' to be explicit.",
    )
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip adm_ids already present in an existing koppen_labels.csv",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Also write koppen_labels.geojson for QGIS spot-checking",
    )
    parser.add_argument(
        "--no_filter", action="store_true",
        help="Include countries with fewer than 8 years of yield data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "koppen_labels.csv"

    # Resume: load already-processed rows
    resume_cache = {}
    if args.resume and csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                resume_cache[(row["adm_id"], row["crop"], row["country"])] = row
        print(f"[info] Resume: {len(resume_cache)} cached rows loaded from {csv_path}")

    # Collect points
    print(f"\n[info] Crops: {args.crop}")
    points = _collect_all_points(
        args.crop, args.countries, apply_filter=not args.no_filter
    )
    print(f"[info] Total unique (adm_id, crop, country) points: {len(points)}")
    if not points:
        print("[error] No points to process. Check data directory and country filter.")
        sys.exit(1)

    # Download / retrieve cached raster
    raster_path = _download_raster(verbose=True)

    # Sample
    results, failed = sample_koppen(points, raster_path, resume_cache)

    # Write
    write_csv(results, csv_path)
    if args.validate:
        write_geojson(results, output_dir / "koppen_labels.geojson")
        print(f"[info] VALIDATE: import koppen_labels.geojson into QGIS and cross-check "
              f"against https://koeppen-geiger.vu-wien.ac.at/present.htm")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Points processed : {len(points)}")
    print(f"Successful       : {len(results)}")
    print(f"Failed           : {len(failed)}")
    if failed:
        print(f"[warning] {len(failed)} failed points; see stderr above.")

    dist = defaultdict(int)
    for r in results:
        dist[r.get("koppen_label", "NoData")] += 1
    print(f"\nKöppen distribution across all sampled locations:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        desc = KOPPEN_DESCRIPTIONS.get(label, "")
        print(f"  {label:5s}  {count:5d}   {desc}")
    print(f"\nOutput: {csv_path}")


if __name__ == "__main__":
    main()
