#!/usr/bin/env python3
"""
fetch_gaez_predictors.py

Fetch all available GAEZ v5 static predictors for CyBench crop locations and
compile them into a wide-format predictor table (one row per adm_id) per crop.

Concept categories covered (ICON framework):
  C3  Agro-climatic potential: suitability class/index, attainable yield,
      growing-period characteristics, AEZ / Koppen classification
  C4  Yield-limiting constraints: thermal / moisture / agro-climatic factors,
      water deficit during crop cycle, heat-stress days, frost frequency
  C5  Soil quality: 8 soil quality criteria (SQ0–SQ7, low & high input)
  C6  Land-use context: multi-cropping potential (rain-fed / irrigated)
  C7  Technology / management gap: yield achievement ratio, actual 2020 yield

GAEZ v5 API:
  GISMGR  https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/GAEZ-V5
  GCS     gs://fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/

Output:
  cybench/data/gaez_v5_predictors/gaez_v5_{crop}.csv   (one per crop)

Usage:
  python fetch_gaez_predictors.py --crop maize
  python fetch_gaez_predictors.py --crop maize wheat
  python fetch_gaez_predictors.py --crop maize --periods 2001-2020
  python fetch_gaez_predictors.py --crop maize --resume
  python fetch_gaez_predictors.py --crop maize --no_filter
  python fetch_gaez_predictors.py --list_predictors

Notes:
  • Rasters are cached under .gaez_v5_raster_cache/ (default).
    Large 1-km TIFs (AEZ33, AEZ57) can be ~500 MB each; allocate ~10 GB cache.
  • Band 1 is sampled for all rasters.  Multi-band rasters (SQX) produce
    separate columns per band.
  • GAEZ v5 uses different crop codes for 2020-actual mapsets (RES06/07):
    wheat → WHE (not WHT).  This is handled automatically.
  • Run precompute.sh (or a SLURM job) to avoid long runtimes on the login node.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Third party ───────────────────────────────────────────────────────────────
import requests
import rasterio
from tqdm import tqdm

# ── CyBench ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import cybench.config  # noqa: F401  (registers paths)
from cybench.config import set_forecast_type

set_forecast_type("0-days")  # avoids "Unrecognized lead time 'end-of-season'" error

# =============================================================================
# CONSTANTS
# =============================================================================

GISMGR_BASE = "https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/GAEZ-V5"
GCS_MAPSET = "https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET"
GCS_MAP    = "https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAP"
GAEZ_VERSION = "v5"

CYBENCH_ROOT    = Path(__file__).resolve().parent.parent.parent
DATA_DIR        = CYBENCH_ROOT / "cybench" / "data"
OUTPUT_DIR      = DATA_DIR / "gaez_v5_predictors"
DEFAULT_CACHE   = Path(__file__).resolve().parent / ".gaez_v5_raster_cache"

MIN_YEARS_THRESHOLD = 8
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Crop codes used in standard GAEZ v5 mapsets (RES02-*, RES05-*)
CROP_CODES = {
    "maize": "MZE",
    "wheat": "WHT",
}

# Crop codes used in 2020-actual mapsets (RES06-*, RES07-*) differ for wheat
CROP_CODES_ACTUALS = {
    "maize": "MZE",
    "wheat": "WHE",
}

# Period dimension member codes
PERIOD_CODES  = {"2001-2020": "HP0120", "1981-2000": "HP8100"}
PERIOD_LABELS = {"HP0120": "2001_2020",  "HP8100": "1981_2000"}

# Water-supply / input-management codes (crop-specific mapsets)
# Standard mapsets use HRLM/LRLM; 2020 actual mapsets use WSR/WST
WSIM_LABELS = {
    "HRLM": "rf_hi",   # rain-fed, high input
    "LRLM": "rf_lo",   # rain-fed, low input
    "HILM": "ir_hi",   # irrigated, high input
    "WSR":  "rf",      # rain-fed  (RES06/07 actual-yield mapsets)
    "WST":  "total",   # all water sources combined (RES06/07)
}

# =============================================================================
# PREDICTOR CATALOG
# =============================================================================
# Each entry defines one variable (or set of variables for multi-band) to fetch.
#
# Fields
# ------
# col_prefix     Base output column name; suffixes _PERIOD and optionally _WSIM
#                are appended at runtime.
# mapset         GAEZ-V5 mapset code.  Prefix "MAP:" signals a static map
#                fetched via the /maps/{CODE}/raster endpoint.
# mapset_fb      Fallback mapset when primary has no raster for this crop.
# crop_specific  True when the raster carries a CROP dimension.
# wsim           WSIM preference list (ordered); None = no WSIM dimension.
# periods        PERIOD_CODES keys to fetch ([] = static / no period dimension).
# crop_code_key  Which CROP_CODES dict to use ('standard' or 'actuals').
# concept        Concept category label.
# description    Human-readable description.

PREDICTOR_CATALOG = [
    # ── C3: Suitability ──────────────────────────────────────────────────────
    {
        "col_prefix":    "suit_class",
        "mapset":        "RES05-SIX30AS",
        "mapset_fb":     "RES05-SIX",
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Crop suitability, class index  (1=Very High → 8=Not Suitable, 9=No cropland / Water)",
    },
    {
        "col_prefix":    "suit_index",
        "mapset":        "RES05-SXX30AS",
        "mapset_fb":     "RES05-SXX",
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Crop suitability, continuous index (0–100)",
    },
    {
        "col_prefix":    "suit_class_cropland",
        "mapset":        "RES05-SCX30AS",
        "mapset_fb":     "RES05-SCX",
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Crop suitability class on current cropland",
    },
    {
        "col_prefix":    "suit_index_cropland",
        "mapset":        "RES05-SUX",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Crop suitability continuous index on current cropland",
    },
    # ── C3: Attainable yield ─────────────────────────────────────────────────
    {
        "col_prefix":    "attainable_yield",
        "mapset":        "RES05-YLX30AS",
        "mapset_fb":     "RES05-YLX",
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Agro-ecological attainable yield (kg dry-weight / ha)",
    },
    {
        "col_prefix":    "attainable_yield_cropland",
        "mapset":        "RES05-YCX30AS",
        "mapset_fb":     "RES05-YCX",
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Agro-ecological attainable yield on current cropland (kg/ha)",
    },
    {
        "col_prefix":    "potential_yield",
        "mapset":        "RES02-YLD",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Agro-climatic potential yield (kg/ha) – before soil/terrain reduction",
    },
    # ── C3: Crop growth cycle ────────────────────────────────────────────────
    {
        "col_prefix":    "growth_start_doy",
        "mapset":        "RES02-CBD",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Beginning of crop growth cycle (day of year, 1–365)",
    },
    {
        "col_prefix":    "growth_cycle_days",
        "mapset":        "RES02-CYL",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Length of crop growth cycle (days)",
    },
    {
        "col_prefix":    "temp_sum_cycle",
        "mapset":        "RES02-TSC",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Crop-specific accumulated temperature during growth cycle (degree-days)",
    },
    {
        "col_prefix":    "crop_et_actual",
        "mapset":        "RES02-ETA",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Crop-specific actual evapotranspiration (mm)",
    },
    # ── C4: Yield-limiting constraints ────────────────────────────────────────
    {
        "col_prefix":    "constraint_total",
        "mapset":        "RES02-FC0",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Total climate-related yield constraint factor (0–1; 0 = unconstrained)",
    },
    {
        "col_prefix":    "constraint_thermal",
        "mapset":        "RES02-FC1",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Crop yield constraint factor – thermal",
    },
    {
        "col_prefix":    "constraint_moisture",
        "mapset":        "RES02-FC2",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Crop yield constraint factor – moisture",
    },
    {
        "col_prefix":    "constraint_agroclimatic",
        "mapset":        "RES02-FC3",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Crop yield constraint factor – agro-climatic",
    },
    {
        "col_prefix":    "water_deficit_cycle",
        "mapset":        "RES05-WDL",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Water deficit / net irrigation requirement during crop cycle (mm)",
    },
    {
        "col_prefix":    "crop_water_deficit",
        "mapset":        "RES02-WDE",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["HRLM", "LRLM"],
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Crop water deficit (mm)",
    },
    # ── C7: Technology / management gap (2020 actuals) ───────────────────────
    {
        "col_prefix":    "yield_achievement_ratio",
        "mapset":        "RES07-YGA-CL",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["WSR", "WST"],   # rainfed preferred
        "periods":       [],               # 2020 actual – no period dimension
        "crop_code_key": "actuals",
        "concept":       "C7",
        "description":   "Yield achievement ratio: actual / attainable yield (2020, 0–1)",
    },
    {
        "col_prefix":    "actual_yield_2020",
        "mapset":        "RES06-YLD",
        "mapset_fb":     None,
        "crop_specific": True,
        "wsim":          ["WSR", "WST"],
        "periods":       [],
        "crop_code_key": "actuals",
        "concept":       "C7",
        "description":   "Downscaled actual crop yield (kg/ha, 2020)",
    },
    # ── C4: Heat stress and frost (crop-agnostic) ─────────────────────────────
    {
        "col_prefix":    "heat_days_35c",
        "mapset":        "RES01-X35",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Days with Tmax > 35 °C (20-year average)",
    },
    {
        "col_prefix":    "heat_days_40c",
        "mapset":        "RES01-X40",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Days with Tmax > 40 °C (20-year average)",
    },
    {
        "col_prefix":    "heat_days_45c",
        "mapset":        "RES01-X45",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Days with Tmax > 45 °C",
    },
    {
        "col_prefix":    "frost_days",
        "mapset":        "RES01-N00",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Days with Tmin < 0 °C",
    },
    {
        "col_prefix":    "frost_number",
        "mapset":        "RES01-FST",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Air frost number (higher = more frost risk)",
    },
    {
        "col_prefix":    "consec_dry_days",
        "mapset":        "RES01-NDD",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C4",
        "description":   "Longest consecutive-dry-day period (days)",
    },
    # ── C3: Growing period characteristics ───────────────────────────────────
    {
        "col_prefix":    "lgp_class",
        "mapset":        "RES01-LGP",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Total growing-period-days class (1–7)",
    },
    {
        "col_prefix":    "lgp_days",
        "mapset":        "RES01-LGD",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Total growing-period days",
    },
    {
        "col_prefix":    "lgp_start_doy",
        "mapset":        "RES01-LGB",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Beginning date of longest LGP component (day of year)",
    },
    {
        "col_prefix":    "lgp_longest_len",
        "mapset":        "RES01-LD1",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Length of longest LGP component (days)",
    },
    {
        "col_prefix":    "temp_sum_gt5c",
        "mapset":        "RES01-TS2",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Annual temperature sum, days Ta > 5 °C (degree-days)",
    },
    {
        "col_prefix":    "temp_sum_gt10c",
        "mapset":        "RES01-TS3",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Annual temperature sum, days Ta > 10 °C (degree-days)",
    },
    {
        "col_prefix":    "growing_period_gt5c",
        "mapset":        "RES01-LT2",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Temperature growing period: days with Ta > 5 °C",
    },
    {
        "col_prefix":    "growing_period_gt10c",
        "mapset":        "RES01-LT3",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Temperature growing period: days with Ta > 10 °C",
    },
    {
        "col_prefix":    "npp_rainfed",
        "mapset":        "RES01-NP2",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Net primary production, rain-fed (kg DM / ha)",
    },
    # ── C3: Climate indices ───────────────────────────────────────────────────
    {
        "col_prefix":    "ppet_annual",
        "mapset":        "RES01-RID",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Annual P/PET ratio",
    },
    {
        "col_prefix":    "precip_annual_mm",
        "mapset":        "RES01-PRC",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Annual precipitation (mm)",
    },
    {
        "col_prefix":    "temp_mean_annual_c",
        "mapset":        "RES01-TMP",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020", "1981-2000"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Mean annual temperature (°C)",
    },
    {
        "col_prefix":    "et_actual_ref",
        "mapset":        "RES01-ETA",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Reference actual evapotranspiration (mm)",
    },
    {
        "col_prefix":    "et_potential_ref",
        "mapset":        "RES01-ET0",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Reference potential evapotranspiration (mm)",
    },
    {
        "col_prefix":    "et_deficit_ref",
        "mapset":        "RES01-WDE",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Reference evapotranspiration deficit (mm)",
    },
    # ── C3: Climate classification ────────────────────────────────────────────
    {
        "col_prefix":    "koppen_3char",
        "mapset":        "RES01-KG3",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Koppen-Geiger 3-character classification (GAEZ v5 / AgERA5 2001-2020)",
    },
    {
        "col_prefix":    "koppen_2char",
        "mapset":        "RES01-KG2",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Koppen-Geiger 2-character classification",
    },
    {
        "col_prefix":    "thermal_zones",
        "mapset":        "RES01-MC3",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "Classification by thermal climates and thermal zones",
    },
    # ── C6: Land-use context ──────────────────────────────────────────────────
    {
        "col_prefix":    "multicrop_rainfed",
        "mapset":        "RES01-MCR",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C6",
        "description":   "Multi-cropping class, rain-fed (0=none … 3=triple-crop possible)",
    },
    {
        "col_prefix":    "multicrop_irrigated",
        "mapset":        "RES01-MCI",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       ["2001-2020"],
        "crop_code_key": "standard",
        "concept":       "C6",
        "description":   "Multi-cropping class, with irrigation",
    },
    # ── C5: Soil quality (all 8 SQ criteria, low & high input) ───────────────
    # SQX mapset: dimension SQX ∈ {SQ0…SQ7}, dimension IM-SQ ∈ {LIM, HIM}
    # SQ0 = most-limiting factor (composite); SQ1-SQ7 = individual criteria
    # Each entry below generates two columns: _lim and _hi
    {
        "col_prefix":    "soil_quality_sq0",
        "mapset":        "SQX",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],          # static (no period)
        "crop_code_key": "standard",
        "concept":       "C5",
        "description":   "Soil quality SQ0 – most-limiting factor (composite, low & high input)",
        "_sqx":          "SQ0",       # custom key: soil-quality index within SQX
    },
    {
        "col_prefix":    "soil_quality_sq1",
        "mapset":        "SQX",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C5",
        "description":   "Soil quality SQ1 – nutrient availability",
        "_sqx":          "SQ1",
    },
    {
        "col_prefix":    "soil_quality_sq2",
        "mapset":        "SQX",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C5",
        "description":   "Soil quality SQ2 – nutrient retention capacity",
        "_sqx":          "SQ2",
    },
    {
        "col_prefix":    "soil_quality_sq3",
        "mapset":        "SQX",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C5",
        "description":   "Soil quality SQ3 – rooting conditions",
        "_sqx":          "SQ3",
    },
    {
        "col_prefix":    "soil_quality_sq4",
        "mapset":        "SQX",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C5",
        "description":   "Soil quality SQ4 – oxygen availability",
        "_sqx":          "SQ4",
    },
    {
        "col_prefix":    "soil_quality_sq5",
        "mapset":        "SQX",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C5",
        "description":   "Soil quality SQ5 – excess salts",
        "_sqx":          "SQ5",
    },
    {
        "col_prefix":    "soil_quality_sq6",
        "mapset":        "SQX",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C5",
        "description":   "Soil quality SQ6 – toxicity",
        "_sqx":          "SQ6",
    },
    {
        "col_prefix":    "soil_quality_sq7",
        "mapset":        "SQX",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C5",
        "description":   "Soil quality SQ7 – workability",
        "_sqx":          "SQ7",
    },
    # ── C3: AEZ classification (static maps) ─────────────────────────────────
    {
        "col_prefix":    "aez_33class",
        "mapset":        "MAP:AEZ33-1KM",   # MAP: prefix → maps endpoint
        "mapset_fb":     "MAP:AEZ33",
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "AEZ 33-class (climate × soil × terrain × land cover, ~1 km)",
    },
    {
        "col_prefix":    "aez_57class",
        "mapset":        "MAP:AEZ57",
        "mapset_fb":     None,
        "crop_specific": False,
        "wsim":          None,
        "periods":       [],
        "crop_code_key": "standard",
        "concept":       "C3",
        "description":   "AEZ 57-class (finer soil subdivision, ~1 km)",
    },
]


# =============================================================================
# IN-MEMORY RASTER CATALOG CACHE (avoids redundant API calls per run)
# =============================================================================
_mapset_raster_cache: dict = {}   # mapset_code → list[raster_item]


# =============================================================================
# API HELPERS
# =============================================================================

def gismgr_get(path: str, params: dict = None) -> dict:
    """GET from GISMGR API with retry.  `path` is appended to GISMGR_BASE."""
    url = path if path.startswith("http") else f"{GISMGR_BASE}/{path}"
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("status") != 200:
                raise RuntimeError(f"GISMGR error: {data.get('message')}")
            return data
        except (requests.RequestException, RuntimeError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
    raise RuntimeError(f"GISMGR GET failed after {MAX_RETRIES} retries: {url}") from last_exc


def list_mapset_rasters(mapset_code: str) -> list:
    """
    Return all raster items in a mapset (handles pagination).
    Results are cached in _mapset_raster_cache for the lifetime of the process.
    """
    if mapset_code in _mapset_raster_cache:
        return _mapset_raster_cache[mapset_code]

    all_items = []
    seen_codes: set = set()
    page = 0
    while True:
        try:
            data = gismgr_get(f"mapsets/{mapset_code}/rasters",
                               {"page": page, "size": 100})
        except RuntimeError as exc:
            print(f"  [api-error] {mapset_code}: {exc}", file=sys.stderr)
            break

        resp  = data.get("response", {})
        items = resp.get("items", [])

        new_items = [i for i in items if i["code"] not in seen_codes]
        if not new_items:
            break
        for item in new_items:
            seen_codes.add(item["code"])
        all_items.extend(new_items)

        if not resp.get("hasNext", False):
            break
        page += 1

    _mapset_raster_cache[mapset_code] = all_items
    return all_items


def get_map_download_url(map_code: str) -> str | None:
    """Query the GISMGR maps endpoint and return the GCS download URL."""
    try:
        data = gismgr_get(f"maps/{map_code}/raster")
        return data.get("response", {}).get("downloadUrl")
    except RuntimeError:
        return None


def filter_rasters(rasters: list, crop_code: str | None = None,
                   period_code: str | None = None,
                   wsim: str | None = None) -> list:
    """
    Filter a raster list using substring / token matching on the raster code.
    Historical rasters only (HIST keyword present, or no future SSP pattern).
    """
    future_ssps = {"SSP126", "SSP245", "SSP370", "SSP585"}

    result = []
    for r in rasters:
        code = r["code"]
        code_up = code.upper()

        # Exclude future SSP scenarios
        if any(ssp in code_up for ssp in future_ssps):
            continue

        # Period filter
        if period_code and period_code not in code:
            continue

        # Crop filter (token match with dot delimiters)
        if crop_code and f".{crop_code}." not in code and not code.endswith(f".{crop_code}"):
            continue

        # WSIM filter (token match)
        if wsim and f".{wsim}" not in code and not code.endswith(f".{wsim}"):
            continue

        result.append(r)

    return result


# =============================================================================
# URL BUILDERS
# =============================================================================

def mapset_gcs_url(mapset_code: str, raster_code: str) -> str:
    """Construct the direct GCS URL for a raster in a regular mapset."""
    return f"{GCS_MAPSET}/{mapset_code}/{raster_code}.tif"


def sqx_gcs_url(sq_dim: str, im_code: str) -> str:
    """Construct GCS URL for a SQX soil-quality raster."""
    raster_code = f"GAEZ-V5.SQX.{sq_dim}.{im_code}"
    return f"{GCS_MAPSET}/SQX/{raster_code}.tif"


# =============================================================================
# DOWNLOAD HELPERS
# =============================================================================

def download_raster(url: str, local_path: Path, verbose: bool = True) -> Path:
    """
    Stream-download a raster TIF to `local_path`.  Returns `local_path`.
    Skips if already cached.  Uses a .tmp file to prevent partial downloads.
    """
    local_path = Path(local_path)
    if local_path.exists():
        if verbose:
            print(f"  [cache] {local_path.name}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(".tmp")

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()

            total_b = int(r.headers.get("content-length", 0))
            if total_b > 500 * 1024 * 1024:
                gb = total_b / (1024 ** 3)
                print(f"  [info] Large file ({gb:.1f} GB) – this may take a while")

            pbar = tqdm(
                total=total_b if total_b else None,
                unit="B", unit_scale=True,
                desc=local_path.name, leave=False,
                disable=not verbose,
            )
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()

            tmp.rename(local_path)
            if verbose:
                mb = local_path.stat().st_size / (1 << 20)
                print(f"  [dl-ok] {local_path.name} ({mb:.1f} MB)")
            return local_path

        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt < MAX_RETRIES - 1:
                print(f"  [retry {attempt+1}/{MAX_RETRIES}] {exc}", file=sys.stderr)
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise RuntimeError(
                    f"Download failed after {MAX_RETRIES} attempts: {url}"
                ) from exc


# =============================================================================
# RASTERIO SAMPLING
# =============================================================================

def sample_band1(tif_path: Path, lon_lat_pairs: list) -> list:
    """
    Sample band 1 of a raster at each (lon, lat) coordinate.
    Returns a list of values (int or float) aligned with lon_lat_pairs.
    Out-of-bounds and nodata pixels are returned as None.
    """
    with rasterio.open(tif_path) as ds:
        nodata = ds.nodata
        dtype  = ds.dtypes[0]
        raw    = [pix[0] for pix in ds.sample(lon_lat_pairs)]

    is_int = "int" in dtype
    results = []
    for val in raw:
        if nodata is not None:
            nd = int(nodata) if is_int else float(nodata)
            v  = int(val)    if is_int else float(val)
            if v == nd:
                results.append(None)
                continue
        if is_int:
            results.append(int(val))
        else:
            fval = float(val)
            results.append(None if fval != fval else fval)   # NaN → None

    return results


# =============================================================================
# LOCATION-FILE HELPERS
# =============================================================================

def get_country_codes(crop: str, countries: list = None) -> list:
    crop_dir = DATA_DIR / crop
    if not crop_dir.exists():
        raise FileNotFoundError(f"No data dir for crop '{crop}': {crop_dir}")
    if countries and "all" not in [c.lower() for c in countries]:
        return list(countries)
    codes = [
        d.name for d in crop_dir.iterdir()
        if d.is_dir() and (d / f"location_{crop}_{d.name}.csv").exists()
    ]
    return sorted(codes)


def filter_countries_by_min_years(crop: str, countries: list = None,
                                   min_years: int = MIN_YEARS_THRESHOLD) -> list:
    ypath = (CYBENCH_ROOT / "YIELD-HUB" / "setups" / "configurations"
             / "years_dict.json")
    with open(ypath) as fh:
        years_dict = json.load(fh)
    crop_years   = years_dict.get(crop, {})
    country_list = get_country_codes(crop, countries)
    valid = []
    print(f"\n[info] Filtering countries with >= {min_years} years of data …")
    for cc in country_list:
        n = len(crop_years.get(cc, []))
        if n >= min_years:
            valid.append(cc)
        else:
            print(f"  [skip] {cc}: {n} years")
    print(f"[info] {len(valid)}/{len(country_list)} countries pass filter")
    return valid


def read_location_file(crop: str, country: str) -> list:
    """Return list of (adm_id, lat, lon) from the location CSV."""
    loc = DATA_DIR / crop / country / f"location_{crop}_{country}.csv"
    if not loc.exists():
        raise FileNotFoundError(f"Location file not found: {loc}")
    points = []
    with open(loc) as fh:
        for row in csv.DictReader(fh):
            try:
                points.append(
                    (row["adm_id"], float(row["latitude"]), float(row["longitude"]))
                )
            except (KeyError, ValueError):
                pass
    return points


# =============================================================================
# CORE FETCH LOGIC
# =============================================================================

def resolve_raster(predictor: dict, crop_code: str | None, period_key: str | None,
                   include_wsim_in_col: bool,
                   cache_dir: Path, verbose: bool) -> tuple:
    """
    Find the best raster for a predictor entry, download it to cache_dir, and
    return (local_path, column_name_suffix).
    Returns (None, None) when no raster is available.

    column_name_suffix examples:
      ""                 → static, no period / wsim
      "_2001_2020"       → period only (crop-agnostic or wsim not shown)
      "_2001_2020_rf_hi" → period + wsim (when include_wsim_in_col=True)
    """
    mapset = predictor["mapset"]
    period_code = PERIOD_CODES.get(period_key) if period_key else None

    # ── MAP: static maps ─────────────────────────────────────────────────────
    if mapset.startswith("MAP:"):
        map_codes = [mapset[4:]]
        if predictor.get("mapset_fb") and predictor["mapset_fb"].startswith("MAP:"):
            map_codes.append(predictor["mapset_fb"][4:])

        for mc in map_codes:
            url = get_map_download_url(mc)
            if not url:
                continue
            local = cache_dir / f"MAP_{mc}.tif"
            try:
                download_raster(url, local, verbose=verbose)
                return local, ""
            except RuntimeError as exc:
                print(f"  [skip] {mc}: {exc}", file=sys.stderr)
        return None, None

    # ── SQX: soil quality ────────────────────────────────────────────────────
    # Handled separately (two IM codes: LIM and HIM → two columns per SQ dim)
    if mapset == "SQX":
        # This function returns a dict of {col_suffix → local_path}
        # The caller handles multi-column expansion
        return "__SQX__", predictor["_sqx"]   # sentinel; processed in caller

    # ── Regular mapsets ───────────────────────────────────────────────────────
    mapsets_to_try = [mapset]
    if predictor.get("mapset_fb"):
        mapsets_to_try.append(predictor["mapset_fb"])

    wsim_prefs = predictor.get("wsim") or []

    for ms in mapsets_to_try:
        rasters = list_mapset_rasters(ms)
        if not rasters:
            continue

        # Filter by crop (None for crop-agnostic)
        ccode = crop_code if predictor["crop_specific"] else None
        filtered = filter_rasters(rasters, crop_code=ccode, period_code=period_code)

        # If period filter returns nothing, relax it
        if not filtered and period_code:
            filtered = filter_rasters(rasters, crop_code=ccode)

        if not filtered:
            continue

        # WSIM selection
        chosen_raster  = None
        chosen_wsim    = None
        for wsim in wsim_prefs:
            wsim_matches = filter_rasters(filtered, wsim=wsim)
            if wsim_matches:
                chosen_raster = wsim_matches[0]
                chosen_wsim   = wsim
                break
        if chosen_raster is None:
            # Take first available
            chosen_raster = filtered[0]
            chosen_wsim   = None

        # Build column suffix
        suffix_parts = []
        if period_code:
            suffix_parts.append(PERIOD_LABELS.get(period_code, period_code.lower()))
        if include_wsim_in_col and chosen_wsim and chosen_wsim in WSIM_LABELS:
            suffix_parts.append(WSIM_LABELS[chosen_wsim])
        col_suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

        # Download
        url        = mapset_gcs_url(ms, chosen_raster["code"])
        local_path = cache_dir / f"{ms}_{chosen_raster['code']}.tif"
        try:
            download_raster(url, local_path, verbose=verbose)
            return local_path, col_suffix
        except RuntimeError as exc:
            print(f"  [dl-skip] {chosen_raster['code']}: {exc}", file=sys.stderr)
            continue

    return None, None


def fetch_sqx(sq_dim: str, points: list, cache_dir: Path,
              verbose: bool) -> dict:
    """
    Fetch SQX soil-quality rasters for a given SQ dimension (SQ0–SQ7) for
    both LIM (low input) and HIM (high input).
    Returns {col_suffix → sampled_values_list}.
    """
    lon_lat = [(lon, lat) for (_, lat, lon, *_) in points]
    result  = {}

    for im_code, label in [("LIM", "lim"), ("HIM", "hi")]:
        url   = sqx_gcs_url(sq_dim, im_code)
        fname = f"SQX_{sq_dim}_{im_code}.tif"
        local = cache_dir / fname
        col   = f"_{label}"
        try:
            download_raster(url, local, verbose=verbose)
            result[col] = sample_band1(local, lon_lat)
        except RuntimeError as exc:
            print(f"  [skip] SQX.{sq_dim}.{im_code}: {exc}", file=sys.stderr)

    return result


def fetch_all_predictors(crop: str, countries: list, periods: list,
                         include_wsim_in_col: bool,
                         cache_dir: Path, output_dir: Path,
                         resume: bool, verbose: bool,
                         no_filter: bool) -> None:
    """Main fetch loop: download GAEZ v5 rasters, sample all locations, write CSV."""

    crop_code         = CROP_CODES.get(crop)
    crop_code_actuals = CROP_CODES_ACTUALS.get(crop)
    if not crop_code:
        raise ValueError(f"Unsupported crop '{crop}'. Supported: {list(CROP_CODES)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Load location points ──────────────────────────────────────────────────
    if no_filter:
        country_list = get_country_codes(crop, countries)
    else:
        country_list = filter_countries_by_min_years(crop, countries)

    points = []   # (adm_id, lat, lon, crop, country)
    for cc in country_list:
        try:
            for adm_id, lat, lon in read_location_file(crop, cc):
                points.append((adm_id, lat, lon, crop, cc))
        except FileNotFoundError as exc:
            print(f"[warning] {exc}", file=sys.stderr)

    if not points:
        print(f"[error] No locations found for crop={crop}")
        return

    print(f"\n[info] {len(points)} locations across {len(country_list)} countries")

    # ── Resume: load existing results from JSONL cache ────────────────────────
    resume_file = output_dir / f"gaez_v5_{crop}_cache.jsonl"
    existing: dict = {}

    if resume and resume_file.exists():
        with open(resume_file) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    existing[row["adm_id"]] = row
                except json.JSONDecodeError:
                    pass
        print(f"[info] Resume: {len(existing)} locations already cached")

    pending = [p for p in points if p[0] not in existing]
    print(f"[info] {len(pending)} locations to process")

    if not pending:
        _write_csv(points, existing, output_dir, crop)
        return

    lon_lat_pending = [(lon, lat) for (_, lat, lon, *_) in pending]
    all_vals: dict  = {p[0]: {} for p in pending}   # adm_id → {col: value}

    # ── Process each predictor ────────────────────────────────────────────────
    for pred in PREDICTOR_CATALOG:
        col_prefix = pred["col_prefix"]
        mapset     = pred["mapset"]
        ccode      = (crop_code_actuals if pred["crop_code_key"] == "actuals"
                      else crop_code)

        target_periods: list = pred.get("periods", [])
        # Determine which periods to actually fetch
        if target_periods:
            fetch_periods = [p for p in target_periods if p in periods] or [target_periods[0]]
        else:
            fetch_periods = [None]   # static / 2020-actual: no period

        for period_key in fetch_periods:
            tag = f" [{period_key}]" if period_key else ""
            print(f"\n[predictor] {col_prefix}{tag}  concept={pred['concept']}")

            # ── SQX soil quality (special handler) ──────────────────────────
            if mapset == "SQX":
                sq_dim     = pred["_sqx"]
                sq_results = fetch_sqx(sq_dim, pending, cache_dir, verbose)
                for col_suffix, values in sq_results.items():
                    col_name = f"{col_prefix}{col_suffix}"
                    for (adm_id, *_), val in zip(pending, values):
                        all_vals[adm_id][col_name] = val
                    nn = sum(v is not None for v in values)
                    print(f"  [ok] col={col_name!r}  non-null={nn}/{len(values)}")
                continue

            # ── MAP / regular mapset ─────────────────────────────────────────
            local_path, col_suffix = resolve_raster(
                pred, ccode, period_key,
                include_wsim_in_col, cache_dir, verbose,
            )

            if local_path is None:
                print(f"  [skip] no raster found", file=sys.stderr)
                continue

            col_name = f"{col_prefix}{col_suffix}"
            try:
                values = sample_band1(local_path, lon_lat_pending)
            except Exception as exc:
                print(f"  [sample-error] {exc}", file=sys.stderr)
                continue

            for (adm_id, *_), val in zip(pending, values):
                all_vals[adm_id][col_name] = val

            nn = sum(v is not None for v in values)
            print(f"  [ok] col={col_name!r}  non-null={nn}/{len(values)}")

    # ── Append new results to JSONL cache ─────────────────────────────────────
    ts = datetime.utcnow().isoformat() + "Z"
    with open(resume_file, "a") as fh:
        for adm_id, lat, lon, crop_name, country in pending:
            row = {
                "adm_id":          adm_id,
                "latitude":        lat,
                "longitude":       lon,
                "crop":            crop_name,
                "country":         country,
                "fetch_timestamp": ts,
                "gaez_version":    GAEZ_VERSION,
                **all_vals.get(adm_id, {}),
            }
            existing[adm_id] = row
            fh.write(json.dumps(row) + "\n")

    _write_csv(points, existing, output_dir, crop)


def _write_csv(points: list, rows: dict, output_dir: Path, crop: str) -> None:
    """Write the final wide-format CSV."""
    if not rows:
        print("[warning] No rows to write")
        return

    id_cols   = ["adm_id", "latitude", "longitude", "crop", "country"]
    meta_cols = {"fetch_timestamp", "gaez_version"}
    pred_cols = sorted(
        c for c in next(iter(rows.values()), {}).keys()
        if c not in id_cols and c not in meta_cols
    )
    # Union of all predictor columns (different rows may have different keys)
    all_pred = set()
    for row in rows.values():
        all_pred.update(c for c in row if c not in id_cols and c not in meta_cols)
    pred_cols = sorted(all_pred)

    fieldnames = id_cols + pred_cols + sorted(meta_cols & all_pred)

    csv_path = output_dir / f"gaez_v5_{crop}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for adm_id, lat, lon, crop_name, country in points:
            row = rows.get(adm_id, {})
            out = {k: row.get(k) for k in fieldnames}
            out.update(adm_id=adm_id, latitude=lat, longitude=lon,
                       crop=crop_name, country=country)
            writer.writerow(out)

    n_cols = len(fieldnames)
    print(f"\n[done] {len(points)} rows × {n_cols} columns → {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch all available GAEZ v5 static predictors for CyBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--crop", required=False, nargs="+", choices=list(CROP_CODES),
        help="Crops to process (default: all supported)",
    )
    parser.add_argument(
        "--countries", nargs="+",
        help="Country codes to process (default: all available for the crop)",
    )
    parser.add_argument(
        "--periods", nargs="+",
        default=["2001-2020", "1981-2000"],
        choices=list(PERIOD_CODES),
        help="Historical periods to fetch (default: both 2001-2020 and 1981-2000)",
    )
    parser.add_argument(
        "--show_wsim_in_col", action="store_true",
        help="Append WSIM label (rf_hi / rf_lo) to column names.  "
             "Off by default (single-WSIM runs produce cleaner names).",
    )
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache_dir", default=str(DEFAULT_CACHE),
        help=f"Raster cache directory (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run: skip adm_ids already in the JSONL cache",
    )
    parser.add_argument(
        "--no_filter", action="store_true",
        help=f"Skip the >={MIN_YEARS_THRESHOLD}-year data filter for countries",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed progress (default: on)",
    )
    parser.add_argument(
        "--list_predictors", action="store_true",
        help="Print the predictor catalog and exit (no data is downloaded)",
    )

    args = parser.parse_args()

    if args.list_predictors:
        print(f"\nGAEZ v5 Predictor Catalog  ({len(PREDICTOR_CATALOG)} entries)\n")
        print(f"{'Column prefix':<35} {'Concept':<6} {'Crop-sp':<8} {'Periods'}")
        print("-" * 75)
        for p in PREDICTOR_CATALOG:
            periods_str = (", ".join(p["periods"]) if p["periods"] else "static / 2020")
            print(f"{p['col_prefix']:<35} {p['concept']:<6} "
                  f"{'yes' if p['crop_specific'] else 'no':<8} {periods_str}")
        return

    crops = args.crop if args.crop else list(CROP_CODES)

    for crop in crops:
        print(f"\n{'='*65}")
        print(f"  GAEZ v5 predictors  –  crop: {crop.upper()}")
        print(f"{'='*65}")

        fetch_all_predictors(
            crop              = crop,
            countries         = args.countries,
            periods           = args.periods,
            include_wsim_in_col = args.show_wsim_in_col,
            cache_dir         = Path(args.cache_dir),
            output_dir        = Path(args.output_dir),
            resume            = args.resume,
            verbose           = args.verbose,
            no_filter         = args.no_filter,
        )


if __name__ == "__main__":
    main()
