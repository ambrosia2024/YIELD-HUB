#!/usr/bin/env python3
"""Convert variable_shift CSVs + eda_peryear.json -> dashboard/data/*.json
Usage: python3 build_eda_data.py <src_dir> <dst_dir>
"""
import sys, json, math, shutil
import pandas as pd
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

COUNTRY_NAME = {
  "AO":"Angola","AR":"Argentina","AT":"Austria","AU":"Australia",
  "BE":"Belgium","BF":"Burkina Faso","BG":"Bulgaria","BR":"Brazil",
  "CN":"China","CZ":"Czechia","DE":"Germany","DK":"Denmark",
  "EE":"Estonia","EL":"Greece","ES":"Spain","ET":"Ethiopia",
  "FI":"Finland","FR":"France","HR":"Croatia","HU":"Hungary",
  "IN":"India","IT":"Italy","LS":"Lesotho","LT":"Lithuania",
  "LV":"Latvia","ML":"Mali","MZ":"Mozambique","NE":"Niger",
  "NL":"Netherlands","PL":"Poland","PT":"Portugal","RO":"Romania",
  "SE":"Sweden","SK":"Slovakia","SN":"Senegal","TD":"Chad",
  "US":"United States","ZA":"South Africa","ZM":"Zambia",
}
CONTINENT = {
  "AO":"Africa","AR":"South America","AT":"Europe","AU":"Oceania",
  "BE":"Europe","BF":"Africa","BG":"Europe","BR":"South America",
  "CN":"Asia","CZ":"Europe","DE":"Europe","DK":"Europe",
  "EE":"Europe","EL":"Europe","ES":"Europe","ET":"Africa",
  "FI":"Europe","FR":"Europe","HR":"Europe","HU":"Europe",
  "IN":"Asia","IT":"Europe","LS":"Africa","LT":"Europe",
  "LV":"Europe","ML":"Africa","MZ":"Africa","NE":"Africa",
  "NL":"Europe","PL":"Europe","PT":"Europe","RO":"Europe",
  "SE":"Europe","SK":"Europe","SN":"Africa","TD":"Africa",
  "US":"North America","ZA":"Africa","ZM":"Africa",
}

dfs = []
for crop in ["maize","wheat"]:
    p = src / f"variable_shift_{crop}.csv"
    if p.exists(): dfs.append(pd.read_csv(p))
df = pd.concat(dfs, ignore_index=True)
df["country_name"] = df["country"].map(lambda c: COUNTRY_NAME.get(c,c))
df["continent"]    = df["country"].map(lambda c: CONTINENT.get(c,"Unknown"))
# Map ndvi/fpar to remote_sensing group
df["source"] = df["source"].replace({"ndvi": "remote_sensing", "fpar": "remote_sensing"})
df["sig"]       = (df["ks_pval"] < 0.05).astype(int)
df["direction"] = (df["mean_test"] > df["mean_train"]).map({True:1,False:-1})

keep = ["crop","country","country_name","continent","source","variable","quantile",
        "ks_stat","ks_pval","wasserstein","mean_train","mean_test",
        "std_train","std_test","n_train","n_test","sig","direction",
        "spatial_shift_median","spatial_shift_min","spatial_shift_max","n_regions",
        "temporal_shift_min","temporal_shift_median","temporal_shift_max"]
df = df[[c for c in keep if c in df.columns]]

ann = ((df["variable"].isin(["yield","yield_detrended"])) & (df["quantile"]=="annual")) | \
      ((~df["variable"].isin(["yield","yield_detrended"])) & (df["quantile"]=="q50"))
annual = df[ann].copy(); annual["quantile"] = "annual"
quant  = df[~df["variable"].isin(["yield_detrended"]) & (df["quantile"]!="annual")].copy()

def rows(d):
    out = []
    for _, r in d.iterrows():
        row = {}
        for c in d.columns:
            v = r[c]
            if hasattr(v,"item"): v = v.item()
            if isinstance(v,float) and math.isnan(v): v = None
            elif isinstance(v,float): v = round(v,6)
            row[c] = v
        out.append(row)
    return out

result = {"annual": rows(annual), "quantile": rows(quant)}
(dst/"eda_variable_shift.json").write_bytes(json.dumps(result,separators=(",",":")).encode())
print(f"eda_variable_shift.json: {len(result['annual'])} annual, {len(result['quantile'])} quantile rows")

shutil.copy(src/"eda_peryear.json", dst/"eda_peryear.json")
print(f"eda_peryear.json: copied ({(src/'eda_peryear.json').stat().st_size//1024} KB)")
