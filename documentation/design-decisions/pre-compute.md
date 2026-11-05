# Pre-compute Design Decisions

This document records the reasoning behind key choices made in the pre-compute
pipeline (YIELD-HUB/pre-compute/). It is meant to be updated after any
experiment or discussion that changes or reaffirms these choices.

---

## Climate baseline period: why 1986-2010 (Koppen) and 1981-2010 (GAEZ)?

### What the scripts do

fetch_koppen.py assigns each CyBench location a static Koppen-Geiger climate
zone label (e.g. Cfb = Oceanic, Dfb = Humid continental) using the Beck et al.
(2018) raster at 1 km resolution.

fetch_aez.py assigns each location an Agro-Climatic Zone (ACZ) suitability
class using FAO GAEZ v4, which is based on observed climate over 1981-2010.

Both produce static per-location labels that are joined to the yield time series
on adm_id for use as features or stratification variables.

### Why a 30-year climatological mean, not recent observations?

Koppen-Geiger zones and GAEZ suitability classes are intended to characterise
the long-run climate type of a location, not its weather in any individual year.
Year-to-year climate variability is already captured by the meteorological
covariates (ERA5, AgERA5, etc.) fed into the yield models. The climate zone
label answers a different question: what kind of place is this structurally?
For that, a 30-year mean is the appropriate tool. The World Meteorological
Organisation standard reference period is 30 years precisely for this reason.

### Why 1986-2010 specifically (Koppen)?

Three reasons pull toward this period:

1. Coverage of the yield data. The CyBench yield time series begins in the early
   1980s and runs through approximately 2016. The 1986-2010 climatology sits
   inside this window. Using a more recent period (e.g. 1991-2020) would produce
   Koppen labels that reflect a climate the earlier years of the yield data did
   not actually experience — a subtle but real mismatch if a region has shifted
   zones under climate change.

2. Alignment with GAEZ. FAO GAEZ v4 uses 1981-2010 as its observed-climate
   baseline and does not provide post-2010 observed-climate scenarios (future
   periods in GAEZ require GCM projections). Using 1986-2010 for Koppen keeps
   both label sets compatible when joined on adm_id.

3. Data availability. Beck et al. (2018) is the only publicly available 1 km
   global Koppen raster and its present-day product covers 1986-2010. A newer
   Beck et al. (2023) update covers 1991-2020 but is a separate dataset and
   would require switching the download source entirely, not just adjusting a
   parameter. Adopting 2023 would be a deliberate decision to re-run the
   pre-compute pipeline and re-validate the labels, not a drop-in change.

### When to revisit

Revisit the period choice if:
- The yield dataset is extended significantly beyond 2020, making the 2010
  cutoff too far in the past to be representative.
- A Beck et al. 2023 raster at 1 km is verified available and the pipeline is
  being re-run from scratch anyway.
- GAEZ releases an updated climatology and alignment between the two label sets
  needs to be restored.

---

## Raster cache and zip handling (fetch_koppen.py)

### Background

The Beck 2018 figshare article (ID 6396959) distributes the raster as a single
zip archive named Beck_KG_V1.zip. The original cache check in _download_raster
only recognised already-extracted raster files (.tif/.tiff/.grd/.img). If a
prior run had downloaded the zip but not completed extraction, subsequent runs
would silently skip the zip in the cache, re-enter the figshare API path, and
fail because the zip filename does not contain the period string "1986_2010"
that the name-matching logic expected.

### Fix applied

The zip extraction logic was refactored into a private helper _extract_zip().
The cache check now also recognises a cached .zip and calls _extract_zip
immediately, so a downloaded-but-not-extracted archive is handled correctly on
the next run without hitting the network again. The download path was also
updated to call _extract_zip instead of inlining the same logic, removing the
duplication.
