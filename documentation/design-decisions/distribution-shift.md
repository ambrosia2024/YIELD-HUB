# Distribution-Shift Design Decisions (`compute_shift_distance.py`)

Scope: how YIELD-HUB quantifies cross-country distribution shift as a **dataset
property**, to be attached as context to the cross-country transfer results and
to drive the Δ-skill-vs-shift-distance curve in `evaluation_protocol.md`.

Unit of analysis: one row per **country** (per crop), derived from per-**region**
(adm_id) feature vectors. The score answers: *how distinguishable is country Y's
feature distribution from the pooled distribution of all other countries?*

Status date: 2026-10-19. Backup of the prior implementation:
`pre-compute/.backup_2026-10-19/compute_shift_distance.py`.

---

## D0. What "shift distance" is (and is not)

It is the AUC of a binary **domain classifier** (classifier two-sample test,
C2ST) trained to separate "region belongs to country Y" from "region belongs to
any other country in the pool".

- AUC ≈ 0.50 → Y looks like the pool → easy transfer.
- AUC ≈ 1.00 → Y is highly out-of-distribution → hard transfer, expect large
  Δ-skill degradation.

It is a property of the **data**, not of any yield model. Computed once per crop,
joined to transfer results on (crop, country).

---

## D1. Temporal features: quantiles, not the mean

**Decision.** Summarize each daily temporal variable by the quantile set
`[2.5, 25, 50, 75, 97.5]` instead of a single mean.

**Why.** The previous implementation collapsed each region's entire time series
to a mean over both year and date. That discards exactly the information a shift
detector exists to find: two countries with identical mean Tmax but very
different variance or hot/dry tails became indistinguishable. Since the whole
downstream purpose is **extreme-year** transfer, the tails are the signal. The
mean is roughly q50; the quantile vector strictly dominates it, and the 2.5/97.5
tails are the genuinely new, non-redundant information.

**Quantile set choice.** We deliberately dropped the collinear mid-quantiles
(e.g. 40/50/60) an earlier draft considered. Adjacent quantiles carry
overlapping information and inflate dimensionality for little gain. `[2.5, 25,
50, 75, 97.5]` keeps both tails + the quartiles + the median: 5 values per
variable, ~44 features total, down from ~88. This materially eases the p≫n
problem for small countries (see D4).

**Axis of aggregation.** Records are daily (one observation per date), so
quantiles are taken **across the dates within each (region, year)** growing
season — not within a date (which has a single value). The per-(region, year)
quantile vectors are then **averaged across years** to one row per region.

- Row unit stays **region**, not region-year. Region-year would quadruple the
  sample count (tempting for p≫n) but the repeated-region rows are correlated,
  which both inflates AUC and breaks the CV independence assumption.
- Averaging quantiles across years means a single freak year cannot dominate a
  region's fingerprint: the q97.5 becomes "the typical hot-day of the season,"
  which is the intended climate signature.

---

## D2. Three feature classes (static variables get NO quantiles)

Quantiles are applied **only** where there is a within-season daily distribution
to summarize. The `dfs_x` schema splits three ways and the code branches on the
index layout (`feat_df.index.nlevels`):

| Class | dfs_x keys | Index | Treatment |
|-------|-----------|-------|-----------|
| Static | `soil` (awc, bulk_density) | 1-level (adm_id) | single value, **no quantiles** |
| Per-year | `crop_season` calendar cols | 2-level (adm_id, year) | single mean across years, **no quantiles** (no daily axis) |
| Daily temporal | `meteo` (tmin/tmax/tavg/prec/cwb/rad), `fpar`, `ndvi` | 3-level (adm_id, year, date) | 5 quantiles per (region, year), averaged over years |

Rationale: a static soil/location value has no distribution to quantile; a
per-year calendar value has one value per year (no daily axis), so we average it
rather than quantiling across years. Only the daily-resolution group carries the
within-season distribution the quantiles are meant to capture.

---

## D3. latitude/longitude excluded entirely

**Decision.** Drop `location` (lat/lon) features unconditionally
(`EXCLUDE_KEYS = {"location"}`); there is no opt-in flag.

**Why.** A country occupies a contiguous geographic box, so lat/lon make almost
every country trivially linearly separable. Including them would make shift
distance largely measure "is this country geographically elsewhere" — a
near-tautology — rather than climate/feature-distribution distance, which is
what we actually want to relate to transfer skill. Computing a *distribution*
shift from geographic *coordinates* is self-defeating, so they are removed, not
merely defaulted off. Soil (awc, bulk_density) is retained — it is a
non-geographic static property.

**History.** An earlier revision kept lat/lon behind a `--include_latlon` flag;
it was removed on 2026-10-19. Leaving geography as a toggle in a
distribution-shift metric is a foot-gun, so the flag was deleted rather than
defaulted off.

---

## D4. Classifier: regularized logistic regression, NOT a gradient-boosted tree

**Decision.** `StandardScaler → LogisticRegression(class_weight='balanced',
C=1.0, L2)`. NaNs imputed with per-column **median** before scaling.

**History — this reverses an earlier lean toward XGBoost.** The reversal is
driven directly by D1, not by indecision:

- The only strong reason to prefer a nonlinear model (XGBoost/RandomForest) over
  a linear one is to detect distributional differences a linear-on-**means**
  classifier would miss — e.g. a hotter tail at equal mean. But D1 **hand-
  engineers that exact information into explicit dimensions** (q97.5, q2.5, the
  quartiles). A hotter tail is now a linear weight on the q97.5 feature. We moved
  the nonlinearity into the representation, so a **linear** classifier already
  sees tail/shape shifts.

Given that, logistic regression is the better choice here:

1. **Stability & comparability** — the whole point of a "distance." With
   n_target sometimes < 10 and ~44 features, a tree (even shallow) overfits the
   tiny positive class, pushes AUC toward 1.0 for nearly everyone, compresses the
   dynamic range, and makes the ranking noisy. A regularized linear model is
   lower-variance and more comparable across countries, with the p≫n inflation
   controlled transparently through `C` rather than opaquely through tree depth.
2. **Tractable permutation null** — the null (D5) is the expensive part
   (B refits × k folds × every country). On the **CPU-only login node that
   SIGKILLs heavy concurrent jobs**, a linear model makes B=500–1000 feasible
   where boosting would make it a scheduled job.
3. **No extra dependency** — scikit-learn already covers it; no xgboost install
   in the `cybench` env.

**Cost accepted.** Linear cannot model joint feature interactions ("hot AND dry"
together) beyond what the separate quantile dimensions encode. For a
conservative, comparable distance metric this is an acceptable — arguably
preferable — trade, and it is trivially reversible. If linear proves
underpowered, the fallback is a lightly-tuned **RandomForest** (handles NaN, less
overfit-prone on tiny n than boosting), **not** XGBoost.

**Imputation note.** Median (not mean) imputation, because the 2.5/97.5 tail
features are deliberately heavy-tailed and the mean is not robust to them.

---

## D5. AUC alone is not enough → permutation null

**Decision.** Wrap every country's AUC in a label-**permutation test** and
report significance and a de-biased effect size, not a bare AUC.

A raw CV-mean AUC has three failures, all amplified by D1 (more features) and the
small n_target regime:

1. **No null baseline.** With p≫n, random labels already give AUC > 0.5.
2. **Conflates effect size with significance.** A single number cannot say both
   "is it real" and "how big."
3. **Not comparable across countries** with different region counts (different
   inflation).

**Mechanism.** For each country, shuffle the country labels `n_perm` times,
recompute the CV AUC each time → null distribution. Then:

- `perm_p = (1 + #{AUC_null ≥ AUC_obs}) / (n_perm + 1)` — significance
  ("is this shift beyond chance?"). The +1 smoothing means `perm_p` is never
  exactly 0; minimum resolvable p = 1/(n_perm+1).
- `auc_null_mean = mean(AUC_null)`.
- `auc_debiased = AUC_obs − auc_null_mean` — effect size, with each country's own
  finite-sample inflation subtracted out, making magnitudes comparable.

`perm_p`, `auc_debiased`, and the raw AUC are **non-redundant**: significance vs
de-biased magnitude vs raw separability — each answers a different question,
consistent with the project's metric-redundancy discipline. (`auc_std`, the
fold-std, is kept for continuity but is **not** a substitute: it understates true
uncertainty and knows nothing about chance.)

**Default `n_perm = 500`** (CLI `--n_perm`): enough resolution for a 0.05 cut;
raise to 1000 for smoother `auc_null_mean`. Runtime scales linearly, and because
of D4 this is cheap. Still, a full run across all countries is a deliberate job,
not an instant smoke test, on the login node.

---

## D6. Shift classification: none / near / far

**Decision.** Emit a categorical `shift_class` from `(perm_p, auc_debiased)`:

| class | condition | meaning |
|-------|-----------|---------|
| `none` | `perm_p ≥ alpha` | not distinguishable from the pool (very close) |
| `near` | `perm_p < alpha` and `auc_debiased < near_far_threshold` | real but small shift (borderline) |
| `far`  | `perm_p < alpha` and `auc_debiased ≥ near_far_threshold` | strong, significant OOD shift (hardest transfer) |

Defaults: `alpha = 0.05`, `near_far_threshold = 0.25` (both CLI-overridable).
`near_far_threshold` on the de-biased AUC ≈ 0.25 corresponds to a raw AUC ≈ 0.75
(since `auc_null_mean ≈ 0.5`).

**Why this definition, and why NOT Köppen.** The none/some split is principled —
it is the permutation p-value, not a hand-picked number. Only the near/far split
within "significant" needs a magnitude threshold, which is documented and tunable
rather than a magic constant. We considered defining near/far from Köppen climate
zones (same dominant zone = near, different = far), but rejected it: that would
silently swap "distance of the feature distribution the model sees" for "climate-
category similarity" — a different, coarser (modal-zone) metric. Köppen remains an
**orthogonal context column** to join later for interpreting *why* a country is
OOD, never the definition of the shift magnitude.

---

## D7. Output schema

`cybench/data/shift_distance/shift_distance_{crop}.csv`, one row per country:

```
crop, country, shift_distance_auc, auc_std, auc_null_mean, auc_debiased,
perm_p, shift_class, n_features_shifted, frac_features_shifted,
n_target, n_pool, n_features, n_perm, note, computed_at
```

`n_features_shifted` / `frac_features_shifted` come from the Kuiper breadth
measure (D8). `--extra_output_dir` writes a second identical copy of this CSV to
an arbitrary directory (e.g. a paper-tables folder) so the standalone shift
output can live next to manuscript assets without touching the eval pipeline.

Join to `ranking.py` transfer output on (crop, country) for the
Δ-skill-vs-shift-distance curve. Reading of the curve:

- **far-shift + good Δ-skill** → supports H1 (generalisation).
- **far-shift + poor Δ-skill** → the hard cases motivating the B4–B6 fusion arms.
- **near-shift** → borderline cases to watch.

---

## D8. Kuiper breadth (implemented 2026-10-19)

The classifier AUC measures **overall** separability. To also measure the
**breadth** of a shift (which / how many features individually differ), a
**model-free per-feature two-sample test** runs alongside it:

- Per feature, a **Kuiper** two-sample test (statistic `V = D⁺ + D⁻`) between
  country Y's regions and the pool. Kuiper is preferred over KS because it is
  sensitive to differences in **both** tails, matching the 2.5/97.5 emphasis
  (KS's `max|D|` can miss a compensating two-tailed shift that Kuiper's sum
  catches).
- **Benjamini–Hochberg FDR** correction across features (`scipy.stats.
  false_discovery_control`).
- Emitted columns: `n_features_shifted` (# features with BH q < `breadth_alpha`,
  default 0.05, CLI `--breadth_alpha`) and `frac_features_shifted`.

This is non-redundant with the classifier AUC (breadth-across-features vs
overall separability) and sharpens near-vs-far interpretation: a near-shift
driven by a single tail feature looks very different from a far-shift broad
across many.

**Implementation note.** `astropy.stats.kuiper_two_sample` is **not installed**
in the `cybench` env, so a self-contained Kuiper (ECDF `V` + the standard
asymptotic p-value, Press et al. / Numerical Recipes) is used; if astropy is
ever installed the code prefers its vetted version automatically. The built-in
path was validated (2026-10-19): `V = D⁺ + D⁻`, `max(D⁺,D⁻)` reproduces scipy's
`ks_2samp` statistic exactly, same-distribution p ≈ 0.50, a clearly shifted
sample gives p ≈ 0. Cost is negligible (no CV, no permutation — one test per
feature per country).

---

## D9. Where the performance × shift join lives: ranking.py

**Decision.** The "model performance across far / near / none" table — the paper
deliverable — is produced by a post-hoc join in `process/ranking.py`
(`shift_stratified_performance`), **not** in `metrics.py` and **not** in
`compute_shift_distance.py`.

**Why here and nowhere else.**

- `compute_shift_distance.py` **cannot** produce it: it is a pure dataset-property
  script that runs *before* any model and has zero model results in scope. Its
  `--extra_output_dir` only writes a second copy of the shift classification
  itself (country → shift_class, AUC, perm_p, breadth) — useful, standalone, but
  not performance.
- `metrics.py` must stay clean: it is the per-run tier measurement, and bolting
  shift columns onto its output CSVs would bloat every run with a dataset-level
  property. (User's explicit constraint.)
- The join needs BOTH the shift CSV (country → shift_class) and the per-country
  model deltas (country → Δ-skill). Only the cross-run post-hoc layer sees both.
  Per the two-file evaluation policy, that layer is `ranking.py`, and the join
  mirrors the existing `earliest_useful_origin`: read CSVs, aggregate, write.

**What it emits** (run: `python ranking.py --results_dir <dir>`; shift CSVs found
automatically at `cybench/data/shift_distance/`, override with
`--shift_distance_dir`):

- `shift_stratified_performance.csv` — one row per (crop, tier, metric,
  arm_vs_b0, shift_class), sorted hardest-bucket-first. Columns: `n_countries`,
  `countries` (the member country NAMES, `;`-joined), `mean_delta`,
  `median_delta`, `mean_m_arm`, `mean_m_b0`, `frac_beats_b0`, `n_wins`, and the
  within-bucket shift context `mean_shift_auc` / `mean_auc_debiased` /
  `mean_feats_shifted`. This is the paper table: read `mean_delta` /
  `frac_beats_b0` / `n_wins` down the far → near → none rows to see whether an arm
  generalises better where the shift is larger (H1/H4).
- `ranking_with_shift.csv` — every `ranking.csv` row plus the per-country shift
  context columns, for scatter plots (x = `shift_distance_auc` or `auc_debiased`,
  y = `delta`) and ad-hoc slicing.

**Join semantics.** Shift distance is arm- and partition-agnostic (a dataset
property), so it joins on (crop, country) only. Countries present in the ranking
panel but absent from the shift CSV (e.g. too few regions to score) are bucketed
as `unknown` rather than dropped, so nothing silently vanishes.

---

## Decisions explicitly rejected

- **Mean-only temporal features** — discards the extreme-tail signal that is the
  entire point (see D1).
- **XGBoost / gradient-boosted trees** — overfits tiny n_target, inflates and
  compresses AUC, makes the permutation null expensive; its one advantage is made
  redundant by the quantile features (see D4).
- **Köppen-defined near/far** — conflates feature-distribution distance with
  climate-category similarity; kept as orthogonal context instead (see D6).
- **Pre-computing extreme-year counts** (`base_rate_extreme`, `agreement_rate`)
  here — these are computed **live** per run in `metrics.py` from training-year
  thresholds (no-leakage design) and depend on `--test_years`; a dataset-wide
  pre-compute would require a fixed split and be incorrect.
