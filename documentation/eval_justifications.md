# Justifications for evaluation-protocol and reference fixes

This file records *why* each change was made, with the code locations it touches,
so the decisions can be defended in the paper and audited later. Entries are
ordered by the question that motivated them.

Framing used throughout: the fixed temporal split (train → 2 validation years →
last 5 years test) is read as "train up to year T, forecast every region for
T+1 … T+5". With `--use_recursive_lags` no observed test-year yield is fed back
as a lag, so test year T+h is a genuine h-year-ahead forecast. Per-year metrics
are therefore per-horizon metrics (`horizon = year − last_train_year`, saved
in the per-sample prediction CSV).

---

## 1. "Margin of error": evaluation-side tolerance, not a loss change

**Change.** `process/evaluationProtocol.py::tolerance_accuracy` reports the
fraction of region-years whose absolute error is within ±5 %, ±10 %, ±20 % of
the observed yield and within ±0.5 t/ha. Keys: `within_5pct`, `within_10pct`,
`within_20pct`, `within_0.5abs` (overall and per year).

**Why not in the loss.** The production loss is the multi-quantile pinball
loss (`architectures/tstLayer.py::_compute_weighted_loss`, lines ~836–865;
same in `linearLayer.py`). An ε-insensitive zone in a pinball loss pulls the
τ = 0.1 / 0.9 quantiles toward the centre because errors smaller than ε stop
contributing gradient asymmetrically. That degrades exactly the tail
calibration the uncertainty claims rest on. A tolerance *metric* costs nothing,
is interpretable to agronomists ("within 10 % in 78 % of region-years"), and
pairs with the calibration metrics instead of fighting them.

**Why these thresholds.** ±10 % is the deviation magnitude at which
early-warning systems typically act; ±5 % and ±20 % bracket it so the
sensitivity is visible; ±0.5 t/ha is an absolute reading that does not blow up
for low-yield regions (relative error on a 1 t/ha region is dominated by label
noise). No threshold is used for model selection.

---

## 2. Anomaly reference: what was wrong and what the fix is

**Observed code.** `process/spatiotemporal_metrics.py::compute_anomaly_correlation`
(old line 287) computed `regional_means = df.groupby('region')['y_true'].mean()`
over whatever samples were passed in; both base layers
(`tstLayer.py::_compute_spatiotemporal_metrics_from_preds`, ~1308–1333, and
the `linearLayer.py` twin) pass only test-year predictions. So the reference
was the 5-year *test-window* mean of the observed targets.

**Three findings.**

1. *The redundant columns.* Because the same reference is subtracted from
   `y` and `ŷ`, `(y − ref) − (ŷ − ref) = y − ŷ`. Hence `rmse_anomaly`,
   `mae_anomaly` and `bias_anomaly` were bit-for-bit equal to the plain
   RMSE/MAE/bias. `smape_anomaly` was defined but meaningless (denominator
   ≈ 0 near the reference). These four are removed. Only metrics whose value
   changes under centring remain: `r_an`, `r2_anomaly`, `nrmse_anomaly`, plus
   the new `std_true_anomaly` / `std_pred_anomaly` (their ratio < 1 flags a
   mean-reverting model that scores well on RMSE but never predicts a bad year).

2. *Swapping to the training mean is not the fix.* Yields trend upward, so
   with a training-mean reference every test year sits "above normal" for
   every region. A model that simply follows the trend then gets a high r_an
   for free. The synthetic check in the session log showed a zero-skill
   trend-follower scoring r_an = 0.34 (window) / 0.39 (training mean) /
   0.00 (trend-referenced).

3. *The right reference is a trend fitted on training years only.*
   The anomaly a researcher cares about is "deviation from what you would
   have expected anyway", and "what you expect anyway" is the technology
   trend. The project already has this object (`architectures/trendLayer.py::TrendModel`).

**Change.** `compute_anomaly_correlation` takes an optional per-sample
`reference` array; `compute_all_spatiotemporal_metrics` takes
`trend_reference` and returns an extra `anomaly_trend` block;
`flatten_spatiotemporal_metrics_simple` adds `anomaly/r_an_trend`. The
window-referenced block is **kept** (key unchanged, `anomaly/r_an_overall`)
and relabelled "window ref" in `validateModel.print_metrics_table`.

**Why keep both.** They answer different questions and have complementary
failure modes:

| reference | uses test targets? | defined per year? | credits trend-level skill? |
|---|---|---|---|
| window (test-window regional mean) | yes (diagnostic only) | no (one value per region-year ⇒ NaN) | no — immune to trend-extrapolation error |
| OLS detrend fitted on training years | no (out-of-sample) | yes | yes — a model that extrapolates the trend better than OLS is rewarded |

The paper's out-of-sample anomaly statement should be the detrended one
(`r_an_detrended` in the protocol); the window one is the conservative
cross-check. Reporting both pre-empts the obvious reviewer objection either way.

**Which trend for the reference (and why two).** `TrendModel` only
extrapolates when a Mann-Kendall test finds a significant trend inside a
5–10-year window; otherwise it returns the regional mean. That makes it a
strong *forecast* baseline (it refuses to extrapolate noise) but a weak
*anomaly reference* (half the regions would be referenced to a flat mean and
the trend-inflation problem returns). The protocol therefore uses:

* `ref_trend` — `TrendModel`, Mann-Kendall-gated. Used as a skill baseline
  (`skill_rmse_vs_trend`). Identical to what `--use_residual_trend` adds back,
  so `skill_rmse_vs_trend` isolates what the network adds on top of its own
  trend component.
* `ref_ols` — plain per-region OLS over all training years (regional mean if
  < 5 years). Standard agronomic detrending; always extrapolates. Used as the
  anomaly reference (`r_an_detrended`), for the bad-year event definition, and
  as the centre of the distributional reference. Also reported as a skill
  baseline (`skill_rmse_vs_ols`).

**r_tm guard.** `compute_temporal_correlation` now takes `min_years=3`: with 2
points Pearson r is ±1 by construction, so regions with < 3 test years return
NaN for the correlation (error metrics still computed). With 5 test years this
only affects regions with missing years.

---

## 3. Evaluation protocol (`process/evaluationProtocol.py`)

All reference quantities come from `dm.train_ds` (denormalised with the
training `y_mean`/`y_std`). Validation targets are used only for the
conformal margin; test targets only for scoring.

### 3.1 Skill scores against free forecasts

`skill_rmse_vs_{trend,ols,climatology,rolling3} = 1 − RMSE_model / RMSE_ref`
(and the MAE analogues). Positive = better than the reference; 0 = no better;
negative = worse. References:

* `climatology`: per-region mean of training yields.
* `rolling3`: mean of the region's last 3 *training* years. It is held
  constant across the 5 test years because in the h-years-ahead reading no
  observed test yield is available. (A rolling mean that peeks at observed
  test years would contradict `--use_recursive_lags`.)
* `trend`, `ols`: see §2.

Why RMSE skill rather than raw RMSE: a raw error is not comparable across
countries whose yields differ five-fold, and "beats its own free baseline" is
the only comparison that can support a claim of forecast value.

### 3.2 Bad-year detection

Event: `(y − ref_ols) < −k · σ_region`, k ∈ {0.5, 1.0}. `σ_region` is the
standard deviation of the region's training residuals around its OLS line
(pooled country σ when a region has < 5 training years). Score for AUC:
`−(ŷ − ref_ols)/σ`; hard decision: predicted anomaly below the same
threshold. Reported: AUC, precision, recall, F1, base rate, number of events.
For quantile models an additional AUC uses the lower quantile as the score
(`badyear_*_auc_qlo`): a well-calibrated lower tail should rank bad years
better than the median does.

Why σ-scaled rather than a fixed percentage: it makes "bad" relative to the
region's own interannual variability, so a 10 % drop in a stable region and a
30 % drop in a volatile one are treated consistently. Events are rare (base
rate reported), so AUC is NaN in years with no event — that is correct, not a
bug.

### 3.3 Calibration and CRPS (quantile models)

* `coverage_q{10,50,90}`: empirical P(y ≤ q_τ); `quantile_calibration_error`
  = mean |coverage − τ|. This is the reliability diagram in one number.
* `picp` vs `picp_nominal` (= τ_hi − τ_lo = 0.8 for the 0.1/0.9 pair) and
  `mean_interval_width`. Width is only comparable between models at the same
  coverage, which is why the conformal step exists.
* `crps_pinball = (2/K) Σ_k pinball_{τ_k}(y, q_k)`: the standard
  quantile-score identity for CRPS with the integral replaced by a mean over
  the K available levels. The existing `validateModel.UncertaintyMetrics._compute_crps`
  integrates step-function indicators with the trapezoid rule and is not
  monotone in quantile accuracy; it is left untouched for continuity but the
  protocol uses the pinball form. Both are saved; use `crps_pinball` for
  comparisons.
* `skill_crps_vs_ols`: CRPS skill against a distributional reference
  `q_τ^ref = ref_ols + σ_region · z_τ`, where `z_τ` is the pooled empirical
  quantile of the *standardised* training residuals. Per-region quantiles at
  τ = 0.1/0.9 from ≤ 15 points are too noisy, so the shape is pooled across
  regions and only the scale is per region. This is the "trend + spread"
  forecast that a quantile model must beat to justify itself.

### 3.4 Split-conformal correction (CQR)

For the outer pair (q_lo, q_hi): non-conformity `s = max(q_lo − y, y − q_hi)`
on the validation years; margin = the ⌈(n+1)(1−α)⌉/n empirical quantile of
`s` with α = 1 − (τ_hi − τ_lo); test interval widened by the margin on both
sides (lower bound clipped at 0). Reported: `conformal_margin`,
`picp_conformal`, `mean_interval_width_conformal`, `crps_pinball_conformal`.

Why: it converts "our intervals are 80 % intervals" from an assertion into a
finite-sample guarantee under exchangeability, and the before/after pair tells
you how far off the raw model was. Caveat to state in the paper: the same
validation years drive early stopping, so the margin is mildly optimistic;
with only 2 validation years the margin is coarse (quantile of ~2 × n_regions
scores). Exchangeability across years is an approximation under trend;
the per-year `picp_conformal` row makes any drift visible.

### 3.5 Outputs

* Console table (`print_protocol_table`), overall + per test year.
* `results_dir/protocol/<metric>.csv` — same one-row-per-run, one-column-per-
  year layout as the existing `rmse.csv` etc., so the existing aggregation
  scripts work unchanged. Written via the new generic
  `helpers.save_metric_tables`; `save_test_results_to_csv` was refactored to
  call it (behaviour unchanged).
* `results_dir/predictions/<model>_<crop>_<country>_<run_id>.csv` — per
  region-year: `y_true`, `y_pred`, quantiles, all references, σ, horizon,
  anomalies, conformal bounds. This is the input for cross-run statistics
  (paired bootstrap over years, Wilcoxon across crop–country pairs), which are
  deliberately *not* computed inside a single run because they need ≥ 2 runs.
* Headline overall values are merged into `final_metrics` under
  `protocol/<metric>` so they reach WandB and `print_metrics_table`.

---

## 4. Embedding tap (`return_embeddings`)

**Observed code.** `architectures/multitaskModels.py::MultitaskWrapper.forward`
(~886) calls `base_model(x_ts, x_static, observed_mask, return_embeddings=True)`.
`XLinearYieldModel.forward` (`linearLayer.py` ~1786) had no such argument, so
the call raised `TypeError`, was caught (~906), and the head was fed
`x_static` (~933). A head on static inputs shares no parameters with the
encoder, so no auxiliary gradient could ever reach TGM/VGM. Every XLinear
multitask number produced so far measured "regression + an unrelated
classifier", not multitask learning. PatchTST went through
`MultitaskTSTWrapper`, which re-implements the forward pass to tap the pooled
state — functional, but a second copy of the forward logic to keep in sync.

**Change.** `XLinearYieldModel.forward` and `PatchTSTModel.forward` accept
`return_embeddings: bool = False` and, when true, return
`(predictions, combined)` where `combined` is the pre-head tensor
(`[endo_pooled | exo_pooled | x_static]` for XLinear,
`[pooled encoder state | x_static]` for PatchTST). Default behaviour and all
existing call sites are unchanged (single tensor returned). No auxiliary head
was added — this only makes the shared representation reachable.

**Why `combined` rather than the pooled encoder state alone.** It is the exact
input of the regression head, so an auxiliary head attached here shares the
whole trunk with the regressor and nothing more; it also keeps the static
features available to the head, matching how the regressor sees them.

---

## 5. `predict()` broadcasting bug (`tstLayer.py` ~1410)

`final_pred_z = pred + batch_trends.squeeze(-1)` was correct for MSE
(`(B,) + (B,)`) but for MQL `pred` is `(B, 3)` and `batch_trends.squeeze(-1)`
is `(B,)`, which only broadcasts when B ∈ {1, 3}; any other batch raises.
The eval/test steps already handled this case correctly; only `predict()`
did not, which is the method the protocol uses to collect validation
predictions for conformal calibration. Fixed to mirror the eval step
(`(B, n_q) + (B, 1)` for MQL). `linearLayer.predict` already had the right form.

---

## 6. What was deliberately not changed

* The training loss, the split, the normalisation, and the base-layer
  per-year metrics: unchanged, so existing result CSVs remain comparable.
* `multitaskBaselines.py` / `multitaskModels.py`: untouched. The tap fix is
  in the base models, so the wrapper's existing `return_embeddings=True`
  call now receives real embeddings for XLinear without further edits; the
  heads themselves are out of scope for this pass.
* Walk-forward evaluation: not added, per the decision to keep the 5-year
  fixed split. The protocol is split-agnostic and would work under
  `generate_walk_forward_splits` unchanged.
* `--loss_type mse` remains a live option in both scripts. It is NOT part of
  the study (the loss of record is the pinball/MQL loss). The protocol only
  has to degrade gracefully for it (skip the quantile/conformal block). It was
  exercised once in a smoke test for that reason and nothing more.

---

## 7. Plain-language guide to every protocol metric

Read this with one row of the console table in mind, e.g. the smoke run
`wheat-NL | patchtst | last train year 2013 | forecast origin 2015 (horizons 1..5)`.
"Origin 2015" means the model has seen every yield up to 2015 (training years
plus the two validation years, whose yields enter as lag features); 2016 is
therefore a 1-year-ahead forecast and 2020 a 5-year-ahead forecast. The
`overall` column pools all five years; the year columns are the horizon curve.

### 7.1 Error and skill (how good, compared with what)

**RMSE** — root-mean-square error in t/ha between the median forecast and the
observed yield. Lower is better. Same as before; listed so the reference rows
can be read against it.

**RMSE trend ref (MK)**, **RMSE OLS ref** — the RMSE you would get from two
forecasts that cost nothing and use no weather at all:
* *trend (MK)*: each region's Mann-Kendall-gated trend line extrapolated to
  the test year (this is exactly the trend the network adds back under
  `--use_residual_trend`). Where no significant trend exists it is just the
  regional mean.
* *OLS*: a plain straight line through the region's training years, always
  extrapolated.
Two more references are in the CSVs but not printed: *climatology* (the
region's training mean) and *rolling-3* (mean of the region's last three
training years).

**Skill RMSE vs X** = 1 − RMSE_model / RMSE_X. The single most important row.
* +0.20 means the model's error is 20 % smaller than reference X.
* 0 means it is no better than X.
* −0.39 (PatchTST smoke run, 2 epochs) means it is 39 % *worse* than just
  extrapolating the trend — i.e. the network has not yet learned anything the
  trend did not already know.
A model that beats climatology but not the trend has only learned the trend.
A model that beats the trend has extracted information from weather, remote
sensing or lags. "Skill vs trend (MK)" is the headline number for "did deep
learning add anything".

### 7.2 Tolerance ("margin of error")

**Within ±10 %** — fraction of region-years whose forecast is within 10 % of
the observed yield. 0.71 = 71 % of forecasts were "close enough" by that
rule. **Within ±0.5 t/ha** is the same with an absolute band (useful because
10 % of 1 t/ha is almost nothing). The CSVs also hold ±5 % and ±20 %. These
are the operational reading of accuracy; they do not enter training.

### 7.3 Anomaly skill (does it get the hard part right)

Every yield = slow part (trend / regional level) + fast part (this year's
anomaly, driven by weather). The slow part is easy; the anomaly is what a
forecast has to earn.

**r_an (OLS-detrended)** — Pearson correlation between observed and predicted
anomalies, where anomaly = yield − the region's training-fitted OLS line.
1 = the model ranks good and bad years perfectly; 0 = no idea; this is the
out-of-sample anomaly statement for the paper. It can be computed per year
because the reference does not depend on the test window.

**r_an (window ref)** — same correlation but with anomaly = yield − the
region's mean over the five test years. It is built from the test targets
themselves, so it is a diagnostic cross-check, not a claim. It removes any
error in the trend extrapolation (which the detrended version still credits)
and is therefore the more conservative of the two. It is undefined per year
(one value per region per year ⇒ nothing to correlate), hence the NaNs.
The two should agree in sign; a large gap (e.g. 0.49 vs −0.03 in the XLinear
smoke run) means most of the apparent anomaly skill is really trend-level
skill, not weather-driven skill.

**Anomaly spread ratio** — std of predicted anomalies / std of observed
anomalies. 1.0 = the model is as "bold" as reality; 0.65 = it predicts
anomalies only two-thirds as large as they actually are (mean-reverting). A
model can have a good RMSE and a low spread ratio at the same time — that is
the model that never calls a bad year.

### 7.4 Bad-year detection (the food-security reading)

A region-year is a **bad year** if its anomaly is below −1 σ, where σ is that
region's year-to-year spread around its own trend (estimated from training
years). The table shows the 1 σ rows; 0.5 σ rows are in the CSVs.

**Bad-year base rate** — what fraction of region-years were bad. 0.14 = 14 %.
This sets the difficulty; with 12 regions × 1 year there may be only one bad
case, which is why per-year AUC is often 1.0 or NaN.

**Bad-year AUC** — treat the predicted anomaly as a "badness score" and ask:
if you pick one bad year and one normal year at random, how often does the
model score the bad one as worse? 0.5 = coin flip, 1.0 = perfect ranking,
below 0.5 = systematically backwards. This is the right metric when events
are rare because it does not depend on a threshold.

**Bad-year recall** — of the actual bad years, what fraction did the model
flag (predicted anomaly also below −1 σ)? 0.29 = it caught 2 of 7. Precision
and F1 (in the CSVs) are the complementary "how many flags were false alarms"
reading. For quantile models a second AUC (`*_auc_qlo`, CSV only) uses the
lower quantile as the score; a well-calibrated lower tail should rank bad
years better than the median does.

### 7.5 Calibration (do the uncertainty bands mean what they say)

The model outputs three quantiles: q10, q50, q90. The [q10, q90] band is
supposed to contain the truth 80 % of the time.

**PICP (nominal)** — the fraction of observed yields that actually fell inside
[q10, q90]. Compare with the nominal 0.80. 0.31 (PatchTST smoke run) means the
band caught only 31 % — the model is badly over-confident. 0.63 (XLinear) is
still too narrow. Above 0.80 would be under-confident.

**Interval width** — average q90 − q10 in t/ha. Only meaningful next to PICP:
narrow bands are good *if* they cover.

**Quantile calib. error** — mean over the three levels of |empirical coverage
− τ|, where empirical coverage of q10 is "fraction of observations below
q10" (should be 0.10), etc. 0 = perfectly calibrated at every level; 0.27 =
on average each quantile is 27 points off its target. Per-level coverages are
in the CSVs (`coverage_q10/50/90`) and are the numbers to plot as a
reliability diagram.

**PICP conformal**, **Width conformal** — the same two numbers after the
split-conformal correction: the band is widened by a margin chosen on the
validation years so that validation coverage reaches 80 %. If the test years
behave like the validation years, PICP conformal ≈ 0.80. In the smoke run it
reached 0.71 at the cost of widening the band from 0.49 to 1.64 t/ha — the
raw band was 3× too narrow. The gap between 0.71 and 0.80 is distribution
shift between validation and test years (2016 was a bad year: coverage 0.33).
Report the raw and conformal pairs side by side; "calibrated" should mean the
conformal one.

### 7.6 CRPS (one number for the whole distribution)

**CRPS (pinball)** — the continuous ranked probability score, in t/ha. Think
of it as the MAE of a probabilistic forecast: it rewards putting probability
mass close to the truth and penalises both wrong centre and wrong width.
Lower is better; a perfect point forecast would have CRPS 0. Computed as
2 × the mean pinball loss over the three quantile levels.

**CRPS skill vs OLS** = 1 − CRPS_model / CRPS_reference, where the reference
is "OLS trend centre + the region's historical residual spread" — i.e. the
probabilistic version of the trend baseline. Positive = the model's
distribution is more useful than trend-plus-climatological-spread. The
PatchTST smoke run's −0.32 says its distribution is 32 % worse than that free
baseline; XLinear's +0.16 says 16 % better.

### 7.7 How to read a row across years

The year columns form the horizon curve (1 … 5 years ahead). Expect skill and
r_an to fall and widths to need to grow with horizon if the model is doing
honest forecasting. A flat curve means the forecast is dominated by the slow
part (trend/level) and the in-season information is not being used; a curve
that gets *better* with horizon is a red flag for leakage.

### 7.8 Where each number lives

* Console table: `print_protocol_table` in `process/evaluationProtocol.py`.
* `results_dir/protocol/<metric>.csv`: one row per run, columns = config +
  one per test year + overall (53 files).
* `results_dir/predictions/<model>_<crop>_<country>_<run_id>.csv`: every
  region-year with `y_true`, `y_pred`, `q10/q50/q90`, the four references,
  `ref_sigma`, `horizon`, `anom_true/anom_pred`, and `q10_conf/q90_conf`.
  This is the file to use for cross-run statistics and figures.
