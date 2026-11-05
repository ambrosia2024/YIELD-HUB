# Evaluation protocol — every test, every metric, in order

Companion to `research_question.md` (v2, Part B). That file says WHICH tiers exist and WHY; this
file says exactly WHAT is computed in each tier, as a formula and in plain words.
Previous version (column-by-column reference for the current CSVs) is kept at
`.backup_2026-10-09/evaluation_protocol.md`; column names below are unchanged where the metric exists.

Status tags:  HAVE = computed today (file:line given; re-verify)   FIX = exists, must be changed
              ADD  = not implemented yet                            Each tier lists its HEADLINE.

---

## 0. Ground rules that apply to every number

### 0.1 Notation
    r      a region (admin unit), t a year, o a forecast origin (0.25 / 0.50 / 0.75 / 1.00 of season)
    y_rt   observed yield (t/ha);   ŷ_rt  the model's point forecast = its median (q50) under MQL
    q_τ,rt the model's τ-quantile forecast, τ in {0.10, 0.50, 0.90};  [q10, q90] = the 80 % band
    N      number of (region, year) test cases in the cell being scored
    trend_r(t)  the per-region trend fitted on TRAINING years only (architectures/trendLayer.py)
    a_rt   = y_rt - trend_r(t)      the ANOMALY: how unusual this year was, beyond the slow trend

### 0.2 The decomposition that organizes everything
    y_rt = trend_r(t) + a_rt
  The trend (technology, slow climate drift) is predictable for free. The anomaly is the part that
  matters for food security and the only part a learned model can be credited for. Tier 0 scores
  the total; everything from Tier 2 on scores the anomaly or the behaviour under stress.

### 0.3 A number is one of three things
    HEADLINE  one per tier; appears in the main figure; carries a confidence interval
    SUPPORT   explains or sanity-checks the headline; appendix tables
    CONTEXT   not a score (base rates, reference errors, n); never compared across arms

### 0.4 Every headline is reported as a DELTA with uncertainty
    Δ_m(arm) = m(arm) - m(B0)        computed on the SAME backbone, split, origin, seeds
  Positive Δ for higher-is-better metrics (skill, AUC, r) and negative Δ for lower-is-better metrics
  (Winkler, CRPS) mean the fusion helped. Section 6 defines the confidence interval and the test.
  "arm beats B0" is only written when Section 6 says so; otherwise the cell says "within noise".

### 0.5 Hygiene gates (not metrics; if any fails, nothing downstream is reported)
    [FIX]  Quantiles sorted after forward so q10 <= q50 <= q90 (tstLayer.py clamps >=0 at :1017,:1427
           but never sorts; a crossed band silently corrupts PICP, width, Winkler, CRPS).
    [HAVE] Recursive lags on for every out-of-sample run (`--use_recursive_lags`), else lag features
           of a test year contain observed yields of other test years.
    [HAVE] All references (trend, OLS, climatology, rolling-3, residual spread) fitted on training years
           only (evaluationProtocol.py:85 `ReferenceForecasts.fit`).
    [HAVE] `protocol_rmse` == core `rmse` on the same run (consistency check between the two code paths).

---

## TIER 0 — In-domain accuracy
   "On the years and countries it was trained for, how close is the forecast, and is it closer
    than a free forecast would be?"
   HEADLINE: `protocol_skill_rmse_vs_trend`   SUPPORT: nrmse, r2, mae, tolerance accuracy

### 0.A Reference forecasts (CONTEXT) — the opponents          [HAVE evaluationProtocol.py:85-199]
  Four free forecasts, all fitted on training years of region r only:
    trend_r(t)        Mann-Kendall-gated window + OLS extrapolation; falls back to the regional mean
                      when no significant trend exists. Same object as `--use_residual_trend`.
    ols_r(t)          plain least-squares line through all training years.
    clim_r            mean of training yields (a flat line).
    roll3_r           mean of the last three training years.
  Columns `protocol_rmse_ref_trend / _ols / _climatology / _rolling3` = RMSE of each opponent on the
  test set. Plain words: the score a lazy forecaster would get. Never compared across arms — the
  references do not depend on the arm.

### 0.B Point error                                           [HAVE validateModel.py:182 ModelMetrics]
  rmse    = sqrt( (1/N) Σ (y - ŷ)² )                     t/ha; big misses count extra. Lower better.
          Plain: typical size of a miss, with disasters weighted heavily.
  mae     = (1/N) Σ |y - ŷ|                              t/ha; all misses count equally. Lower better.
          Plain: on a normal year, how many t/ha off.
  mse     = rmse²                                        internal; lower better.
  nrmse   = rmse / mean(y)                               unit-free. Lower better. (validateModel.py:204)
          Plain: the miss as a fraction of a normal harvest; 0.10 = about 10 % off. This is what
          lets maize in NL (12 t/ha) and maize in MW (2 t/ha) sit in one table.
  r2      = 1 - Σ(y - ŷ)² / Σ(y - mean y)²               1 perfect, 0 = as good as the mean, < 0 worse.
          Plain: share of the ups and downs the model got right. In-domain only: across
          countries of different scale mean(y) is dominated by country scale, so r2 is inflated —
          report per country.
  mape    = (1/N) Σ |y - ŷ| / |y| ;  smape = (1/N) Σ |y - ŷ| / ((|y|+|ŷ|)/2)      SUPPORT; lower better.

### 0.C Skill scores — the HEADLINE                            [HAVE evaluationProtocol.py:211,369-370]
  skill_vs_ref = 1 - rmse_model / rmse_ref
    > 0 the model beats that opponent; 0 tied; < 0 worse than the free forecast.
    Columns `protocol_skill_rmse_vs_{trend,ols,climatology,rolling3}` and the `_mae_` twins.
  rmse or nrmse? It makes NO difference to this score. With nrmse = rmse / mean(y), the mean(y) is
  one constant applied to BOTH model and reference on the same cases, so it cancels in the ratio:
  1 - nrmse_model/nrmse_ref = 1 - rmse_model/rmse_ref, at every aggregation level. The label is
  free; keep rmse. (This equals the CY-Bench benchmark's per-country "NRMSE reduction", so the
  number is directly comparable to that paper.)
  What DOES matter is the unit of aggregation, now LOCKED:
    - within a region:  rmse over that region's test years.
    - within a country (HEADLINE):  skill_r = 1 - rmse_model,r / rmse_ref,r per region, then average
      skill_r EQUALLY across regions. Every region counts once, so a model cannot look good just by
      nailing the few high-production regions while failing the small ones. Guard: floor rmse_ref,r
      (or drop regions with < 3 test years) so a near-zero reference cannot blow skill_r up.
    - across countries:  never pooled; report per country, then median and WORST-CELL. Equal-region
      average + worst-cell (incl. the worst REGION within a country) together certify "good
      everywhere", not just "good on average" — a high average can hide a failing breadbasket.
  Benchmark-comparable support: the per-country region-POOLED skill (one rmse over all region-years =
  the CY-Bench "NRMSE reduction") may be reported alongside for continuity, but it is NOT the headline.
  Food-supply (DEFERRED): a harvested-area-weighted NATIONAL-TOTAL skill (aggregate predicted
  yield x area, score the total) answers "how much wheat" and is the only number that sees regional
  errors add up or cancel. Add it only when the paper makes an explicit food-supply claim; the
  methods claim does not need it.
  Plain: a race against four lazy opponents, scored as "how much of the opponent's error did you
  remove". 0.20 vs trend = the model's error is 20 % smaller than just extrapolating the trend.
  Why the trend is THE opponent: beating climatology is easy (every trend does); beating the trend is
  the first evidence the network learned something about THIS season.
  Paper gate (Tier 0): B0 must have skill_vs_trend > 0 and skill_vs_climatology > 0 at origins 0.75
  and 1.00, else the substrate is broken.
  [ADD] Δ-table: Δ skill_vs_trend (arm - B0) per origin, per country, with CI (Section 6).

### 0.D Tolerance accuracy (SUPPORT, the decision-maker's view) [HAVE evaluationProtocol.py:218,373]
  within_p    = (1/N) Σ 1[ |y - ŷ| / |y| <= p ]       p in {0.05, 0.10, 0.20}
  within_0.5  = (1/N) Σ 1[ |y - ŷ| <= 0.5 t/ha ]
  Plain: what fraction of forecasts were "close enough" for a buyer, insurer or ministry. Higher
  better. The tolerance lives here, in evaluation — never inside the training loss.

---

## TIER 1 — Calibrated uncertainty
   "When the model says 'probably between A and B', is it right as often as it claims — and only
    then, how narrow is the band?"
   Only filled for quantile models (`loss_type = mql`); blank under MSE by design (no band to score).
   HEADLINE (in this order): `protocol_picp` -> `protocol_picp_conformal` -> `winkler_score` -> `protocol_skill_crps_vs_ols`
   Rule: never compare widths between two arms whose coverage differs by more than 5 points.

### 1.A Quantile loss — what the model was trained on          [HAVE validateModel.py:1017,984]
  pinball_τ(y, q) = max( τ (y - q), (τ - 1)(y - q) )
    Under-predicting the τ-quantile costs τ per unit; over-predicting costs (1-τ). The minimizer
    is the true τ-quantile. Columns `pinball_q10 / q50 / q90`, `pinball_avg` = mean of the three.
  Plain: three report cards, one for the low line, the middle line and the high line. q50 pinball
  is half the MAE of the median. Lower better. SUPPORT.

### 1.B Calibration — did the promises hold?                   [HAVE evaluationProtocol.py:252-258,408]
  coverage_τ = (1/N) Σ 1[ y <= q_τ ]           target: τ itself (0.10 / 0.50 / 0.90)
  quantile_calibration_error = (1/3) Σ_τ | coverage_τ - τ |          0 = perfectly honest; lower better
  picp = (1/N) Σ 1[ q10 <= y <= q90 ]           target: 0.80 (`protocol_picp_nominal`)
  Plain: the q10 line promised to lie under the truth 10 % of the time; did it? PICP is the same
  question for the whole band: you promised 80 %, how often did reality land inside?
  picp far below 0.80 = overconfident (band too tight); far above = band too wide to be useful.

### 1.C Sharpness — how narrow, given it is honest             [HAVE validateModel.py:1010,1071]
  mean_interval_width = (1/N) Σ (q90 - q10)                       t/ha; lower better AT equal coverage
  pinaw = mean_interval_width / ( max(y) - min(y) )               unit-free version
  Plain: how wide the band is. A narrow band that misses is not sharp, it is wrong — hence the
  ordering rule above.

### 1.D Winkler score — coverage and sharpness in one number    [HAVE validateModel.py:1091]
  For band [l, u] at miscoverage α (= 0.20 for the 80 % band):
    W = (u - l)                      if l <= y <= u
      = (u - l) + (2/α)(l - y)       if y < l
      = (u - l) + (2/α)(y - u)       if y > u
    winkler_score = mean over N. Lower better.
  Plain: your score is the band width, plus a fine of 10x the distance every time the truth escapes.
  Rewards bands that are both narrow and honest; the single uncertainty headline once coverage is OK.

### 1.E CRPS — the whole distribution in one score             [HAVE two versions]
  Exact: CRPS(F, y) = ∫ (F(z) - 1[z >= y])² dz, the squared distance between the forecast CDF and the
  step at the truth. With three quantiles two approximations exist:
    `protocol_crps_pinball` = (2/K) Σ_τ pinball_τ(y, q_τ)   (evaluationProtocol.py:238) — coarse but
        MONOTONE: improving any quantile always improves it. USE THIS ONE for ranking arms.
    `crps` (validateModel.py:1023) — trapezoid over indicator CDF; not monotone. Legacy; appendix only.
  Plain: one grade for the entire "maybe this, maybe that" spread, not just the middle line.
  Lower better. Collapses to MAE for a point forecast.

### 1.F Reference distribution and CRPS skill                   [HAVE evaluationProtocol.py:192-194,415-418]
  Reference band = ols_r(t) + σ_r · z_τ where σ_r is the std of region r's training residuals and
  z_τ the pooled standardized residual quantiles. Columns `protocol_crps_pinball_ref_ols`,
  `protocol_picp_ref_ols`, `protocol_mean_interval_width_ref_ols`.
  skill_crps_vs_ols = 1 - crps_model / crps_ref                    > 0 = the model's uncertainty beats
                                                                  "trend line + usual wobble"
  Plain: can the network's sense of doubt beat the farmer's rule of thumb? This is the Tier-1
  analogue of skill_vs_trend. [ADD] a `skill_crps_vs_trend` twin using trend_r(t) as centre.

### 1.G Split-conformal correction (CQR) — making the band honest after the fact   [HAVE :286,341-352,422-428]
  On validation years compute s_i = max( q10_i - y_i, y_i - q90_i ) and
    margin = the ceil((n+1)(1-α))/n empirical quantile of s      (Romano et al. 2019)
  Test band becomes [q10 - margin, q90 + margin]; marginal coverage >= 1-α is guaranteed.
  Columns `protocol_conformal_margin` (t/ha), `protocol_picp_conformal`,
  `protocol_mean_interval_width_conformal`, `protocol_crps_pinball_conformal`.
  Plain: using years the model never trained on, learn how much the band must be widened (or can be
  tightened) so the 80 % promise is kept; then report coverage and width again.
  What the paper reads from it: raw picp vs picp_conformal = how dishonest the arm was before the
  fix; width_conformal across arms at (now equal) coverage = who is genuinely sharper.
  [ADD] Reliability diagram (coverage_τ vs τ at more τ levels once `--quantiles` is exposed). Appendix.

---

## TIER 2 — Generalization under shift
   "Does the skill survive when the test is harder than the training: a new year, a new country,
    an extreme year?" Nothing new is measured here: the Tier-0 headline and the anomaly correlation
    are recomputed on harder PARTITIONS, and reported as MEAN and WORST CELL.
   HEADLINE: `protocol_skill_rmse_vs_trend` + `protocol_r_an_detrended`, per partition, mean and worst.

### 2.A Anomaly metrics — scoring the hard part only            [HAVE spatiotemporal_metrics.py; evaluationProtocol.py:376-384]
  Let â_rt = ŷ_rt - trend_r(t) be the predicted anomaly.
  r_an_detrended = Pearson corr( â_rt , a_rt ) over all test cases, anomaly vs the TRAIN-fitted trend
          (`protocol_r_an_detrended`).  Range -1..1; higher better.  THE anomaly headline.
          Plain: forget the slow drift everyone can predict — did the model know which years would
          be surprisingly good or bad?
  r_an_window = same, but anomaly = deviation from each region's TEST-window mean
          (`protocol_r_an_window`). SUPPORT. Measures within-window ranking; a trend-follower scores
          well here for free, so it is never the headline.
  r2_anomaly_detrended = 1 - Σ(a - â)² / Σ(a - mean a)²                     share of anomaly variance explained
  nrmse_anomaly_detrended = rmse(a, â) / mean(y)                           anomaly error as a fraction of yield
  anomaly_spread_ratio = std(â) / std(a)                                   target 1.0
          Plain: does the model dare to call big deviations, or does it hedge toward average?
          < 1 timid (typical for RMSE-trained models), > 1 jumpy.
  r_sp = median over years t of corr_r( ŷ_rt , y_rt )   (`protocol_r_sp`)   higher better
          Plain: in one year, did the model rank the regions right (who had the good harvest)?
  r_tm = median over regions r of corr_t( ŷ_rt , y_rt ) (`protocol_r_tm`)   CONTEXT ONLY
          Computed from 3-5 test years per region; statistical noise. Never reported as a result.
  [FIX] Make the TRAIN-fitted trend the anomaly reference everywhere (not the train mean, which
        inflates r_an for any trend-following model). Drop rmse/mae/bias_anomaly columns: they equal
        the plain rmse/mae/bias because y and ŷ are de-meaned by the same constant.

### 2.B Partitions (strata) — the same metrics, harder test sets
  2a TEMPORAL   [HAVE: the fixed trailing split IS strict year-ahead]
     Train years < T-5, test the last 5. Every Tier-0/1 number already is a temporal-OOD number.
     k-years-ahead is NOT a partition (no in-season covariates exist k years out -> trend only).
  2b SPATIAL — single-source TRANSFER MATRIX                       [ADD runner]
     M[X -> Y] = skill_vs_trend of the specialist trained on country X, scored on country Y's test
     years, for all X, Y in the 8-country panel. Diagonal = in-domain. No pooling plumbing needed.
     Derived numbers:
       asymmetry_XY   = M[X->Y] - M[Y->X]        (transfer is not symmetric)
       hardest_target = argmin_Y mean_X M[X->Y]  (the country nobody transfers to)
       best_source    = argmax_X mean_Y M[X->Y]
     Report spatial-transfer and anomaly-transfer SEPARATELY: the CY-Bench zero-shot result
     (FR->IT, US->AR) is that NRMSE roughly doubles and anomaly correlation collapses to ~0 while
     spatial structure is partly retained, so one pooled skill number hides the real pattern.
     Plain: train on France, forecast Hungary — how much worse than training on Hungary?
     Always in skill units: raw t/ha across countries measures scale mismatch, not skill.
  2c EXTREME                                                       [ADD partition builder]
     Label-extreme year:   a_rt < -1 · σ_r    (σ_r = std of training residuals of region r)
     Driver-extreme year:  heat-stress days in the top decile OR cumulative water balance / RUE in
                           the bottom decile of region r's training years (physics channels,
                           featureEngineering.py:179-321)
     agreement_rate = |label ∩ driver| / |label ∪ driver|      (Jaccard; CONTEXT, and a finding)
     Metrics on each extreme set are POOLED across regions with n stated (per-region sets are tiny).
     Plain: a hot year is not always a bad year; we score both definitions and say how often they agree.

### 2.C Difficulty axis — making OOD graded, not binary          [HAVE pre-compute/compute_shift_distance.py]
  shift_distance(Y) = AUC of a balanced logistic classifier trained to tell "region's feature vector
     comes from country Y" vs "from any other pooled country", on soil + per-region growing-season
     QUANTILE features [2.5,25,50,75,97.5] of each daily driver (tmin/tmax/tavg/prec/cwb/rad/fpar/ndvi),
     averaged over years. lat/lon are EXCLUDED on purpose (geography makes every country trivially
     separable, so AUC would measure "elsewhere" not "different"). 0.5 = indistinguishable (easy),
     1.0 = completely different (hard). DATASET property; computed once per crop, no model involved.
  AUC alone is NOT trusted — it is wrapped in a label-permutation null (shuffle country labels n_perm times):
     perm_p       = P(AUC_null >= AUC_obs)       significance: is the shift beyond chance?
     auc_debiased = AUC_obs - mean(AUC_null)     comparable effect size (removes the small-n inflation)
     shift_class  = none (perm_p >= 0.05) / near (sig, auc_debiased < 0.25) / far (sig, auc_debiased >= 0.25)
  n_features_shifted = # features where a two-sample KUIPER test (tail-sensitive, beats KS) + BH-FDR flags
     country Y vs the pool — the BREADTH of the shift, non-redundant with the classifier's overall separability.
  zone(Y) = Koppen / agro-ecological class — ORTHOGONAL context only (gives the hard cases names); it is
     deliberately NOT the near/far definition (that is distributional distance, not climate-category similarity).
  Plain: a ruler for "how foreign is this country to what the model saw", plus a p-value on whether the
     foreignness is real, a size for how foreign, and a count of how many of its signals are foreign.
  Full rationale and rejected alternatives: documentation/design-decisions/distribution-shift.md.

### 2.D Tier-2 summaries (what the paper shows)                   [ADD]
  worst_cell(m)   = min over regions / partitions / countries of m   (means hide collapse; Chakravarty;
                    the worst REGION within a country is what certifies the big breadbasket didn't fail)
  Δ-vs-shift curve: x = shift_distance(Y) (or auc_debiased), y = Δ skill_vs_trend(arm, Y) over the
                    transfer-matrix targets; one line per arm. The slope says whether fusion helps MORE
                    where the shift is larger (H4).         [HAVE ranking.py: shift_stratified_performance]
  shift-stratified table: ranking.py joins shift_distance_{crop}.csv onto the per-(crop, country) deltas
                    on (crop, country) and buckets them by shift_class, writing
                    shift_stratified_performance.csv — one row per (crop, tier, metric, arm_vs_b0,
                    shift_class): n_countries, the member country NAMES, mean/median Δ, frac_beats_b0,
                    n_wins, and mean_shift_auc — plus ranking_with_shift.csv (every ranking row + shift
                    context). This IS the paper's "performance across far / near / none" table; it lives in
                    ranking.py (post-hoc join), never in metrics.py (per-run output stays clean).
  worst-zone table: mean and worst skill per zone per arm.
  r2 is allowed to be negative here and is printed as such.

---

## TIER 3 — Operational value
   "Would a food-security analyst have been warned, and how early in the season?"
   HEADLINE: `protocol_badyear_1sig_auc`   +   earliest_useful_origin

### 3.A Bad-year detection                                       [HAVE evaluationProtocol.py:261-284,388-402]
  Event_rt = 1[ a_rt < -k σ_r ],   k in {0.5, 1.0}  (columns `..._0.5sig_...` and `..._1sig_...`)
  Score (how worried the model is) = -â_rt ; hard flag = 1[ â_rt < -k σ_r ].
  base_rate = (1/N) Σ Event                       CONTEXT: how common bad years are in the test set
  auc       = P( score(bad year) > score(normal year) )   = area under the ROC curve.
              0.5 coin flip, 1.0 perfect ranking. Column `..._auc`.
  auc_qlo   = same AUC using the LOW quantile -q10 as the warning signal (`..._auc_qlo`, MQL only).
              Plain: is the model's "worst-case line" an earlier alarm than its middle guess?
  precision = TP / (TP + FP)   Plain: when it cried wolf, how often was there a wolf?
  recall    = TP / (TP + FN)   Plain: of the real disasters, how many did it catch?
  f1        = 2 · precision · recall / (precision + recall)
  Plain for AUC: rank all region-years by how worried the model is; AUC is the chance a real bad
  year is ranked above a normal one. Reported pooled with n_events stated; if n_events < 10 the
  cell is printed in grey.

### 3.B Earliest useful origin                                    [ADD]
  earliest_useful_origin(country) = min { o in {0.25, 0.50, 0.75, 1.00} : auc_1sig(o) >= 0.70 }
     (threshold fixed before any result is seen), or "never".
  Plain: how far into the season must we be before the model's warnings are worth acting on?
  Fusion that pulls this earlier is the most practically valuable kind (H3).

### 3.C Shortfall capture (SUPPORT, magnitude on the bad tail)    [ADD; CY-Bench rebuttal metric]
  On the bottom decile of observed anomalies within each country (the worst region-years):
    tail = { (r,t) : a_rt in the lowest 10 % of observed anomalies for that country }
    C    = mean_{tail}( -â_rt ) / mean_{tail}( -a_rt )        a_rt, â_rt both negative on the tail
  C = 1 the forecast called the full average depth of the bad events; C = 0 no shortfall predicted
  (flat / trend); C > 1 over-calls the depth; C < 0 predicted a surplus during a shortfall.
  Why it earns a place: 3.A auc scores RANKING (were bad years ranked above normal), not MAGNITUDE.
  An RMSE/pinball-trained model can rank well yet systematically under-call disaster depth (it hedges
  toward the mean — the same timidity as anomaly_spread_ratio < 1, §2.A). C is the bad-tail twin of
  anomaly_spread_ratio as one operational number: of the shortfall that happened, how much did you see.
  Rules: anomaly referenced to the TRAIN-fitted trend (consistent with §2.A [FIX], not a test-window
  mean); pooled across regions with n stated; if n_tail < 10 the cell is printed in grey. SUPPORT
  under the 3.A auc headline — no new headline. (CY-Bench authors' rebuttal table: Transformer/TST
  C ~ 0.35 maize / 0.00 wheat vs LPJmL ~ 0.84 / 0.49 — a sequence backbone barely captures depth,
  which is exactly where process fusion is expected to help.)

---

## TIER 4 — Mechanism (ICON concept audit)
   "WHAT does the network rely on, and did fusion change that?" Post-hoc, no retraining.
   HEADLINE: conditional variance share per concept group + unexplained share; and the correlation
             between reliance shift and the Tier-2 gain.

### 4.A Concept groups (built from input columns)                 [ADD annotation builder]
  NON-MECHANISTIC = SPATIAL {lat, lon}, CARRY-OVER {lag_yield_k}, MGMT {sos, eos, cutoff}
  MECHANISTIC     = WEATHER {tmin, tmax, tavg, prec, rad, cwb, and the physics indices — they are
                    deterministic functions of weather, so they are the same concept}, SOIL {awc,
                    bulk_density}, RS {fpar, ndvi, ssm, rsm}, LPJML {simulated states}

### 4.B ICON decomposition (Rane et al., arXiv:2608.26083)        [ADD port]
  For a layer representation H (N x d) and concept blocks C_1..C_G, ICON fits a PLS model and
  partitions Var(H) by Type-I sums of squares in a fixed order, conditional on the outcome:
    share_g        = variance of H explained by C_g GIVEN all previously entered concepts and y
    unexplained    = 1 - Σ_g share_g
  Plain: of everything the layer "thinks about", what fraction is about weather, what fraction is
  about which region this is, what fraction is about last year's yield — after removing overlaps
  (which is why PHYSICS cannot be separated from WEATHER: the overlap is total).
  Lower unexplained is better for US (the concept set covers the layer), but a large unexplained
  share is reported, not hidden: it means the concept set is incomplete or the layer is nonlinear.

### 4.C Reliance shift and the mechanism test                     [ADD]
  reliance_shift(arm) = Σ_{g in MECHANISTIC} share_g(arm) - Σ_{g in MECHANISTIC} share_g(B0)
  mechanism_corr      = Spearman corr over countries Y of ( reliance_shift(arm, Y) , Δ skill_transfer(arm, Y) )
  Plain: did fusion make the model think more about weather and less about "where am I / what did I
  get last year" — and is the size of that change related to how much better it did on new countries?
  Cleanest version: B4/B5/B6 vs B0 (identical inputs, different training). B1/B3 vs B0 add inputs, so
  part of any shift is mechanical.

---

## 6. Statistics that turn a difference into a claim                    [ADD significance harness]
  Applied to every HEADLINE Δ in every tier.

  6.1 Seeds. Every cell is trained with s >= 3 seeds (5 for B0 and the headline arm);
      m is the mean over seeds; the seed std is printed as ±.
  6.2 Year-block bootstrap (per partition).
      Resample test YEARS with replacement (B = 1000). For each draw recompute m(arm) and m(B0) on
      the same drawn years and take Δ_b. CI = [2.5th, 97.5th] percentile of {Δ_b}.
      Why years, not region-years: all regions share one weather year, so region-years within a year
      are correlated; resampling them separately gives CIs that are far too narrow.
      Plain: pretend we had a slightly different set of test years — would the fusion still win?
  6.3 Wilcoxon signed-rank (cross-country claims).
      Since there is no pooling, the panel is the per-country Δ across the countries scored (the
      8-country panel / transfer-matrix targets for maize). Statistic on the signs and ranks of
      Δ_country. p < 0.05 required.
      Plain: across countries, is "fusion better" the rule rather than a few lucky cases?
  6.4 Definition of a WIN.
      Δ is a win iff the bootstrap CI excludes 0 AND (for cross-country claims) Wilcoxon p < 0.05.
      Otherwise the cell is reported as "within noise". No bolding of insignificant best values.
  6.5 Power caveat. With a 5-year test window the bootstrap has 5 resamplable units; if P0.4 shows
      a known-trivial change (e.g. +cwb) already "within noise", switch the headline split to
      walk-forward over the last 8 years before running the fusion arms (research_question.md D-SPLIT).

---

## 7. One-page map: test case -> what is asked -> headline -> status

  Hygiene       is the number even valid               quantile order, recursive lags, train-only refs   FIX/HAVE
  Tier 0        closer than the free forecast?         skill_rmse_vs_trend (Δ vs B0)                      HAVE + ADD Δ
  Tier 1        honest band, then narrow band?         picp -> picp_conformal -> winkler -> crps skill     HAVE + ADD skill_crps_vs_trend
  Tier 2a       new year                               Tier-0 headline on the trailing split              HAVE
  Tier 2b       new country                            transfer matrix; Δ-vs-shift; worst cell            ADD
  Tier 2c       extreme year (label / driver)          Tier-0 headline + r_an_detrended, pooled, n         ADD partitions
  Tier 3        would we have been warned, when        badyear_1sig_auc; earliest_useful_origin; shortfall C  HAVE + ADD origin,C
  Tier 4        what did it rely on, did that change   ICON shares; reliance_shift; mechanism_corr        ADD
  Statistics    is the difference real                 year-block bootstrap CI; Wilcoxon; WIN rule        ADD

## 8. Direction of "good" (quick reference)
  Lower better:  rmse, mse, mae, nrmse, mape, smape, pinball_*, crps, protocol_crps_pinball,
                 winkler_score, quantile_calibration_error, mean_interval_width / pinaw (at equal coverage),
                 shift_distance (as a difficulty label, not a model score), earliest_useful_origin
  Higher better: r2, protocol_skill_* (> 0), protocol_within_*, r_sp, r_an_detrended, r2_anomaly_detrended,
                 badyear_* auc/precision/recall/f1, reliance_shift (if H4 holds), mechanism_corr
  Target value:  picp / protocol_picp -> 0.80; coverage_q10/q50/q90 -> 0.10/0.50/0.90;
                 anomaly_spread_ratio -> 1.0; shortfall_capture C -> 1.0
  Context only:  protocol_rmse_ref_*, base_rate, n_events, r_tm, agreement_rate, unexplained share

## 9. The empty-cell rule
  Blank Tier-1 columns (pinball, crps, picp, pinaw, winkler, coverage, conformal) mean the run used
  `loss_type = mse`: a point forecast has no band to score. Re-run with `--loss_type mql`. It is a
  configuration difference, not a bug, and applies to every backbone including CNN-LSTM.
