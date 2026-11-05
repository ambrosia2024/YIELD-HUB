# YIELD-HUB Research Plan v2: When, Where and How Does Process Knowledge Help Deep In-Season Yield Forecasters?

Status: DESIGN v2 (no experiments run). Target venue: TMLR. Rewritten 2026-10-09; v1 (with the full
pipeline inventory, file:line map and verified pitfalls) is preserved at
`.backup_2026-10-09/research_question.md` — read it for WHAT EXISTS; this file only says WHAT TO DO.
Every claim about the pipeline below is a hypothesis to re-verify in code before relying on it.

Env: `conda activate cybench` is broken; use `/home/vsaxena1/miniconda3/envs/cybench/bin/python`.
Login node is CPU-only and SIGKILLs concurrent jobs. Nothing in Part C runs on the login node.

---

## A. THE PAPER

### A.1 Research question (a question, not a conclusion)

  WHEN (forecast origin), WHERE (in-domain vs under spatial/temporal/extreme shift) and HOW (fusion
  stage and process source) does process-based agronomic knowledge improve deep sub-national crop
  yield forecasters — and when does it not?

The null result is a first-class outcome. "Process fusion is within noise in-domain but improves
worst-country OOD skill and bad-year recall" is a publishable TMLR finding; so is "input concat is
all you need". The paper commits to the protocol, not to the answer.

### A.2 What is actually new (positioning against what exists — verified citations)

What already exists, and must be cited rather than claimed as ours:
- Process-model outputs as ML input features (input-stage fusion, tree models, end-of-season or fixed
  dates): Shahhosseini, Hu, Huber & Archontoulis 2021, Sci. Rep. 11:1606 (APSIM + ML, US Corn Belt);
  Paudel et al. 2021, Agric. Syst. 187:103016 (WOFOST outputs + ML, the MARS/MCYFS workflow — by the
  CY-Bench authors themselves); Feng et al. 2019, Agric. For. Meteorol. (APSIM + RF, Australian wheat).
- Representation-stage fusion via pretraining on process-model simulations then fine-tuning on
  observations (KGML): Liu et al. 2024, Nat. Commun. (carbon cycle; doi 10.1038/s41467-023-43860-5);
  KGML for waterlogging yield projections (2024); KGML county-level yield, arXiv:2503.16328 (2025);
  KGML overview arXiv:2403.15989. Review of knowledge-informed hybrid ML in agriculture: Comput.
  Electron. Agric., Dec 2024 (S0168169924009979).
- OOD methodology for yield prediction: Chakravarty 2026, TMLR (graded OOD, worst-case reporting).
- Concept-level shortcut audit: ICON decomposition, Rane et al., arXiv:2608.26083.
- Expert-index-as-prior inside a supervised probabilistic model: DK-VRN, Zhang et al. 2024, RSE 311:114252
  (drought, not yield — inspiration only).

What no prior work does, and what this paper is:
  N1. A CONTROLLED BASELINE LADDER B0->B6 on ONE public multi-country benchmark (CY-Bench), where
      each rung adds exactly one fusion idea on the same backbone / inputs / split / loss / seeds, so
      the fusion factor is isolated: B0 observations only; B1 cheap analytic indices (input); B2 late
      target fusion (process residual target); B3 LPJmL states as inputs (early); B4 transfer
      (pretrain on simulation, fine-tune); B5 alignment (contrastive sim<->obs, fine-tune); B6
      objective (agronomy loss term). The ladder spans both the SOURCE axis (cheap indices in B1 vs
      full process model LPJmL in B2-B6) and the FUSION STAGE (input / target / representation /
      alignment / objective). Prior work tests one rung at a time, on private or single-country data.
  N2. The IN-SEASON axis: every fusion is scored at origins 25/50/75/100 % of the season with
      LEAKAGE-SAFE process-model driving (process states truncated at the origin -> B3a; simulated
      final yield only via climatology-filled reruns -> B3b, clearly labelled; full-season sim is the
      B3c ORACLE, never headline). KGML/APSIM/WOFOST hybrid studies are end-of-season or use
      full-season simulations that silently see the future.
  N3. A TIERED PROTOCOL that ties fusion to trustworthiness, not just RMSE: calibration (quantile +
      conformal), cross-country transfer (transfer matrix) with worst-case reporting, label- AND
      driver-defined extremes, bad-year detection — all as fusion-vs-baseline DELTAS with year-block
      significance.
  N4. A MECHANISM link: ICON concept-reliance shift (non-mechanistic -> mechanistic) measured per
      fusion arm and correlated with the OOD/extreme gains. Nobody has asked whether process fusion
      changes WHAT the network relies on, or whether that predicts WHERE it generalizes.

The paper is N1+N2+N3; N4 is the explanatory closer. If N4 is cut for time the paper still stands.

### A.3 Pre-registered hypotheses (write these in the paper BEFORE results; each has a null outcome)

  H1 (source)   LPJmL input fusion (B3) beats analytic-index fusion (B1) on in-domain skill at every
                origin. Null: B1 ~ B3 -> "cheap indices suffice", a useful negative result for practitioners.
  H2 (stage)    Deeper fusion (B4 transfer / B5 alignment / B6 objective) beats input/target concat
                (B1/B2/B3 with the same signal) under cross-country transfer, but not in-domain.
                Null: concat is all you need (R4).
  H3 (origin)   Fusion gain is largest at early origins (25-50 %) and shrinks toward 0 at 100 %, where
                observed weather/RS already carries the season. Null: flat gain -> fusion adds a bias
                correction, not seasonal information.
  H4 (where)    Fusion gains are larger under shift (worst-country transfer, extreme years) than in-domain,
                and the ICON reliance shift toward WEATHER/LPJML predicts the per-country transfer gain.
                Null: no reliance shift -> fusion acts as a regularizer, not as knowledge.
  H5 (trust)    Fusion improves raw PICP / Winkler; conformal closes any remaining gap equally for all
                arms. Null: calibration is a property of the loss, not of the knowledge.

### A.4 Design decisions (fixed)

- D-ARMS. Seven baselines on a ladder, one fusion idea added at a time, PatchTST headline backbone:
    B0 BASELINE      observations (weather + RS + static + lags) only. The delta reference.
    B1 FAST FUSION   + cheap hand-computed analytic agronomy indices (GDD, RUE, assimilation index,
                       heat-stress) as extra inputs. In code.
    B2 LATE TARGET   set the target to (yield - calibrated LPJmL simulation), reusing the existing
       FUSION        residual-trend add-back path with the process signal in place of the OLS trend.
                       Helps only if the LPJmL prediction is any good. (Was F4c.)
    B3 EARLY FUSION  + LPJmL outputs as extra inputs. Three sub-baselines by how much of the future
                       the simulation is allowed to see:
                       B3a STATES-TO-ORIGIN   only the simulation's state up to the forecast date
                                              (leakage-free; needs no reruns)
                       B3b SIM-YIELD-CLIMFILL simulated final yield, rest of season filled by
                                              climatology (leakage-safe; needs climatology-filled reruns)
                       B3c SIM-YIELD-FULLSEASON  simulated final yield using the whole real season
                                              (ORACLE: it cheats by seeing the future; reported only
                                              as an upper ceiling, never headline)
    B4 TRANSFER      pretrain the backbone on the simulation (LPJmL, or index-derived) across all
       FUSION        simulated region-years, then fine-tune on observed CY-Bench yield (KGML-style).
                       Secondary variant: feature distillation from a frozen process-trained encoder.
    B5 ALIGNMENT     align simulated and CY-Bench data as a pretraining task via contrastive learning,
       FUSION        then fine-tune for the CYP task. (See R3: contrastive alignment can discard
                       view-specific anomaly variance, the inter-annual signal we care about; judge B5
                       specifically on anomaly/temporal skill, not only overall RMSE.)
    B6 OBJECTIVE     agronomy-specific loss as an additional loss term. Two variants:
       FUSION        B6a monotonicity penalty (dŷ/d heat_stress <= 0 beyond threshold, via input-
                       gradient penalty on the static heat-stress count);
                       B6b teacher-consistency: lambda * pinball(ŷ_q50, LPJmL_sim) with lambda swept.
  Combinations (B1+B4, B3a+B2, ...) are a secondary grid, run only after single arms are understood.

- D-CROP. Maize first, end to end. Wheat is the replication crop, run only on the arms that produce
  a finding in maize (B0, B1, B3a, best of B4/B5/B6). Crops are never pooled and never mixed. Verify
  whether CY-Bench mixes grain and silage maize across countries; if so, stratify or exclude.

- D-SPECIALIST. No pooling across countries. The model is a per-country, crop-specific SPECIALIST
  (single-country load, train-only y standardization, recursive lags). A fixed panel of 8 countries,
  chosen in D0 to span zones and data sizes (not all 31 — see compute budget), supplies the
  specialists for the transfer matrix (Tier 2b); the in-domain tiers run per country across the
  source countries.

- D-LPJML (feasibility, decided now so the data request is right the first time).
  LPJmL driven by observed weather up to day t depends only on weather up to t. So the STATE
  trajectory (LAI, above-ground biomass, phenological stage, soil-water, water stress, N stress,
  simulated daily NPP) truncated at the forecast origin is leakage-free with NO reruns -> B3a is
  always feasible. Simulated FINAL yield at origin f needs a rerun with climatology for the rest of
  the season -> B3b is feasible only if the provider can rerun; otherwise B3b is dropped and B3c is
  reported as a labelled oracle. Data to request from the LPJmL contact (checklist in C.P0.0):
  daily or dekadal states + final yield, same spatial support as CY-Bench admin units (or 0.5 deg
  grid + a crop-area-weighted admin aggregation), same calendar years as CY-Bench, the forcing
  dataset name, whether runs are irrigated/rainfed/mixed, and whether any CY-Bench statistics were
  used in calibration (if yes, every B2/B3/B4/B6 result is circular until a train-years-only
  calibration is redone — R2). Calibration METHOD matters as much as circularity: in the CY-Bench
  benchmark, LPJmL maize R2 moves from -0.31 (mean/variance) to +0.34 (affine OLS); pin the
  calibration (recommend affine OLS, fit on TRAIN years only) and report its sensitivity.

- D-CONFOUNDS for B1 vs B3 (RQ H1). LPJmL uses its own forcing and 0.5 deg support; CY-Bench uses
  AgERA5 at admin level. Any B3 gain may be forcing or resolution, not process. Control: add the
  LPJmL-forcing weather channels themselves as a B0' arm (if obtainable); if not, say the confound
  in the paper and limit H1 to "LPJmL as delivered vs indices as computed".

- D-EVAL. One protocol, five tiers, run on EVERY arm regardless of earlier-tier result (no gating —
  H2/H4 need arms that lose in-domain but win under transfer). One headline per tier. Every number is
  a DELTA vs B0 on identical backbone/split/seeds, with a CI. See Part B.

- D-SPLIT. Within each country: train years < T-5, val = last 2 train years, test = last 5 years
  (current fixed split). DECISION PENDING for the user: with 5 test years the year-block bootstrap
  has 5 resamplable units; if P0.4 shows B1 deltas inside the CI, switch the headline to
  walk-forward over the last 8 years (`generate_walk_forward_splits` exists) to buy power. Recursive
  lags ON for every out-of-sample number.

- D-SEEDS. 3 seeds for every cell; 5 for B0 and the headline arm. Report mean +/- std AND the paired test.

- D-WORDS. Lag yield and lat/lon are legitimate origin-available predictors, not "shortcuts". Call
  them NON-MECHANISTIC concepts; reserve "shortcut" for the cross-country OOD tier where they fail.

- D-PHYSICS-NAMING. `_compute_farquhar_series` (featureEngineering.py:257) is an FvCB C3 model with
  C3 constants (Kc=404.9, Gamma=42.75) applied to maize, a C4 crop. For the paper either (i) add a
  C4 variant for maize or (ii) rename the channel "radiation-temperature assimilation index" and
  drop the Farquhar label. Do not publish "Farquhar proxy" for maize.

---

## B. EVALUATION PROTOCOL v2 (what exists -> what to add)

Legend: HAVE = exists (file:line from v1 inventory, re-verify). ADD = net-new. FIX = exists but wrong/unsafe.

### B.0 Hardening before ANY number is reported
  [FIX] Quantile-crossing monotone sort after forward (tstLayer.py clamps >=0 at :1017,:1427, never sorts).
  [FIX] Expose `--quantiles` (default [0.1,0.5,0.9]; tstLayer.py:228, head :504).
  [HAVE] Recursive lags (`--use_recursive_lags`); ENFORCE for all headline runs.
  [HAVE] Trend fit on train years only (`TrendModel`, on_train_start); protocol refs train-only.
  [ADD] A single results schema row per (crop, arm, country, origin, seed, partition);
        helpers.py save_consolidated_results is the chokepoint — extend, don't fork.
  [ADD] Significance harness: paired year-block bootstrap (resample whole years, 1000 draws) for
        per-partition deltas; Wilcoxon signed-rank over the per-country deltas (the 8-country panel /
        transfer-matrix targets) for cross-country claims. A delta is a "win" only if the bootstrap
        CI excludes 0 AND the Wilcoxon p < 0.05.

### B.1 Tier 0 — In-domain accuracy          HEADLINE: skill-vs-trend (RMSE), co-report NRMSE
  [HAVE] NRMSE, R2, RMSE (validateModel.py); skill vs trend/OLS/climatology/rolling-3
         (evaluationProtocol.py `skill_score`, `ReferenceForecasts`).
  [DECIDED] Skill aggregated EQUAL-REGION (per-region skill then averaged within a country), reported
         per country with WORST-CELL (incl. worst region) — a high average must not hide a failing
         breadbasket. Per-country region-pooled skill kept only as benchmark-comparable support. An
         area-weighted national-total skill is deferred to an explicit food-supply claim. (see eval §0.C)
  [ADD]  Delta table vs B0 with CI; per origin; per country.
  Gate for the PAPER (not for advancing arms): B0 beats climatology and trend at 75-100 % origin,
  otherwise the substrate is broken and nothing downstream is interpretable.

### B.2 Tier 1 — Calibrated uncertainty      HEADLINE: PICP@80 raw -> conformal, then Winkler, then CRPS skill
  [HAVE] pinball, CRPS, PICP, PINAW, Winkler (MQL path); split-conformal CQR on val years
         (evaluationProtocol.py:286,:341); quantile coverage per tau (:252).
  [ADD]  CRPS skill vs trend+empirical-residual quantiles as a reference distribution (refs exist
         — verify `Reference-distribution comparisons` section in evaluation_protocol.md covers it).
  [ADD]  Reliability diagram per arm per origin (appendix).
  Rule: never compare sharpness (Winkler/PINAW) between arms whose PICP differs by > 5 pts.

### B.3 Tier 2 — Generalization under shift   HEADLINE: same Tier-0 skill-vs-trend + trend-referenced r_an,
                                             reported as MEAN and WORST-CELL per partition
  Partitions (strata, not new metrics):
    2a TEMPORAL   strict year-ahead (the fixed split). k-ahead is NOT a partition here (no covariates
                  -> trend + climatology; out of scope).
    2b SPATIAL    single-source TRANSFER MATRIX: specialist(X) scored on country Y for all pairs
                  in the 8-country panel, skill/anomaly units only. Cheap; needs no pooling.
                  Report spatial-transfer and anomaly-transfer separately (CY-Bench zero-shot: NRMSE
                  ~doubles, anomaly -> ~0, spatial partly retained). Difficulty axis from D0.2 shift
                  distance; zone names from D0.3.
    2c EXTREME    label-defined (anomaly < -1 sigma of train residuals) AND driver-defined (top-decile
                  heat-stress / bottom-decile RUE or cumulative water balance) years, POOLED, n stated,
                  plus the label-vs-driver agreement rate.
  [HAVE] r_sp, r_tm, r_an (spatiotemporal_metrics.py); OLS-detrended anomaly block in protocol.
  [FIX]  r_an referenced to the TRAIN-fitted trend (not train mean); `r_tm` never reported (3-5 yrs).
  [ADD]  transfer-matrix runner; shift-distance (domain-classifier AUC on static + seasonal-summary
         features, source countries vs each target); zone assignment (Koppen or GAEZ AEZ);
         extremes support counts; worst-cell table; delta-skill-vs-shift-distance curve.

### B.4 Tier 3 — Operational value             HEADLINE: bad-year AUC, plus earliest-useful-origin
  [HAVE] `bad_year_detection` (precision/recall/F1/AUC) (evaluationProtocol.py:261); tolerance
         accuracy within +/-10 % and +/-0.5 t/ha (:218).
  [ADD]  Earliest-useful-origin per country = first origin where bad-year AUC >= 0.7 (threshold
         fixed before results); per-arm.
  [ADD]  Shortfall capture C = mean(-â)/mean(-a) on the bottom-decile observed anomalies (trend-
         referenced), SUPPORT under the AUC headline: magnitude, not just ranking, of the bad tail.

### B.5 Tier 4 — Mechanism (ICON)              HEADLINE: conditional variance shares per concept group +
                                             unexplained share; reliance-shift vs Tier-2 gain correlation
  Concept groups: NON-MECHANISTIC = {SPATIAL lat/lon, CARRY-OVER lag_yield_k, MGMT calendar};
  MECHANISTIC = {WEATHER incl. cwb, SOIL, RS, LPJML}. PHYSICS indices are deterministic functions of
  WEATHER (featureEngineering.py:179-321) -> fold them INTO WEATHER for ICON; PHYSICS-vs-WEATHER
  conditional share is ~0 by construction and must not be interpreted.
  Cleanest ICON test is B4/B5/B6 vs B0 (identical inputs, different training): any reliance shift is
  purely a training effect.
  [ADD]  ICON port (authors' code, arXiv:2608.26083); layer-representation extractor (PatchTST
         `return_embeddings` exists; XLinear forward must be fixed first, R6); concept-annotation
         builder; per-layer runs; shift-vs-gain correlation across countries.
  [REPORT] unexplained share honestly (R8). Large = concept set inadequate or encoding nonlinear.

### B.6 Reporting rules
  - One main figure: per-tier delta vs B0 (y) vs forecast origin (x), one line per arm, one panel
    per tier {T0 skill, T1 Winkler@calibrated, T2 worst-country skill, T3 bad-year AUC}.
  - One main table: the transfer matrix (8x8) for B0 and the best arm, skill units.
  - Every cross-country number in skill/anomaly units; raw t/ha only in appendix.
  - "Within noise" is written as such; no bolding of insignificant best cells.
  - Appendix: full metric dump per evaluation_protocol.md; XLinear/CNN-LSTM robustness; wheat replication.

---

## C. EXPERIMENT CHECKLIST (priority-ordered; tick in place)

Priority key:  P0 = cannot write the paper without it   P1 = the core claim (N1-N3)
               P2 = strengthens acceptance (robustness, H4)   P3 = optional / second paper
Each line: [ ] id  what  -> deliverable  (depends on)

### P0 — SUBSTRATE AND FEASIBILITY (no results claimed from this block)
  [ ] P0.0  COMPUTE + DATA FEASIBILITY (do this week, before any build)
            - time ONE per-country specialist training run on the SLURM GPU partition;
              record min/run. Fill the budget table in C.END with real numbers.
            - send the LPJmL data request (D-LPJML checklist). Record: states available? reruns
              possible? forcing dataset? calibration used CY-Bench labels? spatial support?
            - verify grain-vs-silage maize per country in CY-Bench metadata.
            -> feasibility memo; go/no-go on B3b and on wheat.
  [ ] P0.1  D0 DATA CHARACTERIZATION (read-only, fast)
            - years per country per crop (maize: 42 countries, 31 with >=12 yr, 7 with <=6 yr;
              wheat: 29 / 26 / 2 — from years_dict.json). Sources = >=12 yr; transfer-only <=6 yr;
              drop 0-yr.
            - shift distance: domain-classifier AUC, source countries vs each country, on static +
              season summaries.
            - zone assignment (Koppen / AEZ) per country.
            - extremes support: label-extreme and driver-extreme year counts per country; agreement.
            - pick the 8-COUNTRY PANEL for specialists/transfer matrix: span zones, data sizes,
              include the hardest and the easiest target by shift distance.
            -> data-characterization table, shift ranking, zone map, panel list.
  [ ] P0.2  UNCERTAINTY HARDENING: quantile sort + `--quantiles` flag; sanity PICP on one NL run. (B.0)
  [ ] P0.3  PER-COUNTRY SPECIALIST DATAMODULE: confirm single-country load, train-only y
            standardization, trend-residual target, honest val/test split, recursive lags across the
            test years. (Single-country load exists; verify the trend-residual + recursive-lag path.)
  [ ] P0.4  B0 BASELINE, per-country specialists, maize, origins {25,50,75,100}, 5 seeds, MQL, through
            Tiers 0-1 and 2a. CHECK the power question (D-SPLIT): if seed-std + bootstrap CI on a
            trivial perturbation (e.g. +cwb) already spans 0, switch to walk-forward NOW.
            -> B0 reference table; split decision.
  [ ] P0.5  B0 SPECIALISTS on the 8-country panel, same origins, 3 seeds (if not already covered by
            P0.4). -> the transfer-matrix source models.
  [ ] P0.6  SIGNIFICANCE HARNESS (year-block bootstrap + Wilcoxon) wired to the results schema. (B.0)
  [ ] P0.7  Rename/replace the C3 "Farquhar" channel for maize (D-PHYSICS-NAMING) BEFORE B1 runs,
            so B1 numbers do not have to be regenerated.

### P1 — THE CORE CLAIM (N1-N3; H1, H3, H5)
  [ ] P1.1  B1 FAST FUSION, per-country, maize, origins x 3 seeds: variants {+gdd, +rue, +assim,
            +heat, +all} (mirror featureCache.py:333-342). -> Tier 0/1 deltas vs B0 per origin. (P0.4)
  [ ] P1.2  B3a STATES-TO-ORIGIN, per-country, maize, origins x 3 seeds. -> Tier 0/1 deltas; H1 vs B1. (P0.0 data)
  [ ] P1.3  B3b SIM-YIELD-CLIMFILL (only if reruns exist) and B3c ORACLE (labelled). -> value of the
            simulated final yield beyond states; the oracle ceiling.
  [ ] P1.4  TIER 2b TRANSFER MATRIX for B0, B1 (and B3a) specialists on the 8-panel.
            -> 8x8 skill matrices, asymmetry, hardest target, worst-country table, delta-skill vs
            shift-distance curve; no pooling plumbing needed. (P0.5, P0.1)
  [ ] P1.5  B2 LATE TARGET FUSION (target = y - calibrated LPJmL_sim via the residual add-back path).
            Cheapest LPJmL arm; large expected effect if LPJmL is any good; run before B4/B5/B6. (P0.0 data)
  [ ] P1.6  TIER 2c EXTREMES + TIER 3 BAD-YEAR for B0, B1, B3a from the SAME predictions (no new
            training): pooled label/driver extremes, bad-year AUC per origin, earliest-useful-origin,
            shortfall capture C on the bottom decile.
  [ ] P1.7  TIER 1 CONFORMAL for all P1 arms: PICP raw vs conformal; Winkler among calibrated arms. (P0.2)
  -> With P1 done the paper has: N1 (source axis), N2 (origin axis, leakage-safe), N3 (tiers 0-3).
     Writing starts here.

### P2 — STAGE AXIS AND MECHANISM (deeper-fusion rungs, N4; H2, H4)
  [ ] P2.1  B4 TRANSFER FUSION: pretrain PatchTST on LPJmL simulated yield over all simulated
            region-years (and, as a second source, on index-derived targets), fine-tune on observed.
            Per-country, maize, origins x 3 seeds. -> Tier 0-3 deltas; input-vs-representation on the
            same signal (vs B3a). (P1.2)
  [ ] P2.2  B5 ALIGNMENT FUSION: contrastive pretext aligning simulated and CY-Bench representations,
            then fine-tune for CYP. Per-country, maize, 3 seeds. -> Tier 0-3 deltas; judge on
            anomaly/temporal skill specifically (R3). (P2.1)
  [ ] P2.3  B6a MONOTONICITY PENALTY (heat-stress direction), lambda sweep {1e-3,1e-2,1e-1}, per-country,
            maize, 3 seeds. -> Tier 0-3 deltas; does a constraint help only under shift?
  [ ] P2.4  B6b TEACHER-CONSISTENCY to LPJmL_sim, lambda sweep, per-country, maize, 3 seeds. -> compare
            with B2 and B3a: three ways to use the same signal.
  [ ] P2.5  ICON PORT + concept annotations + extractor; run on B0, B1, B3a, B4, B2 per layer.
            -> reliance profiles; shift-vs-transfer-gain correlation across countries (H4). (P1.4, P2.1)
  [ ] P2.6  B0' FORCING CONTROL (LPJmL forcing weather as B0 input) if the forcing data is obtainable;
            else write the confound paragraph. (D-CONFOUNDS)
  [ ] P2.7  WHEAT REPLICATION on {B0, B1, B3a, best stage arm}, per-country, origins x 3 seeds, Tiers 0-3.
  [ ] P2.8  BACKBONE ROBUSTNESS: CNN-LSTM for {B0, B1, B3a, best arm}; XLinear only after the
            `return_embeddings` fix (R6) and only for input-stage arms.

### P3 — OPTIONAL / SECOND PAPER
  [ ] P3.1  Combination grid (B1+B4, B3a+B2, B3a+B4) — do fusions stack?
  [ ] P3.2  Pre-season (0 %) origin (data_fraction=0 + multi-year summaries + LPJmL climatology run).
  [ ] P3.3  Few-shot fine-tuning curves on transfer-only countries (<=6 yr) from a specialist/B4 model.
  [ ] P3.4  Feature-distillation B4 variant (frozen process-trained encoder) as a comparison to B5.
  [ ] P3.5  Specialists for every source country (beyond the 8-panel).

### C.END — COMPUTE BUDGET (fill from P0.0; the plan is not approved until these are numbers)
  Per-country specialist runs (maize), per country:
     B0(5 seeds) + B1x5 + B2 + B3a + B3b + B3c + B4x2 + B5 + B6a x3 + B6b x3 (3 seeds)
     = 5 + 3*(5+1+1+1+1+2+1+3+3) = 59 runs, x 4 origins = 236 runs per country.
  Transfer matrix:       8x8 = 64 scoring passes per arm (no training beyond the specialists above).
  Panel (specialists):   8 countries x headline arms {B0,B1,B3a} x 4 origins x 3 seeds for the matrix.
  Wheat replication:     ~4 arms x 3 seeds x 4 origins per country.
  Order of magnitude:    measure one run first (P0.0), then multiply by the panel size; the transfer
                         matrix is scoring-only and cheap. Cut origins on the deeper rungs first.

---

## D. RISKS (carry into the paper's limitations)
  R1  LPJmL future leakage: only B3a (states to origin) is leakage-free without reruns; B3b needs
      climatology-filled reruns; B3c is an oracle and is labelled so everywhere.
  R2  Circular calibration: if the LPJmL runs were calibrated on CY-Bench yields including test years,
      every B2/B3/B4/B6b number is contaminated. Ask; if yes, request a train-years-only recalibration.
      Also pin the calibration METHOD (affine OLS, train-only); it alone moves LPJmL R2 by ~0.65.
  R3  Contrastive alignment (B5) discards view-specific anomaly variance; judge B5 on anomaly/temporal
      skill, not overall RMSE, and keep B4 pretrain/distill as the safer representation default.
  R4  Concat is hard to beat; tune B1/B3 honestly; "deeper fusion fails to beat concat" is a result.
  R5  (retired) Pooling dropped — per-country specialists only. The generalist standardization /
      region-id risk no longer applies.
  R6  XLinear has no `return_embeddings`; any XLinear representation-stage result is an artifact until fixed.
  R7  r_tm is noise at 3-5 test years; never reported as headline. Add a permutation null (shuffle
      years within region) for any temporal/anomaly correlation before interpreting it.
  R8  ICON is linear (PLS); report the unexplained share; do not over-claim mechanism.
  R9  Statistical power: 5 test years x ~31 countries; expect many "within noise" cells for B1.
      Pre-commit the walk-forward fallback (D-SPLIT) so it is not a post-hoc choice.
  R10 Forcing/resolution confound in B1-vs-B3 (D-CONFOUNDS).
  R11 C3 photosynthesis constants on a C4 crop (D-PHYSICS-NAMING) — fix before publication.
  R12 Grain vs silage maize mixing across countries — verify in P0.0.

## E. REFERENCES (verified unless marked)
  Shahhosseini M., Hu G., Huber I., Archontoulis S.V. (2021). Coupling machine learning and crop modeling
    improves crop yield prediction in the US Corn Belt. Sci. Rep. 11:1606. doi:10.1038/s41598-020-80820-1
  Paudel D. et al. (2021). Machine learning for large-scale crop yield forecasting. Agric. Syst. 187:103016.
    doi:10.1016/j.agsy.2020.103016
  Feng P. et al. (2019). Dynamic wheat yield forecasts are improved by a hybrid approach using a biophysical
    model and machine learning. Agric. For. Meteorol. [verify exact title/volume]
  Liu L. et al. (2024). Knowledge-guided machine learning can improve carbon cycle quantification in
    agroecosystems. Nat. Commun. doi:10.1038/s41467-023-43860-5
  KGML county-level yield (2025). arXiv:2503.16328.  KGML overview (2024). arXiv:2403.15989.
  Knowledge informed hybrid machine learning in agricultural systems (review). Comput. Electron. Agric.,
    Dec 2024. sciencedirect S0168169924009979
  Chakravarty A. (2026). A Systematic Evaluation of OOD Generalization in Crop Yield Prediction. TMLR.
  Rane R.P. et al. (2026). ICON Decomposition. arXiv:2608.26083.
  Zhang M., Fernandez-Torres M.-A., Camps-Valls G. (2024). DK-VRN. RSE 311:114252.
  CY-Bench (2026). ESSD 18, 3997.
  Internal: `.backup_2026-10-09/research_question.md` (v1 inventory), evaluation_protocol.md,
    justifications.md, hermes_status.txt — all predate this plan; where they disagree, this file wins.
