# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description:
    Tabular-foundation-model baselines for CY-Bench in-season / end-of-season crop
    yield forecasting, built to be benchmarked HEAD-TO-HEAD against the sequence
    baselines in tstBaselines.py / cnnLSTMBaseline.py / linearBaselines.py.

    These models (TabPFN, TabICL, TabDPT, Mitra, LimiX -- all served through the
    TabTune library) are IN-CONTEXT-LEARNING TABULAR models: they consume a flat
    feature table (one row per region-year), not a (T, C) sequence. So we cannot
    feed them the same tensor as the sequence models. Fairness is therefore
    enforced at the INFORMATION + EVALUATION level, not the input-shape level:

      * SAME data, SAME train/val/test split, SAME forecast origin (data_fraction),
        SAME per-country specialist setup, SAME feature GROUPS -- loaded through the
        exact same DailyCYBenchSeqDataModule the sequence models use.
      * The season sequence is TABULARIZED into per-channel temporal summaries
        (binned means + mean/std/min/max/slope/last) computed over the OBSERVED
        window only, plus the static/lag features used verbatim. Granularity
        (--resolution) is the knob that trades columns for intra-season detail.
      * Every model emits the SAME prediction frame (region, year, y_true,
        q10/q50/q90) and is scored by the SAME hardened evaluation (process/metrics
        run_evaluation: references on train+val, skill ladder, conformal, tiers).

    Quantiles (the "MQL or similar" requirement) come three ways, all scored
    identically by the protocol:
      * TabPFN  -> native predictive distribution -> read quantiles.
      * TabICL  -> native pinball quantile head (== the sequence models' MQL).
      * TabDPT / Mitra / LimiX -> point regressors -> split-conformal
        (tabtune ConformalRegressor) to produce calibrated q10/q50/q90.
    The per-model route is recorded; "native learned quantiles vs conformal-from-
    point" is a reported axis, not a hidden difference.

--------------------
IMPORTANT -- TabTune API surface:
    The ONLY TabTune-specific logic lives in the clearly marked
    "========== TABTUNE ADAPTER ==========" section below (build / fit /
    predict_quantiles / embeddings). It is written against the documented
    TabTune 0.2.x API (TabularPipeline, task_type="regression", the uncertainty
    module's ConformalRegressor, episodic/SFT/PEFT fine-tuning). Method names may
    differ slightly in your installed version -- adjust ONLY inside that section;
    everything else (data, tabularization, recursive lags, evaluation hookup) is
    model-agnostic and verified against this repo.

Usage:
    # Zero-shot ICL, native quantiles, end-of-season:
    python tabBaselines.py --crop maize --country NL --model_type tabicl \
        --forecast_type end-of-season --test_years 5 --lag_years 0 \
        --results_dir checkpoints-test/results --wandb_project test-and-delete-later

    # Fine-tuned TabPFN, mid-season, recursive lags, extract embeddings:
    python tabBaselines.py --crop maize --country NL --model_type tabpfn \
        --tab_strategy finetune --forecast_type middle-of-season \
        --lag_years 2 --use_recursive_lags --extract_embeddings \
        --results_dir checkpoints/results

    # Point model with conformal intervals (TabDPT / Mitra / LimiX):
    python tabBaselines.py --crop maize --country NL --model_type limix \
        --forecast_type end-of-season --test_years 5 --lag_years 0
"""

# %% Loading libraries
import os
import sys
import copy
import random
import argparse
import logging
import uuid

from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import torch

# CY-BENCH Dependencies
import cybench.config
from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_TYPE, set_forecast_type, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES,
    KEY_CROP_SEASON, CROP_CALENDAR_DATES,
)

# Load the full season (SOS->EOS); the forecast origin is applied afterwards via
# data_fraction, exactly as in tstBaselines.py.
cybench.config.FORECAST_LEAD_TIME = "0-days"

from cybench.process.alignment_patch import patch_alignment
patch_alignment()

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

# Loading custom functions and classes (same modules the sequence baselines use)
sys.path.append('../../process/')
from metrics import print_metrics_table
from loadData import calculate_fixed_split, DailyCYBenchSeqDataModule
from alignment_patch import verify_forecast_horizon_config
from metrics import run_evaluation, ALL_EVALUATION_TIERS, point_metrics
from helpers import generate_checkpoint_name

sys.path.append('../../architectures/')
from modelconfig import TSTModelConfig

# Models that natively produce quantiles / a predictive distribution.
NATIVE_QUANTILE_MODELS = {"tabpfn", "tabicl"}
# Point regressors that need split-conformal to produce intervals.
POINT_MODELS = {"tabdpt", "mitra", "limix"}
ALL_TAB_MODELS = sorted(NATIVE_QUANTILE_MODELS | POINT_MODELS)

# Named resolutions -> number of temporal bins per channel in the feature table.
# The raw sequence is always loaded at DAILY resolution (365 steps), so every option
# fits comfortably; yearly=1 is a single whole-season value per channel.
RESOLUTION_BINS = {"weekly": 52, "dekad": 36, "monthly": 12, "yearly": 1}

# TabTune's internal model identifiers (VERIFY against your installed version;
# TabTune exposes these names through its registry / TabularPipeline(model=...)).
TABTUNE_MODEL_NAME = {
    "tabpfn": "TabPFNv35Fast",
    "tabicl": "TabICLv2",
    "tabdpt": "TabDPT",
    "mitra":  "MitraV2",
    "limix":  "LimiX",
}


# =============================================================================
# ====================  SEQUENCE -> TABLE  (model-agnostic)  ==================
# =============================================================================
def _masked_channel_summary(series: np.ndarray, mask: np.ndarray,
                            n_bins: int) -> Dict[str, float]:
    """Summarise ONE channel of ONE sample over its OBSERVED timesteps.

    series: (T,) raw values for the channel.
    mask:   (T,) bool, True where the timestep is observed (not padding / not
            past the forecast origin). Padding positions are excluded so the
            summary is bounded by the forecast origin, exactly like the sequence
            models' observed_mask.
    Returns binned means + whole-window mean/std/min/max/slope/last. Empty bins
    or an all-missing channel yield NaN, which TabTune's DataProcessor imputes.
    """
    obs = mask.astype(bool)
    vals = series[obs]
    out: Dict[str, float] = {}
    # Equal-width temporal bins over the observed span (preserves intra-season
    # dynamics; n_bins is the columns-vs-detail knob).
    idx = np.where(obs)[0]
    if idx.size == 0:
        for k in range(n_bins):
            out[f"bin{k}"] = np.nan
        out.update(mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                   slope=np.nan, last=np.nan)
        return out
    lo, hi = idx.min(), idx.max()
    edges = np.linspace(lo, hi + 1, n_bins + 1)
    for k in range(n_bins):
        b0, b1 = edges[k], edges[k + 1]
        sel = obs.copy()
        pos = np.arange(series.shape[0])
        sel &= (pos >= b0) & (pos < b1)
        out[f"bin{k}"] = float(np.nanmean(series[sel])) if sel.any() else np.nan
    out["mean"] = float(np.nanmean(vals))
    out["std"] = float(np.nanstd(vals))
    out["min"] = float(np.nanmin(vals))
    out["max"] = float(np.nanmax(vals))
    # Slope of a least-squares line over the observed index (intra-season trend).
    if vals.size >= 2:
        xi = idx.astype(float)
        out["slope"] = float(np.polyfit(xi, vals, 1)[0])
    else:
        out["slope"] = np.nan
    out["last"] = float(vals[-1])
    return out


def tabularize(X_ts: np.ndarray, X_static: np.ndarray, masks: Optional[np.ndarray],
               ts_names: List[str], static_names: List[str],
               n_bins: int) -> Tuple[np.ndarray, List[str]]:
    """Convert (N, T, C) sequences + (N, S) static features into a flat (N, D) table.

    Temporal channels -> per-channel summaries (binned + mean/std/min/max/slope/last).
    Static/lag features -> used verbatim as columns.
    Column order is deterministic so train/val/test/transfer tables align.
    """
    N, T, C = X_ts.shape
    assert C == len(ts_names), f"ts channel count {C} != names {len(ts_names)}"
    mask_arr = masks if masks is not None else np.ones((N, T), dtype=bool)

    stat_keys = ["mean", "std", "min", "max", "slope", "last"]
    # Build column names once.
    cols: List[str] = []
    for name in ts_names:
        cols += [f"{name}__bin{k}" for k in range(n_bins)]
        cols += [f"{name}__{s}" for s in stat_keys]
    cols += [f"static__{s}" for s in static_names]

    rows = np.empty((N, len(cols)), dtype=np.float32)
    for i in range(N):
        feat: List[float] = []
        for c, name in enumerate(ts_names):
            summ = _masked_channel_summary(X_ts[i, :, c], mask_arr[i], n_bins)
            feat += [summ[f"bin{k}"] for k in range(n_bins)]
            feat += [summ[s] for s in stat_keys]
        # Static features straight through (lag_yield_k, soil, lat/lon, summaries).
        feat += list(X_static[i, :len(static_names)])
        rows[i, :] = feat
    return rows, cols


# =============================================================================
# ==========================  TABTUNE ADAPTER  ================================
#  The ONLY TabTune-version-specific code. Adjust method names here if your
#  installed TabTune differs; the rest of the file is model-agnostic.
# =============================================================================
def tabtune_build_and_fit(model_key: str, strategy: str,
                          X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: Optional[pd.DataFrame], y_val: Optional[np.ndarray],
                          quantiles: List[float], seed: int,
                          epochs: int, lr: float, batch_size: int):
    """Build a TabTune regressor for `model_key` and fit/adapt it on CyBench.

    strategy:
      'inference' -> zero-shot ICL (fit just stores the training context).
      'finetune'  -> supervised fine-tuning on train (turn_by_turn episodic for
                     most backbones; native for TabPFN). No val-based early
                     stopping in TabTune's turn_by_turn path.

    Returns a fitted TabTune pipeline exposing .predict() (and, for native-
    quantile models, quantile output). VERIFY the kwargs against your TabTune.
    """
    from tabtune import TabularPipeline  # TabTune 0.2.1

    tabtune_name = TABTUNE_MODEL_NAME[model_key]

    # Limix's episodic finetune hard-requires n >= support_size + 2 (default
    # support_size=256) and does NOT auto-shrink episodes like TabDPT/Mitra do.
    # On small per-country splits (tens of region-years) it raises instead of
    # training, so fall back to inference with a clear warning — a degraded
    # 65-row episodic SFT would be noise anyway, and the benchmark still gets a
    # Limix row via zero-shot ICL.
    LIMIX_MIN_FINETUNE_ROWS = 258  # 256 support + 2
    n_train = int(len(X_train))
    if model_key == "limix" and strategy != "inference" and n_train < LIMIX_MIN_FINETUNE_ROWS:
        print(f"[TabTune][WARN] Limix episodic {strategy} needs >= {LIMIX_MIN_FINETUNE_ROWS} "
              f"train rows (have {n_train}); falling back to strategy='inference'.")
        strategy = "inference"

    # TabTune 0.2.1: tuning knobs live in tuning_params (validated against
    # TuningConfig), NOT as flat constructor kwargs. Quantiles are requested at
    # predict time via predict_quantiles(), not at construction.
    tuning_params = {
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "seed": seed,
    }
    pipeline = TabularPipeline(
        model_name=tabtune_name,
        task_type="regression",
        tuning_strategy=strategy,           # 'inference' | 'finetune'
        tuning_params=tuning_params,
    )
    # TabTune's DataProcessor handles scaling / imputation / encoding internally.
    # fit(X, y) only — no external validation set in 0.2.1; y must be a Series.
    # The held-out (X_val, y_val) is used downstream (conformal + reference
    # forecasts), not for TabTune fine-tuning.
    pipeline.fit(X_train, pd.Series(np.asarray(y_train).ravel()))
    return pipeline


class _ConformalBands:
    """Split-conformal wrapper: base pipeline's point prediction (q50) plus a
    calibrated absolute-residual interval (q_lo, q_hi). Holds BOTH the fitted
    base pipeline (for the median) and the calibrated ConformalRegressor (for the
    interval), since TabTune's ConformalRegressor only exposes predict_interval().
    """
    def __init__(self, pipeline, conf, quantiles: List[float]):
        self.pipeline = pipeline
        self.conf = conf
        self.quantiles = list(quantiles)

    def predict_bands(self, X: pd.DataFrame):
        lower, upper = self.conf.predict_interval(X)          # 1-alpha central interval
        median = np.asarray(self.pipeline.predict(X), dtype=float).ravel()
        return (np.asarray(lower, dtype=float).ravel(),
                median,
                np.asarray(upper, dtype=float).ravel())


def tabtune_fit_conformal(pipeline, X_cal: pd.DataFrame, y_cal: np.ndarray,
                          quantiles: List[float]):
    """Wrap a POINT regressor with split-conformal to emit calibrated quantiles.

    TabTune 0.2.1 API: ConformalRegressor(pipeline, method='absolute', alpha=...)
    then .calibrate(X_cal, y_cal); intervals via .predict_interval(X) -> (lo, hi).
    alpha is set so the central interval covers (max(q) - min(q)); e.g. for
    [0.1, 0.5, 0.9] -> alpha = 1 - 0.8 = 0.2 (an 80% interval = q10..q90).
    Calibrated on the VALIDATION split only.
    """
    from tabtune.uncertainty import ConformalRegressor
    qs = sorted(quantiles)
    alpha = float(1.0 - (qs[-1] - qs[0]))
    conf = ConformalRegressor(pipeline, method="absolute", alpha=alpha)
    conf.calibrate(X_cal, pd.Series(np.asarray(y_cal).ravel()))
    return _ConformalBands(pipeline, conf, quantiles)


def tabtune_predict_quantiles(model_obj, model_key: str, X: pd.DataFrame,
                              quantiles: List[float]) -> np.ndarray:
    """Return a (N, len(quantiles)) array of predicted quantiles in ORIGINAL scale.

    Native-quantile models return learned quantiles; conformal-wrapped point
    models return q_lo (predict_interval lower), q50 (base point prediction) and
    q_hi (upper). Quantiles are sorted so q10<=q50<=q90.
    """
    if model_key in NATIVE_QUANTILE_MODELS:
        # TabTune 0.2.1: predict_quantiles returns dict {level: array(N,)}.
        preds = model_obj.predict_quantiles(X, quantiles=quantiles)
        # Match requested levels to dict keys tolerantly (avoid float-key KeyErrors).
        keys = list(preds.keys())
        cols = []
        for ql in quantiles:
            k = min(keys, key=lambda kk: abs(float(kk) - float(ql)))
            cols.append(np.asarray(preds[k], dtype=float))
        q = np.column_stack(cols)
    else:
        # Conformal point model: one calibrated interval + the base median.
        # Map the lowest requested quantile -> lower bound, highest -> upper
        # bound, everything in between -> median. Exact for the 3-quantile
        # default [0.1, 0.5, 0.9]; approximate (median) for extra inner levels.
        lo, med, hi = model_obj.predict_bands(X)
        qs = sorted(quantiles)
        q_lo, q_hi = qs[0], qs[-1]
        cols = []
        for ql in quantiles:
            if ql == q_lo:
                cols.append(lo)
            elif ql == q_hi:
                cols.append(hi)
            else:
                cols.append(med)
        q = np.column_stack(cols)
    if q.ndim == 1:  # safety: a single column returned
        q = np.repeat(q[:, None], len(quantiles), axis=1)
    q = np.sort(q, axis=1)  # enforce monotone quantiles (protects PICP/CRPS/Winkler)
    return q


def tabtune_embeddings(pipeline, model_key: str, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract per-row embeddings (N, d). Returns None if unavailable.

    VERIFY: TabPFN exposes get_embeddings / a TabPFNEmbedding transformer; TabICL/
    TabDPT/Mitra/LimiX expose encoder representations via TabTune. Kept best-effort
    so a missing accessor degrades gracefully instead of crashing the run.
    """
    try:
        emb = pipeline.get_embeddings(X)  # VERIFY method name
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Embeddings] extraction unavailable for {model_key}: {e}")
        return None


# =============================================================================
# =====================  PREDICTION-FRAME ADAPTER  ===========================
#  Exposes the exact interface process/metrics.run_evaluation expects from a
#  model: ._per_year_preds, .predict(batch), .quantiles, .loss_type, .device,
#  .eval()/.train(). This lets the tab models reuse the hardened evaluation
#  (references on train+val, skill ladder, conformal, tiers, transfer) UNCHANGED.
# =============================================================================
def impute_with_train_means(X: pd.DataFrame, train_means: Optional[pd.Series]) -> pd.DataFrame:
    """Fill NaN with the TRAIN per-column means (fallback 0 for all-NaN columns).

    This reproduces exactly what the sequence baselines do: they normalize static/
    temporal inputs with the train mean/std and then torch.nan_to_num(nan=0.0),
    which is arithmetically identical to imputing NaN with the train mean before
    normalization. Multi-year lookback features (prev_year_*_Tminus1/2, *_trend_2year)
    are genuinely undefined at the start of the record and for sporadic per-region
    gaps; mean-imputation is the leakage-safe, pipeline-consistent handling, and
    TabICL's internal standardization then places these cells at the neutral 0
    position -- the same spot the sequence models' nan_to_num(0) produces.
    """
    if train_means is None:
        return X.fillna(0.0)
    return X.fillna(train_means).fillna(0.0)


class TabModelAdapter:
    def __init__(self, model_obj, model_key: str, quantiles: List[float],
                 ts_names: List[str], static_names: List[str], n_bins: int,
                 y_mean: float, y_std: float,
                 train_means: Optional[pd.Series] = None,
                 loss_type: str = "mql"):
        self.model_obj = model_obj          # fitted pipeline or conformal wrapper
        self.model_key = model_key
        self.quantiles = list(quantiles)
        self.loss_type = loss_type          # "mql" -> uncertainty tier; "mse" -> point only
        self.ts_names = ts_names
        self.static_names = static_names
        self.n_bins = n_bins
        self.train_means = train_means      # fit on TRAIN only (no leakage)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self._median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        self._per_year_preds: Dict[int, Dict] = {}
        # Lightning-compatibility stubs used by collect_predictions / transfer.
        self.device = torch.device("cpu")
        self.training = False
        self.trainer = None

    # --- Lightning no-ops so the evaluation helpers can call them safely ---
    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def _tab(self, x_ts: np.ndarray, x_static: np.ndarray,
             masks: Optional[np.ndarray]) -> pd.DataFrame:
        rows, cols = tabularize(x_ts, x_static, masks, self.ts_names,
                                self.static_names, self.n_bins)
        df = pd.DataFrame(rows, columns=cols)
        # Same train-mean imputation applied to X_train/X_val at fit time, so
        # test/transfer/embedding frames are handled identically (no leakage:
        # means were fit on TRAIN only).
        return impute_with_train_means(df, self.train_means)

    @torch.no_grad()
    def predict(self, batch):
        """Sequence-batch -> dict, matching BaseTimeSeriesModel.predict so that
        collect_predictions (val conformal) and collect_transfer_predictions
        (zero-shot cross-country) work without modification.

        Predictions are in ORIGINAL yield scale (the tab model is fit on original
        yields), so no denormalization is needed; targets are recovered from the
        normalized y_z using the datamodule's stats that produced the batch.
        """
        x_ts, x_static, y_z, years, adm_ids, lats, lons, mask = batch
        x_ts_np = x_ts.detach().cpu().numpy()
        x_static_np = x_static.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy() if mask is not None else None
        X = self._tab(x_ts_np, x_static_np, mask_np)
        q = tabtune_predict_quantiles(self.model_obj, self.model_key, X, self.quantiles)
        q = np.clip(q, a_min=0.0, a_max=None)  # yields are non-negative
        targets_orig = y_z.detach().cpu().numpy() * self.y_std + self.y_mean
        if self.loss_type == "mse":
            # Point-model mode: expose only the median so run_evaluation treats
            # this as a point forecast (accuracy tier only; uncertainty skipped).
            med = q[:, self._median_idx]
            pred_tensor = torch.tensor(med, dtype=torch.float32)            # (B,)
        else:
            pred_tensor = torch.tensor(q, dtype=torch.float32)              # (B, Q)
        return {
            "predictions": pred_tensor,
            "predictions_z": pred_tensor,
            "targets": torch.tensor(targets_orig, dtype=torch.float32),     # (B,)
            "years": years,
            "adm_ids": adm_ids,
            "lats": lats,
            "lons": lons,
        }


def _static_lag_indices(static_names: List[str], lag_years: int) -> List[Tuple[int, int]]:
    """(lag_k, index-in-static) for lag_yield_k columns, used for recursive lags."""
    out = []
    for lag in range(1, lag_years + 1):
        name = f"lag_yield_{lag}"
        if name in static_names:
            out.append((lag, static_names.index(name)))
    return out


def build_per_year_preds(adapter: TabModelAdapter, test_ds, test_years: set,
                         lag_years: int, use_recursive_lags: bool) -> Dict[int, Dict]:
    """Produce the test prediction frame, honouring recursive lags YEAR-MAJOR so a
    test year is predicted only after all strictly-earlier test years are cached
    (same discipline the sequence models now enforce)."""
    X_ts = test_ds.X_ts.detach().cpu().numpy()
    X_static = test_ds.X_static.detach().cpu().numpy().copy()  # mutated for recursive lags
    masks = test_ds.validity_masks.detach().cpu().numpy() if test_ds.validity_masks is not None else None
    years = test_ds.years.detach().cpu().numpy().astype(int)
    adm_ids = list(test_ds.adm_ids)
    y_orig = test_ds.y.detach().cpu().numpy() * adapter.y_std + adapter.y_mean

    static_names = adapter.static_names
    lag_idx = _static_lag_indices(static_names, lag_years) if (use_recursive_lags and lag_years > 0) else []
    # Original-scale static mean/std for the lag columns (static features are stored
    # RAW, i.e. in original yield units for lag_yield_k), so cached original-scale
    # predictions drop straight in.
    cache: Dict[Tuple[str, int], float] = {}

    per_year: Dict[int, Dict] = {}
    for Y in sorted(set(int(y) for y in years)):
        sel = np.where(years == Y)[0]
        if sel.size == 0:
            continue
        Xs = X_static[sel].copy()
        # Recursive-lag fill: overwrite within-test-window lag columns with cached preds.
        if lag_idx:
            for row_pos, i in enumerate(sel):
                for lag, col in lag_idx:
                    ly = Y - lag
                    if ly in test_years:
                        key = (adm_ids[i], ly)
                        Xs[row_pos, col] = cache.get(key, 0.0)  # 0 == mean before any cache (rare)
        X = adapter._tab(X_ts[sel], Xs, masks[sel] if masks is not None else None)
        q = tabtune_predict_quantiles(adapter.model_obj, adapter.model_key, X, adapter.quantiles)
        q = np.clip(q, a_min=0.0, a_max=None)
        med = q[:, adapter._median_idx]
        per_year[Y] = {
            "preds": med.tolist(),
            "targets": y_orig[sel].tolist(),
            "adm_ids": [adm_ids[i] for i in sel],
            "quantile_preds": q.tolist(),
        }
        for row_pos, i in enumerate(sel):
            cache[(adm_ids[i], Y)] = float(med[row_pos])
    return per_year


# =============================================================================
# ================================  MAIN  =====================================
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CY-Bench tabular-foundation-model baselines (TabTune) with Temporal (Fixed) Split")
    parser.add_argument('--crop', default="maize")
    parser.add_argument('--country', default="NL")
    parser.add_argument('--model_type', default="tabicl", choices=ALL_TAB_MODELS,
                        help="Tabular foundation model backbone (all served by TabTune).")
    parser.add_argument('--tab_strategy', default="inference",
                        choices=['inference', 'finetune'],
                        help="inference=zero-shot ICL; finetune=supervised fine-tuning "
                             "(turn_by_turn episodic; TabPFN uses its native mode). "
                             "(PEFT/LoRA is not supported for regression by these backbones.)")
    parser.add_argument('--forecast_type', default="end-of-season",
                        choices=['end-of-season', 'three-quarter-of-season',
                                 'middle-of-season', 'quarter-of-season'])
    parser.add_argument('--test_years', type=int, default=5)
    # Feature config (mirrors tstBaselines so the information set matches)
    parser.add_argument('--use_sota_features', action='store_true')
    parser.add_argument('--include_spatial_features', action='store_true')
    parser.add_argument('--lag_years', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--use_recursive_lags', action='store_true',
                        help="REQUIRED when --lag_years>0: fill within-test-window lag "
                             "features with the model's own predictions (year-major), "
                             "preventing observed-test-yield leakage.")
    parser.add_argument('--use_cwb_feature', action='store_true')
    parser.add_argument('--drop_tavg', action='store_true')
    parser.add_argument('--use_gdd', action='store_true')
    parser.add_argument('--use_heat_stress_days', action='store_true')
    parser.add_argument('--use_rue', action='store_true')
    parser.add_argument('--use_rtai', action='store_true')
    parser.add_argument('--use_aez', action='store_true',
                        help='Add AEZ crop suitability code as a static feature (1 float). '
                             'Requires pre-computed acz_labels_{crop}.csv.')
    parser.add_argument('--use_koppen', action='store_true',
                        help='Add Köppen decomposition as 3 static features. '
                             'Requires pre-computed koppen_labels.csv.')
    parser.add_argument('--multi_year_summaries', action='store_true')
    parser.add_argument('--multi_year_window', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--multi_year_features', nargs='+', default=['weather'],
                        choices=['weather', 'remote_sensing', 'phenology', 'all'])
    # Tabularization — the ONE temporal knob for these models
    parser.add_argument('--resolution', default='monthly',
                        choices=['weekly', 'dekad', 'monthly', 'yearly'],
                        help="Temporal resolution of the FEATURE TABLE. Each channel's "
                             "observed season is summarised into this many bins: "
                             "weekly (~52), dekad (~36, 10-day periods), monthly (~12), or "
                             "yearly (1 whole-season value). This is the ONLY temporal knob "
                             "for the tabular models: unlike the sequence baselines there is "
                             "no separate --aggregation, because the tab model never sees the "
                             "sequence, only this table. The raw series is always loaded at "
                             "DAILY resolution internally (the finest source, for the most "
                             "faithful summaries) and then binned to --resolution. Finer "
                             "(weekly) = more intra-season detail but more columns; coarser "
                             "(yearly) = fewer, more robust features. Default: monthly.")
    # Quantiles / uncertainty
    parser.add_argument('--quantiles', nargs='+', type=float, default=[0.1, 0.5, 0.9],
                        help="Must include 0.5. Native for TabPFN/TabICL; conformal for point models.")
    parser.add_argument('--loss_type', default='mql', choices=['mql', 'mse'],
                        help="mql (default): pinball loss, emits q10/q50/q90, runs uncertainty tier. "
                             "mse: expose only the median prediction; uncertainty tier is skipped. "
                             "The underlying TabTune model is unchanged -- mse simply discards the "
                             "interval and scores as a point forecast.")
    # Fine-tuning knobs (ignored by --tab_strategy inference)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Fine-tuning learning rate (default 1e-5, TabTune's recommended "
                             "rate for these pretrained backbones; higher risks forgetting).")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    # Embeddings
    parser.add_argument('--extract_embeddings', action='store_true',
                        help="Extract per-row embeddings for train/val/test and save to .npz.")
    parser.add_argument('--embeddings_dir', default=None,
                        help="Where to save embeddings (default: <results_dir>/embeddings).")
    # Output / tracking (mirrors tstBaselines)
    parser.add_argument('--results_dir', default='checkpoints/results')
    parser.add_argument('--save_checkpoint_dir', default='checkpoints-test',
                        help='Directory to save the fitted TabTune pipeline (default: checkpoints-test). '
                             'Set to "" to disable saving.')
    parser.add_argument('--wandb_project', default=None)
    parser.add_argument('--wandb_run_name', default=None)
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--arm', default='B0', help="Fusion-ladder / regime label written to results rows.")
    parser.add_argument('--model_name', default=None, help="Display name in results rows (default: model_type).")
    parser.add_argument('--evaluation_tier', nargs='+',
                        default=['accuracy', 'uncertainity', 'generalization', 'operational'],
                        choices=['accuracy', 'uncertainity', 'generalization', 'operational',
                                 'cross_country_transfer'])
    parser.add_argument('--transfer_countries', nargs='+', default=None)
    parser.add_argument('--generate_predictions', action='store_true')
    args = parser.parse_args()

    # Lag-leakage guard (identical policy to tstBaselines).
    if args.lag_years > 0 and not args.use_recursive_lags:
        parser.error(
            f"\n[LAG LEAKAGE GUARD] --lag_years={args.lag_years} requires --use_recursive_lags.\n"
            f"  Without it, OBSERVED test-year yields enter lag features and invalidate\n"
            f"  the out-of-sample evaluation. Add --use_recursive_lags or set --lag_years 0."
        )

    # Forecast origin -> data_fraction (first fraction of SOS->EOS), same mapping as tst.
    set_forecast_type("0-days")
    forecast_to_fraction = {
        'end-of-season': 1.0, 'three-quarter-of-season': 0.75,
        'middle-of-season': 0.5, 'quarter-of-season': 0.25,
    }
    data_fraction = forecast_to_fraction[args.forecast_type]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id if args.run_id else str(uuid.uuid4())[:8]
    model_name = args.model_name if args.model_name else args.model_type

    q_route = "native" if args.model_type in NATIVE_QUANTILE_MODELS else "conformal"
    print(f"\n{'=' * 70}")
    print(f"CY-BENCH TAB | {args.model_type.upper()} ({args.tab_strategy}) | "
          f"{args.crop}-{args.country} | {args.forecast_type}")
    print(f"quantile route: {q_route}  | resolution={args.resolution} (daily source) | "
          f"lag_years={args.lag_years} (recursive={args.use_recursive_lags})")
    print(f"{'=' * 70}\n")

    # Reuse TSTModelConfig so the datamodule + run_evaluation see the exact same
    # configuration object the sequence baselines use (loss_type='mql' -> quantiles).
    config = TSTModelConfig(
        crop=args.crop, country=args.country,
        model_type=args.model_type, aggregation="daily",  # always summarise from the daily source
        data_fraction=data_fraction,
        use_sota_features=args.use_sota_features,
        include_spatial_features=args.include_spatial_features,
        lag_years=args.lag_years,
        seed=args.seed, batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.epochs,            # record actual fine-tune epochs in hyperparameters.csv
        lr=args.lr,                        # record actual fine-tune LR in hyperparameters.csv
        weight_decay=0.0,                  # TabTune does not expose weight_decay; record as 0
        test_years=args.test_years,
        use_residual_trend=False,          # trend is a sequence-model add-back; N/A here
        use_cwb_feature=args.use_cwb_feature,
        drop_tavg=args.drop_tavg,
        use_recursive_lags=args.use_recursive_lags,
        use_gdd=args.use_gdd,
        use_heat_stress_days=args.use_heat_stress_days,
        use_rue=args.use_rue,
        use_rtai=args.use_rtai,
        use_aez=args.use_aez,
        use_koppen=args.use_koppen,
        multi_year_summaries=args.multi_year_summaries,
        multi_year_window=args.multi_year_window,
        multi_year_features=args.multi_year_features,
        results_dir=args.results_dir,
        loss_type=args.loss_type,
        quantiles=args.quantiles,
        arm=args.arm,
        model_name=model_name,
    )
    verify_forecast_horizon_config(config, resolution_bins=RESOLUTION_BINS,
                                   active_resolution=args.resolution)
    print(f"[Feature Config] Time series vars ({len(config.time_series_vars)}): {config.time_series_vars}")

    # ---- Data: same loader, same split as the sequence baselines ----
    df_y, dfs_x = load_dfs_crop(config.crop, [config.country])
    if df_y is None or len(df_y) == 0:
        print(f"[ERROR] No data for {config.crop}-{config.country}")
        sys.exit(1)
    ds = CYDataset(config.crop, df_y, dfs_x)
    all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))
    fixed_splits = calculate_fixed_split(all_years, test_years=args.test_years, val_years=2)
    print(f"[Split] Train {sorted(fixed_splits['train_years'])} | "
          f"Val {sorted(fixed_splits['val_years'])} | Test {sorted(fixed_splits['test_years'])}")

    dm = DailyCYBenchSeqDataModule(config)
    dm.setup(train_years=fixed_splits['train_years'],
             val_years=fixed_splits['val_years'],
             test_years=fixed_splits['test_years'])

    # Record the TABULARIZATION resolution (weekly/dekad/monthly/yearly) in the
    # config so hyperparameters.csv reflects what the model actually sees, not the
    # internal "daily" load resolution used by the datamodule.
    # The DM has already read config.aggregation for its setup, so this only
    # affects downstream CSV logging — it does not alter any data tensors.
    config.aggregation = args.resolution
    # Log tab-specific fields so they appear in output CSVs (set dynamically
    # because TSTModelConfig is shared with the sequence baselines).
    config.tab_strategy = args.tab_strategy
    config.resolution = args.resolution

    ts_names = dm._get_ts_feature_names()
    static_names = dm._get_static_feature_names()

    # Resolve the named resolution to a bin count. The sequence is loaded at daily
    # resolution (365 steps), so every option fits; the min() is a harmless safety clamp.
    seq_len = int(dm.train_ds.X_ts.shape[1])
    n_bins = min(RESOLUTION_BINS[args.resolution], seq_len)
    print(f"[Tabularize] resolution={args.resolution} -> {n_bins} bins/channel "
          f"(daily source, seq_len={seq_len}).")

    # ---- Tabularize train / val (val = fine-tune selection + conformal calibration) ----
    def arrays(dset):
        X_ts = dset.X_ts.detach().cpu().numpy()
        X_st = dset.X_static.detach().cpu().numpy()
        m = dset.validity_masks.detach().cpu().numpy() if dset.validity_masks is not None else None
        y = dset.y.detach().cpu().numpy() * dm.y_std + dm.y_mean   # ORIGINAL scale
        return X_ts, X_st, m, y

    Xtr_ts, Xtr_st, Mtr, ytr = arrays(dm.train_ds)
    Xva_ts, Xva_st, Mva, yva = arrays(dm.val_ds)
    tr_rows, cols = tabularize(Xtr_ts, Xtr_st, Mtr, ts_names, static_names, n_bins)
    va_rows, _ = tabularize(Xva_ts, Xva_st, Mva, ts_names, static_names, n_bins)
    X_train = pd.DataFrame(tr_rows, columns=cols)
    X_val = pd.DataFrame(va_rows, columns=cols)
    print(f"[Tabularize] feature table: {X_train.shape[1]} columns "
          f"(train={X_train.shape[0]}, val={X_val.shape[0]} rows)")
    # Train-mean imputation (fit on TRAIN only). Multi-year lookback features
    # (prev_year_*_Tminus1/2, *_trend_2year) are genuinely undefined at the start
    # of the record / sporadic per-region gaps; this mirrors the sequence models'
    # normalize-then-nan_to_num(0) == impute-with-train-mean. Applied here to
    # train/val and (via the adapter) to every test/transfer/embedding frame.
    train_means = X_train.mean(axis=0)   # pandas .mean skips NaN
    n_nan_cols = int((X_train.isna().sum() > 0).sum())
    if n_nan_cols:
        print(f"[Impute] mean-imputing {n_nan_cols} column(s) with NaN "
              f"(train-fit means; consistent with sequence baselines).")
    X_train = impute_with_train_means(X_train, train_means)
    X_val = impute_with_train_means(X_val, train_means)

    # ---- Build + fit the TabTune model ----
    print(f"[TabTune] building {TABTUNE_MODEL_NAME[args.model_type]} (strategy={args.tab_strategy})...")
    pipeline = tabtune_build_and_fit(
        args.model_type, args.tab_strategy, X_train, ytr,
        X_val if args.tab_strategy != "inference" else None,
        yva if args.tab_strategy != "inference" else None,
        config.quantiles, args.seed, args.epochs, args.lr, args.batch_size,
    )

    # ---- Save the fitted TabTune pipeline (mirrors --save_checkpoint_dir on the
    #      sequence scripts). TabTune has no per-epoch Lightning checkpoints; it
    #      persists the whole fitted pipeline to a single file via .save(). We reuse
    #      the shared generate_checkpoint_name for a consistent filename, supplying
    #      the three sequence-only fields with the tab path's true values, and add a
    #      strat: tag so inference / finetune / peft runs don't collide. ----
    if args.save_checkpoint_dir:
        os.makedirs(args.save_checkpoint_dir, exist_ok=True)
        _ckpt_args = copy.copy(args)
        _ckpt_args.aggregation = "daily"        # tab path always summarises the daily source
        _ckpt_args.weight_decay = 0.0           # no weight-decay knob in the tab path
        _ckpt_args.use_residual_trend = False   # trend add-back is a sequence-model concept
        ckpt_name = f"{generate_checkpoint_name(_ckpt_args)}_strat:{args.tab_strategy}_runid:{run_id}"
        ckpt_path = os.path.join(args.save_checkpoint_dir, f"{ckpt_name}.tabtune")
        try:
            pipeline.save(ckpt_path)
            print(f"[Checkpoint] saved TabTune pipeline -> {ckpt_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[Checkpoint Warning] failed to save pipeline: {e}")
    # Point models -> calibrate conformal intervals on the VALIDATION split.
    model_obj = pipeline
    if args.model_type in POINT_MODELS:
        print(f"[TabTune] fitting split-conformal on validation ({X_val.shape[0]} rows)...")
        model_obj = tabtune_fit_conformal(pipeline, X_val, yva, config.quantiles)

    adapter = TabModelAdapter(
        model_obj, args.model_type, config.quantiles,
        ts_names, static_names, n_bins, dm.y_mean, dm.y_std,
        train_means=train_means,
        loss_type=config.loss_type,
    )

    # ---- Test predictions (recursive lags, year-major) -> prediction frame ----
    print("[Predict] generating test prediction frame...")
    adapter._per_year_preds = build_per_year_preds(
        adapter, dm.test_ds, set(int(y) for y in fixed_splits['test_years']),
        args.lag_years, args.use_recursive_lags,
    )

    # ---- Evaluation: the SAME hardened protocol the sequence models use ----
    print("\n[Eval] running tiered evaluation (references on train+val, skill, conformal, tiers)...")
    eval_metrics = run_evaluation(
        adapter, dm, config,
        tiers=args.evaluation_tier,
        run_id=run_id, timestamp=timestamp,
        generate_predictions=args.generate_predictions,
        transfer_countries=args.transfer_countries,
    )

    # ---- Embeddings (optional; both model families expose them for your downstream plans) ----
    if args.extract_embeddings:
        emb_dir = args.embeddings_dir or os.path.join(args.results_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        te_ts, te_st, te_m, _ = arrays(dm.test_ds)
        te_rows, _ = tabularize(te_ts, te_st, te_m, ts_names, static_names, n_bins)
        X_test = impute_with_train_means(pd.DataFrame(te_rows, columns=cols), train_means)
        emb = {
            "train": tabtune_embeddings(pipeline, args.model_type, X_train),
            "val": tabtune_embeddings(pipeline, args.model_type, X_val),
            "test": tabtune_embeddings(pipeline, args.model_type, X_test),
        }
        emb = {k: v for k, v in emb.items() if v is not None}
        if emb:
            meta = dict(
                train_regions=list(dm.train_ds.adm_ids), train_years=dm.train_ds.years.tolist(),
                val_regions=list(dm.val_ds.adm_ids), val_years=dm.val_ds.years.tolist(),
                test_regions=list(dm.test_ds.adm_ids), test_years=dm.test_ds.years.tolist(),
            )
            path = os.path.join(emb_dir, f"{args.model_type}_{args.crop}_{args.country}_{run_id}.npz")
            np.savez_compressed(path, **emb, **{f"meta_{k}": np.array(v, dtype=object) for k, v in meta.items()})
            print(f"[Embeddings] saved {list(emb.keys())} -> {path}")
        else:
            print("[Embeddings] none extracted (accessor unavailable for this backbone).")

    # ---- Report + wandb ----
    final_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
    print_metrics_table(f"FINAL (TAB) RESULTS: {args.crop}-{args.country} [{args.model_type}]", final_metrics)

    # Headline accuracy metrics for ALL THREE splits, named like the sequence
    # baselines (train/val/test over nrmse/r2/smape). TEST comes from the
    # authoritative accuracy tier (matches the CSVs); TRAIN/VAL are computed here
    # from the model's median prediction via the same tabularize+impute path.
    median_idx = config.quantiles.index(0.5) if 0.5 in config.quantiles else len(config.quantiles) // 2

    def _split_point_metrics(dset):
        xs_ts, xs_st, xs_m, y_true = arrays(dset)
        rows_, _ = tabularize(xs_ts, xs_st, xs_m, ts_names, static_names, n_bins)
        Xs = impute_with_train_means(pd.DataFrame(rows_, columns=cols), train_means)
        qv = tabtune_predict_quantiles(model_obj, args.model_type, Xs, config.quantiles)
        y_pred = np.clip(qv[:, median_idx], a_min=0.0, a_max=None)   # yields are non-negative
        return point_metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))

    try:
        tr_pm, va_pm = _split_point_metrics(dm.train_ds), _split_point_metrics(dm.val_ds)
    except Exception as e:  # noqa: BLE001
        print(f"[WandB Warning] train/val metric computation failed: {e}")
        tr_pm, va_pm = {}, {}

    split_metrics = {}
    for mk in ("nrmse", "r2", "smape"):
        if tr_pm.get(mk) is not None:
            split_metrics[f"train/{mk}"] = tr_pm[mk]
        if va_pm.get(mk) is not None:
            split_metrics[f"val/{mk}"] = va_pm[mk]
        tv = eval_metrics.get(f"accuracy/{mk}")
        if tv is not None:
            split_metrics[f"test/{mk}"] = tv

    try:
        import wandb
        wandb_project = args.wandb_project if args.wandb_project else "CYBENCH-LSTF-AAAI2027"
        base_run_name = args.wandb_run_name if args.wandb_run_name else f"{args.model_type}-{args.crop}-{args.country}"
        wandb.init(project=wandb_project,
                   name=(f"{base_run_name}-{run_id}" if args.run_id else base_run_name),
                   group=f"{args.crop}-{args.country}", config=vars(args))
        # Log ONLY the three bare-named headline keys (nrmse / r2 / smape) pulled
        # from the authoritative accuracy tier, matching the convention in
        # tstBaselines.py. Full tiered depth (skill, uncertainty, generalization,
        # operational) is written to the tier CSVs by run_evaluation and NOT
        # flooded to wandb.
        wandb_summary = {
            'nrmse': eval_metrics.get('accuracy/nrmse'),
            'r2':    eval_metrics.get('accuracy/r2'),
            'smape': eval_metrics.get('accuracy/smape'),
        }
        wandb.log({k: v for k, v in wandb_summary.items()
                   if v is not None and isinstance(v, (int, float, np.floating))})
        wandb.finish()
    except Exception as e:  # noqa: BLE001
        print(f"[WandB Warning] skipped wandb logging: {e}")

    print(f"\n{'=' * 70}")
    print(f"Experiment complete: {args.crop}-{args.country} | {args.model_type} "
          f"({args.tab_strategy}, quantiles={q_route})")
    print(f"Test years: {sorted(fixed_splits['test_years'])}")
    print(f"{'=' * 70}\n")
