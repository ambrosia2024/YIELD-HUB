from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import logging

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

try:
    import pymannkendall as trend_mk
    HAS_PYMANNKENDALL = True
except ImportError:
    HAS_PYMANNKENDALL = False
    trend_mk = None
    logging.warning("pymannkendall not installed. Trend detection will be disabled. "
                   "Install with: pip install pymannkendall")

from cybench.config import (
    GDD_BASE_TEMP, GDD_UPPER_LIMIT, LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

class TrendModel:
    """
    Temporal trend estimator using OLS regression and Mann-Kendall significance test.

    Decomposes yields into: yield = trend(location, year) + residual
    - trend   captures technology drift / climate change over years
    - residual captures weather-driven cross-sectional variation

    The neural model learns residuals; trend is added back at inference.
    """
    # Minimum years required to fit a meaningful OLS trend
    MIN_TREND_WINDOW_SIZE = 5
    # Maximum window size to balance trend stability and temporal adaptability
    MAX_TREND_WINDOW_SIZE = 10

    def __init__(self):
        self._train_df = None
        # Cache of computed trends keyed by (loc, year). _predict_trend is called
        # once per training sample per batch per epoch; the trend is deterministic
        # given the fitted train_df, so memoising avoids repeating the Mann-Kendall
        # window search every call. Reset on every fit().
        self._trend_cache = {}

    def fit(self, dataset, **fit_params) -> Tuple['TrendModel', Dict]:
        """
        Fit trend model on training dataset.

        Args:
            dataset: Dataset with location, year, target information
            **fit_params: Additional fitting parameters (unused)

        Returns:
            Tuple of (self, empty dict for sklearn compatibility)
        """
        data_items = list(dataset) if hasattr(dataset, '__iter__') else dataset
        rows = [{KEY_LOC: item[KEY_LOC], KEY_YEAR: item[KEY_YEAR], KEY_TARGET: item[KEY_TARGET]}
                for item in data_items]
        self._train_df = pd.DataFrame(rows)
        self._trend_cache = {}
        return self, {}

    def _estimate_trend(self, trend_x: list, trend_y: list, test_x: int) -> float:
        """
        Estimate trend at test_x using OLS regression.

        Args:
            trend_x: List of years for trend fitting
            trend_y: List of yield values corresponding to trend_x years
            test_x: Year to predict trend for

        Returns:
            float: Predicted trend value at test_x
        """
        if len(trend_y) < self.MIN_TREND_WINDOW_SIZE:
            return np.mean(trend_y)
        X = add_constant(trend_x)
        model = OLS(trend_y, X).fit()
        pred_x = add_constant(np.array([test_x]).reshape(1, 1), has_constant="add")
        return model.predict(pred_x)[0]

    def _find_optimal_trend_window(self, train_labels: np.ndarray, window_years: list,
                                   extend_forward: bool = False) -> Optional[list]:
        """
        Find optimal trend window using Mann-Kendall significance test.

        Args:
            train_labels: Array of (year, yield) pairs
            window_years: Available years for window selection
            extend_forward: If True, use earliest years; if False, use latest years

        Returns:
            Optimal window years list, or None if no significant trend found
        """
        # FIXED: Check for pymannkendall availability with clear error message
        if not HAS_PYMANNKENDALL:
            raise ImportError(
                "pymannkendall is required for trend detection but is not installed. "
                "Install it with: pip install pymannkendall"
            )

        min_p = float("inf")
        opt_trend_years = None
        for i in range(self.MIN_TREND_WINDOW_SIZE,
                       min(self.MAX_TREND_WINDOW_SIZE, len(window_years)) + 1):
            trend_x = window_years[:i] if extend_forward else window_years[-i:]
            trend_y = train_labels[np.isin(train_labels[:, 0], trend_x)][:, 1]
            if len(trend_y) == 0:
                continue
            result = trend_mk.original_test(trend_y)
            if result.h and result.p < min_p:
                min_p = result.p
                opt_trend_years = trend_x
        return opt_trend_years

    def _predict_trend(self, test_data):
        trend_predictions = np.zeros((len(test_data), 1))
        for i, item in enumerate(test_data):
            loc, test_year = item[KEY_LOC], item[KEY_YEAR]
            cache_key = (loc, int(test_year))
            if cache_key in self._trend_cache:
                trend_predictions[i, 0] = self._trend_cache[cache_key]
                continue

            sel = self._train_df[self._train_df[KEY_LOC] == loc]
            if sel.empty:
                trend = self._train_df[KEY_TARGET].mean()
            else:
                train_labels = sel[[KEY_YEAR, KEY_TARGET]].values
                train_years = sorted(sel[KEY_YEAR].unique())

                # LEAVE-THAT-YEAR-OUT (Option A). When test_year is itself one of the
                # training years -- which is the case for EVERY sample while building
                # residual targets during training -- we must exclude that year's own
                # observation before fitting the trend. Otherwise the trend equals the
                # label, the residual target y - trend collapses to 0, and the network
                # learns to predict 0 (ignoring all inputs). Dropping test_year makes
                # the trend a genuine smooth fit that does NOT contain the label, and
                # mirrors inference exactly (where the queried year is never a train
                # year). For true inference (test_year not in train_years) the two
                # lines below are a no-op, so the evaluation-reference behaviour is
                # unchanged.
                if test_year in train_years:
                    keep = train_labels[:, 0] != test_year
                    train_labels = train_labels[keep]
                    train_years = [y for y in train_years if y != test_year]

                if len(train_years) == 0:
                    trend = sel[KEY_TARGET].mean()
                else:
                    lt = [y for y in train_years if y < test_year]
                    gt = [y for y in train_years if y > test_year]
                    if (len(lt) < self.MIN_TREND_WINDOW_SIZE and
                            len(gt) < self.MIN_TREND_WINDOW_SIZE):
                        # Fallback mean uses the year-excluded labels to stay
                        # leave-one-out consistent.
                        trend = float(np.mean(train_labels[:, 1]))
                    else:
                        # Interpolate between backward and forward trends instead of cascading
                        # provides better estimates when test year falls between training data blocks
                        lt_trend, gt_trend = None, None

                        if len(lt) >= self.MIN_TREND_WINDOW_SIZE:
                            wy = self._find_optimal_trend_window(train_labels, lt, False)
                            if wy is not None:
                                wv = train_labels[np.isin(train_labels[:, 0], wy)][:, 1]
                                lt_trend = self._estimate_trend(list(wy), list(wv), test_year)

                        if len(gt) >= self.MIN_TREND_WINDOW_SIZE:
                            wy = self._find_optimal_trend_window(train_labels, gt, True)
                            if wy is not None:
                                wv = train_labels[np.isin(train_labels[:, 0], wy)][:, 1]
                                gt_trend = self._estimate_trend(list(wy), list(wv), test_year)

                        # Interpolate if both directions available, otherwise use whichever is available
                        if lt_trend is not None and gt_trend is not None:
                            # Weight by proximity: years closer to test_year get more weight
                            lt_dist = test_year - max(lt)
                            gt_dist = min(gt) - test_year
                            total = lt_dist + gt_dist
                            trend = (gt_dist / total) * lt_trend + (lt_dist / total) * gt_trend
                        elif lt_trend is not None:
                            trend = lt_trend
                        elif gt_trend is not None:
                            trend = gt_trend
                        else:
                            trend = float(np.mean(train_labels[:, 1]))

            self._trend_cache[cache_key] = trend
            trend_predictions[i, 0] = trend
        return trend_predictions

    def predict(self, dataset) -> Tuple[np.ndarray, Dict]:
        """
        Predict trend values for dataset.

        Args:
            dataset: Dataset with location, year information

        Returns:
            Tuple of (trend_predictions array, empty dict for sklearn compatibility)
        """
        items = list(dataset) if hasattr(dataset, '__iter__') else dataset
        return self._predict_trend(items).flatten(), {}