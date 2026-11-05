"""
--------------------
Author: XYZ 
Description: An updated version for cybench.datasets.alignment.add_cutoff_days. The original function in cybench/datasets/alignment.py doesn't handle "end-of-season" and "three-quarter-of-season" lead times. This file provides an updated version that handles all cases.
            Usage: from process.alignment_patch import patch_alignment
                   patch_alignment()  # Call before loading data

Python version: 3.12.0 
"""

import pandas as pd


def add_cutoff_days_extended(df: pd.DataFrame, lead_time: str):
    """Add a column with cutoff days relative to end of season.

    This is an extended version that handles all lead_time options:
    - "end-of-season": cutoff_days = 0 (forecast at harvest)
    - "three-quarter-of-season": cutoff_days = 75% of season
    - "middle-of-season": cutoff_days = 50% of season
    - "quarter-of-season": cutoff_days = 25% of season

    Args:
        df (pd.DataFrame): time series data
        lead_time (str): lead_time option

    Returns:
        the same DataFrame with column added
    """
    if "day" in lead_time:
        df["cutoff_days"] = int(lead_time.split("-")[0])
    else:
        assert "season" in lead_time
        if lead_time == "end-of-season":
            df["cutoff_days"] = 0
        elif lead_time == "three-quarter-of-season":
            df["cutoff_days"] = (df["season_length"] * 3 // 4).astype(int)
        elif lead_time == "middle-of-season":
            df["cutoff_days"] = (df["season_length"] // 2).astype(int)
        elif lead_time == "quarter-of-season":
            df["cutoff_days"] = (df["season_length"] // 4).astype(int)
        else:
            raise Exception(f'Unrecognized lead time "{lead_time}"')

    return df


def patch_alignment():
    """Monkey-patch cybench.datasets.alignment.add_cutoff_days with our extended version.

    This should be called before loading any data from cybench.datasets.
    """
    try:
        import cybench.datasets.alignment as alignment_module
        alignment_module.add_cutoff_days = add_cutoff_days_extended
        print("[Patch] Successfully extended cybench.datasets.alignment.add_cutoff_days")
    except ImportError as e:
        print(f"[Patch Warning] Could not import cybench.datasets.alignment: {e}")
        print("[Patch Warning] The patch will be applied when the module is imported")
        # Store the extended function for later application
        import sys
        sys.modules['_alignment_patch'] = type(sys)('alignment_patch')
        sys.modules['_alignment_patch'].add_cutoff_days_extended = add_cutoff_days_extended
        sys.modules['_alignment_patch']._patch_pending = True


# Auto-patch on import if cybench.datasets.alignment is already imported
def _auto_patch():
    try:
        import sys
        if 'cybench.datasets.alignment' in sys.modules:
            import cybench.datasets.alignment as alignment_module
            if alignment_module.add_cutoff_days.__name__ != 'add_cutoff_days_extended':
                alignment_module.add_cutoff_days = add_cutoff_days_extended
                print("cybench.datasets.alignment.add_cutoff_days (module already imported)")
    except Exception:
        pass


_auto_patch()


def test_alignment_patch():
    """
    Test that the alignment patch is correctly applied and handles all forecast types.

    This can be called independently or imported and called from other modules:
        from cybench.process.alignment_patch import test_alignment_patch
        test_alignment_patch()

    Returns:
        bool: True if patch is working correctly, False otherwise
    """
    import pandas as pd

    print("\n" + "=" * 70)
    print("ALIGNMENT PATCH TEST")
    print("=" * 70)

    # Ensure patch is applied
    patch_alignment()

    try:
        import cybench.datasets.alignment as alignment_module

        # Check patch was applied
        if alignment_module.add_cutoff_days.__name__ != 'add_cutoff_days_extended':
            print(f"\n FAIL: Patch not applied")
            print(f" Function name: {alignment_module.add_cutoff_days.__name__}")
            return False

        print(f"\n Patch status: APPLIED")
        print(f" Function name: {alignment_module.add_cutoff_days.__name__}")

        # Test each forecast type
        test_cases = [
            ("end-of-season", 100, 0),
            ("three-quarter-of-season", 100, 75),
            ("middle-of-season", 100, 50),
            ("quarter-of-season", 100, 25),
        ]

        print(f"\n[TEST] Testing add_cutoff_days with all forecast types:")
        print(f"\n  {'Lead Time':30s} | {'Season Len':>10s} | {'Expected':>9s} | {'Actual':>7s} | {'Status'}")
        print("  " + "-" * 75)

        all_passed = True
        for lead_time, season_length, expected_cutoff in test_cases:
            df = pd.DataFrame({'season_length': [season_length] * 5})
            result = alignment_module.add_cutoff_days(df, lead_time)
            actual_cutoff = result['cutoff_days'].iloc[0]

            status = "✓" if actual_cutoff == expected_cutoff else "✗"
            print(f" {lead_time:30s} | {season_length:>10d} | {expected_cutoff:>9d} | {actual_cutoff:>7d} | {status}")

            if actual_cutoff != expected_cutoff:
                all_passed = False

        if all_passed:
            print(f"\n PASS: All forecast types work correctly")
        else:
            print(f"\n FAIL: Some forecast types failed")

        return all_passed

    except Exception as e:
        print(f"\n ERROR: {e}")
        return False


def verify_forecast_horizon_config(config, fold_idx=None, sample_data=None,
                                   resolution_bins=None, active_resolution=None):
    """
    Verifies and display forecast horizon configuration for a given fold. Shows how data_fraction translates to actual sequence lengths.
    Called at each fold during walk-forward validation to confirm the forecast horizon is being applied correctly.

    Args:
        config: Model config object with data_fraction and aggregation attributes
        fold_idx: Optional fold index for display
        sample_data: Optional dict with 'sos_date' and 'eos_date' to show dates
        resolution_bins: Optional {resolution_name: n_bins} map (tabular path only).
            When given, the daily/weekly/dekad list is shown as the LOAD source and a
            second "Tabular resolution" block lists the post-load binning with the
            active one marked. Sequence scripts omit it -> output is unchanged.
        active_resolution: The --resolution actually used (key of resolution_bins);
            marked "← This one activated" in the tabular-resolution block.

    Returns:
        dict: Summary of forecast horizon configuration
    """
    from cybench.process.featureEngineering import MAX_SEQ_LENS

    # Get the data_fraction from config
    data_fraction = getattr(config, 'data_fraction', 1.0)
    aggregation = getattr(config, 'aggregation', 'dekad')

    # Map data_fraction back to forecast_type name for display
    fraction_to_type = {
        1.0: ('end-of-season', '100%'),
        0.75: ('three-quarter-of-season', '75%'),
        0.5: ('middle-of-season', '50%'),
        0.25: ('quarter-of-season', '25%'),
    }
    forecast_type, percentage = fraction_to_type.get(
        data_fraction,
        (f'custom ({data_fraction:.0%})', f'{data_fraction:.0%}')
    )

    # Calculate expected sequence lengths for each aggregation
    agg_lengths = {}
    for agg, max_len in MAX_SEQ_LENS.items():
        agg_lengths[agg] = max(1, int(max_len * data_fraction))

    # Print diagnostic information
    fold_str = f"Fold {fold_idx} - " if fold_idx is not None else ""

    print(f"\n{'─' * 70}")
    print(f"[{fold_str}Forecast Horizon Configuration]")
    print(f" forecast_type:  {forecast_type}")
    print(f" data_fraction:  {data_fraction} ({percentage} of season)")
    print(f" aggregation:    {aggregation}")
    print(f"\n  Expected sequence lengths (after trimming):")

    for agg, length in agg_lengths.items():
        max_len = MAX_SEQ_LENS[agg]
        if resolution_bins is not None:
            # Tabular path: daily/weekly/dekad is the LOAD source; the active
            # resolution is a post-load binning, marked in the block below.
            indicator = "   ← loaded (source)" if agg == aggregation else ""
        else:
            indicator = " ← This one activated" if agg == aggregation else ""
        print(f" {agg:>8s}: {length:>3d} / {max_len} time steps{indicator}")

    # Tabular path only: show the post-load per-channel binning with the active one.
    if resolution_bins is not None:
        print(f"\n  Tabular resolution (daily series binned per channel):")
        for res, nbins in resolution_bins.items():
            arrow = "   ← This one activated" if res == active_resolution else ""
            unit = "bin" if nbins == 1 else "bins"
            print(f" {res:>8s}: {nbins:>3d} {unit}{arrow}")

    # Show date range if sample_data provided
    if sample_data and 'sos_date' in sample_data and 'eos_date' in sample_data:
        sos = sample_data['sos_date']
        eos = sample_data['eos_date']

        # Calculate cutoff date based on data_fraction
        if isinstance(sos, pd.Timestamp) and isinstance(eos, pd.Timestamp):
            season_days = (eos - sos).days
            cutoff_days = int(season_days * data_fraction)
            cutoff_date = sos + pd.Timedelta(days=cutoff_days)

            print(f"\n Date range (sample):")
            print(f" Start of season: {sos.strftime('%Y-%m-%d')}")
            print(f" End of season:   {eos.strftime('%Y-%m-%d')}")
            print(f" Forecast at:     {cutoff_date.strftime('%Y-%m-%d')} ({cutoff_days} days from SOS)")

    # Verify config has the expected data_fraction
    expected_fractions = [1.0, 0.75, 0.5, 0.25]
    if data_fraction not in expected_fractions:
        print(f"\n WARNING: Non-standard data_fraction={data_fraction}")
    else:
        print(f"\n Configuration valid")

    print(f"{'─' * 70}")

    return {
        'data_fraction': data_fraction,
        'forecast_type': forecast_type,
        'aggregation': aggregation,
        'sequence_lengths': agg_lengths,
        'resolution_bins': resolution_bins,
        'active_resolution': active_resolution,
    }
