"""Binning logic for numeric and categorical feature transformation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitting Logic
# ---------------------------------------------------------------------------

def fit_numeric_bins(
    series: pd.Series,
    n_bins: int,
    special_codes: list[int | float] | None = None,
) -> dict[str, Any]:
    """Fit quantile-based bins for numeric data while isolating special codes.

    Parameters
    ----------
    series : pd.Series
        Numeric input data to be binned.
    n_bins : int
        Target number of bins for the quantile split.
    special_codes : list of (int or float) or None
        Values to be treated as independent bins (e.g., error codes, missing flags).

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'is_numeric', 'special_codes', and the bin 'bins' array.
    """
    binning_data = {
        "is_numeric": True,
        "special_codes": special_codes or [],
        "bins": None,
    }

    clean_series = series
    if special_codes:
        clean_series = series[~series.isin(special_codes)]

    if clean_series.empty:
        binning_data["bins"] = np.array([-np.inf, np.inf])
        return binning_data

    try:
        _, bins = pd.qcut(clean_series, q=n_bins, duplicates="drop", retbins=True)
    except (ValueError, IndexError):
        # Fallback to standard cut if qcut fails due to distribution issues
        _, bins = pd.cut(
            clean_series, bins=min(n_bins, clean_series.nunique()), retbins=True
        )

    bins = bins.tolist()
    bins[0], bins[-1] = -np.inf, np.inf
    binning_data["bins"] = np.array(bins)

    return binning_data


def fit_categorical_bins(
    series: pd.Series,
    special_codes: list[Any] | None = None,
) -> dict[str, Any]:
    """Fit categorical mapping by identifying unique labels and isolating special codes.

    Parameters
    ----------
    series : pd.Series
        Categorical or object-type input data.
    special_codes : list of Any or None
        Values to be treated as independent bins.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'is_numeric', 'special_codes', and 'categories' list.
    """
    s = series.fillna("missing").astype(str)

    binning_data = {
        "is_numeric": False,
        "special_codes": special_codes or [],
        "categories": s.unique().tolist(),
    }
    return binning_data


# ---------------------------------------------------------------------------
# Application Logic
# ---------------------------------------------------------------------------

def apply_bins(
    series: pd.Series,
    config: dict[str, Any],
) -> np.ndarray:
    """Apply fitted binning rules to a new series to generate bin IDs.

    Parameters
    ----------
    series : pd.Series
        Input data to transform.
    config : dict[str, Any]
        Configuration dictionary generated during the fit phase.

    Returns
    -------
    np.ndarray
        Array of integer bin IDs. Special codes are mapped to negative IDs.
    """
    s = series.copy()
    output_codes = np.full(s.shape, -1, dtype=int)

    # Handle Special Codes
    special_mask = s.isin(config["special_codes"])
    if special_mask.any():
        mapping = {val: -(i + 2) for i, val in enumerate(config["special_codes"])}
        output_codes[special_mask] = s[special_mask].map(mapping)

    remaining_mask = ~special_mask
    if not remaining_mask.any():
        return output_codes

    # Handle Normal Data
    if config["is_numeric"]:
        numeric_s = pd.to_numeric(s[remaining_mask], errors="coerce")
        # pd.cut with labels=False returns float array with NaNs for out-of-bounds/NaNs
        codes = pd.cut(numeric_s, bins=config["bins"], labels=False, include_lowest=True)
        
        output_codes[remaining_mask] = np.nan_to_num(codes, nan=-1).astype(int)
    else:
        cat_s = s[remaining_mask].fillna("missing").astype(str)
        lookup = {cat: i for i, cat in enumerate(config["categories"])}
        output_codes[remaining_mask] = np.array([lookup.get(x, -1) for x in cat_s])

    return output_codes


def merge_non_significant_bins(
    codes: np.ndarray,
    y: np.ndarray,
    min_pct: float,
) -> np.ndarray:
    """Merge bins that do not meet the minimum population threshold.

    Parameters
    ----------
    codes : np.ndarray
        Array of bin IDs.
    y : np.ndarray
        Binary target array for consistency (unused in simple merging but kept for API).
    min_pct : float
        Minimum percentage of total population required per bin.

    Returns
    -------
    np.ndarray
        Array of bin IDs with small bins merged into their nearest neighbor.
    """
    unique_codes = np.unique(codes)
    if len(unique_codes) <= 1:
        return codes

    total_count = len(codes)
    new_codes = codes.copy()

    for code in unique_codes:
        mask = (new_codes == code)
        if (mask.sum() / total_count) < min_pct:
            # Find closest bin ID to merge into
            dist = np.abs(unique_codes - code)
            dist[unique_codes == code] = np.inf
            nearest_neighbor = unique_codes[np.argmin(dist)]
            new_codes[mask] = nearest_neighbor

    return new_codes