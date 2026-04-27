"""Binning logic for numeric and categorical feature transformation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


SUPPORTED_BINNING_METHODS = {"quantile", "tree"}


def _finalize_numeric_bins(bins: list[float] | np.ndarray) -> np.ndarray:
    """Return sorted bin edges with open-ended outer boundaries."""
    bins_array = np.asarray(bins, dtype=float)
    bins_array = bins_array[~np.isnan(bins_array)]
    bins_array = np.unique(bins_array)

    if len(bins_array) < 2:
        return np.array([-np.inf, np.inf])

    bins_array[0], bins_array[-1] = -np.inf, np.inf
    return bins_array


def _fit_quantile_bins(clean_series: pd.Series, n_bins: int) -> np.ndarray:
    try:
        _, bins = pd.qcut(clean_series, q=n_bins, duplicates="drop", retbins=True)
    except (ValueError, IndexError):
        # Fallback to standard cut if qcut fails due to distribution issues.
        _, bins = pd.cut(
            clean_series, bins=min(n_bins, clean_series.nunique()), retbins=True
        )

    return _finalize_numeric_bins(bins)


def _fit_tree_bins(
    clean_series: pd.Series,
    y: np.ndarray,
    n_bins: int,
    min_bin_pct: float | None,
    random_state: int | None,
    tree_criterion: str,
    tree_max_depth: int | None,
    tree_min_samples_leaf: int | float | None,
    tree_min_samples_split: int | float,
) -> np.ndarray:
    valid_mask = clean_series.notna().to_numpy()
    x = clean_series[valid_mask].to_numpy(dtype=float).reshape(-1, 1)
    y_clean = np.asarray(y)[valid_mask]

    if len(x) == 0 or len(np.unique(x)) <= 1 or len(np.unique(y_clean)) < 2 or n_bins <= 1:
        return np.array([-np.inf, np.inf])

    min_samples_leaf = tree_min_samples_leaf
    if min_samples_leaf is None:
        min_samples_leaf = min_bin_pct if min_bin_pct else 1

    tree = DecisionTreeClassifier(
        criterion=tree_criterion,
        max_depth=tree_max_depth,
        max_leaf_nodes=n_bins,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=tree_min_samples_split,
        random_state=random_state,
    )
    tree.fit(x, y_clean)

    split_mask = tree.tree_.children_left != tree.tree_.children_right
    thresholds = tree.tree_.threshold[split_mask]

    if len(thresholds) == 0:
        return np.array([-np.inf, np.inf])

    return np.concatenate(([-np.inf], np.sort(np.unique(thresholds)), [np.inf]))


def fit_numeric_bins(
    series: pd.Series,
    n_bins: int,
    special_codes: list[int | float] | None = None,
    *,
    binning_method: str = "quantile",
    y: np.ndarray | None = None,
    min_bin_pct: float | None = None,
    random_state: int | None = 42,
    tree_criterion: str = "gini",
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int | float | None = None,
    tree_min_samples_split: int | float = 2,
) -> dict[str, Any]:
    """Fit numeric bins while isolating special codes.

    Parameters
    ----------
    series : pd.Series
        Numeric input data to be binned.
    n_bins : int
        Target number of bins for the numeric split.
    special_codes : list of (int or float) or None
        Values to be treated as independent bins (e.g., error codes, missing flags).
    binning_method : {"quantile", "tree"}, default="quantile"
        Numeric binning strategy.
    y : np.ndarray or None
        Binary target array. Required when `binning_method="tree"`.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'is_numeric', 'special_codes', and the 'bins' boundaries.
    """
    binning_data = {
        "is_numeric": True,
        "special_codes": special_codes or [],
        "binning_method": binning_method,
        "bins": None,
    }

    special_mask = series.isin(special_codes or [])
    clean_series = pd.to_numeric(series[~special_mask], errors="coerce")

    if clean_series.dropna().empty:
        binning_data["bins"] = np.array([-np.inf, np.inf])
        return binning_data

    if binning_method == "quantile":
        bins = _fit_quantile_bins(clean_series.dropna(), n_bins)
    elif binning_method == "tree":
        if y is None:
            raise ValueError("Target y is required when binning_method='tree'.")
        clean_y = np.asarray(y)[(~special_mask).to_numpy()]
        bins = _fit_tree_bins(
            clean_series,
            clean_y,
            n_bins,
            min_bin_pct,
            random_state,
            tree_criterion,
            tree_max_depth,
            tree_min_samples_leaf,
            tree_min_samples_split,
        )
    else:
        raise ValueError(
            f"binning_method must be one of {sorted(SUPPORTED_BINNING_METHODS)}, "
            f"got {binning_method!r}."
        )

    binning_data["bins"] = bins

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
        "binning_method": "categorical",
        "categories": s.unique().tolist(),
    }
    return binning_data


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
    output_bins = np.full(s.shape, -1, dtype=int)

    # Handle Special Codes
    special_mask = s.isin(config["special_codes"])
    if special_mask.any():
        mapping = {val: -(i + 2) for i, val in enumerate(config["special_codes"])}
        output_bins[special_mask] = s[special_mask].map(mapping)

    remaining_mask = ~special_mask
    if not remaining_mask.any():
        return output_bins

    # Handle Normal Data
    if config["is_numeric"]:
        numeric_s = pd.to_numeric(s[remaining_mask], errors="coerce")
        # pd.cut with labels=False returns float array with NaNs for out-of-bounds/NaNs
        bins = pd.cut(numeric_s, bins=config["bins"], labels=False, include_lowest=True)
        
        # Safely convert to int after replacing NaNs with -1
        output_bins[remaining_mask] = np.nan_to_num(bins, nan=-1).astype(int)
    else:
        cat_s = s[remaining_mask].fillna("missing").astype(str)
        lookup = {cat: i for i, cat in enumerate(config["categories"])}
        output_bins[remaining_mask] = np.array([lookup.get(x, -1) for x in cat_s])

    return output_bins


def merge_non_significant_bins(
    bin_ids: np.ndarray,
    min_pct: float,
    *,
    protect_special_bins: bool = True,
) -> np.ndarray:
    """Merge bins that do not meet the minimum population threshold.

    Parameters
    ----------
    bin_ids : np.ndarray
        Array of bin IDs.
    min_pct : float
        Minimum percentage of total population required per bin.

    Returns
    -------
    np.ndarray
        Array of bin IDs with small bins merged into their nearest neighbor.
    """
    unique_bins = np.unique(bin_ids)
    if len(unique_bins) <= 1:
        return bin_ids

    total_count = len(bin_ids)
    new_bin_ids = bin_ids.copy()

    for bin_id in unique_bins:
        if protect_special_bins and bin_id < 0:
            continue

        mask = (new_bin_ids == bin_id)
        if (mask.sum() / total_count) < min_pct:
            # Find closest bin ID to merge into
            # Use float to allow np.inf assignment
            candidate_bins = unique_bins[unique_bins >= 0] if protect_special_bins else unique_bins
            candidate_bins = candidate_bins[candidate_bins != bin_id]
            if len(candidate_bins) == 0:
                continue

            dist = np.abs(candidate_bins.astype(float) - float(bin_id))
            nearest_neighbor = candidate_bins[np.argmin(dist)]
            new_bin_ids[mask] = nearest_neighbor

    return new_bin_ids

def get_bin_labels(config: dict[str, Any], series: pd.Series = None, bin_ids: np.ndarray = None) -> dict[int, str]:    
    """Converts bin configurations into human-readable range strings."""
    labels = {}
    if config["is_numeric"]:
        for i, val in enumerate(config["special_codes"]):
            labels[-(i + 2)] = f"Special: {val}"
        bins = config["bins"]
        for i in range(len(bins) - 1):
            labels[i] = f"[{bins[i]}, {bins[i+1]})"
        labels[-1] = "Missing/Other"
    else:
        s_str = series.fillna("missing").astype(str)
        temp_df = pd.DataFrame({"label": s_str, "bid": bin_ids})
        labels = (
            temp_df.groupby("bid")["label"]
            .unique()
            .apply(lambda x: ", ".join(sorted(x)))
            .to_dict()
        )
    return labels
